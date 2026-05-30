"""추천 결과(직무·트랙·학습 로드맵) 각 영역에 자연어 근거 단락을 부착한다.

상위 노드(직무 매칭·트랙 시너지·학습 로드맵)가 state 에 채워 둔 도메인
데이터만 LLM 컨텍스트로 사용한다. 노드는 별도 검색 호출을 하지 않으며,
상위 노드가 산출한 검증된 사실 외 정보 생성은 시스템 프롬프트로 차단한다.

처리 흐름 (6단계):
    1. 입력 수집 — ``recommended_jobs`` / ``primary_combos`` /
       ``secondary_combos`` / ``roadmap`` / ``normalized_profile``.
    2. 빈 영역 판정 — 세 영역 모두 비어 있으면 정적 안내로 즉시 반환하고
       LLM 을 호출하지 않는다 (빈 컨텍스트 호출 비용·환각 방지).
    3. 컨텍스트 직렬화 — 영역별 화이트리스트 필드만 사람-읽기 좋은 문자열로
       추려서 프롬프트 변수로 흐른다. 학습 로드맵은 단계별 뷰 외에 학기별
       대표 단계 (학기 부제용) · 과목별 단계 (인과 흐름용) · 인과 흐름 anchor
       (최상위 직무 + 주 추천 트랙 조합) 컨텍스트도 함께 직렬화한다.
    4. 캐비잇 결정 — 직무 후보 중 fallback 채택분이 있으면 ``yes`` 플래그를
       흘려 시스템 프롬프트의 추가 입력 안내 한 문장이 부착되도록 한다.
    5. LLM 호출 — ``ChatPromptTemplate.format_messages`` 로 메시지를 만들어
       구조화 출력을 강제하는 ``LLMClient`` 에 단일 invoke. 출력은 영역별
       단락 외에 학기 카드 헤더용 부제 (``semester_subtitles``) 와 과목 상세
       모달용 인과 흐름 (``course_flows``) 두 종류를 함께 담는다.
    6. state 부분 반환 — ``Explanation.model_dump(mode="json")`` 결과 dict 만.

부작용 격리:
    - LLM 호출은 ``__call__`` 의 단 1곳. 직렬화·빈 영역 판정·캐비잇 결정은
      모두 module-level 순수 함수로 분리되어 mock 없이 테스트 가능하다.
"""

from __future__ import annotations

from typing import Any

from tracktory.graph.models import Explanation
from tracktory.graph.state import GraphState
from tracktory.llm.llm_client import LLMClient
from tracktory.prompts.explanation import EXPLANATION_PROMPT

_EMPTY_CONTEXT_MARKER = "데이터 없음"
_EMPTY_ALL_FALLBACK_TEXT = "이번 추천 결과가 비어 있어 설명을 생성하지 않았습니다."

# 학습 깊이 단계 식별자 → 사용자 표시용 단계명. 학기 부제·과목 인과 흐름의
# {단계명} 슬롯에 영어 식별자가 아니라 한국어 라벨이 흐르도록 노드에서 미리
# 치환한다 (LLM 이 식별자→한국어 변환을 추측하지 않도록 결정론적 매핑).
_STAGE_LABELS: dict[str, str] = {
    "foundation": "기초",
    "core": "핵심",
    "application": "응용",
    "industry": "산학",
}
# 단계 단조 증가 순서. 한 학기에 여러 단계 과목이 섞일 때 대표 단계를
# 고를 경우의 tie-break 기준 (이른 단계 우선).
_STAGE_ORDER = ["foundation", "core", "application", "industry"]


# ---------------------------------------------------------------------------
# 순수 계산 — 빈 영역 판정 / 직렬화 / 캐비잇 결정
# ---------------------------------------------------------------------------


def _jobs_empty(jobs: list[dict[str, Any]] | None) -> bool:
    return not jobs


def _tracks_empty(
    primary: list[dict[str, Any]] | None,
    secondary: list[dict[str, Any]] | None,
) -> bool:
    return not primary and not secondary


def _roadmap_empty(roadmap: dict[str, Any] | None) -> bool:
    """로드맵 dict 자체가 없거나, 4 단계 모두 추천 과목이 비어 있으면 빈 영역."""
    if not roadmap:
        return True
    stages = roadmap.get("stages") or []
    return all(not (stage.get("courses") or []) for stage in stages)


def _serialize_jobs(jobs: list[dict[str, Any]] | None) -> str:
    """직무 후보를 화이트리스트 필드만 추려 사람-읽기 좋은 문자열로 직렬화.

    LLM 환각 표면을 좁히기 위해 ``job_id`` 등 사용자에게 노출되지 않는
    내부 식별자는 제외하고, 트랙·로드맵 단락의 cross-reference 가 필요한
    필드만 흘린다.
    """
    if not jobs:
        return _EMPTY_CONTEXT_MARKER
    lines: list[str] = []
    for job in jobs:
        tech = ", ".join(job.get("tech_stacks") or []) or "(데이터 없음)"
        competencies = ", ".join(job.get("competency_tags") or []) or "(데이터 없음)"
        # 설명 근거로는 이수 과목 부스팅으로 굴절되지 않은 검색 원시 점수
        # (similarity) 를 노출한다. 후보 순위·표시는 match_score(부스팅 반영)
        # 기준이지만, 관심사-직무 적합도의 근거 설명에는 순수 검색 유사도가 맞다.
        similarity = job.get("similarity", 0.0)
        lines.append(
            f"- {job.get('job_name', '(이름 없음)')}"
            f" / 유사도={similarity:.2f}"
            f" / 기술스택=[{tech}]"
            f" / 역량=[{competencies}]"
        )
    return "\n".join(lines)


def _serialize_combos(
    primary: list[dict[str, Any]] | None,
    secondary: list[dict[str, Any]] | None,
) -> str:
    """주 추천 + 보조 추천을 묶어 슬롯 분류와 함께 직렬화."""
    combos = list(primary or []) + list(secondary or [])
    if not combos:
        return _EMPTY_CONTEXT_MARKER
    lines: list[str] = []
    for ranked in combos:
        combo = ranked.get("combo") or {}
        track_a = (combo.get("track_a") or {}).get("track_name", "(이름 없음)")
        track_b = (combo.get("track_b") or {}).get("track_name", "(이름 없음)")
        slot_type = ranked.get("slot_type", "(분류 없음)")
        score = ranked.get("synergy_score", 0.0)
        lines.append(f"- {track_a} + {track_b} / 슬롯={slot_type} / 시너지={score:.2f}")
    return "\n".join(lines)


def _serialize_roadmap(roadmap: dict[str, Any] | None) -> str:
    """학습 로드맵 4 단계를 단계별 과목 리스트와 함께 직렬화."""
    if not roadmap:
        return _EMPTY_CONTEXT_MARKER
    stages = roadmap.get("stages") or []
    if not stages:
        return _EMPTY_CONTEXT_MARKER
    lines: list[str] = []
    for stage in stages:
        courses = stage.get("courses") or []
        if not courses:
            lines.append(f"[{stage.get('stage', '(단계 없음)')}] (추천 과목 없음)")
            continue
        course_strs = [
            f"{course.get('course_name', '(이름 없음)')}(우선순위 {course.get('priority', 0)})"
            for course in courses
        ]
        lines.append(f"[{stage.get('stage', '(단계 없음)')}] " + ", ".join(course_strs))
    return "\n".join(lines)


def _course_stage_map(roadmap: dict[str, Any]) -> dict[str, str]:
    """과목 식별자 → 학습 깊이 단계 식별자 매핑을 단계별 뷰에서 추출.

    학기 분산 뷰 (``semesters``) 의 과목 단위에는 단계 정보가 없으므로, 단계별
    뷰 (``stages``) 를 단일 진실원으로 삼아 과목→단계를 역인덱싱한다.
    """
    mapping: dict[str, str] = {}
    for stage in roadmap.get("stages") or []:
        stage_name = stage.get("stage")
        if not stage_name:
            continue
        for course in stage.get("courses") or []:
            course_id = course.get("course_id")
            if course_id:
                mapping[course_id] = stage_name
    return mapping


def _dominant_stage(course_ids: list[str], stage_map: dict[str, str]) -> str | None:
    """학기 내 과목들의 대표 학습 깊이 단계를 고른다.

    한 학기에 여러 단계 과목이 섞일 수 있어 (기초 마지막 + 핵심 첫 과목 등),
    최빈 단계를 대표로 삼고, 동률이면 더 이른 단계를 택한다. 매핑된 단계가
    하나도 없으면 ``None`` (해당 학기는 부제 생성 대상에서 제외).
    """
    counts: dict[str, int] = {}
    for course_id in course_ids:
        stage = stage_map.get(course_id)
        if stage:
            counts[stage] = counts.get(stage, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda s: (-counts[s], _STAGE_ORDER.index(s)))


def _serialize_semesters(roadmap: dict[str, Any] | None) -> str:
    """학기 분산 plan 을 학기별 대표 단계명·과목과 함께 직렬화.

    학기 부제 출력의 입력 컨텍스트. 각 학기의 대표 단계명을 한국어 라벨로
    미리 치환해 흘려 LLM 이 ``{단계명}`` 슬롯을 그대로 사용하도록 한다.
    """
    if not roadmap:
        return _EMPTY_CONTEXT_MARKER
    semesters = roadmap.get("semesters") or []
    if not semesters:
        return _EMPTY_CONTEXT_MARKER
    stage_map = _course_stage_map(roadmap)
    lines: list[str] = []
    for semester in semesters:
        courses = semester.get("courses") or []
        course_ids = [c.get("course_id") for c in courses if c.get("course_id")]
        dominant = _dominant_stage(course_ids, stage_map)
        if dominant is None:
            continue
        label = _STAGE_LABELS.get(dominant, dominant)
        names = ", ".join(c.get("course_name", "(이름 없음)") for c in courses) or "(과목 없음)"
        lines.append(
            f"- {semester.get('semester')}학기({semester.get('grade')}학년):"
            f" 단계명={label} / 과목=[{names}]"
        )
    if not lines:
        return _EMPTY_CONTEXT_MARKER
    return "\n".join(lines)


def _serialize_courses(roadmap: dict[str, Any] | None) -> str:
    """추천 과목을 과목 식별자·이름·단계명과 함께 직렬화.

    과목 인과 흐름 출력의 입력 컨텍스트. LLM 이 ``course_flows`` 의 각 항목을
    어느 과목에 binding 할지 결정할 수 있도록 ``course_id`` 를 명시하고,
    ``{단계명}`` 슬롯용 한국어 단계 라벨을 함께 흘린다.
    """
    if not roadmap:
        return _EMPTY_CONTEXT_MARKER
    stages = roadmap.get("stages") or []
    lines: list[str] = []
    for stage in stages:
        stage_name = stage.get("stage") or "(단계 없음)"
        label = _STAGE_LABELS.get(stage_name, stage_name)
        for course in stage.get("courses") or []:
            lines.append(
                f"- course_id={course.get('course_id', '(식별자 없음)')}"
                f" / 과목명={course.get('course_name', '(이름 없음)')}"
                f" / 단계명={label}"
            )
    if not lines:
        return _EMPTY_CONTEXT_MARKER
    return "\n".join(lines)


def _serialize_anchor(
    jobs: list[dict[str, Any]] | None,
    primary: list[dict[str, Any]] | None,
) -> str:
    """과목 인과 흐름의 ``{직무명}`` / ``{트랙 조합}`` anchor 를 직렬화.

    인과 흐름은 모든 과목에 동일 anchor (최상위 직무 + 주 추천 트랙 조합) 를
    공유한다. 직무·트랙 데이터가 비어 있어도 흐름 자체는 학습 로드맵 존재
    시 생성되므로, 빈 영역은 명시적 안내 토큰으로 흘린다.
    """
    top_job = jobs[0].get("job_name", "(이름 없음)") if jobs else _EMPTY_CONTEXT_MARKER
    if primary:
        combo = primary[0].get("combo") or {}
        track_a = (combo.get("track_a") or {}).get("track_name", "(이름 없음)")
        track_b = (combo.get("track_b") or {}).get("track_name", "(이름 없음)")
        combo_label = f"{track_a} + {track_b}"
    else:
        combo_label = _EMPTY_CONTEXT_MARKER
    return f"직무명={top_job} / 트랙 조합={combo_label}"


def _needs_caveat(jobs: list[dict[str, Any]] | None) -> bool:
    """직무 후보 중 카테고리 사전 매핑 fallback 으로 채택된 건이 있는지."""
    if not jobs:
        return False
    return any(bool(job.get("fallback_used")) for job in jobs)


# ---------------------------------------------------------------------------
# 노드 진입점
# ---------------------------------------------------------------------------


class LLMExplanationNode:
    """LLM 자연어 설명 생성 노드 — 단일 LLM 호출 + 구조화 출력 검증.

    의존성을 ``__init__`` 으로 주입받으므로 단위 테스트에서 ``LLMClient`` 를
    ``MagicMock(spec=LLMClient)`` 로 교체하면 네트워크 호출 없이 검증 가능하다.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    def __call__(self, state: GraphState) -> dict[str, Any]:
        # 단계 1: 입력 수집
        recommended_jobs = state.get("recommended_jobs")
        primary_combos = state.get("primary_combos")
        secondary_combos = state.get("secondary_combos")
        roadmap = state.get("roadmap")

        jobs_empty = _jobs_empty(recommended_jobs)
        tracks_empty = _tracks_empty(primary_combos, secondary_combos)
        roadmap_empty = _roadmap_empty(roadmap)

        # 단계 2: 세 영역 모두 비어 있으면 LLM 호출 skip + 정적 안내
        if jobs_empty and tracks_empty and roadmap_empty:
            fallback = Explanation(text=_EMPTY_ALL_FALLBACK_TEXT, sections=[])
            return {
                "explanation": fallback.model_dump(mode="json"),
                "trace": ["llm_explanation:empty"],
            }

        # 단계 3: 컨텍스트 직렬화 (빈 영역은 명시적 안내 토큰)
        jobs_context = _serialize_jobs(recommended_jobs)
        tracks_context = _serialize_combos(primary_combos, secondary_combos)
        roadmap_context = _serialize_roadmap(roadmap)
        semesters_context = _serialize_semesters(roadmap)
        courses_context = _serialize_courses(roadmap)
        anchor_context = _serialize_anchor(recommended_jobs, primary_combos)

        # 단계 4: 캐비잇 결정
        caveat_required = "yes" if _needs_caveat(recommended_jobs) else "no"

        # 단계 5: LLM 호출 (단일 진입점)
        messages = EXPLANATION_PROMPT.format_messages(
            jobs_context=jobs_context,
            tracks_context=tracks_context,
            roadmap_context=roadmap_context,
            semesters_context=semesters_context,
            courses_context=courses_context,
            anchor_context=anchor_context,
            caveat_required=caveat_required,
        )
        result = self._client.invoke(messages)

        # 단계 6: state 부분 반환
        return {
            "explanation": result.model_dump(mode="json"),
            "trace": ["llm_explanation:ok"],
        }


__all__ = ["LLMClient", "LLMExplanationNode"]
