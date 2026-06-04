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
       요약 단락 외에 직무·트랙 항목별 개별 근거 (``job_rationales`` /
       ``track_rationales``), 학기 카드 헤더용 부제 (``semester_subtitles``),
       과목 상세 모달용 인과 흐름 (``course_flows``) 을 함께 담는다. 호출·
       파싱 실패는 추천 응답 전체를 떨어뜨리지 않도록 정적 안내 설명으로
       degrade 한다 (트랙 근거는 다음 단계가 조합 데이터로 채운다).
    5.5. 트랙 근거 coverage 보강 — LLM 이 주/보조 추천 조합 일부의 항목별
       트랙 근거 (``track_rationales``) 를 누락해도, 모든 조합이 ``combo_key``
       근거를 갖도록 결정론적으로 채운다. 누락 조합은 그 조합 고유 트랙
       데이터에서 합성하므로 보조 조합끼리 같은 문구로 폴백되지 않는다.
    6. state 부분 반환 — ``Explanation.model_dump(mode="json")`` 결과 dict 만.

부작용 격리:
    - LLM 호출은 ``__call__`` 의 단 1곳. 직렬화·빈 영역 판정·캐비잇 결정·
      트랙 근거 coverage 보강은 모두 module-level 순수 함수로 분리되어 mock
      없이 테스트 가능하다.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

from tracktory.graph.models import Explanation, TrackRationale
from tracktory.graph.state import GraphState
from tracktory.llm.llm_client import LLMClient
from tracktory.prompts.explanation import EXPLANATION_PROMPT

logger = logging.getLogger(__name__)

_EMPTY_CONTEXT_MARKER = "데이터 없음"
_EMPTY_ALL_FALLBACK_TEXT = "이번 추천 결과가 비어 있어 설명을 생성하지 않았습니다."
# LLM 호출·파싱 실패 시에도 추천 응답 자체는 살아 있어야 하므로, 설명만
# 정적 안내로 degrade 한다. 항목별 트랙 근거는 이후 coverage 보강이 조합
# 데이터로 채워 보조 트랙들이 동일 문구로 폴백되는 문제를 막는다.
_GENERATION_FAILED_FALLBACK_TEXT = (
    "지금은 추천 설명을 자세히 생성하지 못했어요. 추천 결과는 아래에서 확인하실 수 있어요."
)

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

# 가운뎃점 계열 유니코드 변형. 트랙 식별자(=학사 원본 트랙명)는 한글 아래아
# (U+318D ㆍ) 를 쓰는데, LLM 이 combo_key 를 옮길 때 흔한 가운뎃점(U+00B7 ·)
# 등으로 정규화해 byte 매칭이 빗나가는 일이 있다. 매칭용 정규화에서 이 계열을
# 한 글자로 접어 LLM 근거를 올바른 조합에 연결하되, 출력 키는 원본을 stamp 한다.
_DOT_VARIANTS = "ㆍ·‧・•"
_DOT_CANONICAL = "·"


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

    각 직무 단위 개별 근거(``job_rationales``)를 추천 직무 항목에 binding 할 수
    있도록 ``job_id`` 를 명시한다 (과목 인과 흐름의 ``course_id`` binding 과 같은
    패턴). 식별자는 binding 키일 뿐 사용자에게 노출되는 이름이 아니므로 환각
    표면을 넓히지 않는다. 그 외 ``tech_stacks`` / ``competency_tags`` / 유사도는
    영역 단락과 직무별 근거의 grounding 근거로 함께 흘린다.
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
            f"- job_id={job.get('job_id', '(식별자 없음)')}"
            f" / 직무명={job.get('job_name', '(이름 없음)')}"
            f" / 유사도={similarity:.2f}"
            f" / 기술스택=[{tech}]"
            f" / 역량=[{competencies}]"
        )
    return "\n".join(lines)


def _track_tags(track: dict[str, Any]) -> str:
    """단일 트랙의 역량·기술스택을 개별 트랙 근거 grounding 용으로 직렬화."""
    competencies = ", ".join(track.get("competencies") or []) or "(데이터 없음)"
    tech = ", ".join(track.get("tech_stacks") or []) or "(데이터 없음)"
    return f"역량=[{competencies}] / 기술스택=[{tech}]"


def _serialize_combos(
    primary: list[dict[str, Any]] | None,
    secondary: list[dict[str, Any]] | None,
) -> str:
    """주 추천 + 보조 추천을 묶어 슬롯 분류와 함께 직렬화.

    각 조합 단위 개별 근거(``track_rationales``)를 binding 할 수 있도록
    ``combo_key`` 를 명시하고, 조합 전체 근거와 구분되는 개별 트랙 근거를
    grounding 하기 위해 두 트랙 각각의 역량·기술스택을 함께 흘린다.
    """
    combos = list(primary or []) + list(secondary or [])
    if not combos:
        return _EMPTY_CONTEXT_MARKER
    lines: list[str] = []
    for ranked in combos:
        combo = ranked.get("combo") or {}
        track_a_obj = combo.get("track_a") or {}
        track_b_obj = combo.get("track_b") or {}
        track_a = track_a_obj.get("track_name", "(이름 없음)")
        track_b = track_b_obj.get("track_name", "(이름 없음)")
        combo_key = combo.get("combo_key", "(식별자 없음)")
        slot_type = ranked.get("slot_type", "(분류 없음)")
        score = ranked.get("synergy_score", 0.0)
        lines.append(
            f"- combo_key={combo_key}"
            f" / 조합={track_a} + {track_b}"
            f" / 슬롯={slot_type} / 시너지={score:.2f}"
            f"\n  · 1트랙 {track_a}: {_track_tags(track_a_obj)}"
            f"\n  · 2트랙 {track_b}: {_track_tags(track_b_obj)}"
        )
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
            f"{course.get('course_name', '(이름 없음)')}({course.get('credits', 0)}학점)"
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


def _track_value_sentence(track_name: str, track: dict[str, Any]) -> str:
    """단일 트랙의 fallback 개별 근거 한 문장을 그 트랙 데이터로 합성한다.

    역량·기술스택이 있으면 상위 몇 개를 근거로 노출하고, 둘 다 비어 있으면
    트랙명만으로 일반 문장을 만든다. 어느 경우든 ``min_length=1`` 을 만족하는
    non-empty 문구를 반환한다.
    """
    tags = (track.get("competencies") or []) + (track.get("tech_stacks") or [])
    if tags:
        joined = ", ".join(tags[:3])
        return f"{track_name} 트랙은 {joined} 등을 키울 수 있어요."
    return f"{track_name} 트랙은 해당 분야의 기반 역량을 키울 수 있어요."


def _fallback_track_rationale(combo: dict[str, Any]) -> TrackRationale:
    """LLM 이 누락한 조합의 항목별 근거를 그 조합 고유 데이터로 합성한다.

    조합 단위 근거 (``combo_rationale``) 와 개별 트랙 근거를 모두 두 트랙명·
    역량·기술스택에서 만들어, 보조 조합끼리 같은 영역 요약 문구로 폴백되지
    않고 서로 구분되는 근거를 갖도록 한다. 트랙명이 조합마다 다르므로 데이터가
    비어 있어도 조합 간 문구가 중복되지 않는다.
    """
    track_a_obj = combo.get("track_a") or {}
    track_b_obj = combo.get("track_b") or {}
    name_a = track_a_obj.get("track_name") or "1트랙"
    name_b = track_b_obj.get("track_name") or "2트랙"
    combo_key = combo.get("combo_key") or f"{name_a}::{name_b}"
    return TrackRationale(
        combo_key=combo_key,
        combo_rationale=(
            f"{name_a}와(과) {name_b}를 함께 선택하면 두 트랙의 강점을 아우르는"
            " 학습 경로를 설계할 수 있어요."
        ),
        track_a_rationale=_track_value_sentence(name_a, track_a_obj),
        track_b_rationale=_track_value_sentence(name_b, track_b_obj),
    )


def _normalize_combo_key(combo_key: str) -> str:
    """combo_key 를 매칭용으로 정규화한다 (가운뎃점 변형 + 호환 형태 흡수).

    트랙 식별자가 한글 아래아(U+318D)를 쓰는데 LLM 이 일반 가운뎃점(U+00B7)
    등으로 옮기면 byte 매칭이 빗나간다. 매칭 단계에서만 이 변형들을 한 글자로
    접어 같은 조합으로 인식하고, 출력 키 자체는 원본을 그대로 stamp 한다.
    """
    folded = "".join(_DOT_CANONICAL if ch in _DOT_VARIANTS else ch for ch in combo_key)
    return unicodedata.normalize("NFKC", folded)


def _ensure_track_rationale_coverage(
    result: Explanation,
    primary: list[dict[str, Any]] | None,
    secondary: list[dict[str, Any]] | None,
) -> Explanation:
    """트랙 근거를 추천 조합 전체와 정확히 1:1 로 맞춘다.

    LLM 은 단일 호출로 모든 조합의 항목별 근거를 enumerate 하도록 지시받지만,
    실측상 (a) 보조 조합 근거를 누락하거나 (b) ``combo_key`` 를 verbatim 으로
    못 옮겨(가운뎃점 변형 등) 소비 측 매칭이 빗나간다. 어느 쪽이든 보조 트랙들이
    같은 영역 요약 문구로 폴백되는 증상으로 나타난다.

    본 함수는 출력을 추천 조합과 일치시킨다 — (1) LLM 근거를 정규화 키로 추천
    조합에 매칭하고(가운뎃점 변형 흡수), 매칭되면 그 문구를 살리되 출력 키는
    추천 조합의 원본 ``combo_key`` 로 stamp 한다, (2) 추천 조합에 없는
    hallucinated 키는 버리며, (3) 누락 조합은 그 조합 고유 데이터로 합성한
    근거로 채운다. 결과 순서는 추천 조합 순서(주 추천 → 보조 추천)를 따른다.
    """
    combos = list(primary or []) + list(secondary or [])
    if not combos:
        # 추천 조합이 없으면 바인딩 대상도 없으므로, LLM 이 만든 트랙 근거는
        # 어디에도 매칭되지 않는 잔여물이다. 빈 리스트로 정리한다.
        if not result.track_rationales:
            return result
        return result.model_copy(update={"track_rationales": []})
    valid: dict[str, dict[str, Any]] = {}
    norm_to_key: dict[str, str] = {}
    for ranked in combos:
        combo = ranked.get("combo") or {}
        combo_key = combo.get("combo_key")
        if combo_key and combo_key not in valid:
            valid[combo_key] = combo
            norm_to_key.setdefault(_normalize_combo_key(combo_key), combo_key)
    llm_by_key: dict[str, TrackRationale] = {}
    for rationale in result.track_rationales:
        auth_key = norm_to_key.get(_normalize_combo_key(rationale.combo_key))
        if auth_key is None or auth_key in llm_by_key:
            continue
        # combo_key 가 변형됐으면 LLM 문구는 살리고 키만 원본으로 교정한다.
        if rationale.combo_key == auth_key:
            llm_by_key[auth_key] = rationale
        else:
            llm_by_key[auth_key] = rationale.model_copy(update={"combo_key": auth_key})
    rebuilt = [
        llm_by_key.get(combo_key) or _fallback_track_rationale(combo)
        for combo_key, combo in valid.items()
    ]
    if rebuilt == result.track_rationales:
        return result
    return result.model_copy(update={"track_rationales": rebuilt})


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
        # LLM 호출·파싱 실패가 추천 응답 전체를 500 으로 떨어뜨리지 않도록,
        # 설명만 정적 안내로 degrade 한다. 트랙 근거는 이후 coverage 보강이
        # 조합 데이터로 채워 소비 측의 combo_key binding 이 유지된다.
        try:
            result = self._client.invoke(messages)
            trace = "llm_explanation:ok"
        except Exception:
            logger.exception("llm_explanation_generation_failed")
            result = Explanation(text=_GENERATION_FAILED_FALLBACK_TEXT)
            trace = "llm_explanation:fallback"

        # 단계 5.5: 트랙 근거 coverage 보강 (보조 조합 누락 방지)
        result = _ensure_track_rationale_coverage(result, primary_combos, secondary_combos)

        # 단계 6: state 부분 반환
        return {
            "explanation": result.model_dump(mode="json"),
            "trace": [trace],
        }


__all__ = ["LLMClient", "LLMExplanationNode"]
