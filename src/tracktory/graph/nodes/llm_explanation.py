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
       추려서 프롬프트 변수로 흐른다.
    4. 캐비잇 결정 — 직무 후보 중 fallback 채택분이 있으면 ``yes`` 플래그를
       흘려 시스템 프롬프트의 추가 입력 안내 한 문장이 부착되도록 한다.
    5. LLM 호출 — ``ChatPromptTemplate.format_messages`` 로 메시지를 만들어
       구조화 출력을 강제하는 ``LLMClient`` 에 단일 invoke.
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

        # 단계 4: 캐비잇 결정
        caveat_required = "yes" if _needs_caveat(recommended_jobs) else "no"

        # 단계 5: LLM 호출 (단일 진입점)
        messages = EXPLANATION_PROMPT.format_messages(
            jobs_context=jobs_context,
            tracks_context=tracks_context,
            roadmap_context=roadmap_context,
            caveat_required=caveat_required,
        )
        result = self._client.invoke(messages)

        # 단계 6: state 부분 반환
        return {
            "explanation": result.model_dump(mode="json"),
            "trace": ["llm_explanation:ok"],
        }


__all__ = ["LLMClient", "LLMExplanationNode"]
