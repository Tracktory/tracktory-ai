"""추천 직무 요구 역량 대비 현재 → 예상 충족도를 산출한다.

추천 결과를 일회성 결과가 아닌 "채워가는 지도" 로 만드는 노드. 상위 노드가
채워 둔 추천 직무(목표 토큰), 학습 로드맵(잔여 추천 과목), 이수 과목만으로
충족도를 계산하며 별도 검색·LLM 호출을 하지 않는다.

목표 토큰 정의:
    추천 직무들의 기술스택 + 역량 태그를 표기 정합 키로 통합한 집합. 직무·트랙·
    과목이 같은 기술을 다른 표기로 적어도(예: "ReactJS" vs "React") 한 토큰으로
    묶이도록 직무 매칭·트랙 시너지와 같은 정합 사전을 재사용한다.

충족도 정의:
    - current  = target ∩ (이수 과목 기술 토큰)
    - expected = target ∩ (이수 과목 + 추천 로드맵 과목 전체 기술 토큰)
    - next_actions = target ∩ (이수 과목 + 다음 액션 shortlist 과목 토큰)
      (합집합 — 노출한 shortlist 만 이수했을 때 도달. current <= next_actions <= expected)
    - 과목별 기여 = (target ∩ 과목 토큰) - current  (현재 미충족분 중 새로 덮는 분)

부작용 격리:
    과목 이름 → 기술 토큰 색인은 생성자에서 1 회 로드하고, 호출 경로(``__call__``)
    는 순수 계산만 수행한다. 핵심 계산은 ``compute_coverage`` 순수 함수로 분리해
    색인을 주입하면 파일 I/O 없이 단위 테스트가 가능하다.

Trace 토큰:
    - ``coverage_analysis:ok`` — 목표 토큰이 1 개 이상이라 분석을 산출.
    - ``coverage_analysis:empty`` — 목표 토큰 부재(추천 직무 없음 또는 직무
      토큰 미보유)로 빈 분석을 반환.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tracktory.common.tech_keywords import (
    canonical_tech_key,
    canonical_tech_keys,
    canonical_tech_token,
)
from tracktory.graph.course_tech import (
    DEFAULT_COURSE_CATALOG_PATH,
    load_course_tech_index,
    resolve_course_tokens,
)
from tracktory.graph.models import (
    CourseCoverageContribution,
    CoverageAnalysis,
    JobCandidate,
    JobCoverage,
    NextActionSuggestion,
)
from tracktory.graph.state import GraphState

# 다음 액션 제안 노출 상한. 충족도를 가장 크게 올리는 상위 과목만 제안해 사용자
# 화면의 인지 부하를 줄인다 (전체 기여도는 course_contributions 로 별도 제공).
_MAX_NEXT_ACTIONS = 3


# ---------------------------------------------------------------------------
# 순수 계산 헬퍼
# ---------------------------------------------------------------------------


def _token_display_map(raw_tokens: list[str]) -> dict[str, str]:
    """원시 기술 토큰 목록을 ``{정합 키: 표시 표기}`` 매핑으로 만든다.

    비교는 정합 키(별칭·대소문자 흡수)로, 사용자 노출은 표시 표기로 한다. 같은
    키의 첫 등장 표기를 표시값으로 보존한다.
    """
    display: dict[str, str] = {}
    for raw in raw_tokens:
        key = canonical_tech_key(raw)
        if key and key not in display:
            display[key] = canonical_tech_token(raw)
    return display


def _job_tokens(job: JobCandidate) -> list[str]:
    """직무가 요구하는 기술스택 + 역량 태그를 한 목록으로 합친다."""
    return [*job.tech_stacks, *job.competency_tags]


def _format_action_message(course_name: str, contribution_ratio: float) -> str:
    """다음 액션 제안 문구 — 과목 이수 시 충족도 증가분을 백분율로 안내."""
    percent = round(contribution_ratio * 100)
    return f"{course_name}을(를) 이수하면 역량 충족도가 약 {percent}% 오릅니다."


def compute_coverage(
    jobs: list[JobCandidate],
    completed_course_names: list[str],
    roadmap_courses: list[tuple[str, str]],
    course_tech_index: dict[str, list[str]],
) -> CoverageAnalysis:
    """추천 직무 요구 역량 대비 현재/예상 충족도 분석을 산출한다.

    색인을 인자로 받는 순수 함수라 파일 I/O 없이 검증 가능하다. 목표 토큰이
    하나도 없으면(추천 직무 부재 또는 직무 토큰 미보유) 빈 분석을 반환한다.

    Args:
        jobs: 추천 직무 후보. ``tech_stacks`` + ``competency_tags`` 가 목표 토큰.
        completed_course_names: 이수 과목 이름 목록 (현재 충족도 산출).
        roadmap_courses: 추천 로드맵의 잔여 과목 ``(course_id, course_name)`` 목록
            (예상 충족도·과목별 기여도 산출). course_id 기준 중복 없음 가정.
        course_tech_index: 과목 이름 → 기술 토큰 색인.

    Returns:
        ``CoverageAnalysis``. 비율 필드는 [0, 1], 목표 토큰 부재 시 모두 0.0.
        ``next_actions_ratio`` 는 노출한 다음 액션 과목까지 이수했을 때의 합집합
        도달 충족도로 ``current_ratio <= next_actions_ratio <= expected_ratio``.
    """
    target_display = _token_display_map([token for job in jobs for token in _job_tokens(job)])
    target_keys = set(target_display)
    required_count = len(target_keys)

    if required_count == 0:
        return CoverageAnalysis(
            required_count=0,
            current_covered=0,
            expected_covered=0,
            current_ratio=0.0,
            expected_ratio=0.0,
        )

    completed_keys = canonical_tech_keys(
        resolve_course_tokens(completed_course_names, course_tech_index)
    )
    current_keys = target_keys & completed_keys

    # 과목별 정합 키를 미리 구해 두고(중복 lookup 회피) 합집합으로 예상 충족도를 잡는다.
    course_keys: list[tuple[str, str, set[str]]] = []
    roadmap_all_keys: set[str] = set()
    for course_id, course_name in roadmap_courses:
        keys = canonical_tech_keys(resolve_course_tokens([course_name], course_tech_index))
        course_keys.append((course_id, course_name, keys))
        roadmap_all_keys |= keys

    reachable_keys = completed_keys | roadmap_all_keys
    expected_keys = target_keys & reachable_keys

    jobs_coverage = [_job_coverage(job, completed_keys, reachable_keys) for job in jobs]
    contributions = _course_contributions(course_keys, target_keys, current_keys, target_display)
    next_actions = [
        NextActionSuggestion(
            course_id=contribution.course_id,
            course_name=contribution.course_name,
            contribution_ratio=contribution.contribution_ratio,
            message=_format_action_message(
                contribution.course_name, contribution.contribution_ratio
            ),
        )
        for contribution in contributions
        if contribution.contribution_ratio > 0.0
    ][:_MAX_NEXT_ACTIONS]

    # 노출한 다음 액션 과목을 모두 이수했을 때 도달하는 충족도. 과목별
    # contribution_ratio 는 토큰이 겹치면 합이 (expected - current) 를 넘으므로,
    # "이 N개 이수 시 도달" 표기는 합집합으로 따로 계산해 over-claim 을 막는다.
    next_action_ids = {action.course_id for action in next_actions}
    next_action_keys = {
        key
        for course_id, _name, keys in course_keys
        if course_id in next_action_ids
        for key in keys & target_keys
    }
    next_actions_reachable = current_keys | next_action_keys

    return CoverageAnalysis(
        required_count=required_count,
        current_covered=len(current_keys),
        expected_covered=len(expected_keys),
        current_ratio=len(current_keys) / required_count,
        expected_ratio=len(expected_keys) / required_count,
        next_actions_covered=len(next_actions_reachable),
        next_actions_ratio=len(next_actions_reachable) / required_count,
        jobs=jobs_coverage,
        course_contributions=contributions,
        next_actions=next_actions,
        gap_tokens=sorted(target_display[key] for key in target_keys - expected_keys),
    )


def _job_coverage(
    job: JobCandidate,
    completed_keys: set[str],
    reachable_keys: set[str],
) -> JobCoverage:
    """추천 직무 한 건의 현재/예상 충족도를 산출한다 (분야별 역량 수준)."""
    job_display = _token_display_map(_job_tokens(job))
    job_keys = set(job_display)
    required = len(job_keys)
    current = len(job_keys & completed_keys)
    expected = len(job_keys & reachable_keys)
    return JobCoverage(
        job_id=job.job_id,
        job_name=job.job_name,
        required_count=required,
        current_covered=current,
        expected_covered=expected,
        current_ratio=(current / required if required else 0.0),
        expected_ratio=(expected / required if required else 0.0),
        missing_tokens=sorted(job_display[key] for key in job_keys - reachable_keys),
    )


def _course_contributions(
    course_keys: list[tuple[str, str, set[str]]],
    target_keys: set[str],
    current_keys: set[str],
    target_display: dict[str, str],
) -> list[CourseCoverageContribution]:
    """잔여 과목별 충족도 기여도를 산출하고 기여 큰 순으로 정렬한다.

    기여도는 현재 충족도 기준 독립 한계 기여(이 과목 하나로 새로 덮는 비율)다.
    동률은 ``course_id`` 사전순으로 deterministic 정렬한다.
    """
    uncovered_now = target_keys - current_keys
    required_count = len(target_keys)
    contributions: list[CourseCoverageContribution] = []
    for course_id, course_name, keys in course_keys:
        added_keys = keys & uncovered_now
        contributions.append(
            CourseCoverageContribution(
                course_id=course_id,
                course_name=course_name,
                added_tokens=sorted(target_display[key] for key in added_keys),
                contribution_ratio=(len(added_keys) / required_count if required_count else 0.0),
            )
        )
    contributions.sort(
        key=lambda contribution: (-contribution.contribution_ratio, contribution.course_id)
    )
    return contributions


def _collect_roadmap_courses(roadmap: dict[str, Any] | None) -> list[tuple[str, str]]:
    """학습 로드맵의 잔여 추천 과목 ``(course_id, course_name)`` 을 중복 없이 모은다.

    학기 분산(``semesters``)을 과목 배치의 단일 진실원으로 삼는다. 같은 과목은
    한 학기에만 배치되지만, 형태가 어긋난 입력에도 안전하도록 course_id 기준으로
    dedup 한다.
    """
    if not roadmap:
        return []
    collected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for plan in roadmap.get("semesters") or []:
        for course in plan.get("courses") or []:
            course_id = course.get("course_id")
            course_name = course.get("course_name")
            if course_id and course_name and course_id not in seen:
                seen.add(course_id)
                collected.append((course_id, course_name))
    return collected


# ---------------------------------------------------------------------------
# 노드 진입점
# ---------------------------------------------------------------------------


class CoverageAnalysisNode:
    """역량 커버리지 분석 노드 — 추천 직무·로드맵·이수 과목으로 충족도 산출.

    과목 이름 → 기술 토큰 색인을 생성자에서 1 회 로드하고, 호출 경로는 순수
    계산만 수행한다. 단위 테스트는 ``course_catalog_path`` 로 합성 카탈로그를
    주입해 외부 I/O 없이 검증한다.
    """

    def __init__(self, course_catalog_path: Path | None = None) -> None:
        self._course_tech_index = load_course_tech_index(
            course_catalog_path or DEFAULT_COURSE_CATALOG_PATH
        )

    def __call__(self, state: GraphState) -> dict[str, Any]:
        jobs_raw: list[dict[str, Any]] = state.get("recommended_jobs") or []
        jobs = [JobCandidate.model_validate(job) for job in jobs_raw]
        normalized: dict[str, Any] = state.get("normalized_profile") or {}
        completed_courses: list[str] = normalized.get("completed_courses") or []
        roadmap_courses = _collect_roadmap_courses(state.get("roadmap"))

        analysis = compute_coverage(
            jobs,
            completed_courses,
            roadmap_courses,
            self._course_tech_index,
        )
        trace = "coverage_analysis:ok" if analysis.required_count > 0 else "coverage_analysis:empty"
        return {
            "coverage_analysis": analysis.model_dump(mode="json"),
            "trace": [trace],
        }


__all__ = ["CoverageAnalysisNode", "compute_coverage"]
