"""단일 기준 직무(anchor) 요구 역량 대비 현재 → 예상 충족도를 산출한다.

추천 결과를 일회성 결과가 아닌 "채워가는 지도" 로 만드는 노드. 상위 노드가
채워 둔 추천 직무, 학습 로드맵(잔여 추천 과목), 이수 과목만으로 충족도를
계산하며 별도 검색·LLM 호출을 하지 않는다.

기준 직무(anchor):
    게이지(현재/예상 충족도)·잔여 과목 기여도·다음 액션·gap 토큰은 추천 직무를
    합치지 않고 **하나의 기준 직무**에 대해 산출한다. 합집합·평균은 "무엇의 몇
    %인지" 를 모호하게 만들기 때문이다. 기본값은 매칭도 1순위(``match_score``
    최댓값, 동률은 입력 순서 우선) 직무이며, 호출자가 ``anchor_job_id`` 로 다른
    추천 직무를 지정하면 그 직무로 다시 산출한다.

분야별 분석(``jobs``)은 예외 — anchor 와 무관:
    추천 직무를 나란히 비교하는 카드 데이터라 **전 추천 직무**의 per-job 충족도를
    각자 자기 토큰 기준으로 담는다. 사용자가 직무 간 충족도를 비교해 기준 직무를
    바꿀지 판단하는 데 쓰이므로 anchor 한 건으로 좁히지 않는다.

목표 토큰(분모) 정의 — 도달 가능 토큰으로 한정:
    기준 직무의 기술스택 + 역량 태그를 표기 정합 키로 통합하되, 그중 학생이
    이수 과목 또는 추천 로드맵 과목으로 **실제 학습 가능한(도달 가능) 토큰만**
    분모로 삼는다. 직무가 요구하는 토큰 중 어떤 과목도 가르치지 않는 토큰은
    분모에서 빼 게이지가 "닿을 수 있는 역량 대비 충족" 을 나타내게 한다. 직무·
    트랙·과목이 같은 기술을 다른 표기로 적어도(예: "ReactJS" vs "React") 한
    토큰으로 묶이도록 직무 매칭·트랙 시너지와 같은 정합 사전을 재사용한다.
    도달 가능 토큰이 과다하면(상한 초과) 학습 빈도(토큰을 가르치는 도달 과목
    수) 상위 N 개만 분모로 남기되, 이수로 이미 충족한 토큰은 우선 보존한다
    (캡 탓에 현재 충족이 과소집계되지 않도록). 캡으로 제외된 토큰은 도달
    가능하지만 분모·gap 어디에도 노출하지 않는 의도적 절단이다.

충족도 정의(target = 도달 가능 목표 토큰):
    - current  = target ∩ (이수 과목 기술 토큰)
    - expected = target ∩ (이수 과목 + 추천 로드맵 과목 전체 기술 토큰)
      = target (분모가 도달 가능 토큰뿐이라 로드맵 전부 이수 시 100% 도달).
      게이지 상단(전체 충족) 표기의 근거다.
    - next_actions = target ∩ (이수 과목 + 다음 액션 shortlist 과목 토큰)
      (합집합 — 노출한 shortlist 만 이수했을 때 도달. current <= next_actions <= expected)
    - 과목별 기여 = (target ∩ 과목 토큰) - current  (현재 미충족분 중 새로 덮는 분)
    - gap = 기준 직무 요구 토큰 중 도달 불가(어떤 과목도 안 가르침) 토큰. 분모에선
      빠지지만 "이 트랙에서 못 채우는 역량" 정보로 보존한다.

부작용 격리:
    과목 이름 → 기술 토큰 색인은 생성자에서 1 회 로드하고, 호출 경로(``__call__``)
    는 순수 계산만 수행한다. 핵심 계산은 ``compute_coverage`` 순수 함수로 분리해
    색인을 주입하면 파일 I/O 없이 단위 테스트가 가능하다.

Trace 토큰:
    - ``coverage_analysis:ok`` — 도달 가능 목표 토큰이 1 개 이상이라 분석을 산출.
    - ``coverage_analysis:no_reachable_tokens`` — 기준 직무는 있으나 요구 토큰을
      가르치는 과목이 없어 도달 가능 분모가 0 (anchor·gap·분야별 분석은 채워 반환).
    - ``coverage_analysis:empty`` — 추천 직무 부재 또는 기준 직무 토큰 미보유로
      빈 분석을 반환.
"""

from __future__ import annotations

from collections import Counter
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

# 도달 가능 목표 토큰(분모)의 상한. 한 직무의 요구 토큰이 방대해 게이지 분모가
# 휩쓸리는 것을 막기 위해, 도달 가능 토큰이 이 값을 넘으면 학습 빈도(토큰을
# 가르치는 도달 과목 수) 상위 N 개만 분모로 남긴다.
_MAX_TARGET_TOKENS = 30


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


def _select_anchor(jobs: list[JobCandidate], anchor_job_id: str | None) -> JobCandidate | None:
    """충족도 산출의 기준 직무(anchor)를 고른다.

    지정한 ``anchor_job_id`` 가 추천 직무에 있으면 그 직무를, 없으면 매칭도
    1순위(``match_score`` 최댓값)를 기본 anchor 로 한다. 동률은 입력 순서를
    보존하므로(``max`` 의 first-max 성질) 상위 노드가 정렬해 둔 순위를 그대로
    따른다. 지정 식별자가 추천 직무에 없으면(예: stale 토글 입력) 기본 anchor
    로 graceful fallback 한다.

    Returns:
        기준 직무. 추천 직무가 하나도 없으면 ``None``.
    """
    if not jobs:
        return None
    if anchor_job_id:
        for job in jobs:
            if job.job_id == anchor_job_id:
                return job
    return max(jobs, key=lambda job: job.match_score)


def _empty_analysis() -> CoverageAnalysis:
    """목표 토큰 부재 시의 graceful 빈 분석. 비율 0.0, 리스트 빈 채로 종료한다."""
    return CoverageAnalysis(
        required_count=0,
        current_covered=0,
        expected_covered=0,
        current_ratio=0.0,
        expected_ratio=0.0,
    )


def _format_action_message(course_name: str, contribution_ratio: float) -> str:
    """다음 액션 제안 문구 — 과목 이수 시 충족도 증가분을 백분율로 안내."""
    percent = round(contribution_ratio * 100)
    return f"{course_name}을(를) 이수하면 역량 충족도가 약 {percent}% 오릅니다."


def _ratio(covered: int, required: int) -> float:
    """충족 토큰 수를 분모로 나눈 비율. 분모 0(도달 가능 토큰 부재)이면 0.0."""
    return covered / required if required else 0.0


def _select_target_keys(
    anchor_keys: set[str],
    reachable_keys: set[str],
    completed_keys: set[str],
    reachable_course_keys: list[set[str]],
) -> set[str]:
    """게이지 분모로 쓸 도달 가능 목표 토큰을 고른다.

    기준 직무 요구 토큰 중 학생이 이수·로드맵 과목으로 학습 가능한(도달 가능)
    토큰만 남긴다. 도달 가능 토큰이 상한(``_MAX_TARGET_TOKENS``)을 넘으면 학습
    빈도(토큰을 가르치는 도달 과목 수) 상위 N 개만 분모로 채택해, 한 직무의
    방대한 요구 토큰에 게이지 분모가 휩쓸리지 않게 한다.

    캡이 걸려도 **이수 과목으로 이미 충족한 토큰은 분모에서 빼지 않는다**. 그래야
    (a) 현재 충족(``current``)이 캡 탓에 과소집계돼 "수강 반응" 의도가 깨지는 일을
    막고, (b) ``current <= required`` 불변을 보장한다. 이미 충족한 토큰이 상한을
    넘는 드문 경우(요구 토큰이 방대하고 그만큼 이수)에는 그 집합을 그대로 반환해
    상한보다 큰 분모를 허용한다 — 불변 보장이 표시 상한보다 우선이다.

    캡으로 제외된 토큰은 도달 가능하지만 분모에도 ``gap_tokens`` 에도 나타나지
    않는다 (gap 은 "도달 불가" 전용). 상위 N 개만 노출하려는 의도적 절단이다.

    Args:
        anchor_keys: 기준 직무 요구 토큰의 정합 키 전체.
        reachable_keys: 이수 + 로드맵 과목이 가르치는 토큰의 정합 키 합집합.
        completed_keys: 이수 과목이 가르치는 토큰의 정합 키 합집합. 캡에서 우선
            보존할 "이미 충족한 목표 토큰" 식별에 쓴다.
        reachable_course_keys: 도달 과목(이수 + 로드맵) 각각의 토큰 정합 키 집합.
            학습 빈도(토큰별 등장 과목 수) 산정에 쓴다.

    Returns:
        분모로 채택한 도달 가능 목표 토큰 키 집합. 상한 초과 시 이미 충족한
        토큰을 우선 보존하고 남은 슬롯을 학습 빈도 상위로 채운다. 동률 빈도는
        토큰 키 사전순으로 끊어 deterministic 하다.
    """
    reachable_target = anchor_keys & reachable_keys
    if len(reachable_target) <= _MAX_TARGET_TOKENS:
        return reachable_target
    guaranteed = reachable_target & completed_keys
    if len(guaranteed) >= _MAX_TARGET_TOKENS:
        return guaranteed
    frequency: Counter[str] = Counter()
    for keys in reachable_course_keys:
        for key in keys & reachable_target:
            frequency[key] += 1
    fill_candidates = reachable_target - guaranteed
    ranked = sorted(fill_candidates, key=lambda key: (-frequency[key], key))
    budget = _MAX_TARGET_TOKENS - len(guaranteed)
    return guaranteed | set(ranked[:budget])


def compute_coverage(
    jobs: list[JobCandidate],
    completed_course_names: list[str],
    roadmap_courses: list[tuple[str, str]],
    course_tech_index: dict[str, list[str]],
    anchor_job_id: str | None = None,
) -> CoverageAnalysis:
    """단일 기준 직무(anchor)의 도달 가능 역량 대비 현재/예상 충족도를 산출한다.

    색인을 인자로 받는 순수 함수라 파일 I/O 없이 검증 가능하다. 분모는 기준 직무
    요구 토큰 *전체* 가 아니라 그중 학생이 이수·로드맵 과목으로 학습 가능한(도달
    가능) 토큰뿐이라, 게이지가 "닿을 수 있는 역량 대비 충족" 을 나타내고 수강
    행동에 반응한다. 게이지·잔여 과목 기여도·다음 액션·gap 토큰은 하나의 기준
    직무에 정렬되고, 분야별 분석(``jobs``)만 예외로 전 추천 직무의 per-job 충족도를
    담는다 (직무 비교용, anchor 무관 + 각 직무 전체 토큰 기준). 추천 직무가 없거나
    기준 직무가 요구 토큰을 전혀 안 가지면 빈 분석을 반환한다.

    Args:
        jobs: 추천 직무 후보. 그중 한 건을 기준 직무로 고른다.
        completed_course_names: 이수 과목 이름 목록 (현재 충족도 산출).
        roadmap_courses: 추천 로드맵의 잔여 과목 ``(course_id, course_name)`` 목록
            (예상 충족도·과목별 기여도 산출). course_id 기준 중복 없음 가정.
        course_tech_index: 과목 이름 → 기술 토큰 색인.
        anchor_job_id: 기준 직무 식별자. ``None`` 또는 추천 직무에 없는 값이면
            매칭도 1순위 직무를 기본 anchor 로 한다.

    Returns:
        ``CoverageAnalysis``. ``anchor_job_id`` / ``anchor_job_name`` 으로 기준
        직무를 함께 싣는다 (커버리지 모달의 "○○ 직무 기준" 라벨용). 비율 필드는
        [0, 1]. ``required_count`` 는 도달 가능 목표 토큰 수이며, 분모가 도달
        가능 토큰뿐이라 ``expected_ratio`` 는 ``required_count > 0`` 일 때 1.0
        (전체 로드맵 이수 시 도달 가능 역량 100% 충족). 기준 직무는 있으나 도달
        가능 토큰이 하나도 없으면 ``required_count == 0`` + 비율 0.0 이되
        ``anchor_job_id`` 와 ``gap_tokens`` (전부 도달 불가)는 채워 반환한다.
        추천 직무 자체가 없거나 기준 직무가 토큰을 전혀 안 가지면 ``anchor_job_id
        == ""`` 의 빈 분석으로 종료한다. ``next_actions_ratio`` 는 노출한 다음
        액션 과목까지 이수했을 때의 합집합 도달 충족도로
        ``current_ratio <= next_actions_ratio <= expected_ratio``.
    """
    anchor = _select_anchor(jobs, anchor_job_id)
    if anchor is None:
        return _empty_analysis()

    target_display = _token_display_map(_job_tokens(anchor))
    anchor_keys = set(target_display)
    if not anchor_keys:
        # 기준 직무가 요구 토큰 자체를 안 가짐 → 산출 불가, graceful 빈 분석.
        return _empty_analysis()

    # 이수·로드맵 과목 정합 키를 과목 단위로 보존한다(중복 lookup 회피). 도달
    # 집합·학습 빈도(분모 캡)·예상 충족도가 모두 이 per-course 키를 공유한다.
    completed_course_keys = [
        canonical_tech_keys(resolve_course_tokens([name], course_tech_index))
        for name in completed_course_names
    ]
    completed_keys: set[str] = (
        set().union(*completed_course_keys) if completed_course_keys else set()
    )

    course_keys: list[tuple[str, str, set[str]]] = []
    roadmap_all_keys: set[str] = set()
    for course_id, course_name in roadmap_courses:
        keys = canonical_tech_keys(resolve_course_tokens([course_name], course_tech_index))
        course_keys.append((course_id, course_name, keys))
        roadmap_all_keys |= keys

    reachable_keys = completed_keys | roadmap_all_keys

    # 분모 = 기준 직무 요구 토큰 중 도달 가능한 토큰(과다 시 학습 빈도 상위 N,
    # 단 이수로 충족한 토큰은 우선 보존).
    target_keys = _select_target_keys(
        anchor_keys,
        reachable_keys,
        completed_keys,
        [*completed_course_keys, *(keys for _id, _name, keys in course_keys)],
    )
    required_count = len(target_keys)
    current_keys = target_keys & completed_keys
    # 분모가 도달 가능 토큰뿐이라 expected(로드맵 전부 이수 후 도달) == 분모.
    expected_keys = target_keys

    # 분야별 분석은 anchor 와 무관 — 전 추천 직무를 각자 토큰 기준으로 담아
    # 직무 간 비교 카드에 쓴다. 현재 충족률 내림차순(동률은 job_id) 정렬.
    jobs_coverage = sorted(
        (_job_coverage(job, completed_keys, reachable_keys) for job in jobs),
        key=lambda coverage: (-coverage.current_ratio, coverage.job_id),
    )
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
        anchor_job_id=anchor.job_id,
        anchor_job_name=anchor.job_name,
        required_count=required_count,
        current_covered=len(current_keys),
        expected_covered=len(expected_keys),
        current_ratio=_ratio(len(current_keys), required_count),
        expected_ratio=_ratio(len(expected_keys), required_count),
        next_actions_covered=len(next_actions_reachable),
        next_actions_ratio=_ratio(len(next_actions_reachable), required_count),
        jobs=jobs_coverage,
        course_contributions=contributions,
        next_actions=next_actions,
        gap_tokens=sorted(target_display[key] for key in anchor_keys - reachable_keys),
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
            anchor_job_id=state.get("anchor_job_id"),
        )
        if analysis.required_count > 0:
            trace = "coverage_analysis:ok"
        elif analysis.anchor_job_id:
            # 기준 직무는 있으나 요구 토큰을 가르치는 과목이 없어 도달 가능 분모 0.
            trace = "coverage_analysis:no_reachable_tokens"
        else:
            trace = "coverage_analysis:empty"
        return {
            "coverage_analysis": analysis.model_dump(mode="json"),
            "trace": [trace],
        }


__all__ = ["CoverageAnalysisNode", "compute_coverage"]
