"""트랙 시너지 결과로부터 학생의 잔여 학기에 분산된 학습 로드맵을 생성한다.

추천된 1 순위 트랙 조합에 권장되는 전공 (필수 + 선택) 과목을 학사 데이터에서
끌어와, 이수 과목·선수과목 관계·학기 용량·학년별 학점 범위·졸업 총 학점이라는
학사 제약을 모두 만족하도록 학생의 잔여 학기에 분산 배치한다.

출력 schema 이중화:
    - ``Roadmap.stages`` — 학습 깊이 4 단계 (foundation / core / application
      / industry) 라벨. 단계는 과목이 배치된 학기의 학년에서 도출하며
      (1→foundation … 4→industry), 자연어 설명 생성 노드가 단계별 과목
      그룹화에 사용한다.
    - ``Roadmap.semesters`` — 학생의 잔여 학기 (current_semester ~ 8) 별 추천
      과목 plan. 사용자 화면의 학기 카드 row 에 직접 매핑된다. 한 학기에는
      여러 단계의 과목이 섞일 수 있다 (예: 학년 후반에 기초 마지막 + 핵심 첫).

알고리즘 — 추천 점수 순 stream:
    후보를 추천 점수 내림차순 (동점 시 priority → course_id) 으로 정렬한 뒤,
    학생의 현재 학기부터 각 과목을 다음 조건을 모두 만족할 때 채택한다.
        1. 학년 제약: 현재 학기의 학년이 ``course.available_grades`` 안에 있다.
        2. 선수 만족: ``course.prereq_ids`` 가 누적 이수 + 채택 집합의 부분집합.
        3. 학기 학사 cap: 학기당 18 학점 (한성대 일반 학기 제도) 미초과.
        4. 학년 cap: 학년별 최대 학점 (학년별 학점 범위 yaml) 미초과.
        5. 졸업 cap: 전공 총합 학점 미초과.
    조건 위배 시 (3) (4) 는 다음 학기 후보로 보류, (1) (2) 는 학년이 진행됨에
    따라 자연 해제된다. 점수가 높은 과목일수록 먼저 시도되어 가능한 이른 학기에
    배치된다. 졸업 cap 도달 시 stream 종료하고 해당 학기에 ``cap_reached``
    마커를 켠다.

부작용 격리:
    - ``course_repo`` 호출은 ``__call__`` 단 1 곳.
    - 순수 계산 함수는 모두 module-level — mock 없이 단위 테스트 가능.

Trace 토큰:
    - ``roadmap:ok`` — 정상 생성.
    - ``roadmap:empty`` — 1 순위 조합 부재 또는 Repository 빈 결과로 안전 종료.
    - ``roadmap:all_completed`` — 후보는 있었으나 이수 과목이 전부 cover 한 경우.
    - ``roadmap:skip`` — ``normalized_profile`` 부재로 그래프 입력 자체 부재.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from tracktory.graph.models import (
    Course,
    Roadmap,
    RoadmapConfig,
    RoadmapCourse,
    RoadmapStage,
    SemesterPlan,
)
from tracktory.graph.state import GraphState
from tracktory.rag.course_repository import CourseRepository

_DEFAULT_ROADMAP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "roadmap.yaml"

StageLabel = Literal["foundation", "core", "application", "industry"]
_STAGE_ORDER: tuple[StageLabel, ...] = ("foundation", "core", "application", "industry")
_MAX_SEMESTER: int = 8

# 본 시스템 추천 대상이 되는 과목 분류. 교양은 사용자 자율 구성 영역으로
# stream 진입 단계에서 제외하여 추천 결과의 학기 카드에 노출되지 않도록 한다.
_RECOMMENDED_COURSE_TYPES: frozenset[str] = frozenset({"전공필수", "전공선택"})

# 배치 학년 → 학습 깊이 단계 라벨. 카탈로그 단계 분류가 아니라 과목이 실제로
# 놓인 학기의 학년에서 단계를 도출해, 산학(industry) 과목 데이터 부재로 산학
# 단계가 항상 비던 문제를 없앤다 (4 학년에 놓인 과목이 산학을 채운다).
_GRADE_TO_STAGE: dict[int, StageLabel] = {
    1: "foundation",
    2: "core",
    3: "application",
    4: "industry",
}

# 추천 정렬·표시 점수 가중치. 전공 필수가 선택보다 추천 우선순위가 높고,
# 두 트랙이 동시에 권장하는 과목은 조합 시너지가 커 가산점을 준다.
# 모두 직관 할당값으로, 운영 데이터 확보 후 ablation 대상이다.
_COURSE_TYPE_SCORE: dict[str, float] = {"전공필수": 0.6, "전공선택": 0.4}
_BOTH_TRACKS_BONUS: float = 0.4

logger = logging.getLogger(__name__)


def _compute_course_score(course: Course, primary_track_ids: list[str]) -> float:
    """과목의 추천 정렬·표시 점수를 0~1 범위로 산출한다.

    전공 필수/선택 기본 점수에 두 1 순위 트랙이 모두 권장하는 과목이면
    가산점을 더한다. 두 트랙 공통 추천 = 조합 시너지의 핵심 신호다.
    """
    base = _COURSE_TYPE_SCORE.get(course.course_type, _COURSE_TYPE_SCORE["전공선택"])
    shared = len(primary_track_ids) >= 2 and all(
        tid in course.track_ids for tid in primary_track_ids
    )
    return base + (_BOTH_TRACKS_BONUS if shared else 0.0)


def _order_candidates(courses: list[Course], score_by_id: dict[str, float]) -> list[Course]:
    """후보를 추천 점수 내림차순으로 정렬한다 — stage stream 의 입력 순서.

    점수가 같으면 ``priority`` → ``course_id`` 사전식으로 deterministic 순서를
    보장한다. 점수가 높은 과목일수록 먼저 시도되어 가능한 이른 학기에 배치된다.
    """
    return sorted(
        courses,
        key=lambda course: (-score_by_id[course.course_id], course.priority, course.course_id),
    )


# ---------------------------------------------------------------------------
# 순수 계산 함수
# ---------------------------------------------------------------------------


def _filter_completed(courses: list[Course], completed_set: set[str]) -> list[Course]:
    """이수 과목 집합에 포함된 후보를 제거한다.

    이수 과목 자체는 추천 결과에 노출하지 않지만, 후수 과목의 선수 만족 신호
    로는 별도로 사용되므로 호출자는 ``completed_set`` 을 그대로 보존한다.
    """
    return [course for course in courses if course.course_id not in completed_set]


def _filter_recommended_types(courses: list[Course]) -> list[Course]:
    """본 시스템 추천 대상 (전공 필수 / 선택) 만 통과시킨다.

    교양은 사용자 자율 구성 영역이라 학습 로드맵의 학기 카드에 노출하지 않는다.
    """
    return [c for c in courses if c.course_type in _RECOMMENDED_COURSE_TYPES]


def _semester_to_grade(semester: int) -> int:
    """학기 번호 (1~8) 를 학년 번호 (1~4) 로 매핑한다.

    한 학년 = 두 학기 가정. 1 학년 1·2 학기 = grade 1, 4 학년 1·2 학기 = grade 4.
    """
    return (semester + 1) // 2


def _distribute_across_semesters(
    candidates: list[Course],
    *,
    current_semester: int,
    completed_set: set[str],
    config: RoadmapConfig,
    score_by_id: dict[str, float],
) -> list[SemesterPlan]:
    """추천 점수 순 stream 으로 학기에 분산 — 본 노드의 분산 알고리즘 본체.

    상세 알고리즘은 모듈 docstring 참조. 본 함수는 학기 단위 plan 만 반환하고
    학습 깊이 라벨 출력은 호출자가 별도로 추출한다 (단일 책임 분리).

    Args:
        candidates: 추천 점수 내림차순으로 이미 정렬된 과목 후보 (이수 분리 +
            교양 제외 완료).
        current_semester: 학생의 잔여 학기 시작점 (1~8).
        completed_set: 이수 과목 ID 집합. 선수 만족 신호의 출발점으로 사용된다.
        config: 학기 cap + 학년 범위 + 졸업 총 학점.
        score_by_id: 과목 식별자 → 추천 점수. 학기 plan 의 ``RoadmapCourse`` 에
            그대로 실어 사용자 표시·정렬에 사용한다.

    Returns:
        학기 번호 오름차순 ``SemesterPlan`` 리스트. 졸업 cap 도달 학기에
        ``cap_reached=True``. 잔여 학기가 짧아 졸업 cap 미달 시 마지막 학기에
        ``graduation_insufficient=True``. 후보 자체가 비어 있으면 빈 리스트.
    """
    if not candidates:
        return []

    sem_cap = config.capacity.max_credits_per_semester_default
    grad_total = config.graduation.total_credits_major

    # 미채택 후보 풀과 본 학기 후보 풀을 분리한다 — 학년 / 선수 / 학년 cap
    # 위배는 다음 학기에 다시 시도해야 한다.
    remaining: list[Course] = list(candidates)
    satisfied: set[str] = set(completed_set)
    grade_credits: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    plans: list[SemesterPlan] = []
    accumulated_total = 0
    cap_reached_seen = False

    for semester in range(current_semester, _MAX_SEMESTER + 1):
        grade = _semester_to_grade(semester)
        grade_cap = config.grade_credits_range[grade].max_credits
        sem_courses: list[Course] = []
        sem_credits = 0

        # 본 학기 stream — 한 번에 한 과목씩 추천 점수 순으로 가장 앞부터
        # 시도하고, 채택 시 remaining 에서 제거한다.
        idx = 0
        while idx < len(remaining):
            if accumulated_total >= grad_total:
                break
            course = remaining[idx]

            # 학년 제약 검증
            if grade not in course.available_grades:
                idx += 1
                continue
            # 선수 만족 검증
            if not all(pid in satisfied for pid in course.prereq_ids):
                idx += 1
                continue
            # 학기 학사 cap 검증
            if sem_credits + course.credits > sem_cap:
                idx += 1
                continue
            # 학년 cap 검증
            if grade_credits[grade] + course.credits > grade_cap:
                idx += 1
                continue
            # 졸업 cap 검증
            if accumulated_total + course.credits > grad_total:
                idx += 1
                continue

            # 채택 — remaining 에서 제거하고 누적 카운터 갱신
            sem_courses.append(course)
            sem_credits += course.credits
            grade_credits[grade] += course.credits
            accumulated_total += course.credits
            satisfied.add(course.course_id)
            remaining.pop(idx)
            # idx 는 증가시키지 않는다 — pop 으로 다음 후보가 같은 위치에 옴

        cap_reached_now = accumulated_total >= grad_total and not cap_reached_seen
        if cap_reached_now:
            cap_reached_seen = True

        plans.append(
            SemesterPlan(
                semester=semester,
                grade=grade,
                courses=_to_roadmap_courses(sem_courses, grade, score_by_id),
                credits_total=sem_credits,
                cap_reached=cap_reached_now,
                graduation_insufficient=False,
            )
        )

        if accumulated_total >= grad_total:
            break

    # 졸업 cap 미달 — 마지막 학기에 미달 마커를 켜서 학사 상담 권유 UI 가
    # 발동하도록 한다. plans 가 비어 있을 일은 위에서 후보 빈 케이스를 미리
    # 거른다는 invariant 로 막혀 있다.
    if not cap_reached_seen and plans:
        plans[-1] = plans[-1].model_copy(update={"graduation_insufficient": True})

    return plans


def _to_roadmap_courses(
    courses: list[Course], grade: int, score_by_id: dict[str, float]
) -> list[RoadmapCourse]:
    """``Course`` 를 사용자 노출용 ``RoadmapCourse`` 로 변환한다.

    단계 라벨은 과목이 배치된 학기의 학년에서 도출하고 (산학 단계 공백 방지),
    학점과 추천 점수를 그대로 실어 사용자 화면의 과목 단위 표기에 매핑한다.
    배치 순서가 이미 점수 내림차순이므로 추가 재정렬은 하지 않는다.
    """
    stage = _GRADE_TO_STAGE[grade]
    return [
        RoadmapCourse(
            course_id=c.course_id,
            course_name=c.course_name,
            credits=c.credits,
            stage=stage,
            score=score_by_id[c.course_id],
        )
        for c in courses
    ]


def _extract_stages_from_semesters(semester_plans: list[SemesterPlan]) -> list[RoadmapStage]:
    """학기 분산 결과에서 학습 깊이 단계 라벨 출력을 재구성한다.

    학기 plan 의 각 ``RoadmapCourse`` 는 이미 배치 학년에서 도출한 ``stage`` 를
    들고 있으므로, 그 값으로 그룹화하여 4 단계 뷰를 만든다. 학기 분산이 실제로
    채택한 과목만 포함되며, 어느 단계도 비어 있을 수 있다.

    이 변환이 분산 출력과 단계 출력을 같은 데이터의 두 뷰로 유지한다 — 자연어
    설명 노드가 단계 라벨에 의존하더라도 학기 분산이 발견한 컷 (선수 / 학년 /
    cap) 이 자동으로 반영된다.
    """
    courses_by_stage: dict[StageLabel, list[RoadmapCourse]] = {stage: [] for stage in _STAGE_ORDER}
    for plan in semester_plans:
        for course in plan.courses:
            courses_by_stage[course.stage].append(course)

    return [RoadmapStage(stage=stage, courses=courses_by_stage[stage]) for stage in _STAGE_ORDER]


def _empty_roadmap(combo_key: str | None = None) -> Roadmap:
    """빈 단계로만 구성된 안전 종료용 로드맵.

    조합 식별자가 식별된 경우 (예: 후보가 모두 이수되어 빈 결과) 호출자가
    그대로 흘려 후속 노드가 binding 을 추적할 수 있게 한다.
    """
    return Roadmap(
        stages=[RoadmapStage(stage=stage, courses=[]) for stage in _STAGE_ORDER],
        semesters=[],
        derived_from_combo_key=combo_key,
    )


def _extract_primary_track_ids(primary_combos: list[dict[str, Any]]) -> list[str]:
    """``primary_combos`` 의 1 순위 항목에서 두 트랙 식별자를 뽑는다.

    트랙 시너지 노드는 ``RankedCombo.model_dump(mode="json")`` 결과 dict 를 흘리며,
    구조는 ``{"combo": {"track_a": {...}, "track_b": {...}, ...}, ...}`` 다.
    형태가 어긋나면 빈 리스트를 반환하여 호출자가 안전 종료할 수 있게 한다.
    """
    if not primary_combos:
        return []
    top = primary_combos[0]
    combo = top.get("combo") or {}
    track_a = combo.get("track_a") or {}
    track_b = combo.get("track_b") or {}
    track_ids = [tid for tid in (track_a.get("track_id"), track_b.get("track_id")) if tid]
    return track_ids


def _extract_primary_combo_key(primary_combos: list[dict[str, Any]]) -> str | None:
    """``primary_combos`` 의 1 순위 항목에서 조합 식별자를 뽑는다.

    후속 자연어 설명 노드가 "이 로드맵이 어느 트랙 조합에서 파생되었는가" 를
    명시적으로 알 수 있도록 ``Roadmap.derived_from_combo_key`` 로 흘릴 값이다.
    형태가 어긋나면 ``None`` 을 반환한다.
    """
    if not primary_combos:
        return None
    top = primary_combos[0]
    combo = top.get("combo") or {}
    key = combo.get("combo_key")
    return key if isinstance(key, str) and key else None


# ---------------------------------------------------------------------------
# 노드 진입점
# ---------------------------------------------------------------------------


class RoadmapNode:
    """학습 로드맵 노드 — 트랙 시너지 1 순위 결과로부터 학기 분산 로드맵 생성.

    의존성을 ``__init__`` 으로 주입받으므로 단위 테스트에서 ``CourseRepository``
    를 mock 으로 교체하면 외부 I/O 없이 검증 가능하다.
    """

    def __init__(
        self,
        course_repo: CourseRepository,
        config_path: Path | None = None,
    ) -> None:
        self._repo = course_repo
        self._config = RoadmapConfig.load_from_yaml(config_path or _DEFAULT_ROADMAP_CONFIG_PATH)

    def __call__(self, state: GraphState) -> dict[str, Any]:
        # 단계 1: 입력 검증
        normalized: dict[str, Any] | None = state.get("normalized_profile")
        if not normalized:
            return {
                "errors": ["roadmap skipped: normalized_profile missing"],
                "trace": ["roadmap:skip"],
            }

        primary_combos_raw: list[dict[str, Any]] | None = state.get("primary_combos")
        primary_combos = primary_combos_raw or []
        track_ids = _extract_primary_track_ids(primary_combos)
        combo_key = _extract_primary_combo_key(primary_combos)
        if not track_ids:
            # 트랙 미선택 1 학년 또는 시너지 산출 부재 — 빈 로드맵으로 흐름 유지
            return {
                "roadmap": _empty_roadmap().model_dump(mode="json"),
                "trace": ["roadmap:empty"],
            }

        # 단계 2: 과목 메타 로드 (외부 I/O — 진입점 단 1 곳)
        courses = self._repo.list_for_tracks(track_ids)
        if not courses:
            return {
                "roadmap": _empty_roadmap(combo_key).model_dump(mode="json"),
                "trace": ["roadmap:empty"],
            }

        # 단계 3: 교양 제외 — 본 시스템 추천 대상은 전공 (필수 + 선택) 만
        major_only = _filter_recommended_types(courses)
        if not major_only:
            return {
                "roadmap": _empty_roadmap(combo_key).model_dump(mode="json"),
                "trace": ["roadmap:empty"],
            }

        # 단계 4: 이수 과목 분리 — 후보에서는 제외, 선수 만족 신호로는 보존
        completed_courses = normalized.get("completed_courses") or []
        completed_set: set[str] = set(completed_courses)
        remaining = _filter_completed(major_only, completed_set)
        if not remaining:
            return {
                "roadmap": _empty_roadmap(combo_key).model_dump(mode="json"),
                "trace": ["roadmap:all_completed"],
            }

        # 단계 5: 추천 점수 산출 + 점수 순 정렬 + 잔여 학기 분산
        score_by_id = {c.course_id: _compute_course_score(c, track_ids) for c in remaining}
        ordered = _order_candidates(remaining, score_by_id)
        current_semester = state.get("current_semester") or normalized.get("current_semester") or 1
        semester_plans = _distribute_across_semesters(
            ordered,
            current_semester=current_semester,
            completed_set=completed_set,
            config=self._config,
            score_by_id=score_by_id,
        )

        # 단계 6: 학습 깊이 라벨 출력 재구성 (자연어 설명 노드 호환)
        stages = _extract_stages_from_semesters(semester_plans)

        # 단계 7: Roadmap 객체로 변환 후 state 부분 반환
        roadmap = Roadmap(
            stages=stages,
            semesters=semester_plans,
            derived_from_combo_key=combo_key,
        )

        return {
            "roadmap": roadmap.model_dump(mode="json"),
            "trace": ["roadmap:ok"],
        }


__all__ = ["CourseRepository", "RoadmapNode"]
