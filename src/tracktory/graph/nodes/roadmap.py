"""트랙 시너지 결과로부터 4 단계 학습 로드맵을 생성한다.

추천된 1 순위 트랙 조합에 권장되는 과목들을 학사 데이터에서 끌어와, 이수
과목·선수과목 관계·학기 용량·졸업 총 학점이라는 학사 제약을 모두 만족하는
4 단계 (foundation → core → application → industry) 추천 로드맵으로 묶는다.

처리 흐름 (7 단계):
    1. 입력 검증 — ``primary_combos`` / ``normalized_profile`` 부재 시 안전 종료.
    2. 1 순위 조합의 두 트랙 식별자 수집.
    3. 과목 메타 로드 (외부 I/O — 진입점 단 1 곳).
    4. 이수 과목을 후보에서 제외 (선수 만족 신호로는 보존).
    5. 단계별 그룹화 + 단계 순서로 선수과목 위배 컷 + 학기 용량 cap.
    6. 4 단계 누적으로 졸업 총 학점 cap.
    7. state 부분 반환.

부작용 격리:
    - ``course_repo`` 호출은 ``__call__`` 단 1 곳.
    - 순수 계산 함수는 모두 module-level — mock 없이 단위 테스트 가능.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Protocol

from tracktory.graph.models import (
    Course,
    Roadmap,
    RoadmapConfig,
    RoadmapCourse,
    RoadmapStage,
)
from tracktory.graph.state import GraphState

_DEFAULT_ROADMAP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "roadmap.yaml"

StageLabel = Literal["foundation", "core", "application", "industry"]
_STAGE_ORDER: tuple[StageLabel, ...] = ("foundation", "core", "application", "industry")

logger = logging.getLogger(__name__)


class CourseRepository(Protocol):
    """과목 메타 데이터 접근 인터페이스.

    학습 로드맵 노드는 본 Protocol 만 의존하고 구체 구현은 외부에서 주입받는다.
    단위 테스트에서는 mock 으로 교체하여 외부 I/O 없이 검증한다. 운영에서는
    한성대 강의계획서 적재 결과로부터 ``Course`` 를 채워 반환하는 구현이 들어간다.

    Repository 구현체의 책임:
        - ``stage`` (4 단계 학습 깊이) 분류
        - ``prereq_ids`` (정규화된 선수과목 식별자) 추출
        - ``priority`` (낮은 숫자 우선) 할당
    """

    def list_for_tracks(self, track_ids: list[str]) -> list[Course]:
        """주어진 트랙 식별자들에 권장되는 과목 목록을 반환한다.

        두 트랙 모두에 권장되는 과목이 있더라도 ``course_id`` 기준으로 dedup 된
        리스트를 반환할 책임은 구현체에 있다.
        """
        ...


# ---------------------------------------------------------------------------
# 순수 계산 함수
# ---------------------------------------------------------------------------


def _filter_completed(courses: list[Course], completed_set: set[str]) -> list[Course]:
    """이수 과목 집합에 포함된 후보를 제거한다.

    이수 과목 자체는 추천 결과에 노출하지 않지만, 후수 과목의 선수 만족 신호
    로는 별도로 사용되므로 호출자는 ``completed_set`` 을 그대로 보존한다.
    """
    return [course for course in courses if course.course_id not in completed_set]


def _group_by_stage(courses: list[Course]) -> dict[StageLabel, list[Course]]:
    """4 단계 라벨로 후보를 그룹화한다. 없는 단계는 빈 리스트를 보장한다."""
    grouped: dict[StageLabel, list[Course]] = {stage: [] for stage in _STAGE_ORDER}
    for course in courses:
        grouped[course.stage].append(course)
    return grouped


def _validate_prereqs(candidates: list[Course], satisfied_set: set[str]) -> list[Course]:
    """선수과목이 만족된 후보만 통과시킨다.

    ``satisfied_set`` 은 이수 과목 + 이전 단계에서 이미 채택한 과목의 합집합이다.
    후보의 모든 ``prereq_ids`` 가 ``satisfied_set`` 의 부분집합일 때만 통과한다.
    """
    return [c for c in candidates if all(pid in satisfied_set for pid in c.prereq_ids)]


def _apply_capacity_cap(candidates: list[Course], max_credits: int) -> list[Course]:
    """학기 학점 cap 까지 priority 오름차순으로 채택한다.

    동일 priority 는 ``course_id`` 알파벳 순으로 deterministic 정렬한다. 후보를
    탐색하다 다음 과목을 더하면 cap 을 넘는 경우, 그 과목은 건너뛰고 더 작은
    과목으로 cap 안을 마저 채울 수 있도록 계속 진행한다.
    """
    ordered = sorted(candidates, key=lambda c: (c.priority, c.course_id))
    accepted: list[Course] = []
    accumulated = 0
    for course in ordered:
        if accumulated + course.credits > max_credits:
            continue
        accepted.append(course)
        accumulated += course.credits
    return accepted


def _apply_graduation_cap(
    stage_groups: dict[StageLabel, list[Course]], total_cap: int
) -> dict[StageLabel, list[Course]]:
    """4 단계 누적 학점이 졸업 cap 을 넘지 않도록 단계 순서대로 추가한다.

    단계 순서 (foundation → industry) 로 순회하면서 누적 학점이 cap 안에 들어오는
    범위까지 채택한다. 한 단계 안에서는 priority 오름차순 (큰 숫자 = 우선순위 낮음)
    으로 정렬한 뒤, cap 초과 직전까지만 채택하고 나머지는 컷한다. 누적이 cap 에
    도달한 이후의 단계는 모두 빈 리스트로 둔다.
    """
    result: dict[StageLabel, list[Course]] = {stage: [] for stage in _STAGE_ORDER}
    accumulated = 0
    for stage in _STAGE_ORDER:
        if accumulated >= total_cap:
            continue
        ordered = sorted(stage_groups.get(stage, []), key=lambda c: (c.priority, c.course_id))
        for course in ordered:
            if accumulated + course.credits > total_cap:
                continue
            result[stage].append(course)
            accumulated += course.credits
    return result


def _to_roadmap_courses(courses: list[Course]) -> list[RoadmapCourse]:
    """``Course`` 를 사용자 노출용 ``RoadmapCourse`` 로 변환한다.

    Repository 가 부여한 ``priority`` 를 그대로 패스하여 "1 순위 / 2 순위" 표기
    의미를 보존한다. 단계 안 표시 순서를 결정론적으로 만들기 위해 priority →
    course_id 순으로 재정렬한다.
    """
    ordered = sorted(courses, key=lambda c: (c.priority, c.course_id))
    return [
        RoadmapCourse(course_id=c.course_id, course_name=c.course_name, priority=c.priority)
        for c in ordered
    ]


def _empty_roadmap() -> Roadmap:
    """빈 단계로만 구성된 안전 종료용 로드맵."""
    return Roadmap(stages=[RoadmapStage(stage=stage, courses=[]) for stage in _STAGE_ORDER])


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


# ---------------------------------------------------------------------------
# 노드 진입점
# ---------------------------------------------------------------------------


class RoadmapNode:
    """학습 로드맵 노드 — 트랙 시너지 1 순위 결과로부터 4 단계 로드맵 생성.

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

        primary_combos: list[dict[str, Any]] | None = state.get("primary_combos")
        track_ids = _extract_primary_track_ids(primary_combos or [])
        if not track_ids:
            # 1 학년 트랙 미선택 또는 시너지 노드 산출 부재 — 빈 로드맵으로 그래프 흐름 유지
            return {
                "roadmap": _empty_roadmap().model_dump(mode="json"),
                "trace": ["roadmap:empty"],
            }

        # 단계 2: 과목 메타 로드 (외부 I/O — 진입점 단 1 곳)
        courses = self._repo.list_for_tracks(track_ids)
        if not courses:
            return {
                "roadmap": _empty_roadmap().model_dump(mode="json"),
                "trace": ["roadmap:empty"],
            }

        # 단계 3: 이수 과목 분리 — 후보에서는 제외, 선수 만족 신호로는 보존
        completed_courses = normalized.get("completed_courses") or []
        completed_set: set[str] = set(completed_courses)
        remaining = _filter_completed(courses, completed_set)
        grouped = _group_by_stage(remaining)

        # 단계 4: 단계 순서로 선수과목 위배 컷 + 학기 용량 cap
        max_credits = self._config.capacity.max_credits_per_semester_default
        per_stage_accepted: dict[StageLabel, list[Course]] = {stage: [] for stage in _STAGE_ORDER}
        satisfied_set: set[str] = set(completed_set)
        for stage in _STAGE_ORDER:
            candidates = grouped[stage]
            valid = _validate_prereqs(candidates, satisfied_set)
            accepted = _apply_capacity_cap(valid, max_credits)
            per_stage_accepted[stage] = accepted
            satisfied_set.update(c.course_id for c in accepted)

        # 단계 5: 4 단계 누적으로 졸업 총 학점 cap
        total_cap = self._config.graduation.total_credits_two_tracks
        capped = _apply_graduation_cap(per_stage_accepted, total_cap)

        # 단계 6: Roadmap 객체로 변환 후 state 부분 반환
        roadmap = Roadmap(
            stages=[
                RoadmapStage(stage=stage, courses=_to_roadmap_courses(capped[stage]))
                for stage in _STAGE_ORDER
            ]
        )

        return {
            "roadmap": roadmap.model_dump(mode="json"),
            "trace": ["roadmap:ok"],
        }
