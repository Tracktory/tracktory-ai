"""``RoadmapNode.__call__`` 진입점 통합 테스트.

``CourseRepository`` 를 in-memory factory 로 교체하여 외부 I/O 없이 검증한다.
실 ``roadmap.yaml`` 의 학기 용량 18 학점 / 졸업 30 학점 cap 을 그대로 사용한다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tracktory.graph.models import Course
from tracktory.graph.nodes.roadmap import CourseRepository, RoadmapNode


def _normalized_profile(
    *,
    completed_courses: list[str] | None = None,
    current_tracks: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "admission_year": 2025,
        "college": "C1",
        "department": "컴퓨터공학부",
        "current_tracks": current_tracks or [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": completed_courses or [],
    }


def _primary_combos(
    track_a_id: str = "T_A",
    track_b_id: str = "T_B",
    *,
    college_id: str = "C1",
    department_id: str = "D1",
) -> list[dict[str, Any]]:
    """1 순위 시너지 조합 한 건을 노드 시너지 노드가 흘리는 dict 형식으로 구성."""

    def _track_dict(track_id: str) -> dict[str, Any]:
        return {
            "college_id": college_id,
            "department_id": department_id,
            "major_id": department_id,
            "track_id": track_id,
            "track_name": track_id,
            "course_ids": [],
            "meta_text": "",
            "meta_vector": [],
            "competencies": [],
            "tech_stacks": [],
        }

    ids_sorted = sorted([track_a_id, track_b_id])
    return [
        {
            "combo": {
                "track_a": _track_dict(track_a_id),
                "track_b": _track_dict(track_b_id),
                "combo_key": "::".join(ids_sorted),
            },
            "synergy_score": 0.8,
            "slot_type": "primary",
            "rank": 1,
        }
    ]


def _build_node(
    courses_by_track: dict[str, list[Course]],
    real_roadmap_yaml_path: Path,
    make_course_repo: Callable[..., CourseRepository],
) -> RoadmapNode:
    repo = make_course_repo(courses_by_track)
    return RoadmapNode(course_repo=repo, config_path=real_roadmap_yaml_path)


# ---------------------------------------------------------------------------
# 시나리오 1: 정상 경로
# ---------------------------------------------------------------------------


def test_node_returns_four_stages_in_canonical_order_with_completed_filtered(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course("c1", credits=3, stage="foundation", priority=1),
            make_course("c2", credits=3, stage="foundation", priority=2),
            make_course(
                "c3",
                credits=3,
                stage="core",
                prereq_ids=["c1"],
                priority=1,
            ),
            make_course(
                "c5",
                credits=3,
                stage="application",
                prereq_ids=["c3"],
                priority=1,
            ),
        ],
        "T_B": [
            make_course("c4", credits=3, stage="industry", priority=1),
        ],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(completed_courses=["c2"]),
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    assert result["trace"] == ["roadmap:ok"]
    assert result["roadmap"]["derived_from_combo_key"] == "T_A::T_B"
    stages = result["roadmap"]["stages"]
    assert [s["stage"] for s in stages] == ["foundation", "core", "application", "industry"]

    foundation_ids = {course["course_id"] for course in stages[0]["courses"]}
    assert foundation_ids == {"c1"}  # c2 는 이수 완료로 제외

    core_ids = {course["course_id"] for course in stages[1]["courses"]}
    assert core_ids == {"c3"}

    application_ids = {course["course_id"] for course in stages[2]["courses"]}
    assert application_ids == {"c5"}

    industry_ids = {course["course_id"] for course in stages[3]["courses"]}
    assert industry_ids == {"c4"}


# ---------------------------------------------------------------------------
# 시나리오 2: 1 학년 (current_tracks=[]) + primary_combos None
# ---------------------------------------------------------------------------


def test_node_returns_empty_roadmap_when_primary_combos_missing(
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    node = _build_node({}, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(current_tracks=[]),
            "primary_combos": None,
        }
    )

    assert result["trace"] == ["roadmap:empty"]
    assert result["roadmap"]["derived_from_combo_key"] is None
    stages = result["roadmap"]["stages"]
    assert [s["stage"] for s in stages] == ["foundation", "core", "application", "industry"]
    assert all(s["courses"] == [] for s in stages)


# ---------------------------------------------------------------------------
# 시나리오 3: normalized_profile 부재
# ---------------------------------------------------------------------------


def test_node_skips_when_normalized_profile_missing(
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    node = _build_node({}, real_roadmap_yaml_path, make_course_repo)

    result = node({"primary_combos": _primary_combos()})

    assert result["trace"] == ["roadmap:skip"]
    assert "roadmap" not in result
    assert "normalized_profile missing" in result["errors"][0]


# ---------------------------------------------------------------------------
# 시나리오 4: 이수 과목이 후수의 prereq 만족 신호
# ---------------------------------------------------------------------------


def test_completed_course_satisfies_prereq_for_downstream(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """이수 과목은 추천에는 안 나오지만, 후수 과목의 prereq 만족 신호로 사용된다."""
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course("c1", credits=3, stage="foundation"),
            make_course("c2", credits=3, stage="core", prereq_ids=["c1"]),
        ],
        "T_B": [
            make_course("c3", credits=3, stage="application", prereq_ids=["c2"]),
            make_course("c4", credits=3, stage="industry"),
        ],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(completed_courses=["c1"]),
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    stages = result["roadmap"]["stages"]
    foundation_ids = {c["course_id"] for c in stages[0]["courses"]}
    assert foundation_ids == set()  # c1 은 이수 완료
    core_ids = {c["course_id"] for c in stages[1]["courses"]}
    assert core_ids == {"c2"}  # c1 (이수) 가 prereq 만족 신호로 흘러 c2 가 추천됨
    application_ids = {c["course_id"] for c in stages[2]["courses"]}
    assert application_ids == {"c3"}  # c2 가 이전 단계 추천이므로 c3 prereq 만족


# ---------------------------------------------------------------------------
# 시나리오 5: prereq 위배 컷
# ---------------------------------------------------------------------------


def test_node_cuts_course_with_unsatisfied_prereq(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """이수 / 이전 단계 어디에도 없는 선수과목을 가진 후보는 컷된다."""
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course(
                "c2",
                credits=3,
                stage="core",
                prereq_ids=["MISSING"],
                priority=1,
            ),
            make_course("c3", credits=3, stage="core", priority=2),
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(),
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    core_ids = {c["course_id"] for c in result["roadmap"]["stages"][1]["courses"]}
    assert "c2" not in core_ids  # MISSING prereq 위배 → 컷
    assert core_ids == {"c3"}


# ---------------------------------------------------------------------------
# 시나리오 6: 학기 용량 cap (18 학점)
# ---------------------------------------------------------------------------


def test_node_respects_per_stage_capacity_cap(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """한 단계 학점 합이 18 을 초과하지 않는다. priority 오름차순으로 채택."""
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            # foundation 후보 8 개, 각 3 학점 = 24 학점 (18 cap 초과)
            make_course(f"f{i}", credits=3, stage="foundation", priority=i)
            for i in range(1, 9)
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(),
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    foundation = result["roadmap"]["stages"][0]["courses"]
    total = sum(3 for _ in foundation)  # 모든 과목 3 학점
    assert total <= 18
    # priority 오름차순으로 채택 — f1..f6 (18 학점)
    ids = [c["course_id"] for c in foundation]
    assert ids[:6] == ["f1", "f2", "f3", "f4", "f5", "f6"]


# ---------------------------------------------------------------------------
# 시나리오 7: 졸업 총합 cap (30 학점)
# ---------------------------------------------------------------------------


def test_node_respects_graduation_total_cap(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """4 단계 누적이 30 학점을 초과하지 않는다."""
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            # 각 단계마다 6 개, 각 3 학점 = 18 학점 가능 / 단계당 cap 18 / 4 단계 합 72 가능
            *[make_course(f"f{i}", credits=3, stage="foundation", priority=i) for i in range(1, 7)],
            *[make_course(f"c{i}", credits=3, stage="core", priority=i) for i in range(1, 7)],
            *[
                make_course(f"a{i}", credits=3, stage="application", priority=i)
                for i in range(1, 7)
            ],
            *[make_course(f"i{i}", credits=3, stage="industry", priority=i) for i in range(1, 7)],
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(),
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    stages = result["roadmap"]["stages"]
    total_credits = sum(3 for stage in stages for _ in stage["courses"])
    assert total_credits <= 30


# ---------------------------------------------------------------------------
# 시나리오 8: Repository 빈 결과
# ---------------------------------------------------------------------------


def test_node_returns_empty_roadmap_when_repo_returns_no_courses(
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    node = _build_node({"T_A": [], "T_B": []}, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(),
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    assert result["trace"] == ["roadmap:empty"]
    assert result["roadmap"]["derived_from_combo_key"] == "T_A::T_B"
    stages = result["roadmap"]["stages"]
    assert all(s["courses"] == [] for s in stages)


# ---------------------------------------------------------------------------
# 시나리오 9: 동점 priority — course_id 알파벳 순 deterministic
# ---------------------------------------------------------------------------


def test_node_breaks_priority_ties_by_course_id_alphabetical(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """같은 priority 후보들은 ``course_id`` 알파벳 순으로 안정 정렬된다."""
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course("zeta", credits=3, stage="foundation", priority=1),
            make_course("alpha", credits=3, stage="foundation", priority=1),
            make_course("mike", credits=3, stage="foundation", priority=1),
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(),
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    foundation_ids = [c["course_id"] for c in result["roadmap"]["stages"][0]["courses"]]
    assert foundation_ids == ["alpha", "mike", "zeta"]


# ---------------------------------------------------------------------------
# 시나리오 10: 이수 과목이 모든 후보를 cover — roadmap:all_completed
# ---------------------------------------------------------------------------


def test_node_marks_all_completed_when_completed_covers_all_candidates(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """Repository 가 후보를 반환했지만 사용자가 모두 이수한 경우 trace 가 구분된다.

    Repository 빈 결과 (``roadmap:empty``) 와는 다른 케이스 — 사용자 시점에서는
    동일한 빈 로드맵이지만 운영 디버깅에서 두 케이스를 구분해야 한다.
    """
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course("c1", credits=3, stage="foundation", priority=1),
            make_course("c2", credits=3, stage="core", priority=1),
        ],
        "T_B": [
            make_course("c3", credits=3, stage="application", priority=1),
        ],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(completed_courses=["c1", "c2", "c3"]),
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    assert result["trace"] == ["roadmap:all_completed"]
    assert result["roadmap"]["derived_from_combo_key"] == "T_A::T_B"
    stages = result["roadmap"]["stages"]
    assert [s["stage"] for s in stages] == ["foundation", "core", "application", "industry"]
    assert all(s["courses"] == [] for s in stages)
