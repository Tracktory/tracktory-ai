"""``RoadmapNode.__call__`` 진입점 통합 테스트.

``CourseRepository`` 를 in-memory factory 로 교체하여 외부 I/O 없이 검증한다.
실 ``roadmap.yaml`` 의 학기 용량 18 학점 / 졸업 전공 78 학점 / 학년별 학점
범위 (1: 8~16, 2~3: 24~36, 4: 0~99) 를 그대로 사용한다.
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
    current_semester: int | None = None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
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
    if current_semester is not None:
        profile["current_semester"] = current_semester
    return profile


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
# 시나리오 1: 정상 경로 — 학습 깊이 라벨 출력 + 학기 분산 출력 동시 검증
# ---------------------------------------------------------------------------


def test_node_returns_both_stages_and_semesters_outputs(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course("c1", credits=3, stage="foundation", priority=1),
            make_course("c2", credits=3, stage="foundation", priority=2),
            make_course("c3", credits=3, stage="core", prereq_ids=["c1"], priority=1),
            make_course("c5", credits=3, stage="application", prereq_ids=["c3"], priority=1),
        ],
        "T_B": [
            make_course("c4", credits=3, stage="industry", priority=1),
        ],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(completed_courses=["c2"], current_semester=1),
            "current_semester": 1,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    assert result["trace"] == ["roadmap:ok"]
    assert result["roadmap"]["derived_from_combo_key"] == "T_A::T_B"

    # 학습 깊이 라벨 출력 — 4 단계 순서 유지
    stages = result["roadmap"]["stages"]
    assert [s["stage"] for s in stages] == ["foundation", "core", "application", "industry"]
    assert {c["course_id"] for c in stages[0]["courses"]} == {"c1"}
    assert {c["course_id"] for c in stages[1]["courses"]} == {"c3"}
    assert {c["course_id"] for c in stages[2]["courses"]} == {"c5"}
    assert {c["course_id"] for c in stages[3]["courses"]} == {"c4"}

    # 학기 분산 출력 — 모든 채택 과목이 학기에 등장하고 학기 번호가 시작점 이상
    semesters = result["roadmap"]["semesters"]
    assert len(semesters) >= 1
    assert all(plan["semester"] >= 1 for plan in semesters)
    all_semester_courses = {c["course_id"] for plan in semesters for c in plan["courses"]}
    assert all_semester_courses == {"c1", "c3", "c5", "c4"}


# ---------------------------------------------------------------------------
# 시나리오 2: 1 학년 (트랙 미선택) + primary_combos None
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
    assert result["roadmap"]["semesters"] == []


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
            "normalized_profile": _normalized_profile(completed_courses=["c1"], current_semester=3),
            "current_semester": 3,
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
            "normalized_profile": _normalized_profile(current_semester=3),
            "current_semester": 3,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    core_ids = {c["course_id"] for c in result["roadmap"]["stages"][1]["courses"]}
    assert "c2" not in core_ids  # MISSING prereq 위배 → 컷
    assert core_ids == {"c3"}


# ---------------------------------------------------------------------------
# 시나리오 6: 학기 학사 cap (18 학점) 미초과
# ---------------------------------------------------------------------------


def test_node_respects_per_semester_capacity_cap(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """어느 학기든 학점 합이 18 (한성대 학사 학기 cap) 을 초과하지 않는다."""
    # 학년 4 (학기 7) 시작점 — 학년 4 cap = 99 라 학기 cap 만 작동
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course(f"f{i}", credits=3, stage="foundation", priority=i, available_grades=[4])
            for i in range(1, 9)
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(current_semester=7),
            "current_semester": 7,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    for plan in result["roadmap"]["semesters"]:
        assert plan["credits_total"] <= 18

    # priority 오름차순 — 학기 7 에 f1..f6 (18 학점), 학기 8 에 f7..f8 (6 학점)
    semester_7 = next(p for p in result["roadmap"]["semesters"] if p["semester"] == 7)
    semester_7_ids = [c["course_id"] for c in semester_7["courses"]]
    assert semester_7_ids == ["f1", "f2", "f3", "f4", "f5", "f6"]


# ---------------------------------------------------------------------------
# 시나리오 7: 졸업 전공 합산 cap (78 학점) 미초과 + cap_reached 마커
# ---------------------------------------------------------------------------


def test_node_respects_graduation_total_cap_and_sets_marker(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """누적 추천 학점이 78 을 초과하지 않고, 도달 학기에 cap_reached 마커가 켜진다."""
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            *[make_course(f"f{i}", credits=3, stage="foundation", priority=i) for i in range(1, 7)],
            *[make_course(f"c{i}", credits=3, stage="core", priority=i) for i in range(1, 9)],
            *[
                make_course(f"a{i}", credits=3, stage="application", priority=i)
                for i in range(1, 11)
            ],
            *[make_course(f"i{i}", credits=3, stage="industry", priority=i) for i in range(1, 5)],
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(current_semester=1),
            "current_semester": 1,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    total_credits = sum(p["credits_total"] for p in result["roadmap"]["semesters"])
    assert total_credits <= 78

    cap_reached_plans = [p for p in result["roadmap"]["semesters"] if p["cap_reached"]]
    assert len(cap_reached_plans) == 1
    assert cap_reached_plans[0]["credits_total"] > 0


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
    assert result["roadmap"]["semesters"] == []


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
            "normalized_profile": _normalized_profile(current_semester=1),
            "current_semester": 1,
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
    assert result["roadmap"]["semesters"] == []


# ---------------------------------------------------------------------------
# 시나리오 11: 1 학년 학년 제약 — core / application / industry 미노출
# ---------------------------------------------------------------------------


def test_freshman_does_not_see_higher_grade_courses_in_year_one(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """1 학년 학기에는 학년 제약상 ``available_grades`` 가 2+ 인 과목이 노출되지 않는다.

    한성대 학사 운영상 1 학년은 전공 기초와 1 학년 전용 전공 선택만 수강 가능하다.
    학기 분산 알고리즘이 학년 제약을 자연 강제하는지 검증한다.
    """
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course("f1", credits=3, stage="foundation", priority=1, available_grades=[1]),
            make_course("f2", credits=3, stage="foundation", priority=2, available_grades=[1]),
            # core 는 2 학년부터 — 1 학년 학기 (1, 2) 에는 노출되지 않아야 함
            make_course("c1", credits=3, stage="core", priority=1, available_grades=[2, 3, 4]),
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(current_semester=1),
            "current_semester": 1,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    year_one_semesters = [p for p in result["roadmap"]["semesters"] if p["grade"] == 1]
    year_one_course_ids = {
        course["course_id"] for plan in year_one_semesters for course in plan["courses"]
    }
    assert "c1" not in year_one_course_ids
    assert year_one_course_ids == {"f1", "f2"}

    # core 는 학년 2+ 학기에 등장
    year_two_plus_ids = {
        c["course_id"]
        for plan in result["roadmap"]["semesters"]
        if plan["grade"] >= 2
        for c in plan["courses"]
    }
    assert "c1" in year_two_plus_ids


# ---------------------------------------------------------------------------
# 시나리오 12: 학년별 학점 ceiling 강제 (1 학년 max 16)
# ---------------------------------------------------------------------------


def test_grade_credits_ceiling_respected_for_year_one(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """1 학년 두 학기 학점 합이 학년 max (16) 를 초과하지 않는다."""
    # 1 학년 전용 후보 8 개 * 3 학점 = 24 학점 (학년 cap 16 초과)
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course(f"f{i}", credits=3, stage="foundation", priority=i, available_grades=[1])
            for i in range(1, 9)
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(current_semester=1),
            "current_semester": 1,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    year_one_total = sum(
        p["credits_total"] for p in result["roadmap"]["semesters"] if p["grade"] == 1
    )
    assert year_one_total <= 16


# ---------------------------------------------------------------------------
# 시나리오 13: 졸업 학점 미달 — graduation_insufficient 마커
# ---------------------------------------------------------------------------


def test_graduation_insufficient_marker_when_remaining_too_short(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """잔여 학기가 너무 짧아 졸업 학점 (78) 에 미달하면 마지막 학기에 마커가 켜진다.

    학기 8 부터 시작 (잔여 1 학기) + 후보 학점 < 78 = 졸업 미달 케이스.
    """
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course("c1", credits=3, stage="industry", priority=1),
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(current_semester=8),
            "current_semester": 8,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    plans = result["roadmap"]["semesters"]
    assert len(plans) == 1
    assert plans[0]["semester"] == 8
    assert plans[0]["graduation_insufficient"] is True
    assert plans[0]["cap_reached"] is False


# ---------------------------------------------------------------------------
# 시나리오 14: 교양 제외 — course_type="교양" 후보는 추천 결과에서 제외
# ---------------------------------------------------------------------------


def test_node_excludes_general_education_courses(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """교양 분류 과목은 사용자 자율 구성 영역으로 추천 결과에 노출되지 않는다."""
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course("major", credits=3, stage="foundation", course_type="전공필수"),
            make_course("elective", credits=3, stage="foundation", course_type="전공선택"),
            make_course("general", credits=3, stage="foundation", course_type="교양"),
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(current_semester=1),
            "current_semester": 1,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    all_course_ids = {
        c["course_id"] for plan in result["roadmap"]["semesters"] for c in plan["courses"]
    }
    assert "general" not in all_course_ids
    assert all_course_ids == {"major", "elective"}


# ---------------------------------------------------------------------------
# 시나리오 15: current_semester=4 (페르소나: 2 학년 트랙 확정) — 시작점 분기
# ---------------------------------------------------------------------------


def test_distribution_starts_at_current_semester(
    make_course: Callable[..., Course],
    make_course_repo: Callable[..., CourseRepository],
    real_roadmap_yaml_path: Path,
) -> None:
    """학기 분산은 학생의 현재 학기부터 시작하며 그 이전 학기는 plan 에 없다."""
    courses_by_track: dict[str, list[Course]] = {
        "T_A": [
            make_course(f"c{i}", credits=3, stage="foundation", priority=i) for i in range(1, 4)
        ],
        "T_B": [],
    }
    node = _build_node(courses_by_track, real_roadmap_yaml_path, make_course_repo)

    result = node(
        {
            "normalized_profile": _normalized_profile(current_semester=4),
            "current_semester": 4,
            "primary_combos": _primary_combos("T_A", "T_B"),
        }
    )

    plans = result["roadmap"]["semesters"]
    assert plans, "expected at least one semester plan"
    assert all(plan["semester"] >= 4 for plan in plans)
    assert plans[0]["semester"] == 4
    assert plans[0]["grade"] == 2
