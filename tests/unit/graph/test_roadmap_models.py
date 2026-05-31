"""Roadmap 도메인 모델 invariant 검증.

학습 깊이 단조 증가 4 단계 (foundation → core → application → industry)
순서·중복·누락이 검증으로 차단되는지, 과목 단위의 우선순위 범위가 강제되는지,
학기 단위 분산 plan 의 학기·학년 범위가 강제되는지, 직렬화 round-trip 이
동일 모델을 복원하는지 확인한다.
"""

import pytest
from pydantic import ValidationError

from tracktory.graph.models import Roadmap, RoadmapCourse, RoadmapStage, SemesterPlan


def _course(
    course_id: str,
    *,
    stage: str = "foundation",
    score: float = 0.4,
    credits: int = 3,
) -> RoadmapCourse:
    return RoadmapCourse(
        course_id=course_id,
        course_name=f"과목-{course_id}",
        credits=credits,
        stage=stage,  # type: ignore[arg-type]
        score=score,
    )


def _full_stages() -> list[RoadmapStage]:
    return [
        RoadmapStage(stage="foundation", courses=[_course("c1", stage="foundation")]),
        RoadmapStage(
            stage="core",
            courses=[
                _course("c2", stage="core", score=0.6),
                _course("c3", stage="core", score=0.4),
            ],
        ),
        RoadmapStage(stage="application", courses=[]),
        RoadmapStage(stage="industry", courses=[]),
    ]


def test_roadmap_accepts_four_stages_in_canonical_order() -> None:
    roadmap = Roadmap(stages=_full_stages())
    assert [s.stage for s in roadmap.stages] == [
        "foundation",
        "core",
        "application",
        "industry",
    ]


def test_roadmap_rejects_missing_stage() -> None:
    """단계 1 개 누락은 ValidationError."""
    stages = _full_stages()[:3]  # industry 누락

    with pytest.raises(ValidationError):
        Roadmap(stages=stages)


def test_roadmap_rejects_out_of_order_stages() -> None:
    """순서 뒤바뀜은 ValidationError."""
    stages = _full_stages()
    stages[0], stages[1] = stages[1], stages[0]  # core ↔ foundation

    with pytest.raises(ValidationError):
        Roadmap(stages=stages)


def test_roadmap_rejects_duplicate_stage() -> None:
    """같은 단계 중복은 ValidationError."""
    stages = [
        RoadmapStage(stage="foundation", courses=[]),
        RoadmapStage(stage="foundation", courses=[]),
        RoadmapStage(stage="application", courses=[]),
        RoadmapStage(stage="industry", courses=[]),
    ]

    with pytest.raises(ValidationError):
        Roadmap(stages=stages)


def test_roadmap_stage_rejects_unknown_label() -> None:
    """Literal 외 값은 RoadmapStage 단계에서 차단."""
    with pytest.raises(ValidationError):
        RoadmapStage(stage="advanced", courses=[])  # type: ignore[arg-type]


def test_roadmap_stage_allows_empty_courses() -> None:
    stage = RoadmapStage(stage="industry", courses=[])
    assert stage.courses == []


def test_roadmap_course_rejects_zero_credits() -> None:
    with pytest.raises(ValidationError):
        RoadmapCourse(course_id="c1", course_name="과목", credits=0, stage="foundation", score=0.4)


def test_roadmap_course_rejects_score_below_zero() -> None:
    with pytest.raises(ValidationError):
        RoadmapCourse(course_id="c1", course_name="과목", credits=3, stage="foundation", score=-0.1)


def test_roadmap_course_rejects_score_above_one() -> None:
    with pytest.raises(ValidationError):
        RoadmapCourse(course_id="c1", course_name="과목", credits=3, stage="foundation", score=1.1)


def test_roadmap_course_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        RoadmapCourse(course_id="", course_name="과목", credits=3, stage="foundation", score=0.4)


def test_roadmap_roundtrip_model_dump_validate() -> None:
    """model_dump(mode='json') → model_validate 가 원본과 동등한 객체를 복원한다."""
    original = Roadmap(stages=_full_stages())
    dumped = original.model_dump(mode="json")
    restored = Roadmap.model_validate(dumped)

    assert restored == original
    assert len(restored.stages) == 4
    assert restored.stages[1].courses[0].score == 0.6
    assert restored.stages[1].courses[1].score == 0.4


def test_roadmap_defaults_combo_key_to_none() -> None:
    """파생 조합 식별자는 명시되지 않으면 None — 안전 종료 분기를 위한 기본값."""
    roadmap = Roadmap(stages=_full_stages())
    assert roadmap.derived_from_combo_key is None


def test_roadmap_preserves_combo_key_through_roundtrip() -> None:
    """파생 조합 식별자가 dump → validate 라운드트립으로 보존된다."""
    original = Roadmap(stages=_full_stages(), derived_from_combo_key="big_data::korean_edu")
    dumped = original.model_dump(mode="json")
    restored = Roadmap.model_validate(dumped)

    assert restored.derived_from_combo_key == "big_data::korean_edu"
    assert restored == original


# ---------------------------------------------------------------------------
# SemesterPlan + Roadmap.semesters 검증
# ---------------------------------------------------------------------------


def test_semester_plan_accepts_valid_semester_and_grade() -> None:
    plan = SemesterPlan(semester=4, grade=2, courses=[_course("c1")], credits_total=3)
    assert plan.semester == 4
    assert plan.grade == 2
    assert plan.cap_reached is False
    assert plan.graduation_insufficient is False


def test_semester_plan_rejects_semester_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SemesterPlan(semester=9, grade=4)


def test_semester_plan_rejects_grade_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SemesterPlan(semester=1, grade=5)


def test_semester_plan_rejects_negative_credits() -> None:
    with pytest.raises(ValidationError):
        SemesterPlan(semester=1, grade=1, credits_total=-1)


def test_roadmap_defaults_semesters_to_empty_list() -> None:
    """학기 분산 결과가 비어 있어도 모델은 valid 하다 (안전 종료 경로)."""
    roadmap = Roadmap(stages=_full_stages())
    assert roadmap.semesters == []


def test_roadmap_roundtrip_preserves_semesters_with_markers() -> None:
    """학기 분산 + 마커 (cap_reached / graduation_insufficient) 가 라운드트립으로 보존된다."""
    plans = [
        SemesterPlan(semester=3, grade=2, courses=[_course("c1")], credits_total=3),
        SemesterPlan(
            semester=4,
            grade=2,
            courses=[_course("c2")],
            credits_total=3,
            cap_reached=True,
        ),
    ]
    original = Roadmap(stages=_full_stages(), semesters=plans)
    dumped = original.model_dump(mode="json")
    restored = Roadmap.model_validate(dumped)

    assert len(restored.semesters) == 2
    assert restored.semesters[1].cap_reached is True
    assert restored.semesters[1].graduation_insufficient is False
    assert restored == original
