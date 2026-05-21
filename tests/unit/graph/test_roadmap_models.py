"""Roadmap 도메인 모델 invariant 검증.

학습 깊이 단조 증가 4 단계 (foundation → core → application → industry)
순서·중복·누락이 검증으로 차단되는지, 과목 단위의 우선순위 범위가 강제되는지,
직렬화 round-trip 이 동일 모델을 복원하는지 확인한다.
"""

import pytest
from pydantic import ValidationError

from tracktory.graph.models import Roadmap, RoadmapCourse, RoadmapStage


def _course(course_id: str, priority: int = 1) -> RoadmapCourse:
    return RoadmapCourse(course_id=course_id, course_name=f"과목-{course_id}", priority=priority)


def _full_stages() -> list[RoadmapStage]:
    return [
        RoadmapStage(stage="foundation", courses=[_course("c1")]),
        RoadmapStage(stage="core", courses=[_course("c2", priority=1), _course("c3", priority=2)]),
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


def test_roadmap_course_rejects_zero_priority() -> None:
    with pytest.raises(ValidationError):
        RoadmapCourse(course_id="c1", course_name="과목", priority=0)


def test_roadmap_course_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        RoadmapCourse(course_id="", course_name="과목", priority=1)


def test_roadmap_roundtrip_model_dump_validate() -> None:
    """model_dump(mode='json') → model_validate 가 원본과 동등한 객체를 복원한다."""
    original = Roadmap(stages=_full_stages())
    dumped = original.model_dump(mode="json")
    restored = Roadmap.model_validate(dumped)

    assert restored == original
    assert len(restored.stages) == 4
    assert restored.stages[1].courses[1].priority == 2


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
