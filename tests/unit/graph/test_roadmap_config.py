"""``RoadmapConfig`` 외부화 매핑·검증 단위 테스트.

실 ``roadmap.yaml`` 의 학사 제약 (학기 용량·졸업 학점) 이 Pydantic 모델
검증을 통과하는지, 잘못된 yaml 모양 / 음수 / 0 같은 비정상 값이 명시적
에러로 차단되는지 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tracktory.graph.models import RoadmapConfig


def test_loads_real_roadmap_yaml(real_roadmap_yaml_path: Path) -> None:
    """실 ``roadmap.yaml`` 이 검증을 통과해야 한다."""
    cfg = RoadmapConfig.load_from_yaml(real_roadmap_yaml_path)
    assert cfg.capacity.max_credits_per_semester_default >= 1
    assert (
        cfg.capacity.max_credits_per_semester_high_gpa
        >= cfg.capacity.max_credits_per_semester_default
    )
    assert cfg.graduation.total_credits_two_tracks >= 1


def test_load_raises_when_top_level_is_not_mapping(tmp_path: Path) -> None:
    """리스트 yaml 은 명시적 ValueError 로 차단된다."""
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level mapping"):
        RoadmapConfig.load_from_yaml(path)


def test_capacity_default_zero_raises() -> None:
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(
            {
                "capacity": {
                    "max_credits_per_semester_default": 0,
                    "max_credits_per_semester_high_gpa": 21,
                },
                "graduation": {"total_credits_two_tracks": 30},
            }
        )


def test_capacity_high_gpa_negative_raises() -> None:
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(
            {
                "capacity": {
                    "max_credits_per_semester_default": 18,
                    "max_credits_per_semester_high_gpa": -1,
                },
                "graduation": {"total_credits_two_tracks": 30},
            }
        )


def test_graduation_total_zero_raises() -> None:
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(
            {
                "capacity": {
                    "max_credits_per_semester_default": 18,
                    "max_credits_per_semester_high_gpa": 21,
                },
                "graduation": {"total_credits_two_tracks": 0},
            }
        )


def test_missing_capacity_section_raises() -> None:
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate({"graduation": {"total_credits_two_tracks": 30}})


def test_missing_graduation_section_raises() -> None:
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(
            {
                "capacity": {
                    "max_credits_per_semester_default": 18,
                    "max_credits_per_semester_high_gpa": 21,
                }
            }
        )
