"""``RoadmapConfig`` 외부화 매핑·검증 단위 테스트.

실 ``roadmap.yaml`` 의 학사 제약 (학기 용량·졸업 학점·학년별 학점 범위) 이
Pydantic 모델 검증을 통과하는지, 잘못된 yaml 모양 / 음수 / 0 같은 비정상
값이 명시적 에러로 차단되는지 확인한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tracktory.graph.models import RoadmapConfig


def _base_config_dict() -> dict[str, Any]:
    """검증을 통과하는 minimal config dict 를 반환한다.

    각 테스트는 본 dict 를 받아 한 필드만 override 해서 단일 invariant 만
    공격한다.
    """
    return {
        "capacity": {
            "max_credits_per_semester_default": 18,
            "max_credits_per_semester_high_gpa": 21,
        },
        "graduation": {
            "total_credits_two_tracks": 30,
            "total_credits_major": 78,
        },
        "grade_credits_range": {
            1: {"min": 8, "max": 16},
            2: {"min": 24, "max": 36},
            3: {"min": 24, "max": 36},
            4: {"min": 0, "max": 99},
        },
    }


def test_loads_real_roadmap_yaml(real_roadmap_yaml_path: Path) -> None:
    """실 ``roadmap.yaml`` 이 검증을 통과해야 한다."""
    cfg = RoadmapConfig.load_from_yaml(real_roadmap_yaml_path)
    assert cfg.capacity.max_credits_per_semester_default >= 1
    assert (
        cfg.capacity.max_credits_per_semester_high_gpa
        >= cfg.capacity.max_credits_per_semester_default
    )
    assert cfg.graduation.total_credits_two_tracks >= 1
    assert cfg.graduation.total_credits_major >= cfg.graduation.total_credits_two_tracks
    assert set(cfg.grade_credits_range.keys()) == {1, 2, 3, 4}
    for grade in (1, 2, 3, 4):
        rng = cfg.grade_credits_range[grade]
        assert rng.min_credits <= rng.max_credits


def test_load_raises_when_top_level_is_not_mapping(tmp_path: Path) -> None:
    """리스트 yaml 은 명시적 ValueError 로 차단된다."""
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level mapping"):
        RoadmapConfig.load_from_yaml(path)


def test_capacity_default_zero_raises() -> None:
    config = _base_config_dict()
    config["capacity"]["max_credits_per_semester_default"] = 0
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_capacity_high_gpa_negative_raises() -> None:
    config = _base_config_dict()
    config["capacity"]["max_credits_per_semester_high_gpa"] = -1
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_graduation_two_tracks_zero_raises() -> None:
    config = _base_config_dict()
    config["graduation"]["total_credits_two_tracks"] = 0
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_graduation_major_zero_raises() -> None:
    config = _base_config_dict()
    config["graduation"]["total_credits_major"] = 0
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_missing_capacity_section_raises() -> None:
    config = _base_config_dict()
    del config["capacity"]
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_missing_graduation_section_raises() -> None:
    config = _base_config_dict()
    del config["graduation"]
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_missing_grade_credits_range_raises() -> None:
    config = _base_config_dict()
    del config["grade_credits_range"]
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_grade_credits_range_missing_grade_raises() -> None:
    """학년 1~4 중 하나라도 누락되면 ValueError."""
    config = _base_config_dict()
    del config["grade_credits_range"][3]
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_grade_credits_range_extra_grade_raises() -> None:
    """학년 5 같은 외부 키는 검증으로 차단된다."""
    config = _base_config_dict()
    config["grade_credits_range"][5] = {"min": 0, "max": 18}
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)


def test_grade_credits_range_min_gt_max_raises() -> None:
    """min > max 는 GradeCreditsRange 검증으로 차단된다."""
    config = _base_config_dict()
    config["grade_credits_range"][1] = {"min": 20, "max": 10}
    with pytest.raises(ValidationError):
        RoadmapConfig.model_validate(config)
