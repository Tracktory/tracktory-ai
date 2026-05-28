"""``JobMatchingConfig`` 외부화 매핑·검증 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tracktory.graph.models import JobMatchingConfig


def test_loads_real_synergy_yaml_job_matching_section(real_synergy_yaml_path: Path) -> None:
    """실 ``synergy.yaml`` 의 ``job_matching`` 섹션이 검증을 통과해야 한다."""
    cfg = JobMatchingConfig.load_from_yaml(real_synergy_yaml_path)
    assert cfg.top_k.default >= 1
    assert cfg.top_k.expanded >= cfg.top_k.default
    assert 0.0 <= cfg.min_job_similarity <= 1.0


def test_load_raises_when_job_matching_section_missing(tmp_path: Path) -> None:
    """``job_matching`` 키가 없는 yaml 은 명시적 ValueError 를 발생시킨다."""
    path = tmp_path / "no_section.yaml"
    path.write_text("weights:\n  complementarity: 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="job_matching"):
        JobMatchingConfig.load_from_yaml(path)


def test_load_raises_when_top_level_is_not_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        JobMatchingConfig.load_from_yaml(path)


def test_min_job_similarity_below_zero_raises() -> None:
    with pytest.raises(ValidationError):
        JobMatchingConfig.model_validate(
            {"top_k": {"default": 3, "expanded": 5}, "min_job_similarity": -0.1}
        )


def test_min_job_similarity_above_one_raises() -> None:
    with pytest.raises(ValidationError):
        JobMatchingConfig.model_validate(
            {"top_k": {"default": 3, "expanded": 5}, "min_job_similarity": 1.1}
        )


def test_top_k_default_zero_raises() -> None:
    with pytest.raises(ValidationError):
        JobMatchingConfig.model_validate(
            {"top_k": {"default": 0, "expanded": 5}, "min_job_similarity": 0.3}
        )
