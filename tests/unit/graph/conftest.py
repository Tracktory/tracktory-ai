"""트랙 시너지 노드 단위 테스트의 공통 fixture.

사용 패턴: ``def test_xxx(make_track, make_job, ...)`` — pytest 가 자동 주입한다.
factory pattern (호출 시점 객체 생성) 으로 테스트마다 다른 인자로 트랙을 만들 수 있다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tracktory.graph.models import (
    JobCandidate,
    SynergyConfig,
    Track,
    TrackCombo,
)

_DEFAULT_DIM = 1536


def _make_meta_vector(seed: int | None) -> list[float]:
    """deterministic seed 기반 L2-normalized 트랙 메타 벡터."""
    if seed is None:
        return []
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(_DEFAULT_DIM)
    norm = float(np.linalg.norm(raw))
    if norm == 0.0:
        return []
    normalized: np.ndarray = raw / norm
    return normalized.tolist()


@pytest.fixture
def make_track() -> Callable[..., Track]:
    """Track factory.

    keyword args 로 4-tier ID·메타·역량 등 모든 필드를 override 가능하다.
    ``meta_seed`` 를 주면 deterministic 한 L2-normalized 벡터가 채워진다.
    """

    def _factory(
        track_id: str,
        *,
        college_id: str = "C1",
        department_id: str = "D1",
        major_id: str | None = None,
        track_name: str | None = None,
        course_ids: list[str] | None = None,
        competencies: list[str] | None = None,
        tech_stacks: list[str] | None = None,
        meta_seed: int | None = None,
    ) -> Track:
        return Track(
            college_id=college_id,
            department_id=department_id,
            major_id=major_id if major_id is not None else department_id,
            track_id=track_id,
            track_name=track_name or track_id,
            course_ids=course_ids or [],
            meta_text="",
            meta_vector=_make_meta_vector(meta_seed),
            competencies=competencies or [],
            tech_stacks=tech_stacks or [],
        )

    return _factory


@pytest.fixture
def make_combo() -> Callable[..., TrackCombo]:
    """TrackCombo factory — combo_key 자동 정렬 생성."""

    def _factory(track_a: Track, track_b: Track) -> TrackCombo:
        ids_sorted = sorted([track_a.track_id, track_b.track_id])
        return TrackCombo(
            track_a=track_a,
            track_b=track_b,
            combo_key="::".join(ids_sorted),
        )

    return _factory


@pytest.fixture
def make_job() -> Callable[..., JobCandidate]:
    """JobCandidate factory."""

    def _factory(
        job_id: str = "j1",
        *,
        job_name: str | None = None,
        tech_stacks: list[str] | None = None,
        competency_tags: list[str] | None = None,
        match_score: float = 0.8,
    ) -> JobCandidate:
        return JobCandidate(
            job_id=job_id,
            job_name=job_name or job_id,
            tech_stacks=tech_stacks or [],
            competency_tags=competency_tags or [],
            match_score=match_score,
        )

    return _factory


@pytest.fixture
def make_synergy_config() -> Callable[..., SynergyConfig]:
    """SynergyConfig factory — 단조 제약을 만족하는 기본값에 partial override.

    각 nested config (weights / similarity / mmr / slots) 를 dict 로 partial
    override 가능하다.
    """

    def _factory(
        *,
        weights: dict[str, Any] | None = None,
        similarity: dict[str, Any] | None = None,
        mmr: dict[str, Any] | None = None,
        slots: dict[str, Any] | None = None,
    ) -> SynergyConfig:
        base: dict[str, dict[str, Any]] = {
            "weights": {"complementarity": 0.3, "coverage": 0.5, "redundancy": 0.2},
            "similarity": {
                "w_college": 0.4,
                "w_department": 0.3,
                "w_track": 0.2,
                "w_course_overlap": 0.1,
                "w_meta": 0.05,
            },
            "mmr": {"lambda": 0.6},
            "slots": {
                "primary_count": 2,
                "secondary_count": 5,
                "cross_college_reserved": 1,
                "min_cross_synergy": 0.3,
            },
        }
        if weights:
            base["weights"].update(weights)
        if similarity:
            base["similarity"].update(similarity)
        if mmr:
            base["mmr"].update(mmr)
        if slots:
            base["slots"].update(slots)
        return SynergyConfig.model_validate(base)

    return _factory


@pytest.fixture
def real_synergy_yaml_path() -> Path:
    """실제 ``src/tracktory/config/synergy.yaml`` 의 경로 — 정합 sanity check 용."""
    return Path(__file__).resolve().parents[3] / "src" / "tracktory" / "config" / "synergy.yaml"
