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
    JobMatchingConfig,
    SynergyConfig,
    Track,
    TrackCombo,
)
from tracktory.rag.job_index import InMemoryJobIndex, Job

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


@pytest.fixture
def real_category_mapping_path() -> Path:
    """실제 ``src/tracktory/config/category_to_jobs.yaml`` 의 경로."""
    return (
        Path(__file__).resolve().parents[3]
        / "src"
        / "tracktory"
        / "config"
        / "category_to_jobs.yaml"
    )


@pytest.fixture
def make_job_data() -> Callable[..., Job]:
    """``Job`` 도메인 factory — 직무 매칭 노드 fixture 용.

    ``vector_seed`` 가 주어지면 deterministic L2 normalized 벡터를 채운다.
    """

    def _factory(
        job_id: str,
        *,
        job_name: str | None = None,
        tech_stacks: list[str] | None = None,
        competency_tags: list[str] | None = None,
        vector_seed: int | None = None,
    ) -> Job:
        return Job(
            job_id=job_id,
            job_name=job_name or job_id,
            job_vector=_make_meta_vector(vector_seed),
            tech_stacks=tech_stacks or [],
            competency_tags=competency_tags or [],
        )

    return _factory


@pytest.fixture
def make_in_memory_job_index() -> Callable[[list[Job]], InMemoryJobIndex]:
    """``InMemoryJobIndex`` 단순 wrapper factory."""

    def _factory(jobs: list[Job]) -> InMemoryJobIndex:
        return InMemoryJobIndex(jobs=jobs)

    return _factory


@pytest.fixture
def make_job_matching_config() -> Callable[..., JobMatchingConfig]:
    """``JobMatchingConfig`` factory — 기본값에 partial override.

    실 yaml 을 거치지 않고 in-memory 에서 검증하고 싶을 때 사용.
    """

    def _factory(
        *,
        top_k: dict[str, int] | None = None,
        min_job_similarity: float = 0.3,
    ) -> JobMatchingConfig:
        base_top_k = {"default": 3, "expanded": 5}
        if top_k:
            base_top_k.update(top_k)
        return JobMatchingConfig.model_validate(
            {"top_k": base_top_k, "min_job_similarity": min_job_similarity}
        )

    return _factory
