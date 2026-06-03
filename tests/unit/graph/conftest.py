"""트랙 시너지·직무 매칭 노드 단위 테스트의 공통 fixture.

사용 패턴: ``def test_xxx(make_track, make_job, ...)`` — pytest 가 자동 주입한다.
factory pattern (호출 시점 객체 생성) 으로 테스트마다 다른 인자로 트랙을 만들 수 있다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tracktory.graph.models import (
    Course,
    JobCandidate,
    JobMatchingConfig,
    SynergyConfig,
    Track,
    TrackCombo,
)
from tracktory.graph.nodes.roadmap import CourseRepository
from tracktory.rag.job_search import JobSearchClient, RagSearchError, RagSearchResult

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
def tmp_course_catalog_path(tmp_path: Path) -> Path:
    """이수 과목 → 기술 토큰 색인용 소형 ``courses.yaml`` fixture.

    실 카탈로그 (대용량 · 기술 토큰 미부착) 대신 부스팅 경로 검증에 필요한
    최소 과목만 담아, 노드 생성 비용을 줄이고 이름 → 토큰 다리를 결정적으로
    통제한다. "자료구조" 과목이 같은 표기의 기술 토큰을 가지므로 직무
    ``tech_stacks=["자료구조"]`` 와 정규 키로 매칭된다.
    """
    path = tmp_path / "courses.yaml"
    catalog = {
        "courses": [
            {"course_id": "DS", "course_name": "자료구조", "tech_stacks": ["자료구조"]},
        ]
    }
    path.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# JobSearchClient fake — 직무 매칭 노드 단위 테스트용
# ---------------------------------------------------------------------------


class FakeJobSearchClient:
    """``rag_search_jobs`` 호출 시 미리 정해진 결과 리스트를 ``top_k`` 만큼 반환한다.

    ``raise_error`` 가 True 면 ``RagSearchError`` 를 raise 하여 직무 매칭 노드의
    error → fallback 분기를 검증할 수 있다.
    """

    def __init__(
        self,
        results: list[RagSearchResult] | None = None,
        *,
        raise_error: bool = False,
    ) -> None:
        self._results = list(results or [])
        self._raise_error = raise_error
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def rag_search_jobs(self, query: str, top_k: int = 3) -> list[RagSearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        if self._raise_error:
            raise RagSearchError("fake search failure")
        return list(self._results[:top_k])


@pytest.fixture
def make_search_result() -> Callable[..., RagSearchResult]:
    """``RagSearchResult`` factory — 직무 검색 boundary 결과 단건."""

    def _factory(
        job_id: str,
        *,
        job_name: str | None = None,
        score: float = 0.8,
        description: str | None = None,
        tech_stacks: list[str] | None = None,
        competency_tags: list[str] | None = None,
    ) -> RagSearchResult:
        return RagSearchResult(
            job_id=job_id,
            job_name=job_name or job_id,
            score=score,
            description=description or f"{job_id} 직무 설명",
            tech_stacks=tech_stacks or [],
            competency_tags=competency_tags or [],
        )

    return _factory


@pytest.fixture
def make_fake_job_search_client() -> Callable[..., JobSearchClient]:
    """``FakeJobSearchClient`` factory — Protocol 만 노출한다."""

    def _factory(
        results: list[RagSearchResult] | None = None,
        *,
        raise_error: bool = False,
    ) -> JobSearchClient:
        return FakeJobSearchClient(results=results, raise_error=raise_error)

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
        completed_course_boost_weight: float = 0.15,
    ) -> JobMatchingConfig:
        base_top_k = {"default": 3, "expanded": 5}
        if top_k:
            base_top_k.update(top_k)
        return JobMatchingConfig.model_validate(
            {
                "top_k": base_top_k,
                "min_job_similarity": min_job_similarity,
                "completed_course_boost": {"weight": completed_course_boost_weight},
            }
        )

    return _factory


@pytest.fixture
def make_course() -> Callable[..., Course]:
    """``Course`` factory — 학습 로드맵 노드 fixture 용.

    Repository 가 채울 모든 메타 (stage·credits·prereq_ids·priority·
    available_grades·course_type) 를 키워드 인자로 override 가능하다.
    학년 제약 / 분류 메타가 부재한 케이스를 시뮬레이션하려면 각 인자의
    기본값 (모든 학년 가능 · 전공 선택) 을 그대로 두면 된다.
    """

    def _factory(
        course_id: str,
        *,
        course_name: str | None = None,
        credits: int = 3,
        stage: str = "foundation",
        prereq_ids: list[str] | None = None,
        track_ids: list[str] | None = None,
        priority: int = 1,
        available_grades: list[int] | None = None,
        course_type: str = "전공선택",
    ) -> Course:
        return Course(
            course_id=course_id,
            course_name=course_name or course_id,
            credits=credits,
            stage=stage,  # type: ignore[arg-type]
            prereq_ids=prereq_ids or [],
            track_ids=track_ids or [],
            priority=priority,
            available_grades=available_grades or [1, 2, 3, 4],
            course_type=course_type,  # type: ignore[arg-type]
        )

    return _factory


@pytest.fixture
def make_course_repo() -> Callable[..., CourseRepository]:
    """단순 in-memory ``CourseRepository`` factory.

    ``courses_by_track`` 인자는 ``dict[track_id, list[Course]]`` — ``list_for_tracks``
    가 요청 ``track_ids`` 의 합집합을 ``course_id`` 기준 dedup 하여 반환한다.
    """

    def _factory(courses_by_track: dict[str, list[Course]]) -> CourseRepository:
        class _InMemoryCourseRepo:
            def __init__(self, mapping: dict[str, list[Course]]) -> None:
                self._mapping = mapping

            def list_for_tracks(self, track_ids: list[str]) -> list[Course]:
                seen: dict[str, Course] = {}
                for track_id in track_ids:
                    for course in self._mapping.get(track_id, []):
                        if course.course_id not in seen:
                            seen[course.course_id] = course
                return list(seen.values())

        return _InMemoryCourseRepo(courses_by_track)

    return _factory


@pytest.fixture
def real_roadmap_yaml_path() -> Path:
    """실제 ``src/tracktory/config/roadmap.yaml`` 의 경로 — 정합 sanity check 용."""
    return Path(__file__).resolve().parents[3] / "src" / "tracktory" / "config" / "roadmap.yaml"
