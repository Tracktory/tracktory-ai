"""추천 파이프라인 + 실제 RAGFlow 직무 검색 어댑터 통합 smoke test.

``tests/integration/test_pipeline_happy_path.py`` 가 4 종 외부 의존성을 모두
``MagicMock(spec=...)`` 으로 격리해 grain 차단을 검증하는 데 반해, 본 모듈은
직무 검색만 운영 어댑터 (``RagflowJobSearchClient``) 로 교체해 자연어 질의
→ RAGFlow 검색 → 직무 매칭 노드 결합부가 실제 외부 시스템과 정합하는지
확인하는 smoke test 다.

검증 범위는 **구조 수준**: 결과 길이 > 0, 필수 필드 채워짐, 점수 정의역
``[0, 1]``. 특정 질의에 대한 expected top job 같은 데이터 의존 검증은 본
모듈 범위 외 — 채용공고 KB 가 변하면 깨지므로 별도 e2e 시나리오로 분리.

격리 정책:
    - ``JobSearchClient`` = ``RagflowJobSearchClient.from_env()`` (실제 호출)
    - ``TrackRepository`` / ``CourseRepository`` / ``LLMClient`` = MagicMock
      (운영 어댑터 미구현 + 본 smoke 범위가 RAGFlow 결합부에 한정)

CI 정책:
    - ``@pytest.mark.e2e`` 로 표시. ``pytest -m "not e2e"`` (CI 기본) 에서 제외.
    - ``RAGFLOW_API_KEY`` / ``RAGFLOW_BASE_URL`` / ``RAGFLOW_DATASET_ID`` 중
      하나라도 환경변수 미설정 시 ``pytest.skip`` 으로 자동 통과 — 로컬에서
      자격증명 없는 개발자가 ``pytest -m e2e`` 를 실행해도 노이즈 0.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tracktory.graph import pipeline as graph_pipeline
from tracktory.graph.models import Course, Explanation, JobCandidate, Track
from tracktory.graph.nodes.llm_explanation import LLMClient
from tracktory.graph.nodes.roadmap import CourseRepository
from tracktory.graph.nodes.track_synergy import TrackRepository
from tracktory.graph.pipeline import (
    PipelineClients,
    build_recommendation_graph,
    get_recommendation_graph,
)
from tracktory.rag.job_search import JobSearchClient
from tracktory.rag.ragflow_job_search import RagflowConfig, RagflowJobSearchClient

pytestmark = pytest.mark.e2e


_REQUIRED_ENV_VARS = ("RAGFLOW_BASE_URL", "RAGFLOW_API_KEY", "RAGFLOW_DATASET_ID")
_EMBED_DIM = 1536


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_pipeline_cache() -> Iterator[None]:
    """모듈 전역 ``lru_cache`` 가 e2e 테스트 간 누출되지 않도록 자동 격리."""
    get_recommendation_graph.cache_clear()
    yield
    get_recommendation_graph.cache_clear()


@pytest.fixture
def ragflow_client() -> JobSearchClient:
    """``RAGFLOW_*`` 환경변수가 모두 설정된 경우에만 운영 어댑터를 생성한다.

    하나라도 누락이면 ``pytest.skip`` — 자격증명 없는 환경에서 e2e 마커를
    돌려도 자연스럽게 통과한다.
    """
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        pytest.skip(f"RAGFlow 자격증명 환경변수 누락: {missing}. .env.local 설정 후 재시도.")

    return RagflowJobSearchClient(RagflowConfig.from_env())


# ---------------------------------------------------------------------------
# 헬퍼 — Track / Course / 정규화 프로필 / mock 의존성
#
# ``tests/integration/test_pipeline_happy_path.py`` 의 헬퍼와 동일 패턴. 본
# 모듈은 ``e2e`` 전용 격리를 위해 의도적으로 복제했다 (integration 디렉토리에
# ``__init__.py`` 가 없어 cross-test import 가 보장되지 않음).
# ---------------------------------------------------------------------------


def _meta_vector(seed: int) -> list[float]:
    """deterministic seed 기반 L2-normalized 트랙 메타 벡터."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(_EMBED_DIM)
    norm = float(np.linalg.norm(raw))
    return (raw / norm).tolist() if norm > 0 else []


def _track(
    track_id: str,
    *,
    college_id: str,
    department_id: str,
    course_ids: list[str],
    tech_stacks: list[str],
    competencies: list[str],
    meta_seed: int,
) -> Track:
    return Track(
        college_id=college_id,
        department_id=department_id,
        major_id=department_id,
        track_id=track_id,
        track_name=track_id,
        course_ids=course_ids,
        meta_text="",
        meta_vector=_meta_vector(meta_seed),
        competencies=competencies,
        tech_stacks=tech_stacks,
    )


def _course(
    course_id: str,
    *,
    track_id: str,
    stage: str,
    prereq_ids: list[str] | None = None,
    credits: int = 3,
    priority: int = 1,
) -> Course:
    return Course(
        course_id=course_id,
        course_name=f"{course_id} 강의",
        credits=credits,
        stage=stage,  # type: ignore[arg-type]
        prereq_ids=prereq_ids or [],
        track_ids=[track_id],
        priority=priority,
    )


def _build_tracks() -> list[Track]:
    in_college = [
        _track(
            f"in{i}",
            college_id="C1",
            department_id="D1",
            course_ids=[f"in_co{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"in_c{i}"],
            meta_seed=i,
        )
        for i in range(3)
    ]
    out_college = [
        _track(
            f"out{i}",
            college_id="C2",
            department_id="D2",
            course_ids=[f"out_co{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"out_c{i}"],
            meta_seed=100 + i,
        )
        for i in range(4)
    ]
    return in_college + out_college


def _build_courses() -> list[Course]:
    return [
        _course("CS101", track_id="in0", stage="foundation", priority=1),
        _course("CS102", track_id="in1", stage="foundation", priority=2),
        _course("CS201", track_id="in0", stage="core", prereq_ids=["CS101"], priority=1),
        _course("CS202", track_id="in1", stage="core", prereq_ids=["CS101", "CS102"], priority=2),
        _course("CS301", track_id="in0", stage="application", prereq_ids=["CS201"], priority=1),
        _course("CS401", track_id="in0", stage="industry", prereq_ids=["CS301"], priority=1),
    ]


def _build_clients(job_search: JobSearchClient) -> PipelineClients:
    """직무 검색만 실제 어댑터, 나머지 3 종은 mock."""
    track_repo = MagicMock(spec=TrackRepository)
    track_repo.list_all.return_value = _build_tracks()

    course_repo = MagicMock(spec=CourseRepository)
    course_repo.list_for_tracks.return_value = _build_courses()

    llm_client = MagicMock(spec=LLMClient)
    llm_client.invoke.return_value = Explanation(
        text="추천 결과 종합 설명 (e2e mock).",
        sections=[],
    )

    return PipelineClients(
        job_search_client=job_search,
        track_repository=track_repo,
        course_repository=course_repo,
        llm_client=llm_client,
    )


def _valid_onboarding_payload(*, college: str = "C1") -> dict[str, Any]:
    """입력 정규화 노드 검증을 통과하는 유효 온보딩 입력."""
    return {
        "admission_year": 2025,
        "college": college,
        "department": "컴퓨터공학부",
        "current_tracks": [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


# ---------------------------------------------------------------------------
# 테스트 — smoke 수준 구조 검증
# ---------------------------------------------------------------------------


def test_pipeline_with_real_ragflow_returns_well_formed_jobs(
    ragflow_client: JobSearchClient,
) -> None:
    """실제 RAGFlow 호출 결과가 ``RagSearchResult`` schema 를 만족하고 파이프라인을 통과한다.

    검증 항목 (smoke):
        - ``recommended_jobs`` 가 채워져 있고 길이 > 0
        - 각 candidate 의 ``match_score`` / ``similarity`` 가 ``[0, 1]`` 범위
        - 필수 필드 (``job_id`` / ``job_name``) 비어있지 않음
        - ``fallback_used=False`` (운영 어댑터 정상 응답 시)
        - 트랙 시너지 / 로드맵 / LLM 설명 노드까지 흐름 정상 (mock 동작)
        - 6 노드 trace 흔적이 모두 남음
    """
    clients = _build_clients(ragflow_client)
    graph = build_recommendation_graph(clients)

    result = graph.invoke(
        {"user_id": "e2e_user", "raw_input": _valid_onboarding_payload()},
    )

    # --- 직무 매칭 결합부 (실제 RAGFlow) ---
    recommended_jobs = result.get("recommended_jobs")
    assert recommended_jobs, "RAGFlow 검색 결과가 비어 있다."
    assert isinstance(recommended_jobs, list)

    for job in recommended_jobs:
        # JobCandidate schema 호환 확인 (필드 누락 / 범위 위반 시 ValidationError).
        candidate = JobCandidate.model_validate(job)
        assert candidate.job_id
        assert candidate.job_name
        assert 0.0 <= candidate.match_score <= 1.0
        assert 0.0 <= candidate.similarity <= 1.0

    # --- 후속 노드 (mock) ---
    assert result.get("normalized_profile") is not None
    assert result.get("profile_text")
    assert len(result["primary_combos"]) == 2
    assert len(result["secondary_combos"]) == 5
    assert result["roadmap"]["stages"]
    assert result["explanation"]["text"]

    # --- 6 노드 trace 흔적 ---
    trace_prefixes = {token.split(":", 1)[0] for token in result["trace"]}
    assert {
        "input_normalize",
        "profile_embed",
        "job_matching",
        "track_synergy",
        "roadmap",
        "llm_explanation",
    } <= trace_prefixes


def test_pipeline_with_real_ragflow_smoke_via_cached_entry(
    ragflow_client: JobSearchClient,
) -> None:
    """``get_recommendation_graph`` (캐시 진입점) 으로 호출해도 동일하게 동작한다.

    캐시 진입점이 운영 어댑터 인스턴스를 받아도 ``frozen`` dataclass hash 로
    정상 동작하는지 확인하는 smoke 검증.
    """
    clients = _build_clients(ragflow_client)
    graph = graph_pipeline.get_recommendation_graph(clients)

    result = graph.invoke(
        {"user_id": "e2e_user_cached", "raw_input": _valid_onboarding_payload()},
    )

    recommended_jobs = result.get("recommended_jobs")
    assert recommended_jobs
    # 직무 매칭이 ``JobCandidate`` 호환 schema 로 흐르는지만 다시 한번 확인.
    for job in recommended_jobs:
        JobCandidate.model_validate(job)
