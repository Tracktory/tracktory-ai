"""추천 API 엔드포인트의 그래프 파이프라인 연결 통합 테스트.

``POST /recommend`` 가 FastAPI 의 dependency_overrides 를 통해 mock boundary
4 종을 주입받은 컴파일된 추천 그래프를 실행하고, 4 부분 묶음 응답을 반환하는
계약을 검증한다. 실제 RAGFlow / LLM / Repository 호출은 발생하지 않는다.

테스트 격리:
    모듈 전역 lru_cache 와 ``app.dependency_overrides`` 가 테스트 간 누출되지
    않도록 autouse fixture 가 매 테스트 진입·종료 시 정리한다.

``@pytest.mark.integration`` 으로 표시 — CI 기본 실행에서 제외하려면
``pytest -m "not integration"`` 을 사용한다.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tracktory.api.dependencies import (
    get_pipeline_clients,
    get_pipeline_config,
    get_recommendation_pipeline,
)
from tracktory.api.main import app
from tracktory.graph.models import Course, Explanation, Track
from tracktory.graph.pipeline import PipelineClients, get_recommendation_graph
from tracktory.llm.llm_client import LLMClient
from tracktory.rag.course_repository import CourseRepository
from tracktory.rag.job_search import JobSearchClient, RagSearchResult
from tracktory.rag.track_repository import TrackRepository

pytestmark = pytest.mark.integration


_EMBED_DIM = 1536

# 추천 API 응답 SLO — functional-spec API-004 의 30초 timeout 과 동일 grain.
_SLO_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_pipeline_cache() -> Iterator[None]:
    """단일 컴파일 lru_cache 가 테스트 간 누출되지 않도록 자동 격리."""
    get_recommendation_graph.cache_clear()  # type: ignore[attr-defined]
    yield
    get_recommendation_graph.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _reset_dependency_overrides() -> Iterator[None]:
    """전역 ``app.dependency_overrides`` 가 테스트 간 누출되지 않도록 격리."""
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers — Mock boundary 4 종 + 유효 페이로드
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
    """cross-college 후보가 존재하는 7 트랙 셋업."""
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
    """4 단계를 모두 덮는 6 과목 셋업."""
    return [
        _course("CS101", track_id="in0", stage="foundation", priority=1),
        _course("CS102", track_id="in1", stage="foundation", priority=2),
        _course("CS201", track_id="in0", stage="core", prereq_ids=["CS101"], priority=1),
        _course("CS202", track_id="in1", stage="core", prereq_ids=["CS101", "CS102"], priority=2),
        _course("CS301", track_id="in0", stage="application", prereq_ids=["CS201"], priority=1),
        _course("CS401", track_id="in0", stage="industry", prereq_ids=["CS301"], priority=1),
    ]


def _build_clients(
    *,
    explanation_text: str = "추천 결과 종합 설명입니다.",
) -> PipelineClients:
    """4 종 의존성을 ``MagicMock(spec=...)`` 로 채운 컨테이너 생성."""
    job_search = MagicMock(spec=JobSearchClient)
    job_search.rag_search_jobs.return_value = [
        RagSearchResult(
            job_id="backend_developer",
            job_name="백엔드 개발자",
            score=0.85,
            description="서버 사이드 시스템을 설계하고 운영합니다.",
            tech_stacks=["py", "sql"],
            competency_tags=["문제해결능력"],
        ),
    ]

    track_repo = MagicMock(spec=TrackRepository)
    track_repo.list_all.return_value = _build_tracks()

    course_repo = MagicMock(spec=CourseRepository)
    course_repo.list_for_tracks.return_value = _build_courses()

    llm_client = MagicMock(spec=LLMClient)
    llm_client.invoke.return_value = Explanation(text=explanation_text, sections=[])

    return PipelineClients(
        job_search_client=job_search,
        track_repository=track_repo,
        course_repository=course_repo,
        llm_client=llm_client,
    )


def _valid_payload() -> dict[str, Any]:
    """RecommendRequest 검증을 통과하는 유효 페이로드."""
    return {
        "admission_year": 2025,
        "college": "C1",
        "department": "컴퓨터공학부",
        "current_tracks": [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


def _override_clients(clients: PipelineClients) -> None:
    """dependency_overrides 로 boundary 주입 — 모듈 전역 app 에 적용."""
    app.dependency_overrides[get_pipeline_clients] = lambda: clients
    app.dependency_overrides[get_pipeline_config] = lambda: None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recommend_happy_path_returns_four_part_envelope() -> None:
    """유효 입력 + mock boundary 로 200 + 4 부분 묶음 envelope 을 반환한다."""
    clients = _build_clients()
    _override_clients(clients)

    start = time.perf_counter()
    with TestClient(app) as client:
        response = client.post("/recommend", json=_valid_payload())
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    body = response.json()
    assert body["is_success"] is True
    data = body["data"]
    assert data["jobs"], "직무 후보가 비어 있으면 안 된다"
    assert len(data["primary_combos"]) == 2
    assert len(data["secondary_combos"]) == 5
    assert data["roadmap"]["stages"], "로드맵 4 단계가 채워져야 한다"
    assert data["explanation"]["text"] == "추천 결과 종합 설명입니다."

    # 30 초 SLO 자동 검증 — mock 환경에서 훨씬 빠르게 통과해야 한다.
    assert elapsed < _SLO_SECONDS


def test_recommend_validation_failure_returns_422_envelope() -> None:
    """필수 필드 누락 시 422 + 표준 에러 envelope 반환."""
    clients = _build_clients()
    _override_clients(clients)

    payload = _valid_payload()
    payload["interests"] = []  # min_length=1 위반

    with TestClient(app) as client:
        response = client.post("/recommend", json=payload)

    # 프로젝트 envelope 표준은 RequestValidationError 를 BAD_REQUEST_ERROR (400)
    # 로 매핑한다 (exception_handlers.validation_handler). FastAPI 의 기본 422
    # 가 아닌 400 으로 정규화된 envelope 이 반환되는지 검증.
    assert response.status_code == 400
    body = response.json()
    assert body["is_success"] is False


def test_recommend_graph_internal_errors_map_to_500() -> None:
    """그래프 state.errors 가 누적되면 500 + SERVER_ERROR envelope 반환.

    RecommendRequest 통과 후 NormalizedProfile 가 거부하는 케이스를 직접
    재현하기 어려우므로 graph dependency 자체를 fake graph 로 override 하여
    errors 가 채워진 final_state 를 반환하도록 한다.
    """

    class _FakeGraph:
        async def ainvoke(self, _state: dict[str, Any]) -> dict[str, Any]:
            return {"errors": ["normalized profile rejected"]}

    app.dependency_overrides[get_recommendation_pipeline] = lambda: _FakeGraph()

    with TestClient(app) as client:
        response = client.post("/recommend", json=_valid_payload())

    assert response.status_code == 500
    body = response.json()
    assert body["is_success"] is False


def test_recommend_boundary_exception_maps_to_500() -> None:
    """그래프 실행 중 boundary 가 raise 하면 500 + SERVER_ERROR envelope 반환."""

    class _ExplodingGraph:
        async def ainvoke(self, _state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boundary blew up")

    app.dependency_overrides[get_recommendation_pipeline] = lambda: _ExplodingGraph()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/recommend", json=_valid_payload())

    assert response.status_code == 500
    body = response.json()
    assert body["is_success"] is False


def test_recommend_graph_missing_roadmap_maps_to_500() -> None:
    """후속 노드가 silent skip 하여 roadmap/explanation 키가 누락된 final_state 도 500.

    state.errors 가 비어 있지만 그래프 contract 가 깨진 케이스 — router 가 KeyError 로
    generic 500 에 빠지지 않고, 누락 필드를 명시한 도메인 detail 로 500 을 반환해야 한다.
    """

    class _PartialGraph:
        async def ainvoke(self, _state: dict[str, Any]) -> dict[str, Any]:
            return {
                "recommended_jobs": [],
                "primary_combos": [],
                "secondary_combos": [],
                "errors": [],
            }

    app.dependency_overrides[get_recommendation_pipeline] = lambda: _PartialGraph()

    with TestClient(app) as client:
        response = client.post("/recommend", json=_valid_payload())

    assert response.status_code == 500
    body = response.json()
    assert body["is_success"] is False


def test_recommend_compiles_graph_once_across_requests() -> None:
    """동일 boundary 로 2 회 호출 시 lru_cache hit 가 발생해 재컴파일이 없다."""
    clients = _build_clients()
    _override_clients(clients)

    # lifespan shutdown 이 cache_clear 를 호출하므로 cache_info 검증은
    # ``with`` 블록 내부에서 수행해야 유효한 카운터를 본다.
    with TestClient(app) as client:
        client.post("/recommend", json=_valid_payload())
        client.post("/recommend", json=_valid_payload())

        info = get_recommendation_graph.cache_info()  # type: ignore[attr-defined]
        assert info.hits >= 1, f"expected cache hit on 2nd request, got {info}"
