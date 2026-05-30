"""운영 boundary 조립 의존성 단위 테스트.

``get_pipeline_clients`` 가 운영 어댑터 4 종 (직무 검색 / 트랙 저장소 / 과목
저장소 / LLM 설명) 을 올바른 구체 타입으로 조립하고, 앱 수명주기당 1 회만
생성(싱글톤)하는 계약을 검증한다. 각 어댑터는 생성자에서 외부 호출(RAGFlow /
OpenAI)을 하지 않으므로 (자격증명·카탈로그만 읽음) 네트워크 없이 통과한다.

엔드포인트가 boundary 를 ``dependency_overrides`` 로 교체 가능한지는 별도
통합 테스트(``tests/integration/test_recommend_api.py``)가 검증한다.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tracktory.api.dependencies import get_pipeline_clients, reset_pipeline_clients
from tracktory.graph.pipeline import PipelineClients
from tracktory.llm.openai_client import OpenAILLMClient
from tracktory.rag.ragflow_job_search import RagflowJobSearchClient
from tracktory.rag.yaml_course_repository import YamlCourseRepository
from tracktory.rag.yaml_track_repository import YamlTrackRepository


@pytest.fixture
def operational_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``from_env`` 가 요구하는 자격증명을 더미 값으로 채우고 싱글톤을 격리한다.

    더미 값으로도 어댑터 생성자는 통과한다 (외부 호출은 첫 요청 시점). 모듈
    전역 싱글톤 캐시가 테스트 간 누출되지 않도록 진입·종료 시 초기화한다.
    """
    monkeypatch.setenv("RAGFLOW_BASE_URL", "http://localhost:9380")
    monkeypatch.setenv("RAGFLOW_API_KEY", "dummy")
    monkeypatch.setenv("RAGFLOW_DATASET_ID", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    reset_pipeline_clients()
    yield
    reset_pipeline_clients()


def test_assembles_operational_adapter_types(operational_env: None) -> None:
    """조립된 묶음이 운영 어댑터 4 종을 올바른 구체 타입으로 담는다."""
    clients = get_pipeline_clients()

    assert isinstance(clients, PipelineClients)
    assert isinstance(clients.job_search_client, RagflowJobSearchClient)
    assert isinstance(clients.track_repository, YamlTrackRepository)
    assert isinstance(clients.course_repository, YamlCourseRepository)
    assert isinstance(clients.llm_client, OpenAILLMClient)


def test_clients_assembled_once_as_singleton(operational_env: None) -> None:
    """수명주기당 1 회만 조립 — 두 번 호출해도 동일 인스턴스를 반환한다.

    동일 인스턴스 재사용이 깨지면 ``frozen`` ``PipelineClients`` 의 id 기반
    hash 가 매번 달라져 그래프 단일 컴파일 캐시가 요청마다 미스한다.
    """
    first = get_pipeline_clients()
    second = get_pipeline_clients()

    assert first is second


def test_reset_rebuilds_clients(operational_env: None) -> None:
    """캐시 초기화 후에는 묶음을 다시 조립한다 (앱 종료/재시작 격리)."""
    first = get_pipeline_clients()
    reset_pipeline_clients()
    second = get_pipeline_clients()

    assert first is not second


def test_missing_credentials_raise_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """필수 자격증명 누락 시 ``RuntimeError`` 로 실패한다 (startup warm-up 이 catch).

    lifespan 은 본 예외를 catch 해 warm-up 을 skip 하고 startup 을 막지 않는다.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAGFLOW_API_KEY", raising=False)
    monkeypatch.delenv("RAGFLOW_BASE_URL", raising=False)
    monkeypatch.delenv("RAGFLOW_DATASET_ID", raising=False)
    reset_pipeline_clients()

    with pytest.raises(RuntimeError):
        get_pipeline_clients()

    reset_pipeline_clients()
