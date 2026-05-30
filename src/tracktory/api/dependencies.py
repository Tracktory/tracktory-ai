"""FastAPI 의존성 — 추천 파이프라인 boundary 주입.

운영 boundary 4 종 (직무 검색 / 트랙 저장소 / 과목 저장소 / LLM 설명) 을
조립해 추천 그래프에 주입한다. 조립은 앱 수명주기당 1 회만 수행되도록 모듈
수준 ``lru_cache`` 싱글톤으로 캐시하며, 매 요청에 동일 인스턴스를 재사용해
그래프 단일 컴파일 (``get_recommendation_graph`` 의 lru_cache) 캐시 hit 을
보장한다 — ``PipelineClients`` 가 ``frozen`` dataclass 라 필드(=client 인스턴스
id) 기반 hash 이므로, 매번 새 인스턴스를 만들면 캐시 미스로 요청마다 재컴파일된다.

테스트는 ``app.dependency_overrides[get_pipeline_clients]`` 로 mock boundary
묶음을 주입해 외부 I/O 없이 엔드포인트를 검증한다 (싱글톤 빌더는 호출되지 않는다).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends

from tracktory.graph.pipeline import (
    PipelineClients,
    PipelineConfig,
    get_recommendation_graph,
)
from tracktory.llm.openai_client import OpenAILLMClient, OpenAILLMConfig
from tracktory.rag.ragflow_client import RagflowConfig
from tracktory.rag.ragflow_job_search import RagflowJobSearchClient
from tracktory.rag.yaml_course_repository import YamlCourseRepository
from tracktory.rag.yaml_track_repository import YamlTrackRepository

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


@lru_cache(maxsize=1)
def _build_pipeline_clients() -> PipelineClients:
    """운영 boundary 4 종을 조립한 묶음을 1 회만 생성한다.

    각 어댑터는 생성자에서 자격증명/카탈로그만 읽고 외부 호출 (RAGFlow /
    OpenAI) 은 첫 요청 시점으로 미루므로, startup warm-up 단계에서 네트워크
    없이 안전하게 구성된다. ``maxsize=1`` 캐시로 같은 인스턴스를 재사용해
    그래프 단일 컴파일 캐시 키 (frozen ``PipelineClients`` hash) 를 안정화한다.

    Raises:
        RuntimeError: 필수 환경변수 (``RAGFLOW_*`` / ``OPENAI_API_KEY``) 누락 시.
            ``RagflowConfig.from_env`` / ``OpenAILLMConfig.from_env`` 계약.
        TrackCatalogError: 트랙 카탈로그 YAML 부재/손상 시.
        CourseCatalogError: 과목 카탈로그 YAML 부재/손상 시.
    """
    return PipelineClients(
        job_search_client=RagflowJobSearchClient(RagflowConfig.from_env()),
        track_repository=YamlTrackRepository(),
        course_repository=YamlCourseRepository(),
        llm_client=OpenAILLMClient(OpenAILLMConfig.from_env()),
    )


def get_pipeline_clients() -> PipelineClients:
    """운영 boundary 클라이언트 묶음 (싱글톤) 을 반환한다.

    수명주기당 1 회 조립된 동일 인스턴스를 반환한다. 테스트는 본 의존성을
    ``app.dependency_overrides`` 로 교체하므로 싱글톤 빌더가 호출되지 않는다.
    """
    return _build_pipeline_clients()


def reset_pipeline_clients() -> None:
    """싱글톤 boundary 캐시를 비운다.

    앱 종료 시점 (lifespan shutdown) 의 자원 참조 해제 + 테스트 간 격리용.
    다음 ``get_pipeline_clients`` 호출은 묶음을 다시 1 회 조립한다.
    """
    _build_pipeline_clients.cache_clear()


def get_pipeline_config() -> PipelineConfig | None:
    """노드 config 경로 묶음을 반환한다.

    None 반환 시 각 노드 내부의 기본 경로를 사용한다.
    """
    return None


def get_recommendation_pipeline(
    clients: Annotated[PipelineClients, Depends(get_pipeline_clients)],
    config: Annotated[PipelineConfig | None, Depends(get_pipeline_config)],
) -> CompiledStateGraph:
    """컴파일된 추천 그래프를 반환한다.

    동일 clients / config 조합에 대해 lru_cache 로 1회만 컴파일하며,
    이후 호출은 캐시된 인스턴스를 재사용한다.
    """
    return get_recommendation_graph(clients, config)
