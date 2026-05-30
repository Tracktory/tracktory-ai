"""FastAPI 의존성 — 추천 파이프라인 boundary 주입.

boundary 클라이언트(JobSearchClient / TrackRepository / CourseRepository / LLMClient)
조립은 후속 작업에서 제공한다. 현재는 NotImplementedError 를 raise 하여 lifespan
warm-up 이 조용히 skip 되도록 한다.

테스트에서는 app.dependency_overrides 로 교체한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends

from tracktory.graph.pipeline import (
    PipelineClients,
    PipelineConfig,
    get_recommendation_graph,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def get_pipeline_clients() -> PipelineClients:
    """운영 boundary 클라이언트 묶음을 반환한다.

    후속 작업에서 실제 구현체(RAGFlow / DB 저장소 / LLM 클라이언트)를 주입한다.
    구현 전까지 NotImplementedError 를 raise 하며, lifespan 핸들러가 이를 catch 해
    warm-up 을 skip 한다.
    """
    raise NotImplementedError("boundary 클라이언트 조립은 후속 작업에서 제공")


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
