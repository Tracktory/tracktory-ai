"""tracktory LangGraph 정의 — state, 노드, 엣지, 그래프 조립 진입점.

본 패키지의 핵심 진입점:
    - ``GraphState`` (``state``) — 노드 간 공유되는 TypedDict.
    - ``build_recommendation_graph`` / ``get_recommendation_graph`` (``pipeline``)
      — 6 노드를 묶은 컴파일된 그래프. 후자는 ``lru_cache(maxsize=1)`` 로
      앱 수명 동안 1회 컴파일을 보장한다.
    - ``PipelineClients`` / ``PipelineConfig`` (``pipeline``) — 외부 의존성 + 정적 설정.
"""

from tracktory.graph.pipeline import (
    PipelineClients,
    PipelineConfig,
    build_recommendation_graph,
    get_recommendation_graph,
)
from tracktory.graph.state import GraphState

__all__ = [
    "GraphState",
    "PipelineClients",
    "PipelineConfig",
    "build_recommendation_graph",
    "get_recommendation_graph",
]
