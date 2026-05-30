"""추천 파이프라인 그래프 조립 + 컴파일 진입점.

여섯 개 노드 (입력 정규화 / 프로필 직렬화 / 직무 매칭 / 트랙 시너지 /
학습 로드맵 / 자연어 설명) 를 단일 ``StateGraph`` 로 묶어 사용자 요청
단위로 호출 가능한 컴파일된 그래프를 반환한다. 외부 검색 / 저장소 / LLM
의존성은 ``PipelineClients`` 로 주입되어 운영용과 테스트용을 교체할 수
있다.

토폴로지::

    START
      → input_normalize
      → [conditional: route_after_input_normalize]
           "continue" → profile_embed
           "end"      → END (안전 종료)
      → profile_embed
      → job_matching
      → track_synergy
      → roadmap
      → llm_explanation
      → END

조건부 라우팅은 입력 정규화 직후 한 번만 두어 검증 실패를 안전하게
종료시킨다. 후속 노드들은 각자 ``state.get(...)`` 부재 시 조용히 skip
하므로 본 그래프 레벨에서 추가 라우팅이 필요하지 않다.

단일 컴파일 보장:
    ``get_recommendation_graph`` 헬퍼가 ``functools.lru_cache(maxsize=1)``
    로 동일 의존성에 대해 ``StateGraph.compile()`` 을 한 번만 호출한다.
    의존성 dataclass 가 ``frozen=True`` 이므로 hashable 이며 ``lru_cache``
    의 키로 그대로 사용된다. 운영 시 앱 수명주기 결합은 본 모듈의 책임 밖이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from tracktory.graph.edges import route_after_input_normalize
from tracktory.graph.nodes import (
    CourseRepository,
    JobMatchingNode,
    LLMClient,
    LLMExplanationNode,
    ProfileEmbedNode,
    RoadmapNode,
    TrackRepository,
    TrackSynergyNode,
    normalize_input,
)
from tracktory.graph.state import GraphState
from tracktory.rag.job_search import JobSearchClient

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


@dataclass(frozen=True)
class PipelineClients:
    """런타임 외부 의존성 — Protocol 4 종 묶음.

    운영/테스트 boundary 에서 교체되는 단위. 컴파일 시점 이후 client 가
    변경되면 캐시된 그래프 인스턴스가 stale 해질 수 있으므로 ``frozen``
    으로 변이를 차단한다. dataclass 의 ``eq`` / ``hash`` 자동 생성으로
    ``lru_cache`` 의 키로도 사용된다.

    Attributes:
        job_search_client: 직무 검색 boundary. 자연어 질의에 대해 직무
            KB 검색 결과를 반환하며 외부 호출 실패를 표준 예외로 변환한다.
        track_repository: 트랙 메타 저장소. 전체 트랙 목록 / 부분 조회를 제공한다.
        course_repository: 과목 메타 저장소. 트랙별 권장 과목 (단계 / 선수
            관계 / 우선순위 포함) 을 제공한다.
        llm_client: 구조화 출력 LLM 클라이언트. 자연어 설명 노드가 단일
            invoke 로 ``Explanation`` 객체를 받는다.
    """

    job_search_client: JobSearchClient
    track_repository: TrackRepository
    course_repository: CourseRepository
    llm_client: LLMClient


@dataclass(frozen=True)
class PipelineConfig:
    """노드 config 파일 경로 묶음 — 정적 외부화 설정.

    각 노드는 ``None`` 일 때 노드 내부에 정의된 기본 경로를 사용한다.
    ablation / 튜닝 시점에 본 dataclass 를 교체하면 코드 변경 없이
    가중치 / 임계값 / 템플릿을 갈아끼울 수 있다.
    """

    profile_template_path: Path | None = None
    job_matching_config_path: Path | None = None
    job_matching_category_mapping_path: Path | None = None
    synergy_config_path: Path | None = None
    roadmap_config_path: Path | None = None


def build_recommendation_graph(
    clients: PipelineClients,
    config: PipelineConfig | None = None,
) -> CompiledStateGraph:
    """6 개 노드를 묶어 컴파일된 LangGraph 인스턴스를 반환한다.

    노드는 모두 ``(state) -> dict`` 시그니처를 따르므로 LangGraph 가
    부분 state dict 를 자동으로 reducer / overwrite 결합한다. 본 함수는
    조립과 컴파일만 책임지며, 노드 내부 정책은 변경하지 않는다.

    Args:
        clients: 4 종 외부 의존성.
        config: 노드 config 경로. ``None`` 이면 각 노드 기본 경로 사용.

    Returns:
        ``StateGraph(GraphState).compile()`` 결과. ``invoke({"user_id": ..., "raw_input": {...}})``
        로 호출 가능하다.
    """
    resolved_config = config or PipelineConfig()

    profile_embed_node = ProfileEmbedNode(template_path=resolved_config.profile_template_path)
    job_matching_node = JobMatchingNode(
        job_search_client=clients.job_search_client,
        config_path=resolved_config.job_matching_config_path,
        category_mapping_path=resolved_config.job_matching_category_mapping_path,
    )
    track_synergy_node = TrackSynergyNode(
        track_repo=clients.track_repository,
        config_path=resolved_config.synergy_config_path,
    )
    roadmap_node = RoadmapNode(
        course_repo=clients.course_repository,
        config_path=resolved_config.roadmap_config_path,
    )
    llm_explanation_node = LLMExplanationNode(llm_client=clients.llm_client)

    graph = StateGraph(GraphState)
    graph.add_node("input_normalize", normalize_input)
    graph.add_node("profile_embed", profile_embed_node)
    graph.add_node("job_matching", job_matching_node)
    graph.add_node("track_synergy", track_synergy_node)
    graph.add_node("roadmap", roadmap_node)
    graph.add_node("llm_explanation", llm_explanation_node)

    graph.add_edge(START, "input_normalize")
    graph.add_conditional_edges(
        "input_normalize",
        route_after_input_normalize,
        {"continue": "profile_embed", "end": END},
    )
    graph.add_edge("profile_embed", "job_matching")
    graph.add_edge("job_matching", "track_synergy")
    graph.add_edge("track_synergy", "roadmap")
    graph.add_edge("roadmap", "llm_explanation")
    graph.add_edge("llm_explanation", END)

    return graph.compile()


def get_recommendation_graph(
    clients: PipelineClients,
    config: PipelineConfig | None = None,
) -> CompiledStateGraph:
    """동일 의존성에 대해 컴파일된 그래프를 1회만 생성하여 재사용한다.

    Contract:
        - ``config=None`` 과 ``config=PipelineConfig()`` 는 동일한 캐시 키로
          정규화된다. 본 wrapper 가 ``None`` 을 기본 인스턴스로 치환한 뒤
          내부 ``lru_cache`` 헬퍼에 전달하므로 두 호출은 동일 컴파일 결과를
          재사용한다.
        - 운영에서는 동일 ``PipelineClients`` 인스턴스를 재주입하는 형태로
          호출되어야 캐시 hit 한다. ``frozen=True`` dataclass 는 필드 기반
          ``__hash__`` 를 자동 생성하지만, ``MagicMock(spec=...)`` 는
          id 기반 hash 이므로 테스트에서 인스턴스 재사용이 필요하다.

    캐시 비우기: ``get_recommendation_graph.cache_clear()``.
    """
    return _cached_recommendation_graph(clients, config or PipelineConfig())


@lru_cache(maxsize=1)
def _cached_recommendation_graph(
    clients: PipelineClients,
    config: PipelineConfig,
) -> CompiledStateGraph:
    """``build_recommendation_graph`` 를 단 1회만 호출하는 캐시 진입점.

    Contract:
        - 키는 ``(clients, config)`` 의 hash. ``config`` 는 ``None`` 이 아닌
          정규화된 인스턴스만 받아 캐시 키 일관성을 보장한다.
        - ``maxsize=1`` 은 운영에서 동일 의존성 단일 컴파일을 가정한다.
          서로 다른 의존성 조합이 빈번히 들어오면 최후 호출만 캐시된다.
    """
    return build_recommendation_graph(clients, config)


# 외부 호출자가 ``get_recommendation_graph.cache_clear()`` 패턴을 그대로 쓸 수
# 있도록 내부 lru_cache 헬퍼의 ``cache_clear`` / ``cache_info`` 를 wrapper 에
# 노출한다. ``functools.lru_cache`` 의 공개 API 와 동일한 인터페이스.
get_recommendation_graph.cache_clear = _cached_recommendation_graph.cache_clear  # type: ignore[attr-defined]
get_recommendation_graph.cache_info = _cached_recommendation_graph.cache_info  # type: ignore[attr-defined]
