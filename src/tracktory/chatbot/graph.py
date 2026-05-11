"""챗봇 LangGraph 조립.

흐름:
    START
      → classify_intent
      → [route_after_intent]
          → retrieve_rag → generate_response → END   (track / job / course)
          → generate_response → END                  (general_advice)

skeleton 단계에서는 더미 노드만 연결. 실제 구현 시 build 함수가 LLM·RAGFlow
클라이언트를 인자로 받도록 시그니처 확장 예정 (CLAUDE.md §4.4 의존성 주입).
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tracktory.chatbot.edges import route_after_intent
from tracktory.chatbot.nodes import (
    classify_intent,
    generate_response,
    retrieve_rag,
)
from tracktory.chatbot.state import ChatbotState


def build_chatbot_graph() -> CompiledStateGraph:
    """챗봇 StateGraph 를 빌드하여 컴파일된 그래프를 반환한다.

    Returns:
        ``invoke`` / ``stream`` 호출이 가능한 컴파일된 그래프.

    TODO: ``build_chatbot_graph(llm, rag_client)`` 로 시그니처 확장.
    노드를 클래스로 전환하여 의존성을 __init__ 로 주입한다.
    """
    builder: StateGraph = StateGraph(ChatbotState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "retrieve_rag": "retrieve_rag",
            "generate_response": "generate_response",
        },
    )
    builder.add_edge("retrieve_rag", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()
