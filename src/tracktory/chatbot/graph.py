"""챗봇 LangGraph 조립.

흐름:
    START → classify_intent → [route]
                              ├─ retrieve_rag → generate_response → END  (track/job/course)
                              └─ generate_response → END                 (general_advice)

MemorySaver 는 in-process 휘발성 — 운영은 PostgresSaver 등 영속 saver 로 교체.
"""

from langgraph.checkpoint.memory import MemorySaver
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
    """챗봇 그래프 컴파일. ``config={"configurable": {"thread_id": ...}}`` 와 함께 호출.

    TODO: (llm, rag_client, checkpointer) 의존성 주입으로 시그니처 확장.
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

    return builder.compile(checkpointer=MemorySaver())
