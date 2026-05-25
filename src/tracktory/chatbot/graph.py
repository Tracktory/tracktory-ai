"""챗봇 LangGraph 조립.

흐름:
    START → classify_intent → [route]
                              ├─ retrieve_rag → generate_response → END  (track/job/course)
                              └─ generate_response → END                 (general_advice)

Checkpointer 는 외부에서 주입:
    - 운영(콘솔/FastAPI): SqliteSaver 등 영속 saver
    - 테스트: MemorySaver — 미지정 시 fallback (휘발성)
"""

from typing import Any

from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tracktory.chatbot.edges import route_after_intent
from tracktory.chatbot.nodes import (
    ClassifyIntentNode,
    GenerateResponseNode,
    RetrieveRagNode,
)
from tracktory.chatbot.rag.ragflow import RagFlowChatbotRetriever
from tracktory.chatbot.state import ChatbotState
from tracktory.prompts.chatbot.intent import IntentClassification
from tracktory.prompts.chatbot.rag_response import ChatbotResponse


def build_chatbot_graph(
    *,
    classifier: Runnable[dict[str, Any], IntentClassification],
    retriever: RagFlowChatbotRetriever,
    rag_response_chain: Runnable[dict[str, Any], ChatbotResponse],
    general_advice_chain: Runnable[dict[str, Any], ChatbotResponse],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """챗봇 그래프 컴파일. ``config={"configurable": {"thread_id": ...}}`` 와 함께 호출

    classifier: INTENT_CLASSIFIER_PROMPT | LLM 체인
    retriever: RagFlowChatbotRetriever
    rag_response_chain: RAG_RESPONSE_PROMPT | LLM.with_structured_output(ChatbotResponse) — RAG 케이스
    general_advice_chain: GENERAL_ADVICE_PROMPT | LLM.with_structured_output(ChatbotResponse) — 일반 조언 케이스
    checkpointer: 대화 히스토리 영속화 saver. 미지정 시 MemorySaver (휘발성).
        - 콘솔/FastAPI 운영: SqliteSaver 주입
        - 단위 테스트: 미지정 → 기본 MemorySaver
    """
    builder: StateGraph = StateGraph(ChatbotState)

    builder.add_node("classify_intent", ClassifyIntentNode(classifier))
    builder.add_node("retrieve_rag", RetrieveRagNode(retriever))
    builder.add_node(
        "generate_response", GenerateResponseNode(rag_response_chain, general_advice_chain)
    )

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

    return builder.compile(checkpointer=checkpointer or MemorySaver())
