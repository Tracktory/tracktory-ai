"""챗봇 LangGraph 조건부 라우팅"""

from typing import Literal

from tracktory.chatbot.state import ChatbotState


def route_after_intent(
    state: ChatbotState,
) -> Literal["retrieve_rag", "generate_response"]:
    """general_advice 는 RAG 우회 (학사 자료 도움 X), 그 외는 RAGFlow 검색 후 응답."""
    if state.intent == "general_advice":
        return "generate_response"
    return "retrieve_rag"
