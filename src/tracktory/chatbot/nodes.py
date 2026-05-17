""" 챗봇 LangGraph 노드 """

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from tracktory.chatbot.rag.ragflow import RagFlowChatbotRetriever, RagSearchError
from tracktory.chatbot.state import ChatbotState
from tracktory.prompts.chatbot.intent import IntentClassification

logger = logging.getLogger("chatbot")


class ClassifyIntentNode:
    """ 챗봇 의도 분류 노드
    질문의 의도를 4-way 분류 (track / job / course / general)
    """

    def __init__(self, classifier: Runnable[dict[str, Any], IntentClassification]) -> None:
        self._classifier = classifier

    def __call__(self, state: ChatbotState) -> dict:
        last_message = state["messages"][-1].content
        result = self._classifier.invoke({"message": last_message})
        return {
            "intent": result.intent,
            "intent_reason": result.reason,
            "search_keywords": result.search_keywords,
        }


class RetrieveRagNode:
    """ 챗봇 RAG 검색 노드 — 의도별 데이터셋 분기 + 빈 결과·실패 안전망 """

    def __init__(self, retriever: RagFlowChatbotRetriever) -> None:
        self._retriever = retriever

    def __call__(self, state: ChatbotState) -> dict:
        intent = state.get("intent")
        keywords = state.get("search_keywords") or []

        if not intent or intent == "general_advice":
            # general_advice / None — 정상 경로상 도달 X, 안전망
            logger.debug("retrieve_rag 우회 (intent=%s)", intent)
            return {"retrieved_docs": []}

        if not keywords:
            logger.info("retrieve_rag: 검색 키워드 없음 (intent=%s)", intent)
            return {"retrieved_docs": []}

        query = ",".join(keywords)
        try:
            results = self._retriever.search(query)
        except RagSearchError:
            logger.warning(
                "retrieve_rag: 검색 실패 (intent=%s, query=%r)",
                intent,
                query,
                exc_info=True,
            )
            return {"retrieved_docs": []}

        # 본문 빈 청크 제외
        cleaned = [chunk for chunk in results if chunk["content"].strip()]
        return {"retrieved_docs": cleaned}


def generate_response(state: ChatbotState) -> dict:
    """LLM 으로 자연어 응답 생성. AIMessage 도 messages 에 함께 append 해야
    다음 턴 checkpointer 가 어시스턴트 응답까지 복원한다.

    TODO: 의도별 두 노드로 split (rag_answer / general_advice).
    """
    last_user_msg = state["messages"][-1].content if state["messages"] else ""
    response_text = f"(dummy) 응답: {last_user_msg}"
    return {
        "response": response_text,
        "messages": [AIMessage(content=response_text)],
    }
