""" 챗봇 LangGraph 노드 """

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from tracktory.chatbot.rag.ragflow import RagFlowChatbotRetriever, RagSearchError
from tracktory.chatbot.state import ChatbotState
from tracktory.prompts.chatbot.intent import IntentClassification
from tracktory.prompts.chatbot.rag_response import (
    ChatbotResponse,
    format_retrieved_docs,
    format_user_context,
)

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


class GenerateResponseNode:
    """챗봇 응답 생성 노드 — intent 에 따라 RAG / 일반 조언 chain 선택

    - track/job/course → RAG chain (RAG_RESPONSE_PROMPT, retrieved_docs 사용)
    - general_advice → 일반 조언 chain (GENERAL_ADVICE_PROMPT, 자료 미사용)
    """

    def __init__(
        self,
        rag_chain: Runnable[dict[str, Any], ChatbotResponse],
        general_chain: Runnable[dict[str, Any], ChatbotResponse],
    ) -> None:
        self._rag = rag_chain
        self._general = general_chain

    def __call__(self, state: ChatbotState) -> dict:
        user_context_block = format_user_context(state.get("user_context"))

        if state.get("intent") == "general_advice":
            result = self._general.invoke(
                {
                    "messages": state["messages"],
                    "user_context_block": user_context_block,
                }
            )
        else:
            result = self._rag.invoke(
                {
                    "messages": state["messages"],
                    "user_context_block": user_context_block,
                    "retrieved_context": format_retrieved_docs(state.get("retrieved_docs") or []),
                }
            )

        return {
            "response": result.text,
            "response_choices": result.choices,
            "messages": [AIMessage(content=result.text)],
        }
