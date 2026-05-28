"""챗봇 LangGraph 노드"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from tracktory.chatbot.state import ChatbotState
from tracktory.prompts.chatbot.intent import IntentClassification

logger = logging.getLogger(__name__)


class ClassifyIntentNode:
    """챗봇 의도 분류 노드
    질문의 의도를 4-way 분류 (track / job / course / general)

    LLM transient error / 구조화 출력 파싱 실패 시 user 응답이 끊기지 않도록
    general_advice 로 폴백한다. 실패 사유는 intent_reason 에 남겨 디버깅·평가에 사용
    """

    def __init__(self, classifier: Runnable[dict[str, Any], IntentClassification]) -> None:
        self._classifier = classifier

    def __call__(self, state: ChatbotState) -> dict:
        last_message = state["messages"][-1].content
        try:
            result = self._classifier.invoke({"message": last_message})
        except Exception as e:
            # LLM 클라이언트마다 raise 타입이 다름 (RateLimit / Timeout / ValidationError 등)
            # 폴백 정책이 동일하므로 Exception 으로 한 곳에서 처리
            error_type = type(e).__name__
            logger.warning(
                "ClassifyIntentNode 분류 실패 (%s) → general_advice 로 폴백",
                error_type,
                exc_info=True,
            )
            return {
                "intent": "general_advice",
                "intent_reason": f"분류 실패: {error_type}",
                "search_keywords": [],
            }
        return {
            "intent": result.intent,
            "intent_reason": result.reason,
            "search_keywords": result.search_keywords,
        }


def retrieve_rag(state: ChatbotState) -> dict:
    """의도별로 RAGFlow 검색. general_advice 는 우회 (edges).

    TODO: RAGFlow 클라이언트 주입 + 의도별 데이터셋 분기.
    """
    return {"retrieved_docs": []}


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
