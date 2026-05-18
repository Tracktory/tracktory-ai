"""챗봇 LangGraph state schema.

히스토리는 checkpointer 가 thread_id 단위로 자체 관리
백엔드는 'conversation_id' 만 유지하고 매 요청마다 'user_context' 만 주입
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from tracktory.chatbot.config import ChatbotIntent
from tracktory.chatbot.rag.ragflow import RetrievedChunk


class ChatbotState(TypedDict):
    # --- 입력 ---
    user_context: dict  # 온보딩 정보

    # --- 누적 (add_messages reducer) ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- 갱신 ---
    intent: ChatbotIntent | None
    intent_reason: str | None  # 디버깅·평가용
    search_keywords: list[str]
    retrieved_docs: list[RetrievedChunk]
    response: str | None  # API 응답 본문
    response_choices: list[str]  # 후속 질문 후보 1-3 개
