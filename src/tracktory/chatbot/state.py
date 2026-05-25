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

    # --- 갱신 (classify_intent 출력) ---
    intent: ChatbotIntent | None
    intent_reason: str | None  # 디버깅·평가용
    search_keywords: list[str]
    is_catalog_query: bool  # 전체 카탈로그 요청 (트랙 종류 / 직무 종류 등)
    target_grade: int | None  # 질문에 학년 명시 시 1~4, 없으면 None

    # --- 갱신 (retrieve_rag 출력) ---
    retrieved_docs: list[RetrievedChunk]

    # --- 갱신 (generate_response 출력) ---
    response: str | None  # API 응답 본문
    response_choices: list[str]  # 후속 질문 후보 1-3 개
