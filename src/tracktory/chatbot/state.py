"""챗봇 LangGraph state schema.

히스토리는 checkpointer 가 thread_id 단위로 자체 관리
백엔드는 'conversation_id' 만 유지하고 매 요청마다 'user_context' 만 주입

갱신 필드는 default_factory / None 기본값을 강제
호출하는 쪽이 초기값을 누락하거나 노드 순서가 바뀌어도 KeyError 가 나지 않도록 함
(TypedDict 는 default 가 없어 위험)
"""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from tracktory.chatbot.config import ChatbotIntent
from tracktory.chatbot.rag.ragflow import RetrievedChunk


class ChatbotState(BaseModel):
    # --- 입력 ---
    user_context: dict  # 온보딩 정보

    # --- 누적 (add_messages reducer) ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- 갱신 ---
    intent: ChatbotIntent | None = None
    intent_reason: str | None = None  # 디버깅·평가용
    search_keywords: list[str] = Field(default_factory=list)
    retrieved_docs: list[RetrievedChunk] = Field(default_factory=list)
    response: str | None = None  # API 응답용
