"""챗봇 LangGraph state schema.

입력(불변) / 갱신 필드를 주석으로 구분
"""

from typing import Literal, TypedDict

ChatbotIntent = Literal[
    "track_question",   # 트랙
    "job_question",     # 직무
    "course_question",  # 과목
    "general_advice",   # 일반
]


class ChatbotState(TypedDict):
    # --- 입력 (불변) — 매 요청마다 외부에서 주입 ---
    message: str                              # 사용자 질문
    user_context: dict                        # 사용자 정보
    history: list[dict]                       # 이전 대화 턴 (Spring 이 동봉)

    # --- 갱신 — 각 노드가 자기 필드만 overwrite ---
    intent: ChatbotIntent | None              # classify_intent 출력
    intent_reason: str | None                 # classify_intent 출력 (디버깅·평가용)
    retrieved_docs: list[dict]                # retrieve_rag 출력
    response: str | None                      # generate_response 출력
