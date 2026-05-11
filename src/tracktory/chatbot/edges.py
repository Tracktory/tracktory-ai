"""챗봇 LangGraph 조건부 라우팅.

CLAUDE.md §4.3 규약:
- 노드와 분리된 파일에 둔다.
- 반환 타입을 Literal[...] 로 고정 — 오타가 런타임이 아닌 mypy 단계에서 잡힘.
- side effect 없이 state 만 읽고 분기 이름 문자열만 반환.
"""

from typing import Literal

from tracktory.chatbot.state import ChatbotState


def route_after_intent(
    state: ChatbotState,
) -> Literal["retrieve_rag", "generate_response"]:
    """의도 분류 결과에 따라 RAG 경로 vs 직답 경로로 분기한다.

    - general_advice: RAG 우회, 바로 응답 생성 (한성대 학사 자료가 도움 안 되는 영역).
    - 그 외 (track / job / course): RAGFlow 검색 후 응답 생성.
    """
    if state["intent"] == "general_advice":
        return "generate_response"
    return "retrieve_rag"
