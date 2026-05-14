"""챗봇 LangGraph 노드 — skeleton 더미.

LLM·RAGFlow 의존성 주입 시 함수 → 클래스로 전환하고 nodes/ 폴더로 split 예정.
"""

from langchain_core.messages import AIMessage

from tracktory.chatbot.state import ChatbotState


def classify_intent(state: ChatbotState) -> dict:
    """질문의 의도를 4-way 분류 (track / job / course / general).

    TODO: prompts.chatbot.intent.INTENT_CLASSIFIER_PROMPT + with_structured_output 로 교체.
    """
    return {
        "intent": "general_advice",
        "intent_reason": "(dummy) skeleton 단계 — 항상 general_advice 반환",
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
