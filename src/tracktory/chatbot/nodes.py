"""챗봇 LangGraph 노드 — skeleton 더미.

각 노드는 state 의 일부만 읽고 자기 출력 필드만 반환
LLM·RAGFlow 의존성이 들어올 때 함수 → 클래스로 전환하고 nodes/ 폴더로 split 예정.
"""

from tracktory.chatbot.state import ChatbotState


def classify_intent(state: ChatbotState) -> dict:
    """질문의 의도를 4-way 분류한다 (track / job / course / general).

    TODO: prompts.chatbot.intent.INTENT_CLASSIFIER_PROMPT 와
    llm.with_structured_output(IntentClassification) 으로 교체.
    """
    return {
        "intent": "general_advice",
        "intent_reason": "(dummy) skeleton 단계 — 항상 general_advice 반환",
    }


def retrieve_rag(state: ChatbotState) -> dict:
    """RAGFlow 에서 의도별로 관련 문서를 검색한다.

    general_advice 경로는 이 노드를 우회 (edges.route_after_intent).
    TODO: RAGFlow 클라이언트 주입 + 의도별 데이터셋 분기.
    """
    return {"retrieved_docs": []}


def generate_response(state: ChatbotState) -> dict:
    """RAG 결과·의도·질문을 LLM 에 던져 자연어 응답을 생성한다.

    TODO: 실제 LLM 연결 시 두 노드로 split 예정
    분기는 노드 내부가 아니라 엣지에서 처리한다
    - generate_rag_response (track / job / course) — prompts.chatbot.rag_answer
    - generate_general_response (general_advice)   — prompts.chatbot.general_advice
    엣지 route_after_intent 도 두 종착 노드로 분기하도록 갱신.
    """
    return {"response": f"(dummy) 응답: {state['message']}"}
