"""챗봇 LangGraph skeleton 의 E2E 동작을 검증한다.

skeleton 단계 — 더미 노드라 응답 내용은 검증 안 하고 흐름이 통과하는지만 확인.
실제 LLM·RAGFlow 연결 후엔 노드별 단위 테스트로 확장.
"""

from tracktory.chatbot import build_chatbot_graph


def _base_state(message: str, history: list[dict]) -> dict:
    """skeleton 테스트용 ChatbotState 기본값 생성."""
    return {
        "message": message,
        "user_context": {"user_id": "u1"},
        "history": history,
        "intent": None,
        "intent_reason": None,
        "retrieved_docs": [],
        "response": None,
    }


def test_first_turn_no_history() -> None:
    """첫 질문 — history 가 빈 리스트일 때 graph 가 정상 통과한다."""
    graph = build_chatbot_graph()
    result = graph.invoke(
        _base_state(message="빅데이터 트랙이 뭐야?", history=[])
    )

    assert result["intent"] == "general_advice"
    assert result["response"] is not None
    assert result["response"].startswith("(dummy)")


def test_followup_turn_with_history() -> None:
    """후속 질문 — history 에 이전 턴이 들어있어도 graph 가 정상 통과한다."""
    graph = build_chatbot_graph()
    history = [
        {"role": "user", "content": "빅데이터 트랙이 뭐야?"},
        {"role": "assistant", "content": "빅데이터 트랙은 데이터 분석·처리를 다루는 트랙입니다."},
    ]
    result = graph.invoke(
        _base_state(message="그럼 어떤 직무로 가면 좋아?", history=history)
    )

    assert result["intent"] == "general_advice"
    assert result["response"] is not None
    assert result["response"].startswith("(dummy)")
    # history 가 graph 를 거쳐도 변하지 않고 그대로 통과해야 한다 (reducer 없음 → 동일성 유지)
    assert result["history"] == history
