"""챗봇 LangGraph skeleton E2E 동작 검증

더미 노드 단계 — 흐름 통과 여부 + checkpointer 의 thread 별 히스토리 자동 누적만 확인
실제 LLM·RAGFlow 연결 후 노드별 단위 테스트로 확장 예정
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from tracktory.chatbot import build_chatbot_graph


def _initial_input(message: str) -> dict:
    """첫 invoke 에 주입하는 입력 — 누적 필드는 checkpointer 가 관리"""
    return {
        "user_context": {"user_id": 1},
        "messages": [HumanMessage(content=message)],
        "intent": None,
        "intent_reason": None,
        "retrieved_docs": [],
        "response": None,
    }


def test_first_turn() -> None:
    """첫 질문 — 그래프 정상 통과 + 더미 응답 생성 확인"""
    graph = build_chatbot_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "test-first"}}

    result = graph.invoke(_initial_input("빅데이터 트랙이 뭐야?"), config=config)

    # 콘솔 출력용
    print("\n")
    for i, msg in enumerate(result["messages"]):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")

    assert result["intent"] == "general_advice"
    assert result["response"] is not None
    assert result["response"].startswith("(dummy)")


def test_followup_turn_history_auto_persisted() -> None:
    """후속 질문 — 같은 thread_id 호출 시 checkpointer 가 이전 turn 의
    user/assistant 메시지를 자동 복원해 messages 에 누적
    """
    graph = build_chatbot_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "test-followup"}}

    # 첫 턴 → user 1 + assistant 1 = 2 messages
    first = graph.invoke(_initial_input("빅데이터 트랙이 뭐야?"), config=config)

    # 후속 턴 → user 1 개만 추가지만 checkpointer 가 이전 2 개 복원 → 총 4 개
    result = graph.invoke(
        {"messages": [HumanMessage(content="그럼 어떤 직무로 가면 좋아?")]},
        config=config,
    )

    # 콘솔 출력용
    print("\n")
    print(f"[첫 turn 후 messages 수] {len(first['messages'])}")
    print("[첫 turn messages 상세]")
    for i, msg in enumerate(first["messages"]):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")
    print(f"[후속 turn 후 messages 수] {len(result['messages'])}")
    print("[후속 turn messages 상세]")
    for i, msg in enumerate(result["messages"]):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")

    assert len(result["messages"]) == 4
    assert result["messages"][0].content == "빅데이터 트랙이 뭐야?"
    assert result["messages"][-1].content.startswith("(dummy)")


def test_threads_isolated() -> None:
    """다른 thread_id 간 히스토리 격리 확인"""
    graph = build_chatbot_graph()

    config_a: RunnableConfig = {"configurable": {"thread_id": "A"}}
    config_b: RunnableConfig = {"configurable": {"thread_id": "B"}}

    result_a = graph.invoke(_initial_input("A 의 첫 질문"), config=config_a)
    result_b = graph.invoke(_initial_input("B 의 첫 질문"), config=config_b)

    # 콘솔 출력용
    print("\n")
    print("[A messages 상세]")
    for i, msg in enumerate(result_a["messages"]):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")
    print("[B messages 상세]")
    for i, msg in enumerate(result_b["messages"]):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")

    # A 는 자기 turn 만 보유 → B 메시지 누수 없음
    assert len(result_a["messages"]) == 2
    assert result_a["messages"][0].content == "A 의 첫 질문"
    # B 는 자기 turn 만 보유 → A 메시지 누수 없음
    assert len(result_b["messages"]) == 2
    assert result_b["messages"][0].content == "B 의 첫 질문"
