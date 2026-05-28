"""ClassifyIntentNode 단위 테스트"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from tracktory.chatbot.nodes import ClassifyIntentNode
from tracktory.chatbot.state import ChatbotState


def _state(messages: list[BaseMessage]) -> ChatbotState:
    return ChatbotState(user_context={}, messages=messages)


def test_classifier_failure_falls_back_to_general_advice(
    caplog: logging.LogCaptureFixture,
) -> None:
    """classifier 가 예외를 던지면 general_advice 로 폴백 + 사유 기록."""
    classifier = MagicMock()
    classifier.invoke.side_effect = RuntimeError("boom")
    node = ClassifyIntentNode(classifier)

    message = "빅데이터 트랙이 뭐야?"
    result = node(_state([HumanMessage(content=message)]))

    # 콘솔 출력용
    print("\n")
    print(f"[입력] {message}")
    print(f"[출력] intent={result['intent']}")
    print(f"[근거] {result['intent_reason']}")
    print(f"[로그] {[rec.message for rec in caplog.records]}")
    ###

    assert result == {
        "intent": "general_advice",
        "intent_reason": "분류 실패: RuntimeError",
        "search_keywords": [],
    }
    assert any("분류 실패" in rec.message for rec in caplog.records), (
        "폴백 발생 시 warning 로그 누락"
    )


@pytest.mark.parametrize(
    ("messages", "expected_reason"),
    [
        ([], "잘못된 입력: 빈 메시지"),
        ([AIMessage(content="ai 응답")], "잘못된 입력: HumanMessage 아님 (AIMessage)"),
        (
            [HumanMessage(content=[{"type": "text", "text": "multimodal"}])],
            "잘못된 입력: 문자열 아님 (list)",
        ),
    ],
    ids=["empty_messages", "non_human_message", "non_string_message"],
)
def test_invalid_input_falls_back_to_general_advice(
    messages: list[BaseMessage],
    expected_reason: str,
    caplog: logging.LogCaptureFixture,
) -> None:
    """진입부 가드 — 빈 메시지 / HumanMessage 아님 / 문자열 아닌 메시지 폴백"""
    classifier = MagicMock()
    node = ClassifyIntentNode(classifier)

    result = node(_state(messages))

    # 콘솔 출력용
    print("\n")
    print(f"[입력] {[(type(m).__name__, m.content) for m in messages] or '빈 리스트'}")
    print(f"[출력] intent={result['intent']}")
    print(f"[근거] {result['intent_reason']}")
    print(f"[로그] {[rec.message for rec in caplog.records]}")
    ###

    assert result == {
        "intent": "general_advice",
        "intent_reason": expected_reason,
        "search_keywords": [],
    }

    # 가드 위반 시 LLM 호출 차단
    classifier.invoke.assert_not_called()

    # 폴백 시 warning 로그 호출 (사일런트 폴백 방지)
    assert any(expected_reason in rec.message for rec in caplog.records), (
        "가드 폴백 시 warning 로그 누락"
    )
