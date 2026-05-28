"""ClassifyIntentNode 단위 테스트"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from tracktory.chatbot.nodes import ClassifyIntentNode
from tracktory.chatbot.state import ChatbotState


def _state(message: str) -> ChatbotState:
    return {
        "user_context": {},
        "messages": [HumanMessage(content=message)],
        "intent": None,
        "intent_reason": None,
        "search_keywords": [],
        "retrieved_docs": [],
        "response": None,
    }


def test_classifier_failure_falls_back_to_general_advice(
    caplog: logging.LogCaptureFixture,
) -> None:
    """classifier 가 예외를 던지면 general_advice 로 폴백 + 사유 기록."""
    classifier = MagicMock()
    classifier.invoke.side_effect = RuntimeError("boom")
    node = ClassifyIntentNode(classifier)

    message = "빅데이터 트랙이 뭐야?"
    result = node(_state(message))

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
