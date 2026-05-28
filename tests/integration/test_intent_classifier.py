"""ClassifyIntentNode 통합 테스트 — 실 LLM (OpenAI) 호출 + state 매핑까지 검증

마커: @pytest.mark.integration → CI 기본 실행에서 제외
스킵: OPENAI_API_KEY 미설정 시
주의: LLM 비결정성으로 가끔 fail 가능 (temperature=0 으로 변동 최소화)

검증 범위:
    질문 → 프롬프트 + LLM → IntentClassification → 노드의 매핑 → state dict
    (LLM 분류 정확도 + 노드의 reason→intent_reason 이름 변환 동시 검증)

실행:
    uv run pytest tests/integration/test_intent_classifier.py -m integration -s
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from tracktory.chatbot.nodes import ClassifyIntentNode
from tracktory.chatbot.state import ChatbotState
from tracktory.common.config import settings
from tracktory.prompts.chatbot.intent import (
    INTENT_CLASSIFIER_PROMPT,
    IntentClassification,
)

# 골든 케이스 — (message, expected_intent, expected_keywords)
# expected_keywords 는 search_keywords 에 반드시 포함되어야 하는 부분집합
# (LLM 이 더 많이 뽑아도 OK, 핵심 키워드 누락만 잡는다)
_GOLDEN_CASES: list[tuple[str, str, list[str]]] = [
    # track_question
    ("빅데이터 트랙이랑 AI 트랙 차이가 뭐예요?", "track_question", ["빅데이터 트랙", "AI 트랙"]),
    # ("산업공학 트랙은 뭐 배워요?", "track_question", ["산업공학 트랙"]),
    # job_question
    ("백엔드 개발자가 되려면 뭘 공부해야 해요?", "job_question", ["백엔드 개발자"]),
    # ("데이터 엔지니어 직무는 어떤 일을 해요?", "job_question", ["데이터 엔지니어"]),
    # course_question
    ("데이터베이스 과목이 어려운가요?", "course_question", ["데이터베이스"]),
    # ("자료구조 수업 어때요?", "course_question", ["자료구조"]),
    # general_advice
    ("2 학년 때 뭘 준비하면 좋을까요?", "general_advice", []),
    # ("학점 관리 어떻게 해야 해요?", "general_advice", []),
]


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


@pytest.mark.integration
@pytest.mark.skipif(not settings.openai_api_key, reason="OPENAI_API_KEY 미설정")
@pytest.mark.parametrize(("message", "expected_intent", "expected_keywords"), _GOLDEN_CASES)
def test_classify_intent_node_golden(
    message: str, expected_intent: str, expected_keywords: list[str]
) -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    classifier = INTENT_CLASSIFIER_PROMPT | llm.with_structured_output(IntentClassification)
    node = ClassifyIntentNode(classifier)

    result = node(_state(message))

    print("\n")
    print(result)
    print(f"[입력] {message}")
    print(f"[출력] intent={result['intent']} keywords={result['search_keywords']}")
    print(f"[근거] {result['intent_reason']}")

    assert result["intent"] == expected_intent, (
        f"의도 라벨 불일치: 기대={expected_intent}, 실제={result['intent']}"
    )
    assert result["intent_reason"], "intent_reason 비어있음 (reason → intent_reason 매핑 실패 가능)"

    if expected_keywords:
        for kw in expected_keywords:
            assert kw in result["search_keywords"], (
                f"키워드 '{kw}' 누락: 실제={result['search_keywords']}"
            )
    else:
        assert result["search_keywords"] == [], (
            f"general_advice 인데 키워드 비어있지 않음: {result['search_keywords']}"
        )
