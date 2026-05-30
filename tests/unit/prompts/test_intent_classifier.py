"""의도 분류 프롬프트 테스트.

- 순수 단위 — 템플릿·스키마 검증. ``uv run pytest`` 기본 실행.
- 골든셋 LLM 평가 (``@pytest.mark.integration``) — 실제 LLM 호출.
  ``uv run pytest tests/unit/prompts -m integration -s``.
"""

import pytest
from pydantic import ValidationError

from tracktory.common.config import settings
from tracktory.prompts.chatbot.intent import (
    INTENT_CLASSIFIER_PROMPT,
    IntentClassification,
)

# 1. 순수 단위 — 템플릿·스키마 sanity


def test_template_variable_is_message_only() -> None:
    """프롬프트의 ``{messag}`` 같은 오타를 정의 시점에 잡는다."""
    assert set(INTENT_CLASSIFIER_PROMPT.input_variables) == {"message"}


def test_schema_rejects_invalid_intent_label() -> None:
    """오타 라벨은 enum 단계에서 거부 — 라우팅 안전망."""
    with pytest.raises(ValidationError):
        IntentClassification(
            intent="track_questiion",  # type: ignore[arg-type]
            search_keywords=[],
            reason="오타",
        )


# 2. 골든셋 LLM 평가 — 의도 라벨은 정확 일치 (enum 이라 변동성 없음), 키워드는 "기대 토큰이
# 결과에 포함" 으로 약하게 검증. LLM 비결정성으로 추가 토큰이 섞일 수 있어 정확 일치(==)는 회귀
# 신호가 아닌 noise 로 fail 하기 쉬움. general_advice 만 예외로 빈 리스트 강제 (RAG 우회 경로).

_GOLDEN_CASES: list[tuple[str, str, list[str]]] = [
    # 단일 의도 — 4 개 카테고리 각각의 대표 케이스
    # ("빅데이터 트랙이 AI 트랙이랑 뭐가 달라요?", "track_question", ["빅데이터 트랙", "AI 트랙"]),
    # ("백엔드 개발자가 되려면 뭘 공부해야 해요?", "job_question", ["백엔드 개발자"]),
    # ("데이터베이스 과목이 어려운가요?", "course_question", ["데이터베이스"]),
    # ("자료구조 선수과목이 뭐예요?", "course_question", ["자료구조"]),
    # ("2 학년 때 뭘 준비하면 좋을까요?", "general_advice", []),
    # 복합 의도 — 트랙·직무·과목·기술명이 섞일 때 핵심 의도를 잡는지 확인
    ("AI 트랙 가려면 무슨 과목 들어야 해요?", "course_question", ["AI 트랙"]),
    (
        "디자인 트랙 들으면 UX 디자이너 될 수 있어요?",
        "job_question",
        ["디자인 트랙", "UX 디자이너"],
    ),
    ("프론트엔드 개발자 되려면 어떤 트랙 들어야 해요?", "track_question", ["프론트엔드 개발자"]),
    ("백엔드 개발자 되려면 어떤 과목 들어야 해요?", "course_question", ["백엔드 개발자"]),
    ("React 배우면 어떤 직무로 갈 수 있어요?", "job_question", ["React"]),
]


@pytest.mark.integration
@pytest.mark.skipif(not settings.openai_api_key, reason="OPENAI_API_KEY 미설정")
@pytest.mark.parametrize(("message", "expected_intent", "expected_keywords"), _GOLDEN_CASES)
def test_intent_classifier_golden(
    message: str, expected_intent: str, expected_keywords: list[str]
) -> None:
    """프롬프트 변경 후 회귀 확인. 비결정성으로 가끔 fail 가능."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = INTENT_CLASSIFIER_PROMPT | llm.with_structured_output(IntentClassification)
    result: IntentClassification = chain.invoke({"message": message})

    print(f"\n[입력] {message}")
    print(f"[출력] intent={result.intent} keywords={result.search_keywords}")
    print(f"[근거] {result.reason}")

    assert result.intent == expected_intent, (
        f"의도 라벨 불일치: 기대={expected_intent}, 실제={result.intent}"
    )
    if expected_keywords:
        for kw in expected_keywords:
            assert kw in result.search_keywords, (
                f"키워드 '{kw}' 누락: 실제={result.search_keywords}"
            )
    else:
        assert result.search_keywords == [], (
            f"general_advice 인데 키워드 비어있지 않음: {result.search_keywords}"
        )
