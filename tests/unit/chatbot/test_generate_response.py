"""GenerateResponseNode 실 LLM 호출 테스트 — 응답 본문 + 선택지 정성 검증

classify_intent / retrieve_rag 우회. 실 RAGFlow 출력에서 떼어온 canned 청크 +
실 OpenAI 호출로 응답 생성 단계만 격리 검증.

실행:
    # 전체 (3 케이스)
    uv run pytest tests/unit/chatbot/test_generate_response.py -m integration -s

    # 한 케이스만
    uv run pytest tests/unit/chatbot/test_generate_response.py -m integration -s -k track
    uv run pytest tests/unit/chatbot/test_generate_response.py -m integration -s -k general
    uv run pytest tests/unit/chatbot/test_generate_response.py -m integration -s -k no_data
"""

from __future__ import annotations

import textwrap
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from tracktory.chatbot.nodes import GenerateResponseNode
from tracktory.chatbot.rag.ragflow import RetrievedChunk
from tracktory.chatbot.state import ChatbotState
from tracktory.common.config import settings
from tracktory.prompts.chatbot.general_advice import GENERAL_ADVICE_PROMPT
from tracktory.prompts.chatbot.rag_response import (
    RAG_RESPONSE_PROMPT,
    ChatbotResponse,
)


# 실 RAGFlow 출력에서 떼어온 canned 청크
_TRACK_CHUNKS: list[RetrievedChunk] = [
    {
        "content": (
            "[트랙: 빅데이터트랙 | 대학: IT공과대학 | 학부: 컴퓨터공학부] "
            "■ 소개 기본적인 데이터베이스 이론 및 설계 기술의 습득을 시작으로 "
            "다양한 분야에서 발생하는 빅데이터를 수집/분석/활용할 수 있는 데이터 마이닝 기술 및 "
            "실제 프로젝트 수행 교과목 등을 통해 실무 중심의 교육을 실시함. "
            "■ 졸업 후 진로 데이터베이스 분야 전문 개발자 및 관리자, "
            "기업전략 수립을 위한 데이터 마이닝 개발자 및 관리자, 인공지능 S/W 전문가"
        ),
        "document_name": "트랙소개_빅데이터트랙.txt",
        "similarity": 0.39,
        "id": "track_bigdata",
    },
    {
        "content": (
            "[트랙: AIㆍ소프트웨어학과 | 대학: 미래플러스대학 | 학부: 미래플러스대학] "
            "■ 소개 AI·소프트웨어학과는 인공지능(AI)과 소프트웨어(SW) 개발 역량을 겸비한 "
            "융합형 실무 인재 양성을 목표. 체계적인 교육 과정을 통해 소프트웨어 개발부터 "
            "AI 및 빅데이터 활용에 이르는 핵심 역량을 배양하고, 융합 프로젝트 설계 및 "
            "운영 능력을 함양하여 미래 산업 환경에 즉시 적응 가능한 전문가로 성장."
        ),
        "document_name": "트랙소개_AI_소프트웨어학과.txt",
        "similarity": 0.34,
        "id": "track_ai_sw",
    },
]


_CASES: dict[str, dict[str, Any]] = {
    "track": {
        "message": "빅데이터 트랙이랑 AI 트랙 차이가 뭐예요?",
        "intent": "track_question",
        "retrieved_docs": _TRACK_CHUNKS,
        "user_context": {},
    },
    "general": {
        "message": "2 학년 때 뭘 준비하면 좋을까요?",
        "intent": "general_advice",
        "retrieved_docs": [],
        "user_context": {"entry_year": 2026, "current_year": 2, "interests": ["AI", "데이터분석"]},
    },
    "no_data": {
        "message": "AI 트랙 정원이 몇 명이에요?",
        "intent": "track_question",
        "retrieved_docs": [],  # 자료 없는 상태
        "user_context": {},
    },
}


def _state(case: dict[str, Any]) -> ChatbotState:
    return {
        "user_context": case["user_context"],
        "messages": [HumanMessage(content=case["message"])],
        "intent": case["intent"],  # type: ignore[typeddict-item]
        "intent_reason": None,
        "search_keywords": [],
        "retrieved_docs": case["retrieved_docs"],
        "response": None,
        "response_choices": [],
    }


@pytest.mark.integration
@pytest.mark.skipif(not settings.openai_api_key, reason="OPENAI_API_KEY 미설정")
@pytest.mark.parametrize("case_name", list(_CASES.keys()))
def test_generate_response(case_name: str) -> None:
    """canned retrieved_docs + 실 LLM → 응답 본문 + 선택지 생성"""
    from langchain_openai import ChatOpenAI

    case = _CASES[case_name]

    # chain 조립 — intent 별로 다른 chain 주입
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rag_chain = RAG_RESPONSE_PROMPT | llm.with_structured_output(ChatbotResponse)
    general_chain = GENERAL_ADVICE_PROMPT | llm.with_structured_output(ChatbotResponse)
    node = GenerateResponseNode(rag_chain, general_chain)

    # 호출
    state = _state(case)
    result = node(state)

    # 출력
    print("\n" + "=" * 70)
    print(f"[{case_name}] {case['message']!r}")
    print(f"  intent: {case['intent']}  /  청크 수: {len(case['retrieved_docs'])}")
    print("=" * 70)
    print("\n[응답 본문]")
    print(textwrap.fill(result["response"], width=80, initial_indent="  ", subsequent_indent="  "))
    print("\n[후속 선택지]")
    for i, choice in enumerate(result["response_choices"], start=1):
        print(f"  {i}. {choice}")
    print("=" * 70)

    # ───── Assertion ─────
    assert result["response"], "응답 본문 비어있음"
    assert 1 <= len(result["response_choices"]) <= 3, (
        f"choices 1-3 강제 위반: {len(result['response_choices'])}개"
    )
    # AIMessage 가 messages 에 append 되는지
    assert result["messages"] and result["messages"][0].content == result["response"]

    # 자료 없을 때 "해당 데이터 없음" 명시 검증 — RAG 케이스에 한정
    # (general_advice 는 자료 미사용이 정상이라 안내 문구 없이 일반 조언 답해야 함)
    if case_name == "no_data":
        assert "해당 데이터 없음" in result["response"] or "[근거: 없음]" in result["response"], (
            f"자료 없는 RAG 케이스인데 안내 문구 없음: {result['response']}"
        )
