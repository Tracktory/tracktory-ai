"""RetrieveRagNode 실 RAGFlow 호출 테스트

RetrieveRagNode → RagFlowChatbotRetriever → 실 RAGFlow → 청크 반환
전구간 검증

실행:
    # 전체 (4 케이스 — track / job / course / general)
    uv run pytest tests/unit/chatbot/test_retrieve_rag.py -m integration -s

    # 한 케이스만 (-k <case_name>)
    uv run pytest tests/unit/chatbot/test_retrieve_rag.py -m integration -s -k track
    uv run pytest tests/unit/chatbot/test_retrieve_rag.py -m integration -s -k job
    uv run pytest tests/unit/chatbot/test_retrieve_rag.py -m integration -s -k course
    uv run pytest tests/unit/chatbot/test_retrieve_rag.py -m integration -s -k general

    # 여러 케이스 조합
    uv run pytest tests/unit/chatbot/test_retrieve_rag.py -m integration -s -k "track or job"
    uv run pytest tests/unit/chatbot/test_retrieve_rag.py -m integration -s -k "not general"
"""

from __future__ import annotations

import textwrap

import pytest
from langchain_core.messages import HumanMessage

from tracktory.chatbot.config import ChatbotIntent
from tracktory.chatbot.nodes import RetrieveRagNode
from tracktory.chatbot.rag.ragflow import RagFlowChatbotRetriever
from tracktory.chatbot.state import ChatbotState
from tracktory.common.config import settings

pytest.importorskip(
    "ragflow_sdk",
    reason="ragflow-sdk 미설치. `uv add ragflow-sdk` 후 재실행",
)


def _state(
    *,
    message: str,
    intent: ChatbotIntent,
    intent_reason: str | None,
    keywords: list[str],
) -> ChatbotState:
    return ChatbotState(
        user_context={},
        messages=[HumanMessage(content=message)],
        intent=intent,
        intent_reason=intent_reason,
        search_keywords=keywords,
    )


# 테스트 케이스 사전 — `-k <case_name>` 으로 골라서 실행 가능
_CASES: dict[str, dict] = {
    "track": {
        "message": "빅데이터 트랙이랑 AI 트랙 차이가 뭐예요?",
        "intent": "track_question",
        "intent_reason": "트랙 자체의 차이를 묻고 있음",
        "keywords": ["빅데이터 트랙", "AI 트랙"],
    },
    "job": {
        "message": "백엔드 개발자가 되려면 뭘 공부해야 해요?",
        "intent": "job_question",
        "intent_reason": "특정 직무로 가기 위한 역량을 묻고 있음",
        "keywords": ["백엔드 개발자"],
    },
    "course": {
        "message": "데이터베이스 과목이 어려운가요?",
        "intent": "course_question",
        "intent_reason": "특정 과목 자체의 속성을 묻고 있음",
        "keywords": ["데이터베이스"],
    },
    "general": {
        "message": "2 학년 때 뭘 준비하면 좋을까요?",
        "intent": "general_advice",
        "intent_reason": "특정 트랙·직무·과목이 아닌 학년 단위 메타 조언",
        "keywords": [],
    },
}


@pytest.mark.integration
@pytest.mark.skipif(
    not (settings.ragflow_api_key and settings.ragflow_base_url and settings.ragflow_dataset_id),
    reason="RAGFlow env (API_KEY / BASE_URL / DATASET_ID) 미설정",
)
@pytest.mark.parametrize("case_name", list(_CASES.keys()))
def test_retrieve_rag_real_call(case_name: str) -> None:
    """4 가지 의도 케이스로 실 RAGFlow 호출 검증 (general 은 RAG 우회)"""
    case = _CASES[case_name]
    node = RetrieveRagNode(RagFlowChatbotRetriever())

    result = node(_state(**case))
    chunks = result["retrieved_docs"]

    print("\n" + "=" * 70)
    print(f"[{case_name}] {case['message']!r}  (intent={case['intent']})")
    print(f"반환 청크 수: {len(chunks)}")
    print("=" * 70)
    for i, chunk in enumerate(chunks, start=1):
        # 모든 whitespace 정규화 + 80자 단위로 줄바꿈 (truncation 없이 전체 표시)
        content = " ".join(chunk["content"].split())
        wrapped = textwrap.fill(content, width=80, initial_indent="    ", subsequent_indent="    ")
        print(f"\n--- #{i} ---")
        print(f"  sim:     {chunk['similarity']:.2f}")
        print(f"  src:     {chunk['document_name']}")
        print(f"  content ({len(content)}자):")
        print(wrapped)
    print("=" * 70)

    if case["intent"] == "general_advice":
        # RAG 우회 경로 — retriever 호출 자체가 안 됨
        assert chunks == []
    else:
        assert len(chunks) > 0, "RAGFlow 가 0 건 반환 — 데이터셋·threshold·query 확인"
        for chunk in chunks:
            assert chunk["content"].strip(), f"빈 본문 청크가 통과됨: {chunk}"
