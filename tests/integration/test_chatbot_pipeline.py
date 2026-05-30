"""챗봇 파이프라인 통합 테스트 — 질문 → LLM 분류 → RAGFlow 검색

ClassifyIntentNode (실 OpenAI) → RetrieveRagNode (실 RAGFlow) 까지 흘려보고
각 단계 결과 출력 + 기본 assertion

TODO (TK-19 응답 생성 노드 완성 후):
    - generate_response 단계 추가
    - state["response"] assertion (자료 기반 응답 / "해당 데이터 없음" 분기 등)

실행:
    # 전체 (5 케이스)
    uv run pytest tests/integration/test_chatbot_pipeline.py -m integration -s

    # 한 질문만
    uv run pytest tests/integration/test_chatbot_pipeline.py -m integration -s -k "general"
"""

from __future__ import annotations

import textwrap

import pytest
from langchain_core.messages import HumanMessage

from tracktory.chatbot.nodes import ClassifyIntentNode, RetrieveRagNode
from tracktory.chatbot.rag.ragflow import RagFlowChatbotRetriever
from tracktory.chatbot.state import ChatbotState
from tracktory.common.config import settings
from tracktory.prompts.chatbot.intent import (
    INTENT_CLASSIFIER_PROMPT,
    IntentClassification,
)

pytest.importorskip(
    "ragflow_sdk",
    reason="ragflow-sdk 미설치. `uv add ragflow-sdk` 후 재실행",
)


# 자연어 질문 — intent / keywords 는 LLM 이 결정
_QUERIES: list[str] = [
    "빅데이터 트랙이랑 AI 트랙 차이가 뭐예요?",
    "백엔드 개발자가 되려면 뭘 공부해야 해요?",
    "데이터베이스 과목이 어려운가요?",
    "백엔드 개발자 되려면 어떤 과목 들어야 해요?",
    "2 학년 때 뭘 준비하면 좋을까요?",  # general_advice 예상 — RAG 우회 검증
]


def _initial_state(message: str) -> ChatbotState:
    return ChatbotState(user_context={}, messages=[HumanMessage(content=message)])


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        settings.openai_api_key
        and settings.ragflow_api_key
        and settings.ragflow_base_url
        and settings.ragflow_dataset_id
    ),
    reason="OPENAI_API_KEY 또는 RAGFlow env 미설정",
)
@pytest.mark.parametrize("query", _QUERIES)
def test_chatbot_pipeline(query: str) -> None:
    """질문 → LLM 의도 분류 → RAGFlow 검색 파이프라인

    TODO: TK-19 generate_response LLM 노드 완성 후 응답 생성 단계 추가
    """
    from langchain_openai import ChatOpenAI

    # 의도 분류 chain 조립
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    classifier = INTENT_CLASSIFIER_PROMPT | llm.with_structured_output(IntentClassification)

    # 노드 + 의존성 주입
    intent_node = ClassifyIntentNode(classifier)
    retrieve_node = RetrieveRagNode(RagFlowChatbotRetriever())

    # 초기 state
    state = _initial_state(query)

    # ───── 단계 1: 의도 분류 ─────
    intent_result = intent_node(state)
    state.intent = intent_result["intent"]
    state.intent_reason = intent_result["intent_reason"]
    state.search_keywords = intent_result["search_keywords"]

    print("\n" + "=" * 70)
    print(f"질문: {query!r}")
    print("─" * 70)
    print(f"intent:          {state.intent}")
    print(f"intent_reason:   {state.intent_reason}")
    print(f"search_keywords: {state.search_keywords}")
    print("─" * 70)

    # ───── 단계 2: RAG 검색 ─────
    retrieve_result = retrieve_node(state)
    state.retrieved_docs = retrieve_result["retrieved_docs"]
    chunks = state.retrieved_docs

    print(f"반환 청크 수: {len(chunks)}")
    print("=" * 70)
    for i, chunk in enumerate(chunks, start=1):
        content = " ".join(chunk["content"].split())
        wrapped = textwrap.fill(
            content[:240], width=80, initial_indent="    ", subsequent_indent="    "
        )
        print(f"\n--- #{i} ---")
        print(f"  sim: {chunk['similarity']:.2f}")
        print(f"  src: {chunk['document_name']}")
        print(wrapped + "...")
    print("=" * 70)

    # ───── 단계 3: 응답 생성 (TK-19 후 추가) ─────
    # TODO: generate_response 가 LLM 호출로 교체되면 여기서 호출하고 state["response"] 검증

    # ───── Assertion ─────
    assert state.intent in (
        "track_question",
        "job_question",
        "course_question",
        "general_advice",
    ), f"의도 라벨이 4 종류 중 하나여야 함: {state.intent}"

    if state.intent == "general_advice":
        # RAG 우회 — chunks 빈 리스트
        assert chunks == [], f"general_advice 인데 청크 반환됨: {len(chunks)}건"
    else:
        # RAG 호출됨 — 결과 있어야
        assert len(chunks) > 0, (
            f"RAGFlow 0 건 반환 — intent={state.intent}, keywords={state.search_keywords}"
        )
        for chunk in chunks:
            assert chunk["content"].strip(), f"빈 본문 청크: {chunk}"
