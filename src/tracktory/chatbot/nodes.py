"""챗봇 LangGraph 노드"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from tracktory.chatbot.rag.ragflow import RagFlowChatbotRetriever, RagSearchError
from tracktory.chatbot.state import ChatbotState
from tracktory.prompts.chatbot.intent import IntentClassification
from tracktory.prompts.chatbot.rag_response import (
    ChatbotResponse,
    format_retrieved_docs,
    format_user_context,
)

logger = logging.getLogger("chatbot")

# classify_intent 가 LLM 에 전달할 직전 turn 수 (user+AI 합쳐서)
_INTENT_HISTORY_TURNS = 10


def _format_intent_history(messages: list[BaseMessage]) -> str:
    """직전 messages 를 role + content 한 줄씩 history 블록으로 정형화."""
    if not messages:
        return "(직전 대화 없음 — 첫 turn)"
    recent = messages[-_INTENT_HISTORY_TURNS:]
    lines = []
    for msg in recent:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


class ClassifyIntentNode:
    """챗봇 의도 분류 노드

    질문의 의도를 4-way 분류 (track / job / course / general) + 카탈로그 / 학년 시그널.
    멀티턴 지시어 처리를 위해 직전 N turn 의 messages 도 LLM 에 같이 전달한다.
    LLM 호출 실패 시 general_advice 로 graceful 폴백
    """

    def __init__(self, classifier: Runnable[dict[str, Any], IntentClassification]) -> None:
        self._classifier = classifier

    def __call__(self, state: ChatbotState) -> dict:
        messages = state["messages"]
        last_message = messages[-1].content
        # 마지막 user 발화는 별도로 LLM 에 넘기므로 history 에서 제외
        history = _format_intent_history(messages[:-1])

        try:
            result = self._classifier.invoke({"message": last_message, "history": history})
            return {
                "intent": result.intent,
                "intent_reason": result.reason,
                "search_keywords": result.search_keywords,
                "is_catalog_query": result.is_catalog_query,
                "target_grade": result.target_grade,
            }
        except Exception as e:
            logger.warning("Intent classifier failed, falling back to general_advice: %s", e)
            return {
                "intent": "general_advice",
                "intent_reason": f"classifier_failed: {type(e).__name__}",
                "search_keywords": [],
                "is_catalog_query": False,
                "target_grade": None,
            }


class RetrieveRagNode:
    """챗봇 RAG 검색 노드 — 의도 + 사용자 컨텍스트 결합한 다중 검색 전략.

    분기 케이스 (우선순위 순):
        1. catalog: '트랙 종류' 같은 전체 카탈로그 요청
            → doc_type=track_intro 전체 (사용자 college 1차 필터 가능)
        2. user_track_courses: 사용자 트랙 기반 과목 추천 (키워드 없음 + tracks 있음)
            → doc_type=curriculum AND track_name in tracks
        3. grade_syllabus: 학년 명시 과목 추천 (target_grade 있음)
            → doc_type=syllabus AND target_grade contains "N학년"
        4. user_track_jobs: 사용자 트랙 기반 직무 (키워드 없음 + tracks 있음)
            → doc_type=track_intro AND track_name in tracks (졸업 후 진로 섹션)
        5. keyword_search: 기존 동작 (키워드 있음)
            → 일반 vector + keyword 검색

    어느 분기든 실패하면 빈 list 로 graceful 폴백.
    """

    def __init__(self, retriever: RagFlowChatbotRetriever) -> None:
        self._retriever = retriever

    def __call__(self, state: ChatbotState) -> dict:
        intent = state.get("intent")
        is_catalog = state.get("is_catalog_query") or False
        target_grade = state.get("target_grade")
        keywords = state.get("search_keywords") or []
        user_ctx = state.get("user_context") or {}
        user_tracks = user_ctx.get("tracks") or []
        user_college = user_ctx.get("college")

        # general_advice / None — 자료 검색 우회 (정상 경로상 도달 X 안전망)
        if not intent or intent == "general_advice":
            logger.debug("retrieve_rag 우회 (intent=%s)", intent)
            return {"retrieved_docs": []}

        # ── 1. 카탈로그: 트랙 종류 / 전체 목록 ──
        if intent == "track_question" and is_catalog:
            return self._search_track_catalog(college=user_college)

        # ── 2. 사용자 트랙 기반 과목 추천 ──
        if intent == "course_question" and not keywords and user_tracks:
            return self._search_curriculum_by_tracks(user_tracks)

        # ── 3. 학년 명시 과목 추천 ──
        if intent == "course_question" and target_grade:
            return self._search_syllabus_by_grade(target_grade, keywords)

        # ── 4. 사용자 트랙 기반 직무 ──
        if intent == "job_question" and not keywords and user_tracks:
            return self._search_track_intro_by_tracks(user_tracks)

        # ── 5. 기존: 키워드 기반 일반 검색 ──
        if not keywords:
            logger.info("retrieve_rag: 검색 키워드 없음 (intent=%s)", intent)
            return {"retrieved_docs": []}
        return self._search_keywords(keywords, intent)

    # ── 분기별 검색 헬퍼 ──

    def _search_track_catalog(self, *, college: str | None) -> dict:
        """track_intro 메타 전체 (선택적으로 사용자 college 로 1차 필터)"""
        conditions: list[dict[str, Any]] = [
            {"name": "doc_type", "comparison_operator": "is", "value": "track_intro"}
        ]
        if college:
            conditions.append({"name": "college", "comparison_operator": "is", "value": college})
        logger.info("retrieve_rag: 트랙 카탈로그 검색 (college=%s)", college)
        return self._safe_search(
            "트랙 소개",
            metadata_condition={"logic": "and", "conditions": conditions},
            page_size=70,
        )

    def _search_curriculum_by_tracks(self, tracks: list[str]) -> dict:
        """사용자 트랙들의 curriculum 메타 전체"""
        conditions = [
            {"name": "doc_type", "comparison_operator": "is", "value": "curriculum"},
            {"name": "track_name", "comparison_operator": "in", "value": tracks},
        ]
        logger.info("retrieve_rag: 사용자 트랙 curriculum 검색 (tracks=%s)", tracks)
        return self._safe_search(
            "전공 과목",
            metadata_condition={"logic": "and", "conditions": conditions},
            page_size=20,
        )

    def _search_syllabus_by_grade(self, grade: int, keywords: list[str]) -> dict:
        """target_grade 명시된 syllabus 검색 (있으면 키워드도 같이)"""
        conditions = [
            {"name": "doc_type", "comparison_operator": "is", "value": "syllabus"},
            {
                "name": "target_grade",
                "comparison_operator": "contains",
                "value": f"{grade}학년",
            },
        ]
        query = ",".join(keywords) if keywords else "전공 과목"
        logger.info("retrieve_rag: 학년 강의계획서 검색 (grade=%d, keywords=%s)", grade, keywords)
        return self._safe_search(
            query,
            metadata_condition={"logic": "and", "conditions": conditions},
        )

    def _search_track_intro_by_tracks(self, tracks: list[str]) -> dict:
        """사용자 트랙의 track_intro (졸업 후 진로 정보 포함)"""
        conditions = [
            {"name": "doc_type", "comparison_operator": "is", "value": "track_intro"},
            {"name": "track_name", "comparison_operator": "in", "value": tracks},
        ]
        logger.info("retrieve_rag: 사용자 트랙 track_intro 검색 (tracks=%s)", tracks)
        return self._safe_search(
            "졸업 후 진로",
            metadata_condition={"logic": "and", "conditions": conditions},
            page_size=10,
        )

    def _search_keywords(self, keywords: list[str], intent: str) -> dict:
        """키워드 기반 일반 vector + keyword 검색 (기존 동작)"""
        query = ",".join(keywords)
        logger.info("retrieve_rag: 키워드 검색 (intent=%s, query=%r)", intent, query)
        return self._safe_search(query)

    def _safe_search(
        self,
        query: str,
        *,
        metadata_condition: dict[str, Any] | None = None,
        page_size: int = 10,
    ) -> dict:
        """RagSearchError 폴백 + 빈 청크 제거 공통 wrapper"""
        try:
            results = self._retriever.search(
                query,
                metadata_condition=metadata_condition,
                page_size=page_size,
            )
        except RagSearchError:
            logger.warning(
                "retrieve_rag: 검색 실패 (query=%r, metadata=%s)",
                query,
                metadata_condition,
                exc_info=True,
            )
            return {"retrieved_docs": []}

        cleaned = [chunk for chunk in results if chunk["content"].strip()]
        return {"retrieved_docs": cleaned}


class GenerateResponseNode:
    """챗봇 응답 생성 노드 — intent 에 따라 RAG / 일반 조언 chain 선택

    - track/job/course → RAG chain (RAG_RESPONSE_PROMPT, retrieved_docs 사용)
    - general_advice → 일반 조언 chain (GENERAL_ADVICE_PROMPT, 자료 미사용)
    """

    def __init__(
        self,
        rag_chain: Runnable[dict[str, Any], ChatbotResponse],
        general_chain: Runnable[dict[str, Any], ChatbotResponse],
    ) -> None:
        self._rag = rag_chain
        self._general = general_chain

    def __call__(self, state: ChatbotState) -> dict:
        user_context_block = format_user_context(state.get("user_context"))

        if state.get("intent") == "general_advice":
            result = self._general.invoke(
                {
                    "messages": state["messages"],
                    "user_context_block": user_context_block,
                }
            )
        else:
            result = self._rag.invoke(
                {
                    "messages": state["messages"],
                    "user_context_block": user_context_block,
                    "retrieved_context": format_retrieved_docs(state.get("retrieved_docs") or []),
                }
            )

        return {
            "response": result.text,
            "response_choices": result.choices,
            "messages": [AIMessage(content=result.text)],
        }
