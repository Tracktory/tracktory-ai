"""챗봇 RAG 응답 생성 프롬프트 + ChatbotResponse 구조화 출력 (CB-004)"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from tracktory.chatbot.rag.ragflow import RetrievedChunk


class ChatbotResponse(BaseModel):
    """응답 본문 + 후속 선택지"""

    text: str = Field(..., min_length=1, description="응답 본문. 출처는 [근거: #N] 형식")
    choices: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="후속 질문 후보 1-3 개. 빈 리스트 금지",
    )


_SYSTEM = """당신은 한성대 진로 상담 챗봇 Tracktory 입니다. 매 턴 RAGFlow 검색 자료가 컨텍스트로 주어집니다.

# 답변 규칙

1. **자료만 사용**: [검색된 자료] 외 사실 생성 금지. 자료에 없는 정보는 "해당 데이터 없음" 명시.
2. **출처 표시 (필수)**: 본문은 **반드시** 마지막 줄에 [근거: #1, #3] 형식으로 끝나야 함. 활용 없으면 [근거: 없음]. 빠뜨리지 마세요.
3. **간결성**: 3-6 문장. 마크다운 헤더(`#`) 금지. 비교·나열은 짧은 불릿(`- `).
4. **톤**: 존댓말, 차분한 상담사. 이모지·과한 추임새 금지. 자료 기반 사실에 "아마"·"~일 것 같다" 헷지 금지.
5. **멀티턴**: "그것/거기" 지시어는 직전 턴 대상에 매핑. 모호하면 1 문장 되묻기.
6. **개인화**: [사용자 프로필] 활용하되 단정적 재인용 금지.
7. **선택지 (CB-004 필수)**: choices 1-3 개. 자료 기반이면 깊이 파는 방향, 일반이면 구체화 방향. 빈 리스트 금지.
8. **후속 질문은 choices 에만**: 본문(text) 에 "어떤 ~를 원하시나요?", "더 궁금한 게 있나요?" 같은 follow-up 질문 작성 금지. 후속 질문은 choices 필드로만 전달.
"""


_USER_PROFILE_BLOCK = "[사용자 프로필]\n{user_context_block}"


_CONTEXT_BLOCK = """[검색된 자료]
{retrieved_context}

위 자료의 사실만 사용해 답하세요. 없으면 "해당 데이터 없음" 명시. choices 1-3 개 필수."""


RAG_RESPONSE_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("system", _USER_PROFILE_BLOCK),
        MessagesPlaceholder("messages"),
        ("system", _CONTEXT_BLOCK),
    ]
)


def format_retrieved_docs(docs: Sequence[RetrievedChunk]) -> str:
    """청크 리스트를 프롬프트용 단일 문자열로 정형화"""
    if not docs:
        return "(검색 결과 없음)"
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        content = doc["content"].strip()
        if not content:
            continue
        blocks.append(f"#{idx} (source: {doc['document_name']})\n{content}")
    return "\n\n".join(blocks) if blocks else "(검색 결과 없음)"


def format_user_context(user_context: dict[str, Any] | None) -> str:
    """온보딩 dict 를 프롬프트용 라벨된 문자열로 정형화 (schema 는 TK-21 에서 확정)"""
    if not user_context:
        return "(온보딩 정보 없음)"

    # 학적: 입학년도 + 학년 합쳐서 한 줄
    enrollment_parts: list[str] = []
    if entry_year := user_context.get("entry_year"):
        enrollment_parts.append(f"{entry_year}년 입학")
    if grade := user_context.get("grade"):
        enrollment_parts.append(f"{grade}학년")

    # 이수 과목 — list[dict] 를 "과목명(학년-학기)" 형태로 평탄화
    completed_subjects_raw = user_context.get("completed_subjects") or []
    completed_labels = [
        f"{s['name']}({s['year']}-{s['semester']})"
        for s in completed_subjects_raw
        if isinstance(s, dict) and s.get("name")
    ]

    fields: list[tuple[str, Any]] = [
        ("학적", " · ".join(enrollment_parts) if enrollment_parts else None),
        ("단과대", user_context.get("college")),
        ("학부", user_context.get("department")),
        (
            "소속 트랙",
            user_context.get("user_track")
            or user_context.get("preferred_tracks")
            or user_context.get("tracks"),
        ),
        ("관심 분야", user_context.get("interests")),
        ("공부 분야", user_context.get("study_field")),
        ("자신 있는 언어", user_context.get("languages")),
        ("이수 과목", completed_labels),
    ]

    # 취업 선호 — nested dict (job_preference) 또는 평면 키 모두 대응
    job_pref = user_context.get("job_preference") or {}
    company_types = job_pref.get("company_types") if isinstance(job_pref, dict) else None
    values = job_pref.get("values") if isinstance(job_pref, dict) else None
    fields.extend(
        [
            ("희망 기업 유형", company_types or user_context.get("company_types")),
            ("중요 가치", values or user_context.get("job_values")),
        ]
    )

    lines: list[str] = []
    for label, value in fields:
        if not value:
            continue
        rendered = ", ".join(map(str, value)) if isinstance(value, list | tuple) else str(value)
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines) if lines else "(온보딩 정보 없음)"
