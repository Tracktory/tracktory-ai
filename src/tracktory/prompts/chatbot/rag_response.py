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
        description="사용자가 다음에 던질 수 있는 짧은 질문문 1-3 개 "
        "(예: '웹공학트랙은 어떤 과목을 배우나요?'). 명사 단독 금지, 빈 리스트 금지.",
    )


_SYSTEM = """당신은 한성대 진로 상담 챗봇 Tracktory 입니다. 매 턴 RAGFlow 검색 자료가 컨텍스트로 주어집니다.

# 답변 규칙

1. **자료만 사용 + 출처 (필수)**: [검색된 자료] 외 사실 생성 금지. **본문의 마지막 줄은 반드시 `[근거: #1, #3]` 형식**으로 끝낸다 (활용 없으면 `[근거: 없음]`). 이 줄이 빠지면 응답 무효.
2. **고유명사 검증**: 본문·choices 의 트랙·과목·직무·기업명은 **자료에 그대로 등장한 것만** 사용. 부족 시 카테고리 표현으로 우회 ("AI 관련 트랙", "백엔드 직무군"). 자료 비면 "현재 자료로는 답할 수 없습니다" 명시.
3. **간결한 톤 + 평문**: 3-6 문장, 존댓말, 차분한 상담사. 나열은 `- ` 불릿 OK. 마크다운 헤더(`#`) 금지. 이모지·헷지("아마", "~같다") 금지.
   - **마크다운 강조 절대 금지**: `**...**`(볼드), `*...*`(이탤릭), `__...__`, `_..._` 어떤 형태도 사용 금지. 트랙명·과목명·직무명·기술명을 강조하고 싶어도 평문으로 작성하세요.
     올바른 예: "웹공학트랙은 클라이언트-서버 기술을..."  /  잘못된 예: "**웹공학트랙**은 클라이언트-서버 기술을..."
4. **개인화 결합**: [사용자 프로필] 과 [검색된 자료] 를 묶어 답변. 단정적 재인용은 금지.
   - 이수 과목 대조 → "이미 X 들으셨으니 다음은 Y"
   - 소속 트랙 + 졸업 진로 + 관심 분야 결합 → 구체 직무 추천
   - 학년 시점 반영 ("3학년이시니 이번 학기에...", "1학년이라 아직 트랙 결정 전이니...")
5. **멀티턴**: "그것/거기" 지시어는 직전 턴 대상에 매핑. 모호하면 1 문장 되묻기.
6. **선택지 (choices) 1-3 개 필수**: 자료 기반이면 깊이, 일반이면 구체화. 빈 리스트 금지.
   각 항목은 사용자가 다음에 던질 수 있는 짧은 의문문 형태 — 명사 단독 금지. (예시는 choices Field description 참조)
7. **본문(text) 과 choices 분리 — 엄격**:
   - 본문에는 어떤 형태의 질문도 들어가지 않는다 (의문문 한 줄도, 질문 불릿 나열도 모두 금지).
   - **choices 항목을 본문에 복제하지 말 것**. 본문 끝에 후속 질문을 불릿으로 나열하는 패턴 절대 금지.
   - 본문 마지막 줄은 반드시 `[근거: ...]` 여야 한다 (rule 1). 이 줄 뒤에 아무것도 오지 않는다.
   - 잘못된 본문 마무리 예:
     ```
     ...두 트랙 모두 좋은 선택입니다.
     - 웹공학트랙은 어떤 과목을 배우나요?
     - 빅데이터트랙 진로는?
     ```
     → 위처럼 본문 끝에 질문 불릿이 붙으면 choices 와 중복되어 무효. 이런 질문들은 **오직 choices 필드에만** 넣는다.
   - 올바른 본문 마무리 예: "...두 트랙 모두 좋은 선택입니다. [근거: #1, #3]"
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
    """온보딩 dict 를 프롬프트용 라벨된 문자열로 정형화"""
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
        ("소속 트랙", user_context.get("tracks")),
        ("관심 분야", user_context.get("interests")),
        ("공부 분야", user_context.get("study_fields")),
        ("자신 있는 기술", user_context.get("tech_stacks")),
        ("희망 기업 유형", user_context.get("company_types")),
        ("중요 가치", user_context.get("work_values")),
        ("이수 과목", completed_labels),
    ]

    lines: list[str] = []
    for label, value in fields:
        if not value:
            continue
        rendered = ", ".join(map(str, value)) if isinstance(value, list | tuple) else str(value)
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines) if lines else "(온보딩 정보 없음)"
