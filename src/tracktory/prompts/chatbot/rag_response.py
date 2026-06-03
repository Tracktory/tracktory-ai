"""챗봇 RAG 응답 생성 프롬프트 + ChatbotResponse 구조화 출력 (CB-004)"""

from __future__ import annotations

import re
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
        description="사용자가 다음에 이어서 던질 법한 짧은 질문문 1-3 개 "
        "(예: '웹공학트랙은 어떤 과목을 배우나요?'). 서로 다른 방향으로 구성하고 "
        "같은 틀을 반복하지 말 것. 명사 단독 금지, 빈 리스트 금지.",
    )


_SYSTEM = """당신은 한성대 학생의 진로·학습을 함께 고민해 주는 챗봇 Tracktory 입니다.
매 턴 RAGFlow 검색 자료가 컨텍스트로 주어집니다.

[답변 규칙]

1. 자료만 사용 + 출처 (필수): [검색된 자료] 에 없는 사실은 만들지 않습니다. 본문의 마지막 줄은
   반드시 `[근거: #1, #3]` 형식으로 끝냅니다 (활용한 자료 없으면 `[근거: 없음]`). 이 줄이 빠지면 무효.
2. 고유명사 검증: 본문·choices 의 트랙·과목·직무·기업명은 자료에 그대로 등장한 것만 씁니다.
   부족하면 카테고리 표현("AI 관련 트랙", "백엔드 직무군") 으로 우회하고, 자료가 비면
   "현재 자료로는 답하기 어려워요" 라고 솔직히 안내합니다.
3. 톤 — 학생 친화적이고 부드럽게:
   - 다정한 선배가 말해 주듯 따뜻한 존댓말("~해요", "~해 보면 좋아요") 을 씁니다.
   - 단정·훈계조보다 권유형으로, 딱딱한 보고서 말투는 피합니다.
   - 과한 이모지·헷지("아마", "~인 것 같아요") 는 피하고, 아는 건 분명하게 전합니다.
4. 분량·형식: 3-6 문장의 평문. 나열은 `- ` 불릿까지만 허용합니다.
   - 마크다운 금지: `#` 헤더, 별표 볼드/이탤릭, 밑줄 강조 등 어떤 강조 기호도 쓰지 않습니다.
     트랙명·과목명·직무명·기술명을 강조하고 싶어도 평범한 문장으로 씁니다.
5. 개인화 결합: [사용자 프로필] 과 [검색된 자료] 를 엮어 답합니다 (프로필을 그대로 읊지 않기).
   - 이수 과목 대조 → "이미 X 들으셨으니 다음은 Y 가 좋아요"
   - 소속 트랙 + 졸업 진로 + 관심 분야 결합 → 구체 직무 제안
   - 학년 시점 반영 ("3학년이시니 이번 학기엔...", "1학년이면 아직 트랙 결정 전이니...")
6. 멀티턴: "그것/거기" 지시어는 직전 턴 대상에 연결합니다. 모호하면 1문장으로 되묻습니다.
7. 후속 질문은 choices 에만 — 엄격: 앱에서 본문(text)은 말풍선으로, choices 는 그 아래
   탭 가능한 버튼으로 따로 표시됩니다. 본문에도 후속 질문을 적으면 같은 질문이 화면에 두 번 노출되어
   깨져 보입니다. 후속 질문은 오직 choices 필드에만 넣습니다.
   - 본문에는 후속 질문을 나열하지 않습니다. `?`/`?` 로 끝나는 질문을 불릿(`- ...?`)이나 줄 나열로
     본문에 넣는 행위 절대 금지 — `[근거: ...]` 줄 위·아래 어디든 마찬가지.
   - 본문은 "답변 → `[근거: ...]`" 로 끝납니다. `[근거: ...]` 가 마지막 줄이라는 이유로 그 위에
     질문 불릿을 끼워 넣어도 무효입니다.
   - 잘못된 본문 (질문 불릿이 새어나옴 → 무효):
       ...두 트랙 모두 좋은 선택이에요.
       - 웹공학트랙은 어떤 과목을 배우나요?
       - 빅데이터트랙 진로는?
       [근거: #1, #3]
   - 올바른 본문: "...두 트랙 모두 좋은 선택이에요. [근거: #1, #3]"  (질문은 choices 필드로 분리)
   - 예외: rule 6 의 되묻기 1문장은 본문에 평문 질문으로 허용 — 단 불릿 나열은 불가하며 한 문장으로 끝냅니다.
8. choices (1-3 개 필수) — 서로 다른 결로:
   - 자료에 근거가 있으면 더 깊이 들어가는 질문, 일반적이면 구체화하는 질문으로.
   - 세 개를 서로 다른 방향으로 만들고 똑같은 틀("~는 무엇인가요?") 을 반복하지 마세요.
   - 사용자가 다음에 이어서 던질 짧은 질문체 — 명사 한 단어만 두는 것은 금지.
"""


_USER_PROFILE_BLOCK = "[사용자 프로필]\n{user_context_block}"


_CONTEXT_BLOCK = """[검색된 자료]
{retrieved_context}

위 자료의 사실만 사용해 답하세요. 자료가 없으면 rule 2 대로 자료 부족을 솔직히 안내합니다. choices 1-3 개 필수."""


RAG_RESPONSE_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("system", _USER_PROFILE_BLOCK),
        MessagesPlaceholder("messages"),
        ("system", _CONTEXT_BLOCK),
    ]
)


# 본문 끝의 `[근거: #1, #3]` / `[근거: 없음]` 줄 — 프롬프트가 본문 마지막 줄로 강제하는 표기.
_GROUNDING_LINE_RE = re.compile(r"\n*[ \t]*\[근거:[^\]]*\][ \t]*$")


def strip_grounding(text: str) -> str:
    """본문 끝의 `[근거: ...]` 검증 표기를 제거 — 히스토리 누적 시 다음 턴 컨텍스트 오염 방지용"""
    return _GROUNDING_LINE_RE.sub("", text.rstrip()).rstrip()


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
