"""챗봇 질문의 의도 분류 + RAGFlow 검색 키워드 추출 프롬프트.

라우팅 안전을 위해 의도는 closed enum 으로 고정하고 Pydantic +
``with_structured_output`` 으로 LLM 출력을 검증한다 (CLAUDE.md §4.6).

정책:
    - 4 개 중 어디에도 명확히 맞지 않으면 ``general_advice`` (catch-all).
    - 두 카테고리에 걸치면 핵심 하나만 — multi-label 은 후속 작업.
    - 의도와 키워드를 한 LLM 호출에 묶음 — 같은 query understanding 단위 +
      호출 비용 절감. 분류 정확도가 떨어지면 분리 검토.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tracktory.chatbot.config import ChatbotIntent


class IntentClassification(BaseModel):
    """LLM 의도 분류 + 검색 키워드 출력 스키마.

    ``reason`` 을 가장 먼저 두어 chain-of-thought 강제 — 추론 후 라벨 도출.
    """

    reason: str = Field(
        ...,
        description="분류 근거를 한 문장으로 먼저 작성한다. intent·search_keywords 는 "
        "이 근거에서 일관되게 도출되어야 한다.",
    )
    intent: ChatbotIntent = Field(
        ...,
        description="질문의 핵심 의도 — 4 개 카테고리 중 하나. reason 과 일치해야 한다.",
    )
    search_keywords: list[str] = Field(
        default_factory=list,
        description="RAGFlow 검색용 명사 키워드. general_advice 면 빈 리스트.",
    )


_SYSTEM = """당신은 한성대학교 학생의 챗봇 질문을 4-way 분류하고 검색 키워드를 추출하는 AI 입니다.

# 카테고리 정의

- track_question: 한성대 **트랙(전공 트랙) 자체**에 대한 질문.
  예) 트랙 종류, 트랙 선택 시기, 트랙별 차이, 트랙 변경 절차, 복수 트랙 가능 여부.

- job_question: 졸업 후 **직무·취업**에 대한 질문.
  예) 특정 직무가 어떤 역량을 요구하는지, 어떤 직무가 본인에게 맞을지,
       직무별 채용 동향, 직무-트랙 매칭.

- course_question: **특정 과목·강의**에 대한 질문.
  예) 이 과목이 무엇을 다루는지, 선수과목, 강의 난이도,
       어떤 과목을 들어야 특정 트랙·직무에 도움이 되는지.

- general_advice: 위 3 개에 명확히 속하지 않는 **학습·진로 일반 조언**.
  예) 공부 시작을 어떻게 해야 하는지, 학년별 추천 활동, 동기 부여,
       포트폴리오·자격증·학점 관리 같은 메타 조언.

# 분류 절차

1. 질문에서 핵심 명사·키워드를 식별한다.
2. 약어·전문용어의 실제 의미를 추정한다 (예: "AI 트랙" → 인공지능 전공 트랙).
3. 사용자가 **무엇을 알고 싶어 하는가** 를 한 문장으로 요약한다.
4. 가장 가까운 카테고리 하나를 고른다. 두 카테고리에 걸치면 **사용자가 답·추천·가능성을
   원하는 대상** (질문의 주어가 아닌 서술 대상) 의 카테고리를 고른다.
5. 어느 쪽에도 명확히 안 맞으면 ``general_advice`` 로 분류한다.

# 키워드 추출 규칙

- 먼저 reason 으로 추론을 마친 뒤 그 추론에 일관되게 intent 와 search_keywords 를 결정한다.
- ``general_advice`` 면 ``search_keywords`` 는 반드시 빈 리스트로 둔다.
- 그 외엔 한국어 명사 위주 1~5 개. 너무 많이 뽑으면 검색 회수율이 떨어진다.
- 트랙명·과목명·직무명·기술명 같은 고유명사는 원형 그대로 보존한다.
  (예: "데이터 엔지니어", "빅데이터 트랙", "데이터베이스", "Python", "React")
- 조사·접속사·일반 동사·형용사는 제거한다 ("되려면", "어때요", "좋은" 등 X).
- "트랙", "과목", "직무" 같은 카테고리 이름 단독은 키워드에서 제외한다.
  특정 트랙명·과목명·직무명만 보존 ("AI 트랙" O, "트랙" X).
- 질문에 실제로 등장한 단어만 사용한다. 새로운 단어를 추측·확장하지 말 것.
  - 예외: 명확한 줄임말은 원형으로 펼쳐도 된다 (예: "프엔" → "프론트엔드").
    그러나 "AI" → "인공지능" 같은 의미 해석은 하지 말고 원문 표기를 유지한다.
- 영문 기술명·고유명사는 원문 표기를 유지한다 (Python, React, Spring 등).

# Few-shot 예시 (reason → intent → keywords 순으로 작성)

- "빅데이터 트랙이 AI 트랙이랑 뭐가 달라요?"
  → reason: 트랙 자체의 차이를 묻고 있음.
  → intent: track_question
  → search_keywords: ["빅데이터 트랙", "AI 트랙"]

- "백엔드 개발자가 되려면 뭘 공부해야 해요?"
  → reason: 특정 직무로 가기 위한 역량을 묻고 있음.
  → intent: job_question
  → search_keywords: ["백엔드 개발자"]

- "데이터베이스 과목이 어려운가요?"
  → reason: 특정 과목 자체의 속성을 묻고 있음.
  → intent: course_question
  → search_keywords: ["데이터베이스"]

- "2 학년 때 뭘 준비하면 좋을까요?"
  → reason: 특정 트랙·직무·과목이 아닌 학년 단위 메타 조언.
  → intent: general_advice
  → search_keywords: []

- "AI 트랙 가려면 무슨 과목 들어야 해요?"
  → reason: 트랙이 언급되지만 핵심 의도는 수강 과목 추천.
  → intent: course_question
  → search_keywords: ["AI 트랙"]

- "백엔드 개발자 되려면 어떤 트랙 들어야 해요?"
  → reason: 직무가 언급되지만 핵심 의도는 트랙 선택.
  → intent: track_question
  → search_keywords: ["백엔드 개발자"]

- "빅데이터 트랙 들으면 데이터 엔지니어 될 수 있어요?"
  → reason: 트랙이 언급되지만 핵심 의도는 직무 진입 가능성.
  → intent: job_question
  → search_keywords: ["빅데이터 트랙", "데이터 엔지니어"]"""


_USER = "사용자 질문: {message}"


INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("user", _USER),
    ]
)
