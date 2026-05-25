"""챗봇 질문의 의도 분류 + RAGFlow 검색 키워드 추출 프롬프트.

라우팅 안전을 위해 의도는 closed enum 으로 고정하고 Pydantic +
``with_structured_output`` 으로 LLM 출력을 검증한다 (CLAUDE.md §4.6).

정책:
    - 4 개 중 어디에도 명확히 맞지 않으면 ``general_advice`` (catch-all).
    - 두 카테고리에 걸치면 핵심 하나만 — multi-label 은 후속 작업.
    - 의도와 키워드를 한 LLM 호출에 묶음 — 같은 query understanding 단위 +
      호출 비용 절감. 분류 정확도가 떨어지면 분리 검토.
    - 직전 대화 history 를 같이 전달해 지시어("그것", "그 직무" 등) 와 elliptical
      질문 ("그럼?") 의 referent 를 직전 turn 에서 끌어옴.
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
        description="분류 근거를 한 문장으로 먼저 작성한다. intent·search_keywords·"
        "is_catalog_query·target_grade 는 이 근거에서 일관되게 도출되어야 한다.",
    )
    intent: ChatbotIntent = Field(
        ...,
        description="질문의 핵심 의도 — 4 개 카테고리 중 하나. reason 과 일치해야 한다.",
    )
    search_keywords: list[str] = Field(
        default_factory=list,
        description="RAGFlow 검색용 명사 키워드. general_advice 또는 카탈로그 모드면 빈 리스트.",
    )
    is_catalog_query: bool = Field(
        default=False,
        description="'트랙 종류 알려줘' / '어떤 트랙 있어?' / '직무 카탈로그' 같이 "
        "특정 대상이 아닌 전체 목록 요청이면 True. retrieve_rag 가 메타 필터로 "
        "doc_type 전체를 가져오게 된다. 특정 대상 질문이면 False.",
    )
    target_grade: int | None = Field(
        default=None,
        description="'1학년 때 뭐 들어야' / '3학년 추천 과목' 같이 학년이 명시되면 1~4, "
        "명시 안 됐으면 None. 사용자 프로필의 grade 와는 별개 — 질문 자체에 등장한 학년만.",
    )


_SYSTEM = """당신은 한성대학교 학생의 챗봇 질문을 4-way 분류하고 검색 키워드를 추출하는 AI 입니다.

# 카테고리 정의

- track_question: 한성대 **트랙(전공 트랙) 자체**에 대한 질문.
  예) 트랙 종류, 트랙 선택 시기, 트랙별 차이, 트랙 변경 절차, 복수 트랙 가능 여부,
       어떤 트랙 골라야 할지.

- job_question: 졸업 후 **직무·취업**에 대한 질문.
  예) 특정 직무가 어떤 역량을 요구하는지, 어떤 직무가 본인에게 맞을지,
       직무별 채용 동향, 직무-트랙 매칭.

- course_question: **특정 과목·강의** 또는 **수강 계획**에 대한 질문.
  예) 이 과목이 무엇을 다루는지, 선수과목, 강의 난이도, 다음 학기 추천 과목,
       어떤 과목을 들어야 특정 트랙·직무에 도움이 되는지, 앞으로 어떤 과목 들어야 하는지.

- general_advice: 위 3 개에 명확히 속하지 않는 **학습·진로 일반 조언**.
  예) 공부 시작을 어떻게 해야 하는지, 학년별 활동, 동기 부여,
       포트폴리오·자격증·학점 관리·동아리·인턴십 같은 메타 조언.

# 분류 절차

1. 직전 대화 history 가 있으면, 현재 질문의 지시어("그것", "그 직무", "거기", "그럼")·
   생략된 주어를 직전 turn 의 대상에 매핑한다.
2. 질문에서 핵심 명사·키워드를 식별한다 (지시어 매핑 결과 포함).
3. 약어·전문용어의 실제 의미를 추정한다.
4. 사용자가 **무엇을 알고 싶어 하는가** 를 한 문장으로 요약 — reason 에 작성.
5. 가장 가까운 카테고리 하나를 고른다. 두 카테고리에 걸치면 **사용자가 답·추천·가능성을
   원하는 대상** (질문의 주어가 아닌 서술 대상) 의 카테고리를 고른다.
6. is_catalog_query / target_grade 시그널을 판단해 채운다.
7. 어느 쪽에도 명확히 안 맞으면 ``general_advice`` 로 분류.

# is_catalog_query 판단 규칙

True 로 둘 때 — **특정 대상 없는 전체 목록 요청**:
- "트랙 종류가 뭐가 있어?"
- "어떤 트랙 고르면 좋을까?" (트랙 카탈로그 보고 골라야 함)
- "직무 종류 알려줘"
- "들을 수 있는 과목 다 알려줘"

False 로 둘 때 — **특정 대상 또는 좁은 범위**:
- "빅데이터 트랙이 뭐야?" (특정 트랙)
- "백엔드 개발자 채용 동향" (특정 직무)
- "1학년 때 뭐 들어야?" (학년은 좁은 범위, 카탈로그 아님)

# target_grade 판단 규칙

질문에 학년이 직접 등장하면 그 숫자 (1~4), 아니면 None.
- "1학년 때 뭐 준비해야?" → 1
- "2학년 2학기에 어떤 과목?" → 2
- "3학년 추천 트랙 있어?" → 3
- "이번 학기 뭐 들어야?" → None (사용자 학년은 user_context 에서 retrieve_rag 가 따로 활용)
- "다음 학기 뭐 들어야?" → None (역시 user_context.grade 로 retrieve_rag 가 처리)

# 키워드 추출 규칙

- 먼저 reason 으로 추론을 마친 뒤 그 추론에 일관되게 intent · search_keywords 결정.
- ``general_advice`` 면 ``search_keywords`` 는 반드시 빈 리스트로 둔다.
- ``is_catalog_query`` 가 True 면 ``search_keywords`` 는 빈 리스트 — 메타 필터로 검색.
- 그 외엔 한국어 명사 위주 1~5 개. 너무 많이 뽑으면 검색 회수율이 떨어진다.
- 트랙명·과목명·직무명·기술명 같은 고유명사는 원형 그대로 보존.
  (예: "데이터 엔지니어", "빅데이터 트랙", "데이터베이스", "Python", "React")
- 조사·접속사·일반 동사·형용사는 제거 ("되려면", "어때요", "좋은" 등 X).
- "트랙", "과목", "직무" 같은 카테고리 이름 단독은 키워드에서 제외.
  특정 트랙명·과목명·직무명만 보존 ("AI 트랙" O, "트랙" X).
- 질문에 실제로 등장한 단어만 사용 (지시어 매핑 결과 포함). 새 단어 추측·확장 금지.
  - 예외: 명확한 줄임말은 원형으로 펼쳐도 된다 (예: "프엔" → "프론트엔드").
    그러나 "AI" → "인공지능" 같은 의미 해석은 하지 말고 원문 표기를 유지.
- 영문 기술명·고유명사는 원문 표기를 유지 (Python, React, Spring 등).

# Few-shot 예시

## 단순 케이스

- "빅데이터 트랙이 AI 트랙이랑 뭐가 달라요?"
  → reason: 두 트랙 자체의 차이를 묻고 있음.
  → intent: track_question, is_catalog_query: false, target_grade: null
  → search_keywords: ["빅데이터 트랙", "AI 트랙"]

- "백엔드 개발자가 되려면 뭘 공부해야 해요?"
  → reason: 특정 직무로 가기 위한 역량을 묻고 있음.
  → intent: job_question, is_catalog_query: false, target_grade: null
  → search_keywords: ["백엔드 개발자"]

- "데이터베이스 과목이 어려운가요?"
  → reason: 특정 과목 자체의 속성을 묻고 있음.
  → intent: course_question, is_catalog_query: false, target_grade: null
  → search_keywords: ["데이터베이스"]

## 카탈로그 케이스

- "어떤 트랙 고르면 좋을까?"
  → reason: 특정 트랙이 아닌 트랙 전체 카탈로그를 보고 선택 조언을 요청.
  → intent: track_question, is_catalog_query: true, target_grade: null
  → search_keywords: []

- "트랙 종류가 뭐가 있어?"
  → reason: 트랙 카탈로그 요청.
  → intent: track_question, is_catalog_query: true, target_grade: null
  → search_keywords: []

## 학년 명시 케이스

- "1학년 때 어떤 과목 들으면 좋아?"
  → reason: 1학년이 들을 수 있는 과목 추천 요청.
  → intent: course_question, is_catalog_query: false, target_grade: 1
  → search_keywords: []

- "3학년이 들어야 할 빅데이터 트랙 과목"
  → reason: 빅데이터 트랙의 3학년 과목 요청.
  → intent: course_question, is_catalog_query: false, target_grade: 3
  → search_keywords: ["빅데이터 트랙"]

## 사용자 프로필 의존 케이스 (학년·트랙 명시 안 됨 → retrieve_rag 가 user_context 활용)

- "다음 학기에 어떤 과목 들어야 해?"
  → reason: 사용자 트랙·학년 기반 다음 학기 과목 추천. 학년이 직접 명시 안 됨.
  → intent: course_question, is_catalog_query: false, target_grade: null
  → search_keywords: []

- "내 트랙으로 어떤 직무 갈 수 있어?"
  → reason: 사용자 소속 트랙 기반 직무 가능성. 트랙명이 직접 명시 안 됨.
  → intent: job_question, is_catalog_query: false, target_grade: null
  → search_keywords: []

## 멀티턴 — 지시어 처리

- (직전 user: "빅데이터 트랙 졸업하면 어떤 직무 가?", assistant: "데이터 엔지니어, 데이터 분석가...")
  current user: "그 직무 되려면 어떤 과목 들어야 해?"
  → reason: 직전 턴에서 언급된 "데이터 엔지니어" 직무에 필요한 수강 과목을 묻고 있음.
  → intent: course_question, is_catalog_query: false, target_grade: null
  → search_keywords: ["데이터 엔지니어"]

- (직전 user: "AI 트랙이 뭐야?", assistant: "AI 트랙은...")
  current user: "그럼 어떤 직무로 가면 좋아?"
  → reason: 직전 턴의 AI 트랙 기준으로 갈 만한 직무 추천.
  → intent: job_question, is_catalog_query: false, target_grade: null
  → search_keywords: ["AI 트랙"]

## 메타 조언 (자료 무관)

- "2 학년 때 뭘 준비하면 좋을까요?"
  → reason: 특정 트랙·직무·과목 자료가 필요하지 않은 학년 단위 메타 조언.
  → intent: general_advice, is_catalog_query: false, target_grade: 2
  → search_keywords: []

- "IT 대기업 가려면 1학년 때 뭘 해야 해?"
  → reason: 특정 직무가 아닌 일반적 취업 준비 활동 조언.
  → intent: general_advice, is_catalog_query: false, target_grade: 1
  → search_keywords: []
"""


_HISTORY_BLOCK = """[직전 대화 (오래된 → 최근 순)]
{history}

위 history 는 현재 질문의 지시어·생략 주어를 해석할 때만 사용한다."""


_USER = "사용자 질문: {message}"


INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("system", _HISTORY_BLOCK),
        ("user", _USER),
    ]
)
