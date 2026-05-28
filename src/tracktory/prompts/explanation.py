"""LLM 설명 생성 노드의 프롬프트 템플릿.

본 모듈은 영역별 (jobs / tracks / roadmap) 자연어 설명을 생성하는 단일
``ChatPromptTemplate`` 을 정의한다. 프롬프트는 다음 4 가지 변수 슬롯을 받는다.

- ``jobs_context``: 직무 후보 직렬화 문자열 (또는 빈 영역 안내).
- ``tracks_context``: 트랙 조합 직렬화 문자열 (또는 빈 영역 안내).
- ``roadmap_context``: 학습 로드맵 직렬화 문자열 (또는 빈 영역 안내).
- ``caveat_required``: ``"yes"`` 또는 ``"no"`` 문자열. 직무 매칭이 카테고리
  사전 매핑 fallback 으로 채택된 경우 ``yes`` 가 흐른다.

프롬프트는 다음 invariant 를 강제한다.

- 출력은 **친근한 존댓말** 톤, 영역별 단락은 **1~2 문장**.
- 컨텍스트에 명시되지 않은 사실은 생성 금지 — 빈 영역은 해당 ``section``
  자체를 비워야 하며 ``text`` 전체 요약에서도 인용하지 않는다.
- ``caveat_required="yes"`` 인 경우 ``text`` 전체 요약 끝에 추가 입력
  안내 한 줄을 부착한다.

이 프롬프트는 ``with_structured_output(Explanation)`` 와 결합되어 출력
스키마가 Pydantic 으로 강제된다. 본 템플릿 자체는 schema 검증을 다시
하지 않는다.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

_SYSTEM_MESSAGE = """\
당신은 한성대학교 학생에게 진로 추천 결과의 근거를 설명해 주는 상담사입니다.

다음 규칙을 반드시 따르세요.

- 어투: 학생이 친근하게 느낄 수 있는 존댓말. 격식적이거나 딱딱한 어투 금지.
- 분량: 영역별 단락(``sections[*].body``) 은 1~2 문장으로 짧게.
- 사실 근거: 아래 입력으로 주어지는 jobs / tracks / roadmap 컨텍스트 안에
  명시된 정보만 인용합니다. 컨텍스트에 없는 직무명·트랙명·과목명·역량명을
  새로 만들거나 추측하지 마세요. 컨텍스트에 데이터가 없으면 해당 영역의
  단락(``sections``) 자체를 만들지 않습니다.
- 영역 카테고리: ``sections[*].topic`` 은 jobs / tracks / roadmap 셋 중
  하나로만 고정합니다. 각 영역은 최대 한 번만 등장합니다.
- 로드맵 근거: 로드맵 단락을 만들 때, 추천 과목이 어떤 직무 역량 또는
  기술 스택과 연결되는지 jobs 컨텍스트의 ``competency_tags`` /
  ``tech_stacks`` 를 근거로 한 줄 안에 명시하세요.
- ``text`` 전체 요약: 학생이 이번 추천을 한 문장으로 이해할 수 있도록
  간단히 정리합니다. 영역별 단락의 단순 반복은 피하세요.
- 캐비잇: ``caveat_required`` 가 ``yes`` 이면, ``text`` 전체 요약의 마지막에
  "마이페이지에서 관심사·흥미를 추가하시면 더 정확해져요." 라는 톤으로
  추가 입력 안내 한 문장을 부착합니다. ``no`` 이면 부착하지 않습니다.
"""

_HUMAN_MESSAGE = """\
다음 추천 결과에 대해 자연어 설명을 생성해 주세요.

[jobs 컨텍스트]
{jobs_context}

[tracks 컨텍스트]
{tracks_context}

[roadmap 컨텍스트]
{roadmap_context}

[캐비잇 부착 여부]
{caveat_required}
"""


EXPLANATION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_MESSAGE),
        ("user", _HUMAN_MESSAGE),
    ]
)
"""LLM 설명 생성 노드가 사용하는 단일 프롬프트.

``format_messages`` 호출 시 4 변수 슬롯을 모두 채워야 한다. 직렬화는
호출 측 노드의 책임으로 분리되어 있다.
"""
