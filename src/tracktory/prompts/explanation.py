"""LLM 설명 생성 노드의 프롬프트 템플릿.

본 모듈은 추천 결과의 자연어 설명을 생성하는 단일 ``ChatPromptTemplate`` 을
정의한다. 출력은 세 종류로 분리된다.

1. 영역별 단락 (``sections``) — 직무 / 트랙 / 로드맵 영역별 근거.
2. 학기 단위 부제 (``semester_subtitles``) — 학습 로드맵 화면의 학기 카드
   헤더에 노출되는 한 줄 요약. 정보 위계의 상위 레벨.
3. 과목 단위 인과 흐름 (``course_flows``) — 과목 상세 모달에 노출되는
   "관심사 → 직무 → 트랙 조합 → 이 과목" 인과 사슬. 정보 위계의 하위 레벨.

프롬프트는 다음 7 가지 변수 슬롯을 받는다.

- ``jobs_context``: 직무 후보 직렬화 문자열 (또는 빈 영역 안내).
- ``tracks_context``: 트랙 조합 직렬화 문자열 (또는 빈 영역 안내).
- ``roadmap_context``: 학습 로드맵 단계별 뷰 직렬화 (또는 빈 영역 안내).
- ``semesters_context``: 학기별 대표 단계명·과목 직렬화 (학기 부제 입력).
- ``courses_context``: 과목별 식별자·이름·단계명 직렬화 (인과 흐름 입력).
- ``anchor_context``: 인과 흐름의 ``{직무명}`` / ``{트랙 조합}`` anchor.
- ``caveat_required``: ``"yes"`` 또는 ``"no"`` 문자열. 직무 매칭이 카테고리
  사전 매핑 fallback 으로 채택된 경우 ``yes`` 가 흐른다.

프롬프트는 다음 invariant 를 강제한다.

- 출력은 **친근한 존댓말** 톤, 영역별 단락은 **1~2 문장**.
- 컨텍스트에 명시되지 않은 사실은 생성 금지 — 빈 영역은 해당 출력 항목
  자체를 비워야 하며 ``text`` 전체 요약에서도 인용하지 않는다.
- 학기 부제·과목 인과 흐름은 학습 로드맵 컨텍스트가 비어 있으면
  (``데이터 없음``) 빈 리스트로 둔다.
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
- 학기 부제(``semester_subtitles``): [학기별 단계] 컨텍스트에 등장하는 각
  학기마다 항목을 하나씩 만듭니다. ``semester`` 는 해당 학기 번호,
  ``subtitle`` 은 "이번 학기는 트랙 OO 단계입니다" 형식이며 OO 자리에는 그
  학기 컨텍스트의 ``단계명`` (기초·핵심·응용·산학) 을 그대로 넣습니다.
  [학기별 단계] 가 ``데이터 없음`` 이면 빈 리스트로 둡니다.
- 과목 인과 흐름(``course_flows``): [과목별 단계] 컨텍스트에 등장하는 각
  과목마다 항목을 하나씩 만듭니다. ``course_id`` 는 그 과목의 ``course_id``
  를 그대로 넣고, ``flow`` 는 "당신의 관심사 → OO 직무 → OO 트랙 → 이
  과목이 OO입니다" 형식입니다. 직무명·트랙 조합 자리에는 [인과 흐름 anchor]
  컨텍스트의 ``직무명`` / ``트랙 조합`` 값을, 마지막 OO 자리에는 그 과목의
  ``단계명`` 을 그대로 넣습니다. [과목별 단계] 가 ``데이터 없음`` 이면 빈
  리스트로 둡니다.
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

[학기별 단계]
{semesters_context}

[과목별 단계]
{courses_context}

[인과 흐름 anchor]
{anchor_context}

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

``format_messages`` 호출 시 7 변수 슬롯을 모두 채워야 한다. 직렬화는
호출 측 노드의 책임으로 분리되어 있다.
"""
