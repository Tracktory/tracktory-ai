"""교육과정 본문의 과목 줄 파싱 — 트랙/과목 저장소가 공유하는 단일 기준.

교육과정 본문 한 줄에는 과목구분·이름·코드·학점이 모두 있다
("  - [전공선택] 공학프로그래밍 (V070044, 3학점)"). 트랙 저장소는 이 줄에서
과목코드만, 과목 저장소는 전체 메타를 뽑지만, **"어떤 과목구분이 추천 대상
(전공)인가"라는 판정 규칙은 동일해야 한다.** 두 저장소가 각자 정규식·필터를
들고 있으면 규칙이 이중화되어 어긋난다 — 실제로 트랙 저장소가 태그를 보지 않아
교양·기타 과목코드가 ``Track.course_ids`` 에만 새어 들어가는 불일치가 있었다.

본 모듈이 그 규칙(과목 줄 정규식 + 과목구분 → course_type 매핑)을 단일 출처로
보유하고, 두 저장소는 ``parse_course_line`` 만 호출한다. 추천 대상은 전공(기초/
필수/선택)뿐이므로 그 외 과목구분(교양류 등)은 ``None`` 으로 제외된다.

코드 prefix(예: "GEN")가 아니라 **과목구분 태그**로 거른다 — prefix 는 데이터에
따라 흔들리지만(교양이 아닌 비전공 코드도 섞임), 태그는 학사 분류의 권위 있는
신호다.

``stage``(학습 깊이)는 과목구분이 아니라 **이수 학년**에서 도출되므로(MVP 결정:
1↔foundation … 4↔industry) 한 줄만 보는 본 파서가 아니라, 학년/학기 섹션을
추적하는 ``RagflowCourseRepository`` 가 부여한다. 본 모듈은 stage 의 어휘
타입(``StageLabel``)만 공유 어휘로 보유한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = ["CourseTypeLabel", "ParsedCourseLine", "StageLabel", "parse_course_line"]

StageLabel = Literal["foundation", "core", "application", "industry"]
CourseTypeLabel = Literal["전공필수", "전공선택", "교양"]

# 과목 줄: "  - [전공선택] 공학프로그래밍 (V070044, 3학점)"
#   group(1) 과목구분, group(2) 과목명, group(3) 코드, group(4) 학점.
# 괄호 안 화이트스페이스는 데이터마다 들쭉날쭉하므로 관대하게 허용한다.
_COURSE_LINE_RE = re.compile(
    r"-\s*\[([^\]]+)\]\s*(.+?)\s*\(\s*([A-Za-z0-9]+)\s*,\s*(\d+)\s*학점\s*\)"
)

# 과목구분 태그 → course_type. 본 매핑에 없는 태그(교양류 등)는 추천 대상이
# 아니므로 parse_course_line 이 None 을 돌려준다. roadmap 노드는 {전공필수,
# 전공선택}만 통과시키므로 전공기초는 전공선택으로 둔다(전공 계열 유지 — 학습
# 깊이 구분은 stage 가 학년 기반으로 따로 표현한다).
_COURSE_TYPE_BY_TAG: dict[str, CourseTypeLabel] = {
    "전공기초": "전공선택",
    "전공필수": "전공필수",
    "전공선택": "전공선택",
    "전공선택(상호인정)": "전공선택",
}


@dataclass(frozen=True)
class ParsedCourseLine:
    """과목 줄 1건의 파싱 결과(전공 과목으로 확정된 것만).

    ``stage`` 는 본 결과에 담기지 않는다 — 학습 깊이는 한 줄이 아니라 이수 학년
    (학년/학기 섹션 헤더)에서 결정되므로 ``RagflowCourseRepository`` 가 부여한다.
    """

    tag: str
    course_name: str
    course_id: str
    credits: int
    course_type: CourseTypeLabel


def parse_course_line(line: str) -> ParsedCourseLine | None:
    """교육과정 본문 한 줄을 파싱한다. 추천 대상(전공)이 아니면 ``None``.

    과목 줄 형식이 아니거나(매칭 실패), 과목구분이 전공 계열(기초/필수/선택)이
    아니면(교양류·미지 태그) ``None`` 을 돌려준다. 호출 측은 ``None`` 을 "이 줄은
    건너뛴다"로 처리하면 두 저장소가 동일한 전공 판정 기준을 공유하게 된다.
    """
    match = _COURSE_LINE_RE.search(line)
    if match is None:
        return None
    tag = match.group(1).strip()
    course_type = _COURSE_TYPE_BY_TAG.get(tag)
    if course_type is None:
        return None  # 교양류 등 비전공 → 추천 대상 아님, 제외.
    return ParsedCourseLine(
        tag=tag,
        course_name=match.group(2).strip(),
        course_id=match.group(3),
        credits=int(match.group(4)),
        course_type=course_type,
    )
