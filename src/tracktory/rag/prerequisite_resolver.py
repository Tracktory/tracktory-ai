"""선수과목 이름 → course_id 해석 — ``prerequisites.json`` 을 과목 카탈로그에 잇는다.

``prerequisites.json`` 은 ``{과목명: [선수과목명]}`` 구조인데 ``Course.prereq_ids``
는 ``course_id`` 리스트다. 본 모듈이 그 간극을 메워, 과목 카탈로그 생성 시점에
``prereq_ids`` 를 채울 수 있게 한다.

카탈로그(전공)에 없는 선수(교양·타과)와 자기참조는 제외한다 — 로드맵의 선수
만족 검증은 "이수 + 채택" 전공 집합 안에서만 이뤄지므로, 카탈로그 밖 코드를
넣으면 그 과목이 영원히 충족 불가가 되어 배치에서 빠진다. 전공 로드맵에서
교양·타과 선수는 의미가 없으므로 드롭이 옳다.

이름 매칭은 정규화 후 exact 비교다. ``prerequisites.json`` 의 과목명과 카탈로그
``course_name`` 은 둘 다 학사 ``courses.csv`` 의 교과목명에서 파생되어 형식만
다를 뿐 같은 좌표를 공유한다(괄호·공백 차이 정도). 따라서 정규화로 충분하며,
퍼지 매칭은 오매칭 위험이 커 쓰지 않는다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = ["PrereqResolution", "normalize_course_name", "resolve_prereq_ids"]

_PAREN_RE = re.compile(r"[（(（][^）)）]*[）)）]")


def normalize_course_name(name: str) -> str:
    """매칭용 과목명 정규화 — 괄호 안 내용 제거 + 공백 제거 + 소문자.

    ``build_prerequisites.py`` 의 과목명 정규화와 동일한 규칙을 따른다(둘 다
    ``courses.csv`` 교과목명을 기준 좌표로 쓰므로 규칙이 어긋나면 매칭이 깨진다).
    """
    return _PAREN_RE.sub("", name).replace(" ", "").lower()


@dataclass
class PrereqResolution:
    """선수과목 해석 결과 + 드롭 통계(로깅·검증용)."""

    # course_id → 해석된 선수 course_id 리스트(등장 순서 보존, dedup).
    prereq_ids_by_course: dict[str, list[str]] = field(default_factory=dict)
    # (과목명, 카탈로그 밖 선수과목명) — 교양/타과 등으로 드롭된 엣지.
    dropped_edges: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_resolved_edges(self) -> int:
        return sum(len(v) for v in self.prereq_ids_by_course.values())


def resolve_prereq_ids(
    prerequisites_by_name: Mapping[str, list[str]],
    courses: Iterable[tuple[str, str]],
) -> PrereqResolution:
    """과목별 선수 course_id 를 해석한다.

    Args:
        prerequisites_by_name: ``{과목명: [선수과목명, ...]}`` (prerequisites.json).
        courses: ``(course_name, course_id)`` 쌍. 카탈로그(전공)에 실재하는 과목만.

    Returns:
        ``PrereqResolution`` — course_id 별 선수 course_id 리스트와 드롭 통계.
        카탈로그 밖 선수·자기참조는 제외하고, 등장 순서를 보존하며 dedup 한다.
    """
    course_list = list(courses)
    name_to_id: dict[str, str] = {}
    for course_name, course_id in course_list:
        # 정규화 이름 충돌 시 첫 등장 우선(코드가 식별자이므로 표시 영향 없음).
        name_to_id.setdefault(normalize_course_name(course_name), course_id)

    prereq_norm = {
        normalize_course_name(name): prereqs for name, prereqs in prerequisites_by_name.items()
    }

    result = PrereqResolution()
    for course_name, course_id in course_list:
        prereq_names = prereq_norm.get(normalize_course_name(course_name))
        if not prereq_names:
            continue
        resolved: list[str] = []
        for prereq_name in prereq_names:
            prereq_id = name_to_id.get(normalize_course_name(prereq_name))
            if prereq_id is None:
                result.dropped_edges.append((course_name, prereq_name))
                continue
            if prereq_id == course_id:
                continue  # 자기참조 제외.
            if prereq_id not in resolved:
                resolved.append(prereq_id)
        if resolved:
            result.prereq_ids_by_course[course_id] = resolved
    return result
