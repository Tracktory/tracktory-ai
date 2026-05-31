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

동명 과목이 여러 학과·트랙에 존재하면(예: ``운영체제`` 가 V-트랙과 AI·소프트웨어
학과에 동시에) 이름만으로는 어느 코드인지 정해지지 않는다. 이때 **후수 과목과
트랙이 겹치는 동명 후보를 우선** 선택한다 — 그러지 않으면 로드맵이 학생 본인
트랙에 같은 과목이 있는데도 엉뚱한 학과 과목을 선수로 끌어온다. 트랙이 겹치는
후보가 없으면 첫 등장 후보로 fallback 한다(같은 트랙에 동명이 애초에 없는
정상 케이스 보존).
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


def _select_prereq_id(
    candidates: list[tuple[str, frozenset[str]]],
    course_id: str,
    course_tracks: frozenset[str],
) -> str | None:
    """동명 선수 후보 중 후수 과목과 트랙이 가장 많이 겹치는 후보를 선택한다.

    겹치는 후보가 없으면 첫 등장 후보로 fallback 한다(교양·타과 유일 동명처럼
    같은 트랙 후보가 애초에 없는 정상 케이스 보존). 자기참조 후보는 제외하며,
    제외 후 후보가 없으면 ``None`` 을 반환한다. 동률은 등장 순서가 앞선 후보가
    이긴다(``>`` 비교로 첫 등장 우선 유지).
    """
    pool = [(cid, tracks) for cid, tracks in candidates if cid != course_id]
    if not pool:
        return None
    best_id, best_overlap = pool[0][0], len(pool[0][1] & course_tracks)
    for cid, tracks in pool[1:]:
        overlap = len(tracks & course_tracks)
        if overlap > best_overlap:
            best_id, best_overlap = cid, overlap
    return best_id


def resolve_prereq_ids(
    prerequisites_by_name: Mapping[str, list[str]],
    courses: Iterable[tuple[str, str, list[str]]],
) -> PrereqResolution:
    """과목별 선수 course_id 를 해석한다.

    동명 과목이 여러 학과·트랙에 존재할 때, 후수 과목과 트랙이 겹치는 동명 후보를
    우선 선택한다(상세는 모듈 docstring). 그래서 호출자는 트랙 식별자까지 넘긴다.

    Args:
        prerequisites_by_name: ``{과목명: [선수과목명, ...]}`` (prerequisites.json).
        courses: ``(course_name, course_id, track_ids)`` 쌍. 카탈로그(전공)에
            실재하는 과목만. ``track_ids`` 는 동명 후보 중 같은 트랙 후보를
            우선하기 위한 선택 신호다.

    Returns:
        ``PrereqResolution`` — course_id 별 선수 course_id 리스트와 드롭 통계.
        카탈로그 밖 선수·자기참조는 제외하고, 등장 순서를 보존하며 dedup 한다.
    """
    course_list = list(courses)
    # 정규화 이름 → 동명 후보 (course_id, track 집합) 리스트. 등장 순서 보존.
    name_to_candidates: dict[str, list[tuple[str, frozenset[str]]]] = {}
    for course_name, course_id, track_ids in course_list:
        name_to_candidates.setdefault(normalize_course_name(course_name), []).append(
            (course_id, frozenset(track_ids))
        )

    prereq_norm = {
        normalize_course_name(name): prereqs for name, prereqs in prerequisites_by_name.items()
    }

    result = PrereqResolution()
    for course_name, course_id, track_ids in course_list:
        prereq_names = prereq_norm.get(normalize_course_name(course_name))
        if not prereq_names:
            continue
        course_tracks = frozenset(track_ids)
        resolved: list[str] = []
        for prereq_name in prereq_names:
            candidates = name_to_candidates.get(normalize_course_name(prereq_name))
            if not candidates:
                result.dropped_edges.append((course_name, prereq_name))
                continue
            prereq_id = _select_prereq_id(candidates, course_id, course_tracks)
            if prereq_id is None:
                continue  # 후보가 자기참조뿐 → 제외.
            if prereq_id not in resolved:
                resolved.append(prereq_id)
        if resolved:
            result.prereq_ids_by_course[course_id] = resolved
    return result
