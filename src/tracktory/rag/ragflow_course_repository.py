"""RAGFlow 기반 ``CourseRepository`` 구현체 (오프라인 카탈로그 생성기 엔진).

교육과정(curriculum) 문서 본문에서 과목을 추출해 ``Course`` 로 조립한다. 본문
한 줄에 과목구분·이름·코드·학점이 모두 있고("- [전공선택] 공학프로그래밍
(V070044, 3학점)"), 학년/학기 섹션 헤더(" 2학년 1학기")로 이수 학년을 알 수 있어
강의계획서(syllabus) 없이 ``Course`` 를 채운다.

``list_for_tracks(track_ids)`` 는 각 트랙의 교육과정을 파싱해 과목을 모으고
``course_id`` 기준 dedup 한다 — 같은 과목이 여러 트랙/학기에 중복 등장하면
``track_ids`` / ``available_grades`` / ``available_semesters`` 를 union 한다.
교양(선택필수교양 / 일반교양 / 교양필수)은 추천 대상이 아니라 제외한다.

런타임 추천 경로(roadmap 노드)는 ``YamlCourseRepository``(오프라인)를 쓰고, 본
클래스는 ``scripts/generate_course_catalog.py`` 가 ``courses.yaml`` 을 생성할 때
RAGFlow 호출 엔진으로 쓴다.

호출 측은 ``RagflowCourseRepository`` 를 직접 import 하지 않고 ``CourseRepository``
Protocol 타입으로만 주입받는다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from tracktory.graph.models import Course
from tracktory.rag.curriculum_lines import CourseTypeLabel, StageLabel, parse_course_line
from tracktory.rag.hansung_catalog import match_track_name
from tracktory.rag.ragflow_client import RagflowClient, RagflowDocument, RagflowError

__all__ = ["CourseRepositoryError", "RagflowCourseRepository"]

logger = logging.getLogger(__name__)

_CURRICULUM_KEYWORD = "교육과정"
_CURRICULUM_DOC_TYPE = "curriculum"

# 교육과정 본문 형식:
#   " 2학년 1학기"                                   ← 학년/학기 섹션 헤더
#   "  - [전공선택] 공학프로그래밍 (V070044, 3학점)"   ← 과목 줄
# 과목 줄 파싱·전공 판정은 ``curriculum_lines.parse_course_line`` 으로 트랙
# 저장소와 공유한다. 본 모듈은 학년/학기 섹션 추적만 별도로 책임진다.
_SECTION_RE = re.compile(r"(\d)\s*학년\s*(\d)\s*학기")


@dataclass
class _CourseAccum:
    """``course_id`` 단위 누적기 — 여러 트랙/학기 중복 등장을 합친다."""

    course_name: str
    credits: int
    stage: StageLabel
    course_type: CourseTypeLabel
    track_ids: list[str] = field(default_factory=list)
    grades: set[int] = field(default_factory=set)
    semesters: set[int] = field(default_factory=set)


class CourseRepositoryError(Exception):
    """과목 카탈로그 로딩 실패.

    RAGFlow 호출/응답 파싱 실패를 도메인 예외로 변환한다. 학습 로드맵 노드의
    안전 종료 분기가 raw 예외가 아닌 본 예외를 보도록 boundary 에서 좁힌다.
    """


class RagflowCourseRepository:
    """``CourseRepository`` Protocol 의 RAGFlow 교육과정 어댑터.

    HTTP 호출·재시도·에러 변환은 주입된 ``RagflowClient`` 가 담당하고, 본 클래스는
    교육과정 본문 → ``Course`` 변환·dedup 만 책임진다.
    """

    def __init__(self, client: RagflowClient) -> None:
        # 부작용 격리·DI 원칙: 테스트에서 fake client 주입 가능.
        self._client = client

    def list_for_tracks(self, track_ids: list[str]) -> list[Course]:
        """주어진 트랙들의 교육과정에서 추천 대상(전공) 과목을 모아 반환한다.

        ``course_id`` 기준 dedup. 입력 ``track_ids`` 중 교육과정 문서가 없는 것은
        조용히 무시한다. 실패는 ``CourseRepositoryError`` 로 변환한다.
        """
        wanted = list(dict.fromkeys(track_ids))  # 입력 순서 보존 dedup
        if not wanted:
            return []
        try:
            curriculum_by_name = self._index_curriculum_by_track()
            accum: dict[str, _CourseAccum] = {}
            for track_id in wanted:
                doc = curriculum_by_name.get(match_track_name(track_id))
                if doc is None:
                    continue
                body = self._client.fetch_document_text(doc.document_id)
                self._accumulate(body, track_id, accum)
        except RagflowError as exc:
            raise CourseRepositoryError(f"과목 카탈로그 로딩 실패: {exc}") from exc
        return self._to_courses(accum)

    def _index_curriculum_by_track(self) -> dict[str, RagflowDocument]:
        """교육과정 문서를 정규화 트랙명 → 문서로 색인한다(메타만, 본문 미조회).

        파일명에 "교육과정" 이 우연히 든 다른 doc_type(예: 강의계획서)을 doc_type
        메타로 걸러낸다.
        """
        index: dict[str, RagflowDocument] = {}
        for doc in self._client.list_documents(name_keyword=_CURRICULUM_KEYWORD):
            if doc.meta_fields.get("doc_type") != _CURRICULUM_DOC_TYPE:
                continue
            name = doc.meta_fields.get("track_name")
            if isinstance(name, str) and name.strip():
                index[match_track_name(name)] = doc
        return index

    @staticmethod
    def _accumulate(body: str, track_id: str, accum: dict[str, _CourseAccum]) -> None:
        """교육과정 본문 1건을 파싱해 ``accum`` 에 과목을 누적한다(교양 제외).

        현재 학년/학기 섹션을 추적해 각 과목 줄에 이수 학년·절대 학기 번호를
        부여한다. 절대 학기는 1학년 1학기=1, 4학년 2학기=8 이다.
        """
        current_grade: int | None = None
        current_semester: int | None = None
        for line in body.splitlines():
            section = _SECTION_RE.search(line)
            if section:
                current_grade = int(section.group(1))
                term = int(section.group(2))
                current_semester = (current_grade - 1) * 2 + term
                continue
            parsed = parse_course_line(line)
            if parsed is None:
                continue  # 과목 줄 아님 또는 교양류(비전공) → 추천 대상 아님, 제외.
            code = parsed.course_id
            entry = accum.get(code)
            if entry is None:
                entry = _CourseAccum(
                    course_name=parsed.course_name,
                    credits=parsed.credits,
                    stage=parsed.stage,
                    course_type=parsed.course_type,
                )
                accum[code] = entry
            if track_id not in entry.track_ids:
                entry.track_ids.append(track_id)
            if current_grade is not None:
                entry.grades.add(current_grade)
            if current_semester is not None:
                entry.semesters.add(current_semester)

    @staticmethod
    def _to_courses(accum: dict[str, _CourseAccum]) -> list[Course]:
        courses: list[Course] = []
        for code, entry in accum.items():
            grades = sorted(entry.grades) if entry.grades else [1, 2, 3, 4]
            semesters = sorted(entry.semesters) if entry.semesters else list(range(1, 9))
            try:
                courses.append(
                    Course(
                        course_id=code,
                        course_name=entry.course_name,
                        credits=entry.credits,
                        stage=entry.stage,
                        prereq_ids=[],  # 교육과정에 선수관계 없음 → 보류.
                        track_ids=entry.track_ids,
                        priority=1,  # 도출 보류 → 기본값.
                        available_grades=grades,
                        available_semesters=semesters,
                        course_type=entry.course_type,
                    )
                )
            except ValidationError as exc:
                logger.warning("Course 검증 실패 스킵(%s): %s", code, exc)
        return courses
