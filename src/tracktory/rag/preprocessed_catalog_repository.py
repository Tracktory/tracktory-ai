"""전처리된 RAG txt 파일 기반 트랙/과목 카탈로그 저장소.

``scripts/generate_track_catalog.py`` / ``generate_course_catalog.py`` 의 입력을
RAGFlow 문서 API가 아니라 ``data/processed/rag`` 산출물로 고정한다. RAGFlow는
검색 인덱스 소비자로 두고, 정적 카탈로그 생성은 로컬 전처리 결과를 단일 원천으로
삼는다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from tracktory.graph.models import Course, Track
from tracktory.rag.curriculum_lines import CourseTypeLabel, StageLabel, parse_course_line
from tracktory.rag.hansung_catalog import canonical_track_name, match_track_name
from tracktory.rag.preprocessing.extract_rag_metadata import parse_metadata

__all__ = [
    "PreprocessedCatalogError",
    "PreprocessedCourseRepository",
    "PreprocessedTrackRepository",
]

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RAG_DIR = _ROOT / "data" / "processed" / "rag"
_SECTION_RE = re.compile(r"(\d)\s*학년\s*(\d)\s*학기")
_STAGE_BY_GRADE: dict[int, StageLabel] = {
    1: "foundation",
    2: "core",
    3: "application",
    4: "industry",
}


def _stage_from_grade(grade: int) -> StageLabel:
    return _STAGE_BY_GRADE[min(max(grade, 1), 4)]


class PreprocessedCatalogError(Exception):
    """전처리 카탈로그 txt 로딩/파싱 실패."""


@dataclass(frozen=True)
class _TrackDoc:
    track_name: str
    college: str
    department: str | None
    text: str


@dataclass(frozen=True)
class _CurriculumDoc:
    track_name: str
    text: str


@dataclass
class _CourseAccum:
    course_name: str
    credits: int
    course_type: CourseTypeLabel
    track_ids: list[str] = field(default_factory=list)
    grades: set[int] = field(default_factory=set)
    semesters: set[int] = field(default_factory=set)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def _load_track_docs(rag_dir: Path) -> list[_TrackDoc]:
    tracks_dir = rag_dir / "tracks"
    if not tracks_dir.exists():
        raise PreprocessedCatalogError(f"트랙소개 디렉터리 없음: {tracks_dir}")

    docs: list[_TrackDoc] = []
    for path in sorted(tracks_dir.glob("트랙소개_*.txt")):
        text = _read_text(path)
        meta = parse_metadata(path, rag_dir)["metadata"]
        track_name = _required_meta(path, meta, "track_name")
        college = _required_meta(path, meta, "college")
        department_meta = _optional_meta(meta, "department")
        docs.append(_TrackDoc(track_name, college, department_meta, text))
    return docs


def _load_curriculum_docs(rag_dir: Path) -> list[_CurriculumDoc]:
    courses_dir = rag_dir / "courses"
    if not courses_dir.exists():
        raise PreprocessedCatalogError(f"교육과정 디렉터리 없음: {courses_dir}")

    docs: list[_CurriculumDoc] = []
    for path in sorted(courses_dir.glob("교육과정_*.txt")):
        text = _read_text(path)
        meta = parse_metadata(path, rag_dir)["metadata"]
        track_name = _required_meta(path, meta, "track_name")
        docs.append(_CurriculumDoc(track_name, text))
    return docs


def _required_meta(path: Path, meta: dict[str, object], key: str) -> str:
    value = meta.get(key)
    if isinstance(value, str) and value.strip():
        return _clean_meta_value(value)
    raise PreprocessedCatalogError(f"{key} 메타 누락: {path}")


def _optional_meta(meta: dict[str, object], key: str) -> str | None:
    value = meta.get(key)
    if isinstance(value, str) and value.strip():
        return _clean_meta_value(value)
    return None


def _clean_meta_value(value: str) -> str:
    return value.strip().removesuffix("]").strip()


class PreprocessedTrackRepository:
    """``data/processed/rag/tracks|courses/*.txt`` 기반 ``TrackRepository``."""

    def __init__(self, rag_dir: Path | None = None) -> None:
        self._rag_dir = rag_dir or _DEFAULT_RAG_DIR

    def list_all(self) -> list[Track]:
        return self._build_tracks(wanted=None)

    def list_track_doc_names(self) -> list[str]:
        """트랙소개 txt 전체 트랙명(권위 표기). ``list_all`` 은 교육과정·course_id 가 없는 트랙을
        조용히 드롭하므로, 생성기가 '왜 카탈로그에서 빠졌는지'를 전수 대조·리포팅할 때
        모집단으로 쓴다. ``list_all`` 의 track_id 와 같은 정규화 형태라야 차집합 대조가 맞다."""
        return [canonical_track_name(doc.track_name) for doc in _load_track_docs(self._rag_dir)]

    def find_by_track_ids(self, track_ids: list[str]) -> list[Track]:
        wanted = set(track_ids)
        if not wanted:
            return []
        return self._build_tracks(wanted=wanted)

    def _build_tracks(self, *, wanted: set[str] | None) -> list[Track]:
        curriculum_by_name = {
            match_track_name(doc.track_name): doc.text
            for doc in _load_curriculum_docs(self._rag_dir)
        }

        tracks: list[Track] = []
        for doc in _load_track_docs(self._rag_dir):
            # RAG 본문 헤더는 가운뎃점(·)으로 치환돼 있어, 카탈로그가 노출하는 식별자·표시명은
            # 권위 표기(ㆍ)로 되돌려 백엔드·원본과 일치시킨다. 교육과정 매칭은 가운뎃점류를
            # 통일하는 match_track_name 이 흡수하므로 표기 복원과 무관하게 짝지어진다.
            track_name = canonical_track_name(doc.track_name)
            if wanted is not None and track_name not in wanted:
                continue
            curriculum = curriculum_by_name.get(match_track_name(track_name))
            if curriculum is None:
                continue
            course_ids = self._extract_course_ids(curriculum)
            if not course_ids:
                continue
            # 학부(department) 메타가 없으면 트랙 자신이 곧 독립 학과(트랙 구분이 없는
            # 단일 학과)다. college 로 대체하면 한 단과대 아래 별개 단일 학과들이 같은
            # department_id 로 뭉뚱그려져 구조적으로 구분되지 않는다. 트랙명을 학과
            # 식별자로 써 단일 학과가 학과당 트랙 1개로 식별되게 한다.
            department_id = doc.department or track_name
            try:
                tracks.append(
                    Track(
                        college_id=doc.college,
                        department_id=department_id,
                        major_id=department_id,
                        track_id=track_name,
                        track_name=track_name,
                        course_ids=course_ids,
                        meta_text="",
                        meta_vector=[],
                        competencies=[],
                        tech_stacks=[],
                    )
                )
            except ValidationError as exc:
                logger.warning("Track 검증 실패 스킵(%s): %s", track_name, exc)
        return tracks

    @staticmethod
    def _extract_course_ids(curriculum_text: str) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for line in curriculum_text.splitlines():
            parsed = parse_course_line(line)
            if parsed is None or parsed.course_id in seen:
                continue
            seen.add(parsed.course_id)
            ordered.append(parsed.course_id)
        return ordered


class PreprocessedCourseRepository:
    """``data/processed/rag/courses/*.txt`` 기반 ``CourseRepository``."""

    def __init__(self, rag_dir: Path | None = None) -> None:
        self._rag_dir = rag_dir or _DEFAULT_RAG_DIR

    def list_for_tracks(self, track_ids: list[str]) -> list[Course]:
        wanted = list(dict.fromkeys(track_ids))
        if not wanted:
            return []

        curriculum_by_name = {
            match_track_name(doc.track_name): doc.text
            for doc in _load_curriculum_docs(self._rag_dir)
        }
        accum: dict[str, _CourseAccum] = {}
        for track_id in wanted:
            body = curriculum_by_name.get(match_track_name(track_id))
            if body is None:
                continue
            self._accumulate(body, track_id, accum)
        return self._to_courses(accum)

    @staticmethod
    def _accumulate(body: str, track_id: str, accum: dict[str, _CourseAccum]) -> None:
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
                continue

            entry = accum.get(parsed.course_id)
            if entry is None:
                entry = _CourseAccum(
                    course_name=parsed.course_name,
                    credits=parsed.credits,
                    course_type=parsed.course_type,
                )
                accum[parsed.course_id] = entry
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
                        stage=_stage_from_grade(grades[0]),
                        prereq_ids=[],
                        track_ids=entry.track_ids,
                        priority=1,
                        available_grades=grades,
                        available_semesters=semesters,
                        course_type=entry.course_type,
                    )
                )
            except ValidationError as exc:
                logger.warning("Course 검증 실패 스킵(%s): %s", code, exc)
        return courses
