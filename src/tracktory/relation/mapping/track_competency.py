"""
트랙별 커버 역량 매핑 테이블 생성 스크립트.

강의계획서 텍스트에서 직접 기술 키워드를 추출하여 트랙별 역량을 집계한다.

소스 파일:
  - data/raw/hansung/courses.csv              : 트랙 → 과목 목록
  - data/processed/rag/output/syllabi_rename/ : 과목별 강의계획서 txt

처리 흐름:
  1. courses.csv → 트랙별 고유 과목명 목록
  2. syllabi_rename/*.txt → 과목명 → 계획서 텍스트 맵
  3. 트랙 과목명으로 계획서 텍스트 조인 (없는 과목은 스킵)
  4. extract_tech_keywords() 로 텍스트에서 기술 키워드 추출
  5. 트랙별 집계: 역량 → 커버 과목 목록, 등장 횟수, coverage_rate

실행:
    uv run python -m tracktory.relation.mapping.track_competency
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from tracktory.common.tech_keywords import extract_tech_keywords
from tracktory.relation.mapping.config import (
    COURSES_CSV,
    SYLLABUS_DIR,
    TRACK_COMPETENCY_OUT_PATH,
    TRACK_LOG_FILE,
)


class CompetencyEvidence(TypedDict):
    name: str
    covering_courses: list[str]
    course_count: int
    coverage_rate: float


class TrackCompetencyEntry(TypedDict):
    track_name: str
    total_courses: int
    courses_with_syllabus: int
    competencies: list[CompetencyEvidence]


def _load_lecture_csv(path: Path) -> list[dict[str, str]]:
    """강의정보 CSV를 읽어 row 딕셔너리 리스트로 반환한다.

    Args:
        path: courses.csv 경로.

    Returns:
        CSV row 딕셔너리 리스트.
    """
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _parse_course_name(text: str) -> str:
    """txt 텍스트에서 과목명을 추출한다."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("과목명:"):
            return stripped[len("과목명:") :].strip()
    return ""


def _load_syllabus_map(syllabus_dir: Path) -> dict[str, str]:
    """강의계획서 txt 파일 디렉터리에서 과목명 → 텍스트 맵을 반환한다.

    같은 과목명 파일이 여러 개면 텍스트를 이어붙인다.

    Args:
        syllabus_dir: 강의계획서 txt 파일 디렉터리.

    Returns:
        과목명을 키, 강의계획서 전체 텍스트를 값으로 하는 딕셔너리.
    """
    texts_by_course: dict[str, list[str]] = defaultdict(list)

    for txt_file in sorted(syllabus_dir.glob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = txt_file.read_text(encoding="utf-8-sig")

        course_name = _parse_course_name(text)
        if course_name:
            texts_by_course[course_name].append(text)

    return {course: "\n".join(texts) for course, texts in texts_by_course.items()}


def _group_by_track(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """CSV row를 트랙 → 고유 과목명 리스트로 그룹화한다.

    Args:
        rows: _load_lecture_csv() 결과.

    Returns:
        트랙명을 키, 정렬된 과목명 리스트를 값으로 하는 딕셔너리.
    """
    track_courses: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        track = row.get("트랙", "").strip()
        course = row.get("교과목", "").strip()
        if track and course:
            track_courses[track].add(course)
    return {track: sorted(courses) for track, courses in track_courses.items()}


def build_track_competency_map(
    lecture_csv: Path = COURSES_CSV,
    syllabus_dir: Path = SYLLABUS_DIR,
    log_file: Path = TRACK_LOG_FILE,
) -> dict[str, TrackCompetencyEntry]:
    """트랙별 커버 역량 매핑 테이블을 빌드한다.

    Args:
        lecture_csv: courses.csv 경로.
        syllabus_dir: 강의계획서 txt 파일 디렉터리.
        log_file: 강의계획서 미매칭 과목 로그 경로.

    Returns:
        {track_name: TrackCompetencyEntry} 딕셔너리.
    """
    rows = _load_lecture_csv(lecture_csv)
    syllabus_map = _load_syllabus_map(syllabus_dir)
    track_courses = _group_by_track(rows)

    result: dict[str, TrackCompetencyEntry] = {}
    no_syllabus_log: list[str] = []

    for track_name, courses in sorted(track_courses.items()):
        total = len(courses)
        competency_courses: dict[str, set[str]] = defaultdict(set)
        courses_with_syllabus = 0

        for course in courses:
            text = syllabus_map.get(course)
            if not text:
                no_syllabus_log.append(f"[{track_name}] {course}")
                continue
            courses_with_syllabus += 1
            for tech in extract_tech_keywords(text):
                competency_courses[tech].add(course)

        competencies: list[CompetencyEvidence] = [
            CompetencyEvidence(
                name=comp,
                covering_courses=sorted(course_set),
                course_count=len(course_set),
                coverage_rate=round(len(course_set) / courses_with_syllabus, 4)
                if courses_with_syllabus > 0
                else 0.0,
            )
            for comp, course_set in competency_courses.items()
        ]
        competencies.sort(key=lambda c: (-c["course_count"], c["name"]))

        result[track_name] = TrackCompetencyEntry(
            track_name=track_name,
            total_courses=total,
            courses_with_syllabus=courses_with_syllabus,
            competencies=competencies,
        )

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"강의계획서 미매칭 과목 ({len(no_syllabus_log)}건)\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(no_syllabus_log))

    return result


def save_track_competency_map(
    result: dict[str, TrackCompetencyEntry],
    out_path: Path = TRACK_COMPETENCY_OUT_PATH,
) -> None:
    """빌드된 결과를 JSON 파일로 저장한다.

    Args:
        result: build_track_competency_map() 반환값.
        out_path: 저장할 JSON 경로.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    result = build_track_competency_map()
    save_track_competency_map(result)

    total_courses = sum(e["total_courses"] for e in result.values())
    total_with_syllabus = sum(e["courses_with_syllabus"] for e in result.values())

    print(f"\n완료: {TRACK_COMPETENCY_OUT_PATH}")
    print(f"트랙 수          : {len(result)}개")
    print(f"총 과목 수       : {total_courses}개")
    print(f"강의계획서 있음  : {total_with_syllabus}개")
    print(f"강의계획서 없음  : {total_courses - total_with_syllabus}개")
    print(f"미매칭 로그      : {TRACK_LOG_FILE}")
