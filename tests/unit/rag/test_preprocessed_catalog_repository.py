from __future__ import annotations

from pathlib import Path

from tracktory.rag.preprocessed_catalog_repository import (
    PreprocessedCourseRepository,
    PreprocessedTrackRepository,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_preprocessed_track_repository_reads_track_and_curriculum_txt(tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag"
    _write(
        rag_dir / "tracks" / "트랙소개_트랙A_학부A.txt",
        "[트랙: 트랙A | 대학: 대학A | 학부: 학부A]\n\n■ 소개\n소개",
    )
    _write(
        rag_dir / "courses" / "교육과정_트랙A_학부A.txt",
        (
            "[트랙: 트랙A | 대학: 대학A | 학부: 학부A]\n"
            "■ 1학년 1학기\n"
            "  - [전공기초] 기초과목 (C001, 3학점)\n"
            "  - [일반교양] 교양과목 (G001, 2학점)\n"
            "■ 2학년 1학기\n"
            "  - [전공필수] 필수과목 (C002, 3학점)\n"
        ),
    )

    tracks = PreprocessedTrackRepository(rag_dir).list_all()

    assert len(tracks) == 1
    assert tracks[0].track_id == "트랙A"
    assert tracks[0].college_id == "대학A"
    assert tracks[0].department_id == "학부A"
    assert tracks[0].course_ids == ["C001", "C002"]


def test_preprocessed_course_repository_dedups_and_sets_stage(tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag"
    _write(
        rag_dir / "courses" / "교육과정_트랙A_학부A.txt",
        (
            "[트랙: 트랙A | 대학: 대학A | 학부: 학부A]\n"
            "■ 1학년 1학기\n"
            "  - [전공기초] 공유과목 (C001, 3학점)\n"
            "■ 2학년 1학기\n"
            "  - [전공선택] 선택과목 (C002, 3학점)\n"
        ),
    )
    _write(
        rag_dir / "courses" / "교육과정_트랙B_학부B.txt",
        (
            "[트랙: 트랙B | 대학: 대학B | 학부: 학부B]\n"
            "■ 1학년 2학기\n"
            "  - [전공선택] 공유과목 (C001, 3학점)\n"
            "■ 3학년 1학기\n"
            "  - [전공필수] 심화과목 (C003, 3학점)\n"
        ),
    )

    courses = {
        c.course_id: c
        for c in PreprocessedCourseRepository(rag_dir).list_for_tracks(["트랙A", "트랙B"])
    }

    assert set(courses) == {"C001", "C002", "C003"}
    assert courses["C001"].track_ids == ["트랙A", "트랙B"]
    assert courses["C001"].stage == "foundation"
    assert courses["C001"].available_grades == [1]
    assert courses["C001"].available_semesters == [1, 2]
    assert courses["C002"].stage == "core"
    assert courses["C003"].stage == "application"


def test_track_repository_falls_back_department_to_college(tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag"
    _write(
        rag_dir / "tracks" / "트랙소개_학과A.txt",
        "[트랙: 학과A | 대학: 한성대학교]\n\n■ 소개\n소개",
    )
    _write(
        rag_dir / "courses" / "교육과정_학과A.txt",
        (
            "[트랙: 학과A | 대학: 한성대학교]\n"
            "■ 1학년 1학기\n"
            "  - [전공선택] 전공과목 (C001, 3학점)\n"
        ),
    )

    track = PreprocessedTrackRepository(rag_dir).list_all()[0]

    assert track.college_id == "한성대학교"
    assert track.department_id == "한성대학교"


def test_track_repository_skips_tracks_without_matching_curriculum(tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag"
    _write(
        rag_dir / "tracks" / "트랙소개_빈트랙.txt",
        "[트랙: 빈트랙 | 대학: 대학A | 학부: 학부A]\n\n■ 소개\n소개",
    )
    (rag_dir / "courses").mkdir(parents=True)

    assert PreprocessedTrackRepository(rag_dir).list_all() == []


def test_track_repository_skips_matched_curriculum_with_no_major_courses(
    tmp_path: Path,
) -> None:
    rag_dir = tmp_path / "rag"
    _write(
        rag_dir / "tracks" / "트랙소개_교양트랙.txt",
        "[트랙: 교양트랙 | 대학: 대학A | 학부: 학부A]\n\n■ 소개\n소개",
    )
    _write(
        rag_dir / "courses" / "교육과정_교양트랙.txt",
        (
            "[트랙: 교양트랙 | 대학: 대학A | 학부: 학부A]\n"
            "■ 1학년 1학기\n"
            "  - [일반교양] 교양과목 (G001, 2학점)\n"
        ),
    )

    assert PreprocessedTrackRepository(rag_dir).list_all() == []
