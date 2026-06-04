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


def test_track_repository_falls_back_department_to_own_track_name(tmp_path: Path) -> None:
    """학부 메타가 없으면 학과 식별자를 college 가 아닌 트랙명으로 잡는다.

    학부 메타가 없는 트랙은 트랙 구분이 없는 단일 학과다. college 로 대체하면 한
    단과대 아래 별개 단일 학과들이 같은 department_id 로 뭉뚱그려져 구조적으로
    구분되지 않으므로, 트랙명을 학과 식별자로 써 학과당 트랙 1개로 식별되게 한다.
    """
    rag_dir = tmp_path / "rag"
    _write(
        rag_dir / "tracks" / "트랙소개_학과A.txt",
        "[트랙: 학과A | 대학: 창의융합대학]\n\n■ 소개\n소개",
    )
    _write(
        rag_dir / "courses" / "교육과정_학과A.txt",
        (
            "[트랙: 학과A | 대학: 창의융합대학]\n"
            "■ 1학년 1학기\n"
            "  - [전공선택] 전공과목 (C001, 3학점)\n"
        ),
    )

    track = PreprocessedTrackRepository(rag_dir).list_all()[0]

    assert track.college_id == "창의융합대학"
    assert track.department_id == "학과A"
    assert track.major_id == "학과A"


def test_single_departments_in_same_college_get_distinct_department_ids(tmp_path: Path) -> None:
    """학부 메타가 없는 단일 학과들이 같은 단과대 아래에서도 구조적으로 분리된다.

    예전에는 둘 다 college 로 대체돼 같은 department_id 로 뭉뚱그려져 단일 학과를
    구분할 수 없었다. 트랙명 fallback 으로 각자 자기 학과로 식별돼야 한다.
    """
    rag_dir = tmp_path / "rag"
    for dept in ("학과A", "학과B"):
        _write(
            rag_dir / "tracks" / f"트랙소개_{dept}.txt",
            f"[트랙: {dept} | 대학: 창의융합대학]\n\n■ 소개\n소개",
        )
        _write(
            rag_dir / "courses" / f"교육과정_{dept}.txt",
            (
                f"[트랙: {dept} | 대학: 창의융합대학]\n"
                "■ 1학년 1학기\n"
                "  - [전공선택] 전공과목 (C001, 3학점)\n"
            ),
        )

    tracks = {t.track_id: t for t in PreprocessedTrackRepository(rag_dir).list_all()}

    assert tracks["학과A"].department_id == "학과A"
    assert tracks["학과B"].department_id == "학과B"
    assert tracks["학과A"].department_id != tracks["학과B"].department_id


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
