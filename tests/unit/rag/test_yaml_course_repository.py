from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tracktory.rag.yaml_course_repository import CourseCatalogError, YamlCourseRepository

_CATALOG = {
    "courses": [
        {
            "course_id": "C001",
            "course_name": "기초과목",
            "credits": 3,
            "stage": "foundation",
            "course_type": "전공선택",
            "track_ids": ["트랙A", "트랙B"],
            "available_grades": [1],
            "available_semesters": [1, 2],
        },
        {
            "course_id": "C002",
            "course_name": "선택과목",
            "credits": 3,
            "stage": "application",
            "course_type": "전공선택",
            "track_ids": ["트랙A"],
            "available_grades": [2],
            "available_semesters": [3],
        },
        {
            "course_id": "C004",
            "course_name": "비공유과목",
            "credits": 3,
            "stage": "application",
            "course_type": "전공선택",
            "track_ids": ["트랙C"],
            "available_grades": [3],
            "available_semesters": [5],
        },
    ]
}


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "courses.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_list_for_tracks_returns_intersecting_courses(tmp_path: Path) -> None:
    repo = YamlCourseRepository(_write(tmp_path, _CATALOG))

    courses = {c.course_id for c in repo.list_for_tracks(["트랙B"])}

    # 트랙B 는 C001 에만 속함.
    assert courses == {"C001"}


def test_list_for_tracks_unions_across_requested_tracks(tmp_path: Path) -> None:
    repo = YamlCourseRepository(_write(tmp_path, _CATALOG))

    courses = {c.course_id for c in repo.list_for_tracks(["트랙A", "트랙C"])}

    assert courses == {"C001", "C002", "C004"}


def test_loaded_course_has_model_defaults(tmp_path: Path) -> None:
    repo = YamlCourseRepository(_write(tmp_path, _CATALOG))

    c001 = next(c for c in repo.list_for_tracks(["트랙A"]) if c.course_id == "C001")

    # YAML 에 없는 필드는 모델 기본값, YAML 에 있는 학기 제약은 그대로 유지.
    assert c001.prereq_ids == []
    assert c001.priority == 1
    assert c001.available_semesters == [1, 2]


def test_empty_track_ids_returns_empty(tmp_path: Path) -> None:
    repo = YamlCourseRepository(_write(tmp_path, _CATALOG))
    assert repo.list_for_tracks([]) == []


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CourseCatalogError):
        YamlCourseRepository(tmp_path / "nope.yaml")


def test_missing_courses_key_raises(tmp_path: Path) -> None:
    with pytest.raises(CourseCatalogError):
        YamlCourseRepository(_write(tmp_path, {"something": []}))


def test_invalid_entry_is_skipped(tmp_path: Path) -> None:
    catalog = {
        "courses": [
            {"course_id": "BAD"},  # 필수 필드 누락 → 검증 실패 스킵
            _CATALOG["courses"][0],
        ]
    }
    repo = YamlCourseRepository(_write(tmp_path, catalog))

    assert {c.course_id for c in repo.list_for_tracks(["트랙A"])} == {"C001"}
