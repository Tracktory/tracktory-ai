from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tracktory.rag.yaml_track_repository import TrackCatalogError, YamlTrackRepository

_CATALOG = {
    "tracks": [
        {
            "track_id": "응용산업데이터공학트랙",
            "track_name": "응용산업데이터공학트랙",
            "college_id": "IT공과대학",
            "department_id": "산업시스템공학부",
            "major_id": "산업시스템공학부",
            "course_ids": ["CTE0029", "V070044"],
        },
        {
            "track_id": "빅데이터트랙",
            "track_name": "빅데이터트랙",
            "college_id": "IT공과대학",
            "department_id": "컴퓨터공학부",
            "major_id": "컴퓨터공학부",
            "course_ids": [],
        },
    ]
}


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "tracks.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_list_all_loads_tracks_from_yaml(tmp_path: Path) -> None:
    repo = YamlTrackRepository(_write(tmp_path, _CATALOG))

    tracks = repo.list_all()

    assert [t.track_id for t in tracks] == ["응용산업데이터공학트랙", "빅데이터트랙"]
    first = tracks[0]
    assert first.college_id == "IT공과대학"
    assert first.department_id == "산업시스템공학부"
    assert first.major_id == "산업시스템공학부"
    assert first.course_ids == ["CTE0029", "V070044"]
    # YAML 에 없는 필드는 모델 기본값.
    assert first.meta_text == ""
    assert first.meta_vector == []
    assert first.competencies == []
    assert first.tech_stacks == []


def test_find_by_track_ids_filters_and_ignores_unknown(tmp_path: Path) -> None:
    repo = YamlTrackRepository(_write(tmp_path, _CATALOG))

    tracks = repo.find_by_track_ids(["빅데이터트랙", "없는트랙"])

    assert [t.track_id for t in tracks] == ["빅데이터트랙"]


def test_missing_file_raises_catalog_error(tmp_path: Path) -> None:
    with pytest.raises(TrackCatalogError):
        YamlTrackRepository(tmp_path / "does_not_exist.yaml")


def test_top_level_not_mapping_raises(tmp_path: Path) -> None:
    with pytest.raises(TrackCatalogError):
        YamlTrackRepository(_write(tmp_path, ["not", "a", "mapping"]))


def test_missing_tracks_key_raises(tmp_path: Path) -> None:
    with pytest.raises(TrackCatalogError):
        YamlTrackRepository(_write(tmp_path, {"something_else": []}))


def test_invalid_entry_is_skipped(tmp_path: Path) -> None:
    catalog = {
        "tracks": [
            {"track_id": "온전한트랙"},  # 필수 필드 누락 → 검증 실패 스킵
            _CATALOG["tracks"][0],
        ]
    }
    repo = YamlTrackRepository(_write(tmp_path, catalog))

    tracks = repo.list_all()

    assert [t.track_id for t in tracks] == ["응용산업데이터공학트랙"]
