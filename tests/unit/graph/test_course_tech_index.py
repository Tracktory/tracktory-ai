"""``load_course_tech_index`` / ``resolve_course_tokens`` 순수 헬퍼 단위 테스트.

이수 과목 *이름* → 직무 기술 *토큰* 다리의 색인 로딩·이름 정규화·중복 합집합·
미보유 처리와, 색인 lookup 의 이름 정합·미지 과목 무영향을 외부 I/O 없이
``tmp_path`` 기반으로 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tracktory.graph.course_tech import (
    load_course_tech_index,
    resolve_course_tokens,
)


def _write_catalog(tmp_path: Path, courses: object) -> Path:
    path = tmp_path / "courses.yaml"
    path.write_text(yaml.safe_dump({"courses": courses}, allow_unicode=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_course_tech_index
# ---------------------------------------------------------------------------


def test_index_normalizes_course_name_key(tmp_path: Path) -> None:
    """키는 ``course_name.strip().casefold()`` 로 정규화된다."""
    path = _write_catalog(
        tmp_path,
        [{"course_id": "C1", "course_name": "  데이터베이스 ", "tech_stacks": ["MySQL"]}],
    )
    index = load_course_tech_index(path)
    assert index == {"데이터베이스": ["MySQL"]}


def test_index_unions_duplicate_names_dedup_preserve_order(tmp_path: Path) -> None:
    """같은 정규화 이름은 토큰을 합집합으로 묶되 중복 제거·최초 순서 보존."""
    path = _write_catalog(
        tmp_path,
        [
            {"course_id": "C1", "course_name": "프로그래밍", "tech_stacks": ["Java", "Spring"]},
            {"course_id": "C2", "course_name": "프로그래밍", "tech_stacks": ["Spring", "MySQL"]},
        ],
    )
    index = load_course_tech_index(path)
    assert index == {"프로그래밍": ["Java", "Spring", "MySQL"]}


def test_index_skips_missing_or_empty_tech_stacks(tmp_path: Path) -> None:
    """``tech_stacks`` 누락·빈 과목은 색인에 키를 만들지 않는다."""
    path = _write_catalog(
        tmp_path,
        [
            {"course_id": "C1", "course_name": "토큰없음"},
            {"course_id": "C2", "course_name": "빈리스트", "tech_stacks": []},
            {"course_id": "C3", "course_name": "공백토큰", "tech_stacks": ["  "]},
            {"course_id": "C4", "course_name": "정상", "tech_stacks": ["Python"]},
        ],
    )
    index = load_course_tech_index(path)
    assert index == {"정상": ["Python"]}


def test_index_empty_catalog_returns_empty(tmp_path: Path) -> None:
    """과목이 없으면 빈 색인."""
    path = _write_catalog(tmp_path, [])
    assert load_course_tech_index(path) == {}


def test_index_none_yaml_returns_empty(tmp_path: Path) -> None:
    """빈 yaml (``None``) 은 빈 색인으로 관대 처리."""
    path = tmp_path / "courses.yaml"
    path.write_text("", encoding="utf-8")
    assert load_course_tech_index(path) == {}


def test_index_raises_when_top_level_not_courses_mapping(tmp_path: Path) -> None:
    """최상위가 ``courses`` 리스트를 담은 mapping 이 아니면 ValueError."""
    path = tmp_path / "courses.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(ValueError, match="courses"):
        load_course_tech_index(path)


def test_index_raises_when_courses_key_missing(tmp_path: Path) -> None:
    path = tmp_path / "courses.yaml"
    path.write_text(yaml.safe_dump({"other": []}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="courses"):
        load_course_tech_index(path)


# ---------------------------------------------------------------------------
# resolve_course_tokens
# ---------------------------------------------------------------------------


def test_resolve_unions_tokens_across_courses() -> None:
    """여러 이수 과목의 토큰을 합집합으로 묶되 중복 제거·최초 순서 보존."""
    index = {"데이터베이스": ["MySQL", "SQL"], "웹프로그래밍": ["Spring", "MySQL"]}
    tokens = resolve_course_tokens(["데이터베이스", "웹프로그래밍"], index)
    assert tokens == ["MySQL", "SQL", "Spring"]


def test_resolve_unknown_name_contributes_nothing() -> None:
    """색인에 없는 과목 이름은 토큰을 기여하지 않는다."""
    index = {"데이터베이스": ["MySQL"]}
    tokens = resolve_course_tokens(["없는과목", "데이터베이스"], index)
    assert tokens == ["MySQL"]


def test_resolve_robust_to_whitespace_and_casing() -> None:
    """이름 lookup 은 입력의 공백·대소문자 차이를 흡수한다."""
    index = {"database": ["MySQL"]}
    tokens = resolve_course_tokens(["  DataBase  "], index)
    assert tokens == ["MySQL"]


def test_resolve_empty_input_returns_empty() -> None:
    assert resolve_course_tokens([], {"데이터베이스": ["MySQL"]}) == []


def test_resolve_empty_index_returns_empty() -> None:
    assert resolve_course_tokens(["데이터베이스"], {}) == []
