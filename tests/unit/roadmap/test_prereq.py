"""선수과목 alias 정규화 모듈 단위 테스트.

실제 데이터 파일 (``data/processed/prereq_alias.json``) 은 별도 동기화 경로로 흐르며
본 테스트는 외부 I/O 없이 인메모리·임시 파일로 alias 해석과 cycle 감지 계약을 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tracktory.roadmap.prereq import (
    AliasMap,
    detect_alias_cycles,
    load_alias_map,
    resolve_alias,
)


def test_alias_map_preserves_entries() -> None:
    entries = {"java": "자바 프로그래밍", "전공 교과목": None}
    alias_map = AliasMap(entries=entries)
    assert alias_map.entries == entries


def test_alias_map_rejects_empty_key() -> None:
    with pytest.raises(ValidationError):
        AliasMap(entries={"": "자바 프로그래밍"})


def test_load_alias_map_reads_json_file(tmp_path: Path) -> None:
    path = tmp_path / "alias.json"
    path.write_text(
        json.dumps({"java": "자바 프로그래밍", "전공 교과목": None}, ensure_ascii=False),
        encoding="utf-8",
    )
    alias_map = load_alias_map(path)
    assert alias_map.entries == {"java": "자바 프로그래밍", "전공 교과목": None}


def test_load_alias_map_rejects_non_dict_top_level(tmp_path: Path) -> None:
    path = tmp_path / "alias.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_alias_map(path)


def test_load_alias_map_rejects_non_string_value(tmp_path: Path) -> None:
    path = tmp_path / "alias.json"
    path.write_text(json.dumps({"java": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_alias_map(path)


def test_resolve_alias_exact_match() -> None:
    alias_map = AliasMap(entries={"java": "자바 프로그래밍"})
    assert resolve_alias("java", alias_map) == "자바 프로그래밍"


def test_resolve_alias_pass_through_when_not_registered() -> None:
    alias_map = AliasMap(entries={"java": "자바 프로그래밍"})
    assert resolve_alias("선형대수", alias_map) == "선형대수"


def test_resolve_alias_returns_none_when_value_is_none() -> None:
    alias_map = AliasMap(entries={"전공 교과목": None})
    assert resolve_alias("전공 교과목", alias_map) is None


def test_resolve_alias_follows_chain() -> None:
    alias_map = AliasMap(entries={"java": "자바", "자바": "자바 프로그래밍"})
    assert resolve_alias("java", alias_map) == "자바 프로그래밍"


def test_resolve_alias_raises_when_chain_cycles() -> None:
    alias_map = AliasMap(entries={"A": "B", "B": "A"})
    with pytest.raises(RuntimeError):
        resolve_alias("A", alias_map)


def test_detect_alias_cycles_returns_empty_when_acyclic() -> None:
    alias_map = AliasMap(entries={"java": "자바 프로그래밍", "전공 교과목": None})
    assert detect_alias_cycles(alias_map) == []


def test_detect_alias_cycles_finds_self_loop() -> None:
    alias_map = AliasMap(entries={"A": "A"})
    cycles = detect_alias_cycles(alias_map)
    assert cycles == [["A"]]


def test_detect_alias_cycles_finds_two_cycle_without_duplicates() -> None:
    alias_map = AliasMap(entries={"A": "B", "B": "A"})
    cycles = detect_alias_cycles(alias_map)
    assert cycles == [["A", "B"]]


def test_detect_alias_cycles_handles_chain_into_cycle() -> None:
    alias_map = AliasMap(entries={"X": "A", "A": "B", "B": "A"})
    cycles = detect_alias_cycles(alias_map)
    assert cycles == [["A", "B"]]
