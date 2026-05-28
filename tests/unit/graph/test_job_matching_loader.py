"""``_load_category_mapping`` 의 yaml 입력 분기 단위 테스트.

본 PR scope 의 missing line 분기 (빈 파일 / mapping 아닌 top-level / entry 의
``job_id`` 누락) 를 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tracktory.graph.nodes.job_matching import (
    _build_fallback_candidates,
    _load_category_mapping,
)


def test_load_returns_empty_dict_when_yaml_is_blank(tmp_path: Path) -> None:
    """빈 yaml 파일은 ``{}`` 로 안전 변환된다 (yaml.safe_load → None)."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert _load_category_mapping(path) == {}


def test_load_raises_when_top_level_is_not_mapping(tmp_path: Path) -> None:
    """top-level 이 mapping 이 아닌 경우 ValueError — 스키마 위반 조기 감지."""
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define a top-level mapping"):
        _load_category_mapping(path)


def test_fallback_skips_entry_without_job_id() -> None:
    """매핑 entry 에 ``job_id`` 가 없으면 조용히 건너뛴다 — 부분 손상 입력 견고성."""
    mapping = {
        "IT/인터넷": [
            {"rank": 1},  # job_id 누락 — skip
            {"job_id": "backend_developer", "job_name": "백엔드 개발자", "rank": 2},
        ]
    }
    candidates = _build_fallback_candidates("IT/인터넷", mapping, k=5)
    assert len(candidates) == 1
    assert candidates[0].job_id == "backend_developer"


def test_fallback_uses_rank_sentinel_when_rank_missing() -> None:
    """``rank`` 누락 entry 는 sentinel 로 인해 마지막으로 정렬된다."""
    mapping = {
        "테스트": [
            {"job_id": "no_rank", "job_name": "랭크 없음"},  # rank 누락
            {"job_id": "ranked", "job_name": "랭크 있음", "rank": 1},
        ]
    }
    candidates = _build_fallback_candidates("테스트", mapping, k=5)
    assert [c.job_id for c in candidates] == ["ranked", "no_rank"]
