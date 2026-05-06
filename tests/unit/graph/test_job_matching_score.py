"""직무 매칭 노드의 순수 계산 함수 단위 테스트.

module-level 함수만 직접 호출하여 외부 I/O / 의존성 주입 없이 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tracktory.graph.nodes.job_matching import (
    _build_fallback_candidates,
    _cosine_similarity,
    _load_category_mapping,
    _top_k_by_similarity,
)
from tracktory.rag.job_index import Job

_DIM = 8


def _unit_vector(index: int, dim: int = _DIM) -> list[float]:
    """축 방향 단위 벡터 — deterministic cosine 통제용."""
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


def _make_job(job_id: str, vector: list[float]) -> Job:
    return Job(
        job_id=job_id,
        job_name=job_id,
        job_vector=vector,
    )


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_returns_one_for_identical_unit_vector() -> None:
    assert _cosine_similarity(_unit_vector(0), _unit_vector(0)) == pytest.approx(1.0)


def test_cosine_returns_zero_for_orthogonal_unit_vectors() -> None:
    assert _cosine_similarity(_unit_vector(0), _unit_vector(1)) == pytest.approx(0.0)


def test_cosine_returns_negative_for_opposite_unit_vectors() -> None:
    """L2 normalized 가정에서 반대 방향 단위 벡터는 내적이 -1.0."""
    opposite = [-v for v in _unit_vector(0)]
    assert _cosine_similarity(_unit_vector(0), opposite) == pytest.approx(-1.0)


def test_cosine_returns_zero_when_dim_mismatch() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_cosine_returns_zero_when_either_vector_is_empty() -> None:
    assert _cosine_similarity([], _unit_vector(0)) == 0.0
    assert _cosine_similarity(_unit_vector(0), []) == 0.0


# ---------------------------------------------------------------------------
# _top_k_by_similarity
# ---------------------------------------------------------------------------


def test_top_k_sorts_descending_by_similarity() -> None:
    profile = _unit_vector(0)
    jobs = [
        _make_job("low", _unit_vector(1)),  # cosine = 0
        _make_job("high", _unit_vector(0)),  # cosine = 1
    ]
    result = _top_k_by_similarity(jobs, profile, k=2)
    assert [job.job_id for job, _ in result] == ["high", "low"]
    assert [score for _, score in result] == [pytest.approx(1.0), pytest.approx(0.0)]


def test_top_k_clips_negative_cosine_to_zero() -> None:
    """음수 cosine 은 ``similarity`` 필드 제약 [0, 1] 만족을 위해 0 으로 clip 된다."""
    profile = _unit_vector(0)
    jobs = [_make_job("opp", [-v for v in _unit_vector(0)])]  # raw cosine = -1.0
    result = _top_k_by_similarity(jobs, profile, k=1)
    assert result[0][1] == pytest.approx(0.0)


def test_top_k_returns_fewer_when_k_larger_than_pool() -> None:
    profile = _unit_vector(0)
    jobs = [_make_job("only", _unit_vector(0))]
    result = _top_k_by_similarity(jobs, profile, k=10)
    assert len(result) == 1


def test_top_k_breaks_ties_by_job_id_alphabetical() -> None:
    profile = _unit_vector(0)
    # 두 직무 모두 직교 → cosine 0 동점 → job_id 순
    jobs = [
        _make_job("z_job", _unit_vector(1)),
        _make_job("a_job", _unit_vector(2)),
    ]
    result = _top_k_by_similarity(jobs, profile, k=2)
    assert [job.job_id for job, _ in result] == ["a_job", "z_job"]


# ---------------------------------------------------------------------------
# _load_category_mapping
# ---------------------------------------------------------------------------


def test_load_category_mapping_returns_dict_for_valid_yaml(tmp_path: Path) -> None:
    yaml_text = "IT/인터넷:\n  - { job_id: backend_developer, rank: 1 }\n"
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    result = _load_category_mapping(path)
    assert "IT/인터넷" in result
    assert result["IT/인터넷"][0]["job_id"] == "backend_developer"


def test_load_category_mapping_returns_empty_dict_for_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert _load_category_mapping(path) == {}


def test_load_category_mapping_raises_when_top_level_is_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_category_mapping(path)


# ---------------------------------------------------------------------------
# _build_fallback_candidates
# ---------------------------------------------------------------------------


def test_fallback_returns_empty_when_category_unmapped() -> None:
    job_lookup = {"backend_developer": _make_job("backend_developer", _unit_vector(0))}
    assert _build_fallback_candidates("미존재카테고리", {}, k=3, job_lookup=job_lookup) == []


def test_fallback_returns_candidates_sorted_by_rank() -> None:
    mapping = {
        "IT/인터넷": [
            {"job_id": "j_b", "rank": 2},
            {"job_id": "j_a", "rank": 1},
        ]
    }
    job_lookup = {
        "j_a": _make_job("j_a", _unit_vector(0)),
        "j_b": _make_job("j_b", _unit_vector(1)),
    }
    result = _build_fallback_candidates("IT/인터넷", mapping, k=2, job_lookup=job_lookup)
    assert [c.job_id for c in result] == ["j_a", "j_b"]
    assert all(c.fallback_used is True for c in result)
    assert all(c.similarity == 0.0 for c in result)


def test_fallback_skips_jobs_missing_from_index() -> None:
    """매핑된 직무가 인덱스에 없으면 조용히 건너뛰어 가용한 만큼만 반환한다."""
    mapping = {
        "IT/인터넷": [
            {"job_id": "in_index", "rank": 1},
            {"job_id": "missing_from_index", "rank": 2},
        ]
    }
    job_lookup = {"in_index": _make_job("in_index", _unit_vector(0))}
    result = _build_fallback_candidates("IT/인터넷", mapping, k=5, job_lookup=job_lookup)
    assert [c.job_id for c in result] == ["in_index"]


def test_fallback_truncates_to_k() -> None:
    mapping = {"IT/인터넷": [{"job_id": f"j{i}", "rank": i} for i in range(5)]}
    job_lookup = {f"j{i}": _make_job(f"j{i}", _unit_vector(0)) for i in range(5)}
    result = _build_fallback_candidates("IT/인터넷", mapping, k=2, job_lookup=job_lookup)
    assert len(result) == 2
    assert [c.job_id for c in result] == ["j0", "j1"]
