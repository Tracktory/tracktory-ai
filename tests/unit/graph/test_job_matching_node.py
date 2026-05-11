"""``JobMatchingNode.__call__`` 진입점 통합 테스트.

deterministic cosine 통제를 위해 직무 임베딩을 표준 단위 벡터 (e_i) 로 채운다.
실 ``synergy.yaml`` + ``category_to_jobs.yaml`` 을 정합 sanity check 으로 사용.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from tracktory.graph.nodes.job_matching import JobMatchingNode
from tracktory.rag.job_index import InMemoryJobIndex, Job

_DIM = 1536


def _unit_vector(index: int, dim: int = _DIM) -> list[float]:
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


def _job(job_id: str, vector: list[float]) -> Job:
    return Job(job_id=job_id, job_name=job_id, job_vector=vector)


def _profile(interests: list[str] | None = None) -> dict[str, Any]:
    return {
        "admission_year": 2025,
        "college": "C1",
        "department": "컴퓨터공학부",
        "current_tracks": [],
        "interests": interests or ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


def _build_node(
    jobs: list[Job],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> JobMatchingNode:
    return JobMatchingNode(
        job_index=InMemoryJobIndex(jobs=jobs),
        config_path=real_synergy_yaml_path,
        category_mapping_path=real_category_mapping_path,
    )


# ---------------------------------------------------------------------------
# 입력 검증 skip 케이스
# ---------------------------------------------------------------------------


def test_node_skips_when_profile_vector_missing(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    node = _build_node([], real_synergy_yaml_path, real_category_mapping_path)
    result = node({"normalized_profile": _profile()})
    assert result["trace"] == ["job_matching:skip"]
    assert "recommended_jobs" not in result


def test_node_skips_when_normalized_profile_missing(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    node = _build_node([], real_synergy_yaml_path, real_category_mapping_path)
    result = node({"profile_vector": _unit_vector(0)})
    assert result["trace"] == ["job_matching:skip"]


def test_node_skips_when_job_index_is_empty(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    node = _build_node([], real_synergy_yaml_path, real_category_mapping_path)
    result = node(
        {
            "profile_vector": _unit_vector(0),
            "normalized_profile": _profile(),
        }
    )
    assert result["trace"] == ["job_matching:skip"]
    assert "empty job index" in result["errors"][0]


# ---------------------------------------------------------------------------
# 정상 경로 (max similarity >= threshold)
# ---------------------------------------------------------------------------


def test_node_returns_top_k_descending_when_above_threshold(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    matched = _job("matched_job", _unit_vector(0))
    partial = _job("partial_job", _unit_vector(1))  # cosine(e0, e1) = 0 → clip 0
    node = _build_node([matched, partial], real_synergy_yaml_path, real_category_mapping_path)

    result = node(
        {
            "profile_vector": _unit_vector(0),
            "normalized_profile": _profile(),
        }
    )
    assert result["trace"] == ["job_matching:ok"]
    recommended = result["recommended_jobs"]
    assert len(recommended) == 2
    assert recommended[0]["job_id"] == "matched_job"
    assert recommended[0]["fallback_used"] is False
    assert recommended[0]["similarity"] == pytest.approx(1.0)
    assert recommended[1]["similarity"] == pytest.approx(0.0)


def test_node_no_diversity_constraint_returns_duplicates(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """동일 vector 두 직무 모두 top-k 에 포함 — 다양성은 후속 단계 책임."""
    twin_a = _job("twin_a", _unit_vector(0))
    twin_b = _job("twin_b", _unit_vector(0))
    other = _job("other_job", _unit_vector(1))
    node = _build_node([twin_a, twin_b, other], real_synergy_yaml_path, real_category_mapping_path)

    result = node(
        {
            "profile_vector": _unit_vector(0),
            "normalized_profile": _profile(),
        }
    )
    assert result["trace"] == ["job_matching:ok"]
    job_ids = {item["job_id"] for item in result["recommended_jobs"]}
    assert "twin_a" in job_ids
    assert "twin_b" in job_ids


def test_node_match_score_and_similarity_are_equal_in_normal_path(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """``match_score`` 와 ``similarity`` 는 같은 값으로 채워진다 (다운스트림 호환)."""
    target = _job("target", _unit_vector(0))
    node = _build_node([target], real_synergy_yaml_path, real_category_mapping_path)

    result = node(
        {
            "profile_vector": _unit_vector(0),
            "normalized_profile": _profile(),
        }
    )
    item = result["recommended_jobs"][0]
    assert item["match_score"] == item["similarity"]


# ---------------------------------------------------------------------------
# Fallback 경로 (max similarity < threshold)
# ---------------------------------------------------------------------------


def _it_jobs() -> list[Job]:
    """``IT/인터넷`` 카테고리 매핑의 직무 식별자를 모두 보유한 인덱스 시드."""
    return [
        _job("backend_developer", _unit_vector(1)),
        _job("frontend_developer", _unit_vector(2)),
        _job("data_engineer", _unit_vector(3)),
        _job("devops_engineer", _unit_vector(4)),
        _job("ml_engineer", _unit_vector(5)),
    ]


def test_node_triggers_fallback_when_max_similarity_below_threshold(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    node = _build_node(_it_jobs(), real_synergy_yaml_path, real_category_mapping_path)

    # profile = e0, 모든 직무 vector 가 e1~e5 → cosine 모두 0 → fallback
    result = node(
        {
            "profile_vector": _unit_vector(0),
            "normalized_profile": _profile(interests=["IT/인터넷"]),
        }
    )
    assert result["trace"] == ["job_matching:fallback_categorized"]
    recommended = result["recommended_jobs"]
    assert all(item["fallback_used"] is True for item in recommended)
    # 매핑의 rank 1 = backend_developer 가 첫 자리
    assert recommended[0]["job_id"] == "backend_developer"


def test_node_logger_info_called_once_on_fallback(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    node = _build_node(_it_jobs(), real_synergy_yaml_path, real_category_mapping_path)

    with caplog.at_level(logging.INFO, logger="tracktory.graph.nodes.job_matching"):
        node(
            {
                "profile_vector": _unit_vector(0),
                "normalized_profile": _profile(interests=["IT/인터넷"]),
            }
        )

    fallback_logs = [r for r in caplog.records if r.message == "job_matching_category_fallback"]
    assert len(fallback_logs) == 1


def test_node_returns_empty_with_warning_when_category_unmapped(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    node = _build_node(_it_jobs(), real_synergy_yaml_path, real_category_mapping_path)

    with caplog.at_level(logging.WARNING, logger="tracktory.graph.nodes.job_matching"):
        result = node(
            {
                "profile_vector": _unit_vector(0),
                "normalized_profile": _profile(interests=["미존재카테고리"]),
            }
        )

    assert result["trace"] == ["job_matching:fallback_unmapped"]
    assert result["recommended_jobs"] == []
    unmapped_logs = [r for r in caplog.records if r.message == "job_matching_category_unmapped"]
    assert len(unmapped_logs) == 1


def test_node_no_logger_info_on_normal_path(
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    node = _build_node(
        [_job("matched", _unit_vector(0))],
        real_synergy_yaml_path,
        real_category_mapping_path,
    )

    with caplog.at_level(logging.INFO, logger="tracktory.graph.nodes.job_matching"):
        node(
            {
                "profile_vector": _unit_vector(0),
                "normalized_profile": _profile(),
            }
        )

    fallback_logs = [r for r in caplog.records if "fallback" in r.message]
    assert fallback_logs == []
