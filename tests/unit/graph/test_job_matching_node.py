"""``JobMatchingNode.__call__`` 진입점 통합 테스트.

직무 검색 boundary 는 ``FakeJobSearchClient`` 로 통제하여 외부 호출 없이
정상 / 임계값 미만 / 호출 실패 / 카테고리 미매핑 시나리오를 모두 검증한다.
실 ``synergy.yaml`` + ``category_to_jobs.yaml`` 을 정합 sanity check 으로 사용.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tracktory.graph.nodes.job_matching import JobMatchingNode
from tracktory.rag.job_search import JobSearchClient, RagSearchResult


def _profile(
    interests: list[str] | None = None,
    completed_courses: list[str] | None = None,
) -> dict[str, Any]:
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
        "completed_courses": completed_courses or [],
    }


def _build_node(
    client: JobSearchClient,
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> JobMatchingNode:
    return JobMatchingNode(
        job_search_client=client,
        config_path=real_synergy_yaml_path,
        category_mapping_path=real_category_mapping_path,
    )


# ---------------------------------------------------------------------------
# 입력 검증 skip 케이스
# ---------------------------------------------------------------------------


def test_node_skips_when_profile_text_missing(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    client = make_fake_job_search_client()
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)
    result = node({"normalized_profile": _profile()})
    assert result["trace"] == ["job_matching:skip"]
    assert "recommended_jobs" not in result


def test_node_skips_when_profile_text_empty(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """빈 문자열도 skip — None 과 동일 분기."""
    client = make_fake_job_search_client()
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)
    result = node({"profile_text": "", "normalized_profile": _profile()})
    assert result["trace"] == ["job_matching:skip"]


def test_node_skips_when_normalized_profile_missing(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    client = make_fake_job_search_client()
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)
    result = node({"profile_text": "AI 에 관심 있는 학생"})
    assert result["trace"] == ["job_matching:skip"]


# ---------------------------------------------------------------------------
# 정상 경로 (top score >= threshold)
# ---------------------------------------------------------------------------


def test_node_returns_top_k_in_order_when_above_threshold(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    results = [
        make_search_result("matched_job", score=0.85),
        make_search_result("second_job", score=0.42),
    ]
    client = make_fake_job_search_client(results=results)
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)

    result = node(
        {
            "profile_text": "AI 와 데이터 분석에 관심 있는 학생",
            "normalized_profile": _profile(),
        }
    )
    assert result["trace"] == ["job_matching:ok"]
    recommended = result["recommended_jobs"]
    assert len(recommended) == 2
    assert recommended[0]["job_id"] == "matched_job"
    assert recommended[0]["fallback_used"] is False
    assert recommended[0]["similarity"] == pytest.approx(0.85)
    assert recommended[1]["similarity"] == pytest.approx(0.42)


def test_node_passes_profile_text_and_top_k_to_client(
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """직무 검색 boundary 가 받는 인자가 ``profile_text`` 와 default top_k 인지 검증."""
    client = MagicMock(spec=JobSearchClient)
    client.rag_search_jobs.return_value = [make_search_result("only", score=0.9)]
    node = JobMatchingNode(
        job_search_client=client,
        config_path=real_synergy_yaml_path,
        category_mapping_path=real_category_mapping_path,
    )

    profile_text = "백엔드 개발에 흥미가 있는 학생"
    node({"profile_text": profile_text, "normalized_profile": _profile()})

    client.rag_search_jobs.assert_called_once_with(
        query=profile_text,
        top_k=3,  # synergy.yaml 의 job_matching.top_k.default
    )


def test_node_match_score_equals_similarity_without_boost(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """이수 과목이 없어 부스팅이 0 이면 ``match_score`` 와 ``similarity`` 가 일치한다."""
    client = make_fake_job_search_client(results=[make_search_result("target", score=0.7)])
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)

    result = node(
        {
            "profile_text": "AI 에 관심 있는 학생",
            "normalized_profile": _profile(),
        }
    )
    item = result["recommended_jobs"][0]
    assert item["match_score"] == item["similarity"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Fallback 경로
# ---------------------------------------------------------------------------


def test_node_triggers_fallback_when_top_score_below_threshold(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """상위 점수 < min_job_similarity → 카테고리 사전 매핑 fallback."""
    client = make_fake_job_search_client(
        results=[make_search_result("low_score_job", score=0.05)],
    )
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)

    result = node(
        {
            "profile_text": "관심사가 분명치 않은 학생",
            "normalized_profile": _profile(interests=["IT/인터넷"]),
        }
    )
    assert result["trace"] == ["job_matching:fallback_categorized"]
    recommended = result["recommended_jobs"]
    assert all(item["fallback_used"] is True for item in recommended)
    assert recommended[0]["job_id"] == "BE"
    assert recommended[0]["job_name"] == "백엔드 개발자"


def test_node_triggers_fallback_when_results_empty(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """빈 결과도 임계값 미만으로 간주 → fallback 분기."""
    client = make_fake_job_search_client(results=[])
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)

    result = node(
        {
            "profile_text": "프로필",
            "normalized_profile": _profile(interests=["IT/인터넷"]),
        }
    )
    assert result["trace"] == ["job_matching:fallback_categorized"]


def test_node_triggers_fallback_on_rag_search_error(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``RagSearchError`` raise 도 fallback 분기로 전환되어야 한다."""
    client = make_fake_job_search_client(raise_error=True)
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)

    with caplog.at_level(logging.WARNING, logger="tracktory.graph.nodes.job_matching"):
        result = node(
            {
                "profile_text": "프로필",
                "normalized_profile": _profile(interests=["IT/인터넷"]),
            }
        )

    assert result["trace"] == ["job_matching:fallback_categorized"]
    error_logs = [r for r in caplog.records if r.message == "job_matching_rag_search_error"]
    assert len(error_logs) == 1


def test_node_logger_info_called_once_on_fallback(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_fake_job_search_client(results=[make_search_result("low", score=0.05)])
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)

    with caplog.at_level(logging.INFO, logger="tracktory.graph.nodes.job_matching"):
        node(
            {
                "profile_text": "프로필",
                "normalized_profile": _profile(interests=["IT/인터넷"]),
            }
        )

    fallback_logs = [r for r in caplog.records if r.message == "job_matching_category_fallback"]
    assert len(fallback_logs) == 1


def test_node_returns_empty_with_warning_when_category_unmapped(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_fake_job_search_client(results=[make_search_result("low", score=0.05)])
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)

    with caplog.at_level(logging.WARNING, logger="tracktory.graph.nodes.job_matching"):
        result = node(
            {
                "profile_text": "프로필",
                "normalized_profile": _profile(interests=["미존재카테고리"]),
            }
        )

    assert result["trace"] == ["job_matching:fallback_unmapped"]
    assert result["recommended_jobs"] == []
    unmapped_logs = [r for r in caplog.records if r.message == "job_matching_category_unmapped"]
    assert len(unmapped_logs) == 1


def test_node_no_logger_info_on_normal_path(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_fake_job_search_client(results=[make_search_result("ok", score=0.9)])
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)

    with caplog.at_level(logging.INFO, logger="tracktory.graph.nodes.job_matching"):
        node(
            {
                "profile_text": "프로필",
                "normalized_profile": _profile(),
            }
        )

    fallback_logs = [r for r in caplog.records if "fallback" in r.message]
    assert fallback_logs == []


# ---------------------------------------------------------------------------
# 이수 과목 부스팅 (정상 경로 후처리)
# ---------------------------------------------------------------------------


def test_completed_courses_boost_score_on_normal_path(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """이수 과목이 직무 토큰과 겹치면 점수가 raw 검색 점수보다 높아진다."""

    def _result() -> RagSearchResult:
        return make_search_result(
            "backend", score=0.6, tech_stacks=["자료구조"], competency_tags=[]
        )

    node_no_courses = _build_node(
        make_fake_job_search_client(results=[_result()]),
        real_synergy_yaml_path,
        real_category_mapping_path,
    )
    without = node_no_courses(
        {"profile_text": "백엔드", "normalized_profile": _profile(completed_courses=[])}
    )

    node_with_courses = _build_node(
        make_fake_job_search_client(results=[_result()]),
        real_synergy_yaml_path,
        real_category_mapping_path,
    )
    with_courses = node_with_courses(
        {
            "profile_text": "백엔드",
            "normalized_profile": _profile(completed_courses=["자료구조"]),
        }
    )

    raw_score = without["recommended_jobs"][0]["match_score"]
    boosted_score = with_courses["recommended_jobs"][0]["match_score"]
    assert raw_score == pytest.approx(0.6)
    assert boosted_score > raw_score


def test_completed_courses_without_overlap_keep_raw_score(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """겹치지 않는 이수 과목은 점수를 바꾸지 않는다."""
    client = make_fake_job_search_client(
        results=[make_search_result("backend", score=0.6, tech_stacks=["go"])]
    )
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)
    result = node(
        {
            "profile_text": "백엔드",
            "normalized_profile": _profile(completed_courses=["국어"]),
        }
    )
    assert result["recommended_jobs"][0]["match_score"] == pytest.approx(0.6)


def test_completed_courses_do_not_rescue_below_threshold_match(
    make_fake_job_search_client: Callable[..., JobSearchClient],
    make_search_result: Callable[..., RagSearchResult],
    real_synergy_yaml_path: Path,
    real_category_mapping_path: Path,
) -> None:
    """임계값 미만 매칭은 이수 과목이 겹쳐도 fallback 으로 전환된다 (부스팅은 raw 점수 이후)."""
    client = make_fake_job_search_client(
        results=[make_search_result("low", score=0.05, tech_stacks=["자료구조"])]
    )
    node = _build_node(client, real_synergy_yaml_path, real_category_mapping_path)
    result = node(
        {
            "profile_text": "프로필",
            "normalized_profile": _profile(interests=["IT/인터넷"], completed_courses=["자료구조"]),
        }
    )
    assert result["trace"] == ["job_matching:fallback_categorized"]
    assert all(item["fallback_used"] is True for item in result["recommended_jobs"])
