"""``TrackSynergyNode.__call__`` 진입점 통합 테스트.

``TrackRepository`` 를 ``MagicMock(spec=...)`` 로 교체하여 외부 I/O 없이 검증한다.
``input_normalize.py`` / ``profile_embed.py`` 의 skip 패턴 정합 + logger 단일 위치 호출 검증.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

from tracktory.graph.nodes.track_synergy import (
    TrackRepository,
    TrackSynergyNode,
)


def _build_node(
    tracks: list,
    real_synergy_yaml_path: Path,
) -> tuple[TrackSynergyNode, MagicMock]:
    repo = MagicMock(spec=TrackRepository)
    repo.list_all.return_value = tracks
    node = TrackSynergyNode(track_repo=repo, config_path=real_synergy_yaml_path)
    return node, repo


def _normalized_profile(
    college: str = "C1",
    department: str = "D1",
    current_tracks: list[str] | None = None,
) -> dict:
    return {
        "admission_year": 2025,
        "college": college,
        "department": department,
        "current_tracks": current_tracks or [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


def test_node_skips_when_recommended_jobs_missing(real_synergy_yaml_path) -> None:
    node, _ = _build_node(tracks=[], real_synergy_yaml_path=real_synergy_yaml_path)
    result = node({"normalized_profile": _normalized_profile()})
    assert result["trace"] == ["track_synergy:skip"]
    assert "primary_combos" not in result


def test_node_skips_when_normalized_profile_missing(real_synergy_yaml_path) -> None:
    node, _ = _build_node(tracks=[], real_synergy_yaml_path=real_synergy_yaml_path)
    result = node({"recommended_jobs": [{"job_id": "j1", "job_name": "x", "match_score": 0.5}]})
    assert result["trace"] == ["track_synergy:skip"]


def test_node_skips_when_department_missing(real_synergy_yaml_path) -> None:
    node, _ = _build_node(tracks=[], real_synergy_yaml_path=real_synergy_yaml_path)
    profile = _normalized_profile()
    del profile["department"]
    result = node(
        {
            "recommended_jobs": [{"job_id": "j1", "job_name": "x", "match_score": 0.5}],
            "normalized_profile": profile,
        }
    )
    assert result["trace"] == ["track_synergy:skip"]


def test_node_returns_seven_combos_when_data_complete(make_track, real_synergy_yaml_path) -> None:
    """주 추천 2 + 보조 추천 5 = 7 슬롯이 모두 채워진다."""
    in_college = [
        make_track(
            f"in{i}",
            college_id="C1",
            department_id="D1",
            course_ids=[f"co{i}"],
            tech_stacks=["py", "sql"],
            competencies=["c1"],
            meta_seed=i,
        )
        for i in range(4)
    ]
    out_college = [
        make_track(
            f"out{i}",
            college_id="C2",
            department_id="D2",
            course_ids=[f"out_co{i}"],
            tech_stacks=["go"],
            competencies=["c2"],
            meta_seed=100 + i,
        )
        for i in range(4)
    ]
    node, repo = _build_node(
        tracks=in_college + out_college, real_synergy_yaml_path=real_synergy_yaml_path
    )

    result = node(
        {
            "recommended_jobs": [
                {
                    "job_id": "j1",
                    "job_name": "Backend",
                    "tech_stacks": ["py", "sql", "go"],
                    "match_score": 0.7,
                }
            ],
            "normalized_profile": _normalized_profile(college="C1"),
        }
    )
    assert result["trace"][0] == "track_synergy:ok"
    assert len(result["primary_combos"]) == 2
    assert len(result["secondary_combos"]) == 5
    repo.list_all.assert_called_once()


def test_node_logger_info_called_once_on_t2_fallback(
    make_track, real_synergy_yaml_path, caplog
) -> None:
    """학부 cross-dept fallback 발생 시 logger.info 가 정확히 1회 호출된다."""
    # 모든 트랙이 같은 단과대 + 다른 학부 → T1 cross 0 → T2 fallback
    same_college_d1 = [
        make_track(
            f"d1_{i}",
            college_id="C1",
            department_id="D1",
            course_ids=[f"co{i}"],
            tech_stacks=["py"],
            competencies=["c1"],
            meta_seed=i,
        )
        for i in range(3)
    ]
    same_college_d2 = [
        make_track(
            f"d2_{i}",
            college_id="C1",
            department_id="D2",
            course_ids=[f"d2c{i}"],
            tech_stacks=["sql"],
            competencies=["c2"],
            meta_seed=10 + i,
        )
        for i in range(3)
    ]
    node, _ = _build_node(
        tracks=same_college_d1 + same_college_d2,
        real_synergy_yaml_path=real_synergy_yaml_path,
    )

    with caplog.at_level(logging.INFO, logger="tracktory.graph.nodes.track_synergy"):
        result = node(
            {
                "recommended_jobs": [
                    {
                        "job_id": "j1",
                        "job_name": "Backend",
                        "tech_stacks": ["py", "sql"],
                        "match_score": 0.7,
                    }
                ],
                "normalized_profile": _normalized_profile(college="C1"),
            }
        )

    assert result["slot3_fallback_triggered"] is True
    fallback_logs = [
        r for r in caplog.records if r.message == "track_synergy_cross_college_fallback"
    ]
    assert len(fallback_logs) == 1


def test_node_no_logger_info_on_normal_path(make_track, real_synergy_yaml_path, caplog) -> None:
    """정상 cross-college 경로 (T1 채택) 에서는 logger.info 가 호출되지 않는다."""
    # 두 단과대 트랙 모두 jobs.tech_stacks 를 충분히 커버 + 각 트랙 competency 가 unique
    # → 모든 조합 synergy 가 임계값 0.3 충분히 상회 → cross-college 정상 채택
    in_college = [
        make_track(
            f"in{i}",
            college_id="C1",
            department_id="D1",
            course_ids=[f"in_co{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"in_c{i}"],
            meta_seed=i,
        )
        for i in range(3)
    ]
    out_college = [
        make_track(
            f"out{i}",
            college_id="C2",
            department_id="D2",
            course_ids=[f"out_co{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"out_c{i}"],
            meta_seed=100 + i,
        )
        for i in range(3)
    ]
    node, _ = _build_node(
        tracks=in_college + out_college, real_synergy_yaml_path=real_synergy_yaml_path
    )

    with caplog.at_level(logging.INFO, logger="tracktory.graph.nodes.track_synergy"):
        result = node(
            {
                "recommended_jobs": [
                    {
                        "job_id": "j1",
                        "job_name": "Backend",
                        "tech_stacks": ["py", "sql"],
                        "match_score": 0.7,
                    }
                ],
                "normalized_profile": _normalized_profile(college="C1"),
            }
        )

    assert result["slot3_fallback_triggered"] is False
    fallback_logs = [
        r for r in caplog.records if r.message == "track_synergy_cross_college_fallback"
    ]
    assert len(fallback_logs) == 0
