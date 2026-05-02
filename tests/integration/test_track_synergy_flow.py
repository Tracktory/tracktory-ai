"""트랙 시너지 노드 end-to-end 통합 테스트.

LangGraph state 입력 → 노드 호출 → state 부분 출력 검증.
``TrackRepository`` 는 ``MagicMock(spec=...)`` 로 in-memory 구현을 흉내내며,
실제 ``synergy.yaml`` 을 로드하여 가중치·임계값 정합도 함께 검증한다.

``@pytest.mark.integration`` 으로 표시 — ``pyproject.toml`` markers 등록과 정합.
CI 기본 실행에서 제외하려면 ``pytest -m "not integration"`` 사용.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tracktory.graph.models import Track
from tracktory.graph.nodes.track_synergy import TrackRepository, TrackSynergyNode

pytestmark = pytest.mark.integration

_DEFAULT_DIM = 1536
_REAL_SYNERGY_YAML = (
    Path(__file__).resolve().parents[2] / "src" / "tracktory" / "config" / "synergy.yaml"
)


def _meta(seed: int) -> list[float]:
    """deterministic seed 기반 L2-normalized 트랙 메타 벡터."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(_DEFAULT_DIM)
    norm = float(np.linalg.norm(raw))
    return (raw / norm).tolist() if norm > 0 else []


def _track(
    track_id: str,
    *,
    college_id: str,
    department_id: str,
    course_ids: list[str],
    tech_stacks: list[str],
    competencies: list[str],
    meta_seed: int,
) -> Track:
    return Track(
        college_id=college_id,
        department_id=department_id,
        major_id=department_id,
        track_id=track_id,
        track_name=track_id,
        course_ids=course_ids,
        meta_text="",
        meta_vector=_meta(meta_seed),
        competencies=competencies,
        tech_stacks=tech_stacks,
    )


def _build_node(tracks: list[Track]) -> TrackSynergyNode:
    repo = MagicMock(spec=TrackRepository)
    repo.list_all.return_value = tracks
    return TrackSynergyNode(track_repo=repo, config_path=_REAL_SYNERGY_YAML)


def _normalized_profile(college: str = "C1") -> dict[str, Any]:
    return {
        "admission_year": 2025,
        "college": college,
        "department": "컴퓨터공학부",
        "current_tracks": [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


def _state(college: str = "C1") -> dict[str, Any]:
    return {
        "job_candidates": [
            {
                "job_id": "j1",
                "job_name": "Backend",
                "tech_stacks": ["py", "sql"],
                "match_score": 0.7,
            }
        ],
        "normalized_profile": _normalized_profile(college=college),
    }


def test_full_pipeline_t1_cross_college_path() -> None:
    """단과대 cross 후보 존재 → 슬롯 3 = cross-college, fallback 미발생."""
    in_college = [
        _track(
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
        _track(
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
    node = _build_node(in_college + out_college)
    result = node(_state())

    assert result["slot3_fallback_triggered"] is False
    assert result["slot3_fallback_level"] is None
    assert len(result["primary_combos"]) == 2
    assert len(result["secondary_combos"]) == 5
    # 보조 슬롯의 첫 자리는 cross-college (예약 슬롯)
    assert result["secondary_combos"][0]["slot_type"] == "cross_college"


def test_full_pipeline_t2_fallback_path() -> None:
    """단과대 cross 0 + 학부 cross 존재 → fallback_level = 'T2'."""
    same_college_d1 = [
        _track(
            f"d1_{i}",
            college_id="C1",
            department_id="D1",
            course_ids=[f"d1c{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"d1_c{i}"],
            meta_seed=i,
        )
        for i in range(3)
    ]
    same_college_d2 = [
        _track(
            f"d2_{i}",
            college_id="C1",
            department_id="D2",
            course_ids=[f"d2c{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"d2_c{i}"],
            meta_seed=10 + i,
        )
        for i in range(3)
    ]
    node = _build_node(same_college_d1 + same_college_d2)
    result = node(_state())

    assert result["slot3_fallback_triggered"] is True
    assert result["slot3_fallback_level"] == "T2"
    assert len(result["primary_combos"]) == 2
    assert len(result["secondary_combos"]) == 5


def test_full_pipeline_mmr_overflow_path() -> None:
    """모든 트랙이 동일 학부 → cross 후보 0 → fallback_level = 'MMR'."""
    same_dept = [
        _track(
            f"d{i}",
            college_id="C1",
            department_id="D1",
            course_ids=[f"co{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"c{i}"],
            meta_seed=i,
        )
        for i in range(7)
    ]
    node = _build_node(same_dept)
    result = node(_state())

    assert result["slot3_fallback_triggered"] is True
    assert result["slot3_fallback_level"] == "MMR"
    assert len(result["primary_combos"]) == 2
    assert len(result["secondary_combos"]) == 5
    # MMR fallback 시 모든 보조 슬롯이 mmr 분류
    assert all(s["slot_type"] == "mmr" for s in result["secondary_combos"])


def test_full_pipeline_synthetic_47_tracks() -> None:
    """한성대 47 트랙 흉내 — 단과대 4 / 학부 8 / 트랙 47."""
    tracks: list[Track] = []
    track_idx = 0
    for college_idx in range(4):
        for dept_idx in range(2):
            for _ in range(6):
                if len(tracks) >= 47:
                    break
                tracks.append(
                    _track(
                        f"t{track_idx}",
                        college_id=f"C{college_idx}",
                        department_id=f"D{college_idx}_{dept_idx}",
                        course_ids=[f"c{track_idx}_{x}" for x in range(3)],
                        tech_stacks=(["py", "sql"] if track_idx % 2 == 0 else ["py", "go"]),
                        competencies=[f"comp{track_idx}"],
                        meta_seed=track_idx,
                    )
                )
                track_idx += 1
    assert len(tracks) == 47

    node = _build_node(tracks)
    result = node(_state(college="C0"))

    assert len(result["primary_combos"]) == 2
    assert len(result["secondary_combos"]) == 5
    assert all(p["slot_type"] == "primary" for p in result["primary_combos"])
    slot_types = [s["slot_type"] for s in result["secondary_combos"]]
    cross_count = slot_types.count("cross_college")
    mmr_count = slot_types.count("mmr")
    # 예약 슬롯은 최대 1, 나머지는 모두 MMR
    assert cross_count <= 1
    assert cross_count + mmr_count == 5
