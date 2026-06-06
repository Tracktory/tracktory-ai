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

from tracktory.graph.models import JobCandidate, SynergyConfig, Track
from tracktory.graph.nodes.track_synergy import (
    TrackRepository,
    TrackSynergyNode,
    _generate_combos,
    _synergy_score,
)

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
        "recommended_jobs": [
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


def test_primary_stays_same_college_when_cross_track_is_job_irrelevant() -> None:
    """주 추천은 단과대 내부로 닫히고, 직무에 거의 무관한 타 단과대 트랙은 보조로 밀린다.

    타 단과대 트랙(out0)은 직무 토큰 1 개만 공급해, 합집합 분모 시절이라면 완전 분업
    (comp=1.0)으로 둔갑해 주 추천을 점령했을 후보다. 같은 단과대 우선 선택 + 전체 직무
    토큰 분모로, 주 추천 두 자리는 모두 사용자 단과대(C1) 내부 조합이어야 하며 out0 은
    들어오면 안 된다. 그래야 이 조합에서 파생되는 학습 로드맵도 직무 연관 과목으로 채워진다.
    """
    in_college = [
        _track(
            "in0",
            college_id="C1",
            department_id="D1",
            course_ids=["in_co0"],
            tech_stacks=["py", "sql", "java"],
            competencies=["c_in0"],
            meta_seed=1,
        ),
        _track(
            "in1",
            college_id="C1",
            department_id="D1",
            course_ids=["in_co1"],
            tech_stacks=["spring boot", "docker"],
            competencies=["c_in1"],
            meta_seed=2,
        ),
        _track(
            "in2",
            college_id="C1",
            department_id="D2",
            course_ids=["in_co2"],
            tech_stacks=["aws"],
            competencies=["c_in2"],
            meta_seed=3,
        ),
        _track(
            "in3",
            college_id="C1",
            department_id="D2",
            course_ids=["in_co3"],
            tech_stacks=["aws"],
            competencies=["c_in3"],
            meta_seed=4,
        ),
    ]
    # 타 단과대 트랙 — 직무 토큰 'go' 하나만 공급 (사실상 무관).
    cross_irrelevant = _track(
        "out0",
        college_id="C2",
        department_id="D9",
        course_ids=["out_co0"],
        tech_stacks=["go"],
        competencies=["c_out0"],
        meta_seed=100,
    )
    node = _build_node([*in_college, cross_irrelevant])
    state = {
        "recommended_jobs": [
            {
                "job_id": "j1",
                "job_name": "Backend",
                "tech_stacks": ["py", "sql", "java", "spring boot", "docker", "aws", "go"],
                "match_score": 0.8,
            }
        ],
        "normalized_profile": _normalized_profile(college="C1"),
    }
    result = node(state)

    primary = result["primary_combos"]
    assert len(primary) == 2
    for combo in primary:
        assert combo["combo"]["track_a"]["college_id"] == "C1"
        assert combo["combo"]["track_b"]["college_id"] == "C1"
    primary_track_ids = {
        tid
        for combo in primary
        for tid in (combo["combo"]["track_a"]["track_id"], combo["combo"]["track_b"]["track_id"])
    }
    assert "out0" not in primary_track_ids


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

    # 전체 후보 조합의 시너지 점수가 한 값으로 뭉치지 않고 변별된다 — 순위가 점수 근거로
    # 정렬됨을 보장한다 (complementarity 포화 시절엔 거의 모든 조합이 동률이라 순위가
    # combo_key tie-break 로 임의 결정됐다). 상위 k 는 동률 최댓값일 수 있어, 분포는
    # 출력 7 개가 아니라 전체 후보에서 본다.
    cfg = SynergyConfig.load_from_yaml(_REAL_SYNERGY_YAML)
    jobs = [JobCandidate.model_validate(j) for j in _state(college="C0")["recommended_jobs"]]
    combos = _generate_combos(tracks, "C0", [])
    distinct_scores = {round(_synergy_score(c, jobs, cfg.weights), 4) for c in combos}
    assert len(distinct_scores) > 1
