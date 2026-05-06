"""``_sim_4tier`` 5항 합산 + ``_mmr_select`` 의 synergy/다양성 균형 검증.

``SynergyConfig.SimilarityConfig`` 의 단조 제약 violation + single-track 학과
(``major_id == department_id``) degenerate 케이스도 검증한다.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tracktory.graph.models import RankedCombo
from tracktory.graph.nodes.track_synergy import (
    _mmr_select,
    _ScoredCombo,
    _sim_4tier,
)


def test_sim_4tier_full_match_returns_max_weighted_sum(
    make_track, make_combo, make_synergy_config
) -> None:
    """완전 동일 조합 (모든 tier + 메타) 의 sim_4tier 는 가중치 합 + 메타 cos(=1) 이다."""
    a = make_track("a", college_id="C1", department_id="D1", major_id="M1", meta_seed=42)
    b = make_track("b", college_id="C1", department_id="D1", major_id="M1", meta_seed=42)
    same_combo = make_combo(a, b)
    cfg = make_synergy_config()
    # T1/T2/T3 는 두 조합이 동일하므로 모두 1, course_overlap 은 빈 set 이므로 0,
    # meta cos 는 1 (둘 다 같은 seed).
    sim = _sim_4tier(same_combo, same_combo, cfg.similarity)
    expected = (
        cfg.similarity.w_college
        + cfg.similarity.w_department
        + cfg.similarity.w_track
        + cfg.similarity.w_meta * 1.0
    )
    assert sim == pytest.approx(expected, abs=1e-9)


def test_sim_4tier_monotone_constraint_violation_raises(make_synergy_config) -> None:
    """w_college < w_department 같은 단조 제약 위배는 ValidationError 로 fail-fast."""
    with pytest.raises(ValidationError):
        make_synergy_config(similarity={"w_college": 0.1, "w_department": 0.5, "w_track": 0.05})


def test_sim_4tier_handles_degenerate_major_id_equal_department(
    make_track, make_combo, make_synergy_config
) -> None:
    """single-track 학과 (major_id == department_id) 조합도 정상 계산된다."""
    # AI응용학과·융합보안학과 같이 단일 트랙만 운영하는 학과: major_id == department_id
    a = make_track("a", college_id="C1", department_id="D_AI", major_id="D_AI")
    b = make_track("b", college_id="C1", department_id="D_AI", major_id="D_AI")
    c = make_track("c", college_id="C2", department_id="D_OTHER", major_id="D_OTHER")
    d = make_track("d", college_id="C2", department_id="D_OTHER", major_id="D_OTHER")
    same_combo = make_combo(a, b)
    other_combo = make_combo(c, d)
    cfg = make_synergy_config()

    # degenerate 조합도 sim_4tier 산출이 정상이며, 모든 tier 가 disjoint 인 다른 학과
    # 조합과의 sim 보다 자기 조합 sim 이 크다.
    sim_self = _sim_4tier(same_combo, same_combo, cfg.similarity)
    sim_cross = _sim_4tier(same_combo, other_combo, cfg.similarity)
    assert sim_self > sim_cross


def test_mmr_select_pure_synergy_when_lambda_one(
    make_track, make_combo, make_synergy_config
) -> None:
    """λ = 1 → 다양성 항 무시, 시너지 순으로 선택."""
    a = make_track("a")
    b = make_track("b")
    c = make_track("c")
    scored = [
        _ScoredCombo(combo=make_combo(a, b), synergy_score=0.3),
        _ScoredCombo(combo=make_combo(a, c), synergy_score=0.9),
        _ScoredCombo(combo=make_combo(b, c), synergy_score=0.6),
    ]
    cfg = make_synergy_config(mmr={"lambda": 1.0})
    chosen = _mmr_select(
        candidates=scored,
        selected=[],
        n=2,
        lambda_=cfg.mmr.lambda_value,
        sim_cfg=cfg.similarity,
        start_rank=4,
    )
    assert [r.synergy_score for r in chosen] == [0.9, 0.6]
    assert [r.rank for r in chosen] == [4, 5]
    assert all(r.slot_type == "mmr" for r in chosen)


def test_mmr_excludes_already_selected(make_track, make_combo, make_synergy_config) -> None:
    """selected 에 든 combo_key 는 후보에서 제외된다."""
    a = make_track("a")
    b = make_track("b")
    c = make_track("c")
    scored = [
        _ScoredCombo(combo=make_combo(a, b), synergy_score=0.9),
        _ScoredCombo(combo=make_combo(a, c), synergy_score=0.8),
    ]
    pre_selected = [
        RankedCombo(
            combo=make_combo(a, b),
            synergy_score=0.9,
            slot_type="primary",
            rank=1,
        )
    ]
    cfg = make_synergy_config(mmr={"lambda": 1.0})
    chosen = _mmr_select(
        candidates=scored,
        selected=pre_selected,
        n=2,
        lambda_=cfg.mmr.lambda_value,
        sim_cfg=cfg.similarity,
        start_rank=2,
    )
    # (a, b) 는 selected 라 제외 → (a, c) 1 개만 선택 가능
    assert len(chosen) == 1
    assert chosen[0].combo.combo_key == "a::c"
