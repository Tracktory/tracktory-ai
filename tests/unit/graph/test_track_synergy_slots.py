"""primary 선택 + cross-college 슬롯 + T1 → T2 → MMR 흘림 fallback 검증.

순수 계산 함수만 호출 — logger / I/O 없이 분류 로직 자체를 검증.
"""

from __future__ import annotations

from tracktory.graph.nodes.track_synergy import (
    _ScoredCombo,
    _select_cross_college_slot,
    _select_primary,
)


def test_select_primary_returns_top_k_by_synergy(make_track, make_combo) -> None:
    """시너지 상위 k 개 선택 + slot_type='primary' / rank=1..k 부여."""
    a = make_track("a")
    b = make_track("b")
    c = make_track("c")
    scored = [
        _ScoredCombo(combo=make_combo(a, b), synergy_score=0.5),
        _ScoredCombo(combo=make_combo(a, c), synergy_score=0.9),
        _ScoredCombo(combo=make_combo(b, c), synergy_score=0.7),
    ]
    primary = _select_primary(scored, k=2)
    assert [r.synergy_score for r in primary] == [0.9, 0.7]
    assert all(r.slot_type == "primary" for r in primary)
    assert [r.rank for r in primary] == [1, 2]


def test_select_primary_breaks_score_ties_deterministically(make_track, make_combo) -> None:
    """동점 시 combo_key 알파벳 순으로 결정적 정렬 — 같은 입력에 항상 같은 순위.

    점수가 같은 조합은 시너지 모델상 등가라 점수로 더 가를 수 없다. 임의(비결정)
    정렬 대신 combo_key 로 결정적 resolution 을 보장해, 동일 입력이 매 호출 같은
    주 추천을 내도록 고정한다 (점수 동률에 의한 임의 정렬 제거).
    """
    a = make_track("a")
    b = make_track("b")
    c = make_track("c")
    combo_ab = make_combo(a, b)
    combo_ac = make_combo(a, c)
    # 입력 순서를 뒤집어도 결과 순위가 같아야 한다 (비결정성 배제).
    forward = [
        _ScoredCombo(combo=combo_ab, synergy_score=0.5),
        _ScoredCombo(combo=combo_ac, synergy_score=0.5),
    ]
    reverse = list(reversed(forward))

    keys_forward = [r.combo.combo_key for r in _select_primary(forward, k=2)]
    keys_reverse = [r.combo.combo_key for r in _select_primary(reverse, k=2)]
    assert keys_forward == keys_reverse == sorted([combo_ab.combo_key, combo_ac.combo_key])


def test_cross_college_selects_t1_candidate_above_threshold(make_track, make_combo) -> None:
    """T1 cross 후보 (단과대 다른 트랙 포함) 중 임계값 이상 + 시너지 최대 채택."""
    primary_a = make_track("a", college_id="C1")
    primary_b = make_track("b", college_id="C1")
    cross_winner = make_track("c", college_id="C2")
    cross_loser = make_track("d", college_id="C2")

    primary = _select_primary(
        [_ScoredCombo(combo=make_combo(primary_a, primary_b), synergy_score=0.95)],
        k=1,
    )
    candidates = [
        _ScoredCombo(combo=make_combo(primary_a, cross_winner), synergy_score=0.60),
        _ScoredCombo(combo=make_combo(primary_a, cross_loser), synergy_score=0.40),
        # 임계값 미달 → 제외
        _ScoredCombo(combo=make_combo(primary_b, cross_winner), synergy_score=0.20),
    ]
    slot, fallback = _select_cross_college_slot(
        scored=candidates, primary=primary, min_cross_synergy=0.3
    )
    assert fallback is None
    assert slot is not None
    assert slot.synergy_score == 0.60
    assert slot.slot_type == "cross_college"
    assert slot.rank == 3


def test_cross_college_t1_fallback_to_t2_when_no_t1_candidate(make_track, make_combo) -> None:
    """단과대 cross 후보 없음 → 학부 cross-dept 로 relax (fallback_level='T2')."""
    primary_a = make_track("a", college_id="C1", department_id="D1")
    primary_b = make_track("b", college_id="C1", department_id="D1")
    same_college_other_dept = make_track("c", college_id="C1", department_id="D2")

    primary = _select_primary(
        [_ScoredCombo(combo=make_combo(primary_a, primary_b), synergy_score=0.95)],
        k=1,
    )
    candidates = [
        _ScoredCombo(combo=make_combo(primary_a, same_college_other_dept), synergy_score=0.50),
    ]
    slot, fallback = _select_cross_college_slot(
        scored=candidates, primary=primary, min_cross_synergy=0.3
    )
    assert fallback == "T2"
    assert slot is not None
    assert slot.synergy_score == 0.50


def test_cross_college_mmr_fallback_when_no_cross_dept(make_track, make_combo) -> None:
    """학부 cross 도 없음 → 슬롯3 = None, fallback_level='MMR'."""
    primary_a = make_track("a", college_id="C1", department_id="D1")
    primary_b = make_track("b", college_id="C1", department_id="D1")
    same_dept = make_track("c", college_id="C1", department_id="D1")

    primary = _select_primary(
        [_ScoredCombo(combo=make_combo(primary_a, primary_b), synergy_score=0.95)],
        k=1,
    )
    candidates = [
        _ScoredCombo(combo=make_combo(primary_a, same_dept), synergy_score=0.80),
    ]
    slot, fallback = _select_cross_college_slot(
        scored=candidates, primary=primary, min_cross_synergy=0.3
    )
    assert fallback == "MMR"
    assert slot is None


def test_cross_college_excludes_primary_combos(make_track, make_combo) -> None:
    """primary 슬롯에 이미 든 조합은 후보에서 제외된다."""
    a = make_track("a", college_id="C1")
    b = make_track("b", college_id="C2")  # cross
    primary = _select_primary(
        [_ScoredCombo(combo=make_combo(a, b), synergy_score=0.95)],
        k=1,
    )
    candidates = [
        # primary 와 동일 combo_key — 제외되어야 함
        _ScoredCombo(combo=make_combo(a, b), synergy_score=0.95),
    ]
    slot, fallback = _select_cross_college_slot(
        scored=candidates, primary=primary, min_cross_synergy=0.3
    )
    # primary 가 cross-college 자체임 → primary 제외 후 cross 후보 0 → MMR fallback
    assert fallback == "MMR"
    assert slot is None
