"""``build_candidate_pairs`` 의 보강 케이스 단위 테스트.

``test_track_synergy_combos.py`` 는 트랙 시너지 노드 진입부의 대표 분기 4 종
(1트랙 주전공 제약 · self-pair 제외 · 순서 무관 dedup) 을 다룬다. 본 파일은
그 위에 얹는 보강 케이스 — 1트랙 풀 enumerate 정확도, 파트너 풀 범위,
빈 입력 처리, 알 수 없는 트랙 식별자 처리 — 를 담당한다.
"""

from __future__ import annotations

from tracktory.graph.nodes.track_candidates import build_candidate_pairs


def test_freshman_pairs_each_in_college_primary_with_every_partner(make_track) -> None:
    """단과대 소속 트랙이 여러 개면 각 트랙이 1트랙으로 enumerate 된다."""
    primary_1 = make_track("p1", college_id="C1")
    primary_2 = make_track("p2", college_id="C1")
    partner = make_track("q", college_id="C2")

    combos = build_candidate_pairs(
        tracks=[primary_1, primary_2, partner],
        user_college_id="C1",
        current_tracks=[],
    )

    # p1-p2, p1-q, p2-q 총 3 페어
    assert len(combos) == 3
    combo_keys = {c.combo_key for c in combos}
    assert combo_keys == {"p1::p2", "p1::q", "p2::q"}


def test_upperclass_partner_pool_spans_all_tracks_outside_primary(make_track) -> None:
    """트랙 선택 학년: 2트랙(파트너) 풀은 1트랙을 뺀 모든 트랙이다 (학과 경계 무관)."""
    fixed = make_track("fx", college_id="C1")
    same_college = make_track("sc", college_id="C1")
    other_college = make_track("oc", college_id="C2")

    combos = build_candidate_pairs(
        tracks=[fixed, same_college, other_college],
        user_college_id="C1",
        current_tracks=["fx"],
    )

    partner_ids = {c.track_b.track_id for c in combos}
    assert partner_ids == {"sc", "oc"}


def test_freshman_excludes_primary_outside_user_college(make_track) -> None:
    """주전공 학부에 속하지 않은 트랙은 1트랙 자리에 오지 못한다."""
    user_college_track = make_track("u", college_id="C1")
    other_college_track = make_track("o", college_id="C2")

    combos = build_candidate_pairs(
        tracks=[user_college_track, other_college_track],
        user_college_id="C1",
        current_tracks=[],
    )

    assert len(combos) == 1
    assert combos[0].track_a.track_id == "u"


def test_empty_tracks_returns_empty_list(make_track) -> None:
    """전체 트랙 목록이 비어있으면 후보는 빈 리스트다."""
    combos = build_candidate_pairs(
        tracks=[],
        user_college_id="C1",
        current_tracks=[],
    )

    assert combos == []


def test_upperclass_with_unknown_current_track_id_is_ignored(make_track) -> None:
    """``current_tracks`` 식별자가 전체 트랙 목록에 없으면 해당 트랙은 무시된다."""
    real = make_track("real", college_id="C1")
    other = make_track("other", college_id="C2")

    combos = build_candidate_pairs(
        tracks=[real, other],
        user_college_id="C1",
        current_tracks=["does_not_exist"],
    )

    # 유효한 1트랙이 0 개이므로 후보 0 개
    assert combos == []
