"""``_generate_combos`` 의 후보 생성 계약을 검증한다.

1트랙 주전공 제약 (1학년 vs 2학년+) · self-pair 제외 · combo_key dedup 만 검증한다.
Pydantic 자체의 round-trip 은 검증 대상 X.
"""

from __future__ import annotations

from tracktory.graph.nodes.track_synergy import _generate_combos


def test_generate_combos_filters_by_user_college_for_freshman(make_track) -> None:
    """1학년 (current_tracks=[]): 1트랙은 사용자 단과대 소속만 선택된다."""
    in_college = make_track("a", college_id="C1")
    out_college = make_track("b", college_id="C2")
    third = make_track("c", college_id="C2")

    combos = _generate_combos(
        tracks=[in_college, out_college, third],
        user_college_id="C1",
        current_tracks=[],
    )
    primary_track_ids = {c.track_a.track_id for c in combos}
    assert primary_track_ids == {"a"}
    assert len(combos) == 2  # (a, b) and (a, c)


def test_generate_combos_uses_current_tracks_for_upperclass(make_track) -> None:
    """2학년+ (current_tracks 비어있지 않음): 1트랙 풀 = current_tracks."""
    user_track = make_track("user_t", college_id="C1")
    other = make_track("o1", college_id="C2")
    another = make_track("o2", college_id="C2")

    combos = _generate_combos(
        tracks=[user_track, other, another],
        user_college_id="C1",
        current_tracks=["user_t"],
    )
    primary_track_ids = {c.track_a.track_id for c in combos}
    assert primary_track_ids == {"user_t"}


def test_generate_combos_excludes_self_pairs(make_track) -> None:
    """동일 트랙 (track_a.track_id == track_b.track_id) 조합은 제외."""
    only = make_track("solo", college_id="C1")
    combos = _generate_combos(
        tracks=[only],
        user_college_id="C1",
        current_tracks=[],
    )
    assert combos == []


def test_generate_combos_dedup_by_combo_key(make_track) -> None:
    """1트랙 풀에 두 트랙이 모두 있으면 (a, b) 와 (b, a) 가 동일 combo_key 로 dedup."""
    a = make_track("aaa", college_id="C1")
    b = make_track("bbb", college_id="C1")

    combos = _generate_combos(
        tracks=[a, b],
        user_college_id="C1",
        current_tracks=[],
    )
    assert len(combos) == 1
    assert combos[0].combo_key == "aaa::bbb"
