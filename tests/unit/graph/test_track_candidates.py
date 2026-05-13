"""``build_candidate_pairs`` 의 보강 케이스 단위 테스트.

``test_track_synergy_combos.py`` 는 트랙 시너지 노드 진입부의 대표 분기 4 종
(1트랙 주전공 학부 제약 · self-pair 제외 · 순서 무관 dedup) 을 다룬다. 본
파일은 그 위에 얹는 보강 케이스 — 1트랙 풀 enumerate 정확도, 파트너 풀
범위, 빈 입력 처리, 알 수 없는 트랙 식별자 처리, 학부 안 다수 트랙으로
구성된 현실 fixture (예: 컴퓨터공학부 4 트랙) — 를 담당한다.
"""

from __future__ import annotations

from tracktory.graph.nodes.track_candidates import build_candidate_pairs


def test_freshman_pairs_each_in_department_primary_with_every_partner(make_track) -> None:
    """학부 안 트랙이 여러 개면 각 트랙이 1트랙으로 enumerate 된다."""
    primary_1 = make_track("p1", department_id="D1")
    primary_2 = make_track("p2", department_id="D1")
    partner = make_track("q", department_id="D2")

    combos = build_candidate_pairs(
        tracks=[primary_1, primary_2, partner],
        user_department_id="D1",
        current_tracks=[],
    )

    # p1-p2, p1-q, p2-q 총 3 페어
    assert len(combos) == 3
    combo_keys = {c.combo_key for c in combos}
    assert combo_keys == {"p1::p2", "p1::q", "p2::q"}


def test_freshman_with_four_tracks_in_user_department_produces_all_within_and_cross_pairs(
    make_track,
) -> None:
    """컴공 학부 4 트랙 fixture: 학부 내 페어 C(4,2)=6 + 학부 외 트랙과의 페어 4 = 총 10."""
    cs_tracks = [make_track(f"cs{i}", department_id="CS") for i in range(4)]
    other_dept_track = make_track("kor", department_id="KOR")

    combos = build_candidate_pairs(
        tracks=[*cs_tracks, other_dept_track],
        user_department_id="CS",
        current_tracks=[],
    )

    assert len(combos) == 10
    combo_keys = {c.combo_key for c in combos}
    expected_within_cs = {f"cs{i}::cs{j}" for i in range(4) for j in range(i + 1, 4)}
    expected_cross = {f"cs{i}::kor" for i in range(4)}
    assert combo_keys == expected_within_cs | expected_cross


def test_upperclass_partner_pool_spans_all_tracks_outside_primary(make_track) -> None:
    """트랙 선택 학년: 2트랙(파트너) 풀은 1트랙을 뺀 모든 트랙이다 (학과 경계 무관)."""
    fixed = make_track("fx", department_id="D1")
    same_dept = make_track("sd", department_id="D1")
    other_dept = make_track("od", department_id="D2")

    combos = build_candidate_pairs(
        tracks=[fixed, same_dept, other_dept],
        user_department_id="D1",
        current_tracks=["fx"],
    )

    partner_ids = {c.track_b.track_id for c in combos}
    assert partner_ids == {"sd", "od"}


def test_empty_tracks_returns_empty_list(make_track) -> None:
    """전체 트랙 목록이 비어있으면 후보는 빈 리스트다."""
    combos = build_candidate_pairs(
        tracks=[],
        user_department_id="D1",
        current_tracks=[],
    )

    assert combos == []


def test_upperclass_with_unknown_current_track_id_is_ignored(make_track) -> None:
    """``current_tracks`` 식별자가 전체 트랙 목록에 없으면 해당 트랙은 무시된다."""
    real = make_track("real", department_id="D1")
    other = make_track("other", department_id="D2")

    combos = build_candidate_pairs(
        tracks=[real, other],
        user_department_id="D1",
        current_tracks=["does_not_exist"],
    )

    # 유효한 1트랙이 0 개이므로 후보 0 개
    assert combos == []
