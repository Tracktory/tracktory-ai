"""트랙 시너지 노드 진입부의 후보 페어 빌드 계약 검증.

본 파일은 트랙 시너지 노드가 후보 페어 빌드 단계를 호출했을 때 기대되는
대표 분기 4 종을 검증한다. 동일 함수의 엣지·보강 케이스는
``test_track_candidates.py`` 가 담당한다.
"""

from __future__ import annotations

from tracktory.graph.nodes.track_candidates import build_candidate_pairs


def test_generate_combos_filters_by_user_department_for_freshman(make_track) -> None:
    """트랙 미선택 학년 (current_tracks=[]): 1트랙은 학생 주전공 학부 소속만 선택된다."""
    in_dept = make_track("a", department_id="D1")
    out_dept = make_track("b", department_id="D2")
    third = make_track("c", department_id="D2")

    combos = build_candidate_pairs(
        tracks=[in_dept, out_dept, third],
        user_department_id="D1",
        current_tracks=[],
    )
    primary_track_ids = {c.track_a.track_id for c in combos}
    assert primary_track_ids == {"a"}
    assert len(combos) == 2  # (a, b) and (a, c)


def test_generate_combos_uses_current_tracks_for_upperclass(make_track) -> None:
    """트랙 선택 학년 (current_tracks 비어있지 않음): 1트랙 풀 = current_tracks."""
    user_track = make_track("user_t", department_id="D1")
    other = make_track("o1", department_id="D2")
    another = make_track("o2", department_id="D2")

    combos = build_candidate_pairs(
        tracks=[user_track, other, another],
        user_department_id="D1",
        current_tracks=["user_t"],
    )
    primary_track_ids = {c.track_a.track_id for c in combos}
    assert primary_track_ids == {"user_t"}


def test_generate_combos_excludes_self_pairs(make_track) -> None:
    """동일 트랙 (track_a.track_id == track_b.track_id) 조합은 제외."""
    only = make_track("solo", department_id="D1")
    combos = build_candidate_pairs(
        tracks=[only],
        user_department_id="D1",
        current_tracks=[],
    )
    assert combos == []


def test_generate_combos_dedup_by_combo_key(make_track) -> None:
    """1트랙 풀에 두 트랙이 모두 있으면 (a, b) 와 (b, a) 가 동일 combo_key 로 dedup."""
    a = make_track("aaa", department_id="D1")
    b = make_track("bbb", department_id="D1")

    combos = build_candidate_pairs(
        tracks=[a, b],
        user_department_id="D1",
        current_tracks=[],
    )
    assert len(combos) == 1
    assert combos[0].combo_key == "aaa::bbb"
