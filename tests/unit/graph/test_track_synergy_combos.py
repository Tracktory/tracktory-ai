"""``_generate_combos`` 의 후보 생성 계약을 검증한다.

1트랙 주전공 제약 (1학년 vs 2학년+) · self-pair 제외 · combo_key dedup 만 검증한다.
Pydantic 자체의 round-trip 은 검증 대상 X.
"""

from __future__ import annotations

from tracktory.graph.nodes.track_synergy import _generate_combos, _single_department_ids


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
    user_a = make_track("user_a", college_id="C1")
    user_b = make_track("user_b", college_id="C1")
    other = make_track("o1", college_id="C2")
    another = make_track("o2", college_id="C2")

    combos = _generate_combos(
        tracks=[user_a, user_b, other, another],
        user_college_id="C1",
        current_tracks=["user_a", "user_b"],
    )
    primary_track_ids = {c.track_a.track_id for c in combos}
    assert primary_track_ids == {"user_a", "user_b"}
    assert "user_a::user_b" in {c.combo_key for c in combos}


def test_generate_combos_excludes_self_pairs(make_track) -> None:
    """동일 트랙 (track_a.track_id == track_b.track_id) 조합은 제외."""
    # 같은 복수 트랙 학과의 두 트랙 — 단일 학과 제외가 아니라 self-pair 제외만 본다.
    a = make_track("aaa", college_id="C1", department_id="D1")
    b = make_track("bbb", college_id="C1", department_id="D1")
    combos = _generate_combos(
        tracks=[a, b],
        user_college_id="C1",
        current_tracks=[],
    )
    assert all(c.track_a.track_id != c.track_b.track_id for c in combos)


def test_generate_combos_dedup_by_combo_key(make_track) -> None:
    """1트랙 풀에 두 트랙이 모두 있으면 (a, b) 와 (b, a) 가 동일 combo_key 로 dedup."""
    a = make_track("aaa", college_id="C1", department_id="D1")
    b = make_track("bbb", college_id="C1", department_id="D1")

    combos = _generate_combos(
        tracks=[a, b],
        user_college_id="C1",
        current_tracks=[],
    )
    assert len(combos) == 1
    assert combos[0].combo_key == "aaa::bbb"


def test_single_department_ids_identifies_departments_with_one_track(make_track) -> None:
    """학과당 트랙 수가 1 인 department_id 만 단일 학과로 식별한다."""
    multi_a = make_track("ma", department_id="multi")
    multi_b = make_track("mb", department_id="multi")
    solo = make_track("solo", department_id="solo_dept")

    assert _single_department_ids([multi_a, multi_b, solo]) == {"solo_dept"}


def test_generate_combos_excludes_single_department_as_partner(make_track) -> None:
    """단일 학과 트랙은 복수 트랙 학과 트랙의 조합 파트너로 생성되지 않는다."""
    cs_a = make_track("cs_a", college_id="C1", department_id="cs")
    cs_b = make_track("cs_b", college_id="C1", department_id="cs")
    single = make_track("AI응용학과", college_id="C2", department_id="AI응용학과")

    combos = _generate_combos(
        tracks=[cs_a, cs_b, single],
        user_college_id="C1",
        current_tracks=[],
    )

    partner_ids = {c.track_b.track_id for c in combos} | {c.track_a.track_id for c in combos}
    assert "AI응용학과" not in partner_ids
    assert {c.combo_key for c in combos} == {"cs_a::cs_b"}


def test_generate_combos_empty_when_upperclass_user_in_single_department(make_track) -> None:
    """2학년+ 사용자 본인(current_tracks)이 단일 학과면 조합을 생성하지 않는다."""
    user_single = make_track("AI응용학과", college_id="C2", department_id="AI응용학과")
    cs_a = make_track("cs_a", college_id="C1", department_id="cs")
    cs_b = make_track("cs_b", college_id="C1", department_id="cs")

    combos = _generate_combos(
        tracks=[user_single, cs_a, cs_b],
        user_college_id="C2",
        current_tracks=["AI응용학과"],
        user_department_id="AI응용학과",
    )

    assert combos == []


def test_generate_combos_empty_when_freshman_in_single_department(make_track) -> None:
    """1학년(current_tracks=[]) 이라도 본인 학과가 단일 학과면 조합을 생성하지 않는다.

    단일 학과 학생은 같은 단과대의 다른 복수 트랙 학과 트랙과 묶여서는 안 된다 —
    단과대 풀만 보면 파트너가 잡히지만, 본인 학과(department) 가 단일 학과면
    학년과 무관하게 조합 자체가 무의미하다.
    """
    user_single = make_track(
        "뷰티디자인매니지먼트학과",
        college_id="디자인대학",
        department_id="뷰티디자인매니지먼트학과",
    )
    design_a = make_track("dz_a", college_id="디자인대학", department_id="ICT디자인학부")
    design_b = make_track("dz_b", college_id="디자인대학", department_id="ICT디자인학부")

    combos = _generate_combos(
        tracks=[user_single, design_a, design_b],
        user_college_id="디자인대학",
        current_tracks=[],
        user_department_id="뷰티디자인매니지먼트학과",
    )

    assert combos == []


def test_generate_combos_multi_track_department_behavior_unchanged(make_track) -> None:
    """복수 트랙 학과끼리의 조합은 단일 학과가 섞여 있어도 그대로 생성된다."""
    cs_a = make_track("cs_a", college_id="C1", department_id="cs")
    cs_b = make_track("cs_b", college_id="C1", department_id="cs")
    design_a = make_track("dz_a", college_id="C2", department_id="design")
    design_b = make_track("dz_b", college_id="C2", department_id="design")
    single = make_track("solo학과", college_id="C3", department_id="solo학과")

    combos = _generate_combos(
        tracks=[cs_a, cs_b, design_a, design_b, single],
        user_college_id="C1",
        current_tracks=[],
    )

    combo_keys = {c.combo_key for c in combos}
    # 1트랙 풀 = C1 의 cs_a, cs_b. 파트너 = 단일 학과를 뺀 cs/design 트랙.
    assert combo_keys == {"cs_a::cs_b", "cs_a::dz_a", "cs_a::dz_b", "cs_b::dz_a", "cs_b::dz_b"}
    assert all("solo학과" not in key for key in combo_keys)
