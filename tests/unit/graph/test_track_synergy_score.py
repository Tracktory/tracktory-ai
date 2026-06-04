"""``_synergy_score`` 와 3 항 (complementarity / job_coverage / redundancy) 검증.

시너지 식의 정의역 [0, 1] clip · 각 항의 의미적 정합 (직무 토큰 분업도 / 채용공고
기술스택 도달도 / 과목 중복도) 만 검증한다.
"""

from __future__ import annotations

from tracktory.graph.nodes.track_synergy import (
    _complementarity,
    _job_coverage,
    _redundancy,
    _synergy_score,
)


def test_synergy_in_unit_interval(make_track, make_combo, make_job, make_synergy_config) -> None:
    """가중 합산 결과가 1.5 이상 또는 음수가 되어도 [0, 1] 로 clip 된다."""
    a = make_track("a", competencies=["x"], tech_stacks=["py"])
    b = make_track("b", competencies=["y"], tech_stacks=["sql"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["py", "sql"])]

    # 가중치를 의도적으로 합 1.5 가 되게 만든다.
    cfg = make_synergy_config(weights={"complementarity": 1.0, "coverage": 1.0, "redundancy": 0.0})
    score = _synergy_score(combo, jobs, cfg.weights)
    assert 0.0 <= score <= 1.0


def test_complementarity_zero_when_no_jobs(make_track, make_combo) -> None:
    """직무 토큰이 없으면 분업할 기준이 없으므로 complementarity = 0."""
    a = make_track("a", tech_stacks=["py"])
    b = make_track("b", tech_stacks=["sql"])
    combo = make_combo(a, b)
    assert _complementarity(combo, []) == 0.0


def test_complementarity_zero_when_one_track_irrelevant(make_track, make_combo, make_job) -> None:
    """한 트랙이 직무와 무관(직무 기여 0)하면 분업 불성립 → complementarity = 0.

    무관 조합과 보완 조합을 구분하는 핵심 게이트. track_b 의 토큰이 직무 토큰과
    전혀 겹치지 않으면 두 트랙이 직무를 분업한다고 볼 수 없다.
    """
    a = make_track("a", tech_stacks=["py", "sql"])
    b = make_track("b", tech_stacks=["welding", "plumbing"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["py", "sql", "go"])]
    assert _complementarity(combo, jobs) == 0.0


def test_complementarity_zero_when_job_contributions_identical(
    make_track, make_combo, make_job
) -> None:
    """두 트랙의 직무 기여가 동일하면 분업이 아니라 중복 → complementarity = 0."""
    a = make_track("a", tech_stacks=["py", "sql"])
    b = make_track("b", tech_stacks=["py", "sql"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["py", "sql", "go"])]
    assert _complementarity(combo, jobs) == 0.0


def test_complementarity_max_when_tracks_split_job_tokens(make_track, make_combo, make_job) -> None:
    """두 트랙이 직무 토큰을 겹침 없이 나눠 공급하면 complementarity = 1.0 (완전 분업)."""
    a = make_track("a", tech_stacks=["py"])
    b = make_track("b", tech_stacks=["sql"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["py", "sql"])]
    assert _complementarity(combo, jobs) == 1.0


def test_complementarity_partial_when_job_contributions_overlap(
    make_track, make_combo, make_job
) -> None:
    """직무 기여가 일부 겹치면 0 과 1 사이 (부분 분업).

    contrib_a={py, sql}, contrib_b={sql, go} 의 대칭차집합 {py, go}=2 를 전체 직무
    토큰 {py, sql, go, rust}=4 로 나눠 0.5. 분모가 전체 직무 토큰이므로 직무의 절반을
    겹침 없이 분담한 정도를 반영한다 (직무가 요구하는 rust 는 둘 다 못 덮어 분모에만 남는다).
    """
    a = make_track("a", tech_stacks=["py", "sql"])
    b = make_track("b", tech_stacks=["sql", "go"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["py", "sql", "go", "rust"])]
    assert _complementarity(combo, jobs) == 0.5


def test_complementarity_uses_competency_tags_and_tech_stacks(
    make_track, make_combo, make_job
) -> None:
    """직무 토큰·트랙 토큰 모두 기술스택·역량 두 축을 합쳐 매칭한다.

    track_a 는 역량으로, track_b 는 기술스택으로 각각 직무에 기여하므로 양쪽
    축이 모두 anchor 에 반영되어야 분업이 성립한다.
    """
    a = make_track("a", competencies=["데이터 분석 능력"])
    b = make_track("b", tech_stacks=["sql"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["sql"], competency_tags=["데이터 분석 능력"])]
    assert _complementarity(combo, jobs) == 1.0


def test_complementarity_zero_when_both_tracks_irrelevant(make_track, make_combo, make_job) -> None:
    """두 트랙 모두 직무와 무관하면 complementarity = 0."""
    a = make_track("a", tech_stacks=["welding"])
    b = make_track("b", tech_stacks=["plumbing"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["py", "sql"])]
    assert _complementarity(combo, jobs) == 0.0


def test_complementarity_anchor_filters_track_private_tokens(
    make_track, make_combo, make_job
) -> None:
    """직무 토큰 밖의 트랙 고유 토큰은 anchor 교집합에서 제외된다 (포화 회귀 가드).

    track_a={py, x}, track_b={py, y}, 직무={py} 에서 트랙 고유 x·y 가 필터링되어
    두 기여가 모두 {py} 로 동일 → 0.0. 고유 토큰이 필터링되지 않았다면 대칭차집합
    {x, y} 로 2/3 가 나왔을 것이다. 이 차이가 anchor 의 핵심 동작이다.
    """
    a = make_track("a", tech_stacks=["py", "track_private_x"])
    b = make_track("b", tech_stacks=["py", "track_private_y"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["py"])]
    assert _complementarity(combo, jobs) == 0.0


def test_complementarity_discriminates_related_from_unrelated(
    make_track, make_combo, make_job
) -> None:
    """무관 조합(0)과 실제 보완 조합(>0)이 점수로 구분된다 (포화 회귀 가드)."""
    jobs = [make_job(tech_stacks=["py", "sql", "go"])]
    relevant = make_track("a", tech_stacks=["py"])
    complementary_partner = make_track("b", tech_stacks=["sql"])
    irrelevant_partner = make_track("c", tech_stacks=["welding"])

    complementary = _complementarity(make_combo(relevant, complementary_partner), jobs)
    unrelated = _complementarity(make_combo(relevant, irrelevant_partner), jobs)
    assert complementary > unrelated
    assert unrelated == 0.0


def test_complementarity_canonicalizes_notation_differences(
    make_track, make_combo, make_job
) -> None:
    """직무·트랙이 같은 기술을 다른 표기로 적어도 정합 토큰으로 묶여 분업이 성립한다.

    track_a 는 "ReactJS", 직무는 "React" 로 표기가 달라도 같은 토큰으로 통합돼
    track_a 가 직무에 기여한다. 통합이 없으면 contrib_a 가 비어 분업이 0 이 됐을 것이다.
    """
    a = make_track("a", tech_stacks=["ReactJS"])
    b = make_track("b", tech_stacks=["sql"])
    combo = make_combo(a, b)
    jobs = [make_job(tech_stacks=["React", "sql"])]
    assert _complementarity(combo, jobs) == 1.0


def test_complementarity_not_saturated_by_weak_disjoint_partner(
    make_track, make_combo, make_job
) -> None:
    """직무 토큰이 많은데 한 트랙이 1 개만 공급하면 comp 가 1.0 으로 포화되지 않는다.

    분모가 두 기여의 합집합이면 서로소이기만 해도 1.0 이라, 직무의 1/6 만 공급하는
    트랙이 완전 분업으로 둔갑해 무관 조합이 상위를 점령한다. 전체 직무 토큰을 분모로
    삼으면 분담 비율만큼만 점수가 나와, 균형 분업 조합이 약한-서로소 조합보다 높다.
    """
    jobs = [make_job(tech_stacks=["py", "sql", "java", "go", "rust", "kotlin"])]
    strong = make_track("s", tech_stacks=["py", "sql", "java"])
    weak_disjoint = make_track("w", tech_stacks=["go"])
    weak_combo = make_combo(strong, weak_disjoint)

    # 직무 토큰 6 개 중 둘이 겹침 없이 분담하는 토큰은 {py, sql, java, go}=4 → 4/6.
    weak_comp = _complementarity(weak_combo, jobs)
    assert weak_comp == 4 / 6
    assert weak_comp < 1.0

    # 직무 전체를 절반씩 완전 분담하면 comp = 1.0 (포화는 분담률이 100% 일 때만).
    balanced = make_combo(
        make_track("ba", tech_stacks=["py", "sql", "java"]),
        make_track("bb", tech_stacks=["go", "rust", "kotlin"]),
    )
    assert _complementarity(balanced, jobs) == 1.0
    assert _complementarity(balanced, jobs) > weak_comp


def test_job_coverage_uses_tech_stacks_only(make_track, make_combo, make_job) -> None:
    """coverage 분모는 jobs[*].tech_stacks 합집합 (NCS 미사용)."""
    a = make_track("a", tech_stacks=["py"])
    b = make_track("b", tech_stacks=["sql"])
    combo = make_combo(a, b)
    jobs = [
        make_job("j1", tech_stacks=["py", "sql", "go"], competency_tags=["IGNORE_ME"]),
    ]
    # jobs 의 tech_stacks 3 개 중 트랙 합집합 (py, sql) 이 2 개 커버.
    assert _job_coverage(combo, jobs) == 2 / 3


def test_redundancy_higher_when_courses_overlap(make_track, make_combo) -> None:
    """과목 중복이 클수록 redundancy 가 높다 (Jaccard similarity)."""
    a = make_track("a", course_ids=["c1", "c2"])
    b_no_overlap = make_track("b1", course_ids=["c3", "c4"])
    b_full_overlap = make_track("b2", course_ids=["c1", "c2"])

    low = _redundancy(make_combo(a, b_no_overlap))
    high = _redundancy(make_combo(a, b_full_overlap))
    assert low == 0.0
    assert high == 1.0


def test_job_coverage_canonicalizes_notation_differences(make_track, make_combo, make_job) -> None:
    """직무·트랙이 같은 기술을 다른 표기로 적어도 정합 토큰으로 통합해 커버한다.

    원본 표기 그대로 비교하면 "Spring Boot" != "spring boot" 라 교집합이 0 이라
    커버율이 평탄해지는데, 정합 후에는 두 기술 모두 커버되어 2/3 가 된다.
    """
    a = make_track("a", tech_stacks=["Spring Boot"])
    b = make_track("b", tech_stacks=["SQL"])
    combo = make_combo(a, b)
    jobs = [make_job("j1", tech_stacks=["spring boot", "sql", "react"])]

    # 원본 표기 그대로면 대소문자 차이로 교집합이 비어 커버율 0 이 된다.
    raw_overlap = (set(a.tech_stacks) | set(b.tech_stacks)) & set(jobs[0].tech_stacks)
    assert raw_overlap == set()

    # 정합 토큰으로 통합하면 spring boot · sql 두 기술이 커버된다.
    assert _job_coverage(combo, jobs) == 2 / 3


def test_synergy_distinguishes_relevant_from_irrelevant_combo(
    make_track, make_combo, make_job, make_synergy_config
) -> None:
    """직무 기술을 커버하는 조합이 그렇지 못한 조합보다 높은 시너지를 받는다.

    표기 정합 전에는 양쪽 모두 커버율 0 으로 점수가 평탄했던 회귀를 막는다.
    """
    jobs = [make_job("be", tech_stacks=["Spring Boot", "SQL", "Java"])]
    cfg = make_synergy_config()

    relevant = make_combo(
        make_track("ra", tech_stacks=["spring boot", "java"]),
        make_track("rb", tech_stacks=["sql"]),
    )
    irrelevant = make_combo(
        make_track("ia", tech_stacks=["Photoshop"]),
        make_track("ib", tech_stacks=["Premiere Pro"]),
    )

    relevant_score = _synergy_score(relevant, jobs, cfg.weights)
    irrelevant_score = _synergy_score(irrelevant, jobs, cfg.weights)
    assert relevant_score > irrelevant_score
    assert irrelevant_score == 0.0
