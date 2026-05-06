"""``_synergy_score`` 와 3 항 (complementarity / job_coverage / redundancy) 검증.

시너지 식의 정의역 [0, 1] clip · 각 항의 의미적 정합 (Jaccard 거리 / 채용공고
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


def test_complementarity_zero_when_competencies_identical(make_track, make_combo) -> None:
    """두 트랙의 역량이 완전 동일하면 complementarity = 0 (대칭 차집합 0)."""
    a = make_track("a", competencies=["py", "data"])
    b = make_track("b", competencies=["py", "data"])
    combo = make_combo(a, b)
    assert _complementarity(combo) == 0.0


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
