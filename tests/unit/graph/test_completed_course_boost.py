"""``_apply_completed_course_boost`` 순수 헬퍼 단위 테스트.

부스팅 공식 ``boost = weight · |completed ∩ job_tokens| / |job_tokens|`` 의
정확한 가산량·재정렬·경계 조건을 외부 I/O 없이 검증한다. 교집합은 이수 과목
기술 토큰과 직무 토큰을 모두 정규 키 (별칭·대소문자 흡수) 로 환산한 뒤 계산
하므로, "자바" 와 "Java" 처럼 표기만 다른 기술이 같은 토큰으로 매칭되는지도
함께 검증한다. 두 번째 인자는 과목 이름이 아니라 환산된 기술 토큰이다.
"""

from __future__ import annotations

import pytest

from tracktory.graph.models import JobCandidate
from tracktory.graph.nodes.job_matching import _apply_completed_course_boost


def _candidate(
    job_id: str,
    *,
    score: float,
    tech_stacks: list[str] | None = None,
    competency_tags: list[str] | None = None,
) -> JobCandidate:
    return JobCandidate(
        job_id=job_id,
        job_name=job_id,
        tech_stacks=tech_stacks or [],
        competency_tags=competency_tags or [],
        match_score=score,
        similarity=score,
        fallback_used=False,
    )


def test_boost_adds_weight_times_overlap_ratio() -> None:
    """2/4 직무 토큰을 덮으면 boost = weight · 0.5."""
    cand = _candidate(
        "backend",
        score=0.5,
        tech_stacks=["자료구조", "운영체제"],
        competency_tags=["네트워크", "데이터베이스"],
    )
    result = _apply_completed_course_boost([cand], ["자료구조", "운영체제"], weight=0.2)
    # overlap_ratio = 2/4 = 0.5 → boost = 0.2 * 0.5 = 0.1
    assert result[0].match_score == pytest.approx(0.6)
    # similarity 는 원본 검색 점수를 보존한다 (부스팅의 영향을 받지 않음).
    assert result[0].similarity == pytest.approx(0.5)


def test_boost_preserves_similarity_as_raw_search_score() -> None:
    """부스팅이 발생해도 similarity 는 입력 검색 점수 그대로 보존된다."""
    cand = _candidate(
        "backend",
        score=0.5,
        tech_stacks=["자료구조", "운영체제"],
        competency_tags=["네트워크", "데이터베이스"],
    )
    result = _apply_completed_course_boost(
        [cand], ["자료구조", "운영체제", "네트워크", "데이터베이스"], weight=0.2
    )
    # 전체 overlap → match_score = 0.5 + 0.2*1.0 = 0.7, similarity 는 0.5 보존
    assert result[0].match_score == pytest.approx(0.7)
    assert result[0].similarity == pytest.approx(0.5)


def test_full_overlap_adds_full_weight() -> None:
    cand = _candidate("data", score=0.4, tech_stacks=["python"], competency_tags=["sql"])
    result = _apply_completed_course_boost([cand], ["Python", "SQL"], weight=0.15)
    # overlap_ratio = 2/2 = 1.0 → boost = 0.15
    assert result[0].match_score == pytest.approx(0.55)


def test_boost_is_clipped_at_one() -> None:
    cand = _candidate("x", score=0.95, tech_stacks=["a", "b"], competency_tags=[])
    result = _apply_completed_course_boost([cand], ["a", "b"], weight=0.5)
    # 0.95 + 0.5 = 1.45 → clip → 1.0
    assert result[0].match_score == pytest.approx(1.0)


def test_no_overlap_keeps_score() -> None:
    cand = _candidate("x", score=0.6, tech_stacks=["go"], competency_tags=["rust"])
    result = _apply_completed_course_boost([cand], ["자료구조"], weight=0.3)
    assert result[0].match_score == pytest.approx(0.6)


def test_token_match_is_case_insensitive() -> None:
    cand = _candidate("x", score=0.5, tech_stacks=["Python"], competency_tags=[])
    result = _apply_completed_course_boost([cand], ["  python  "], weight=0.2)
    assert result[0].match_score == pytest.approx(0.7)


def test_alias_aligned_token_boosts_canonical_job_token() -> None:
    """별칭 표기 "자바" 가 직무의 정규 표기 "Java" 와 같은 정규 키로 매칭된다."""
    cand = _candidate("x", score=0.5, tech_stacks=["Java"], competency_tags=[])
    result = _apply_completed_course_boost([cand], ["자바"], weight=0.2)
    # 정규 키 매칭 → overlap_ratio = 1.0 → boost = 0.2
    assert result[0].match_score == pytest.approx(0.7)


def test_unrelated_token_yields_no_boost() -> None:
    """직무 어휘와 무관한 토큰은 정규 키 교집합이 비어 점수를 바꾸지 않는다."""
    cand = _candidate("x", score=0.5, tech_stacks=["Java"], competency_tags=[])
    result = _apply_completed_course_boost([cand], ["회계"], weight=0.2)
    assert result[0].match_score == pytest.approx(0.5)


def test_empty_job_tokens_yields_zero_boost() -> None:
    cand = _candidate("x", score=0.5)
    result = _apply_completed_course_boost([cand], ["자료구조"], weight=0.3)
    assert result[0].match_score == pytest.approx(0.5)


def test_boost_resorts_candidates_by_new_score() -> None:
    """부스팅으로 하위 후보가 상위로 역전될 수 있어야 한다."""
    top = _candidate("top", score=0.7, tech_stacks=["go"], competency_tags=[])
    boosted = _candidate(
        "boosted", score=0.6, tech_stacks=["자료구조"], competency_tags=["알고리즘"]
    )
    result = _apply_completed_course_boost([top, boosted], ["자료구조", "알고리즘"], weight=0.3)
    # boosted: 0.6 + 0.3*1.0 = 0.9 > top 0.7 → 순서 역전
    assert [c.job_id for c in result] == ["boosted", "top"]
    assert result[0].match_score == pytest.approx(0.9)


def test_tie_preserves_original_order() -> None:
    """동점은 안정 정렬로 원래 검색 순서를 유지한다."""
    a = _candidate("a", score=0.5)
    b = _candidate("b", score=0.5)
    result = _apply_completed_course_boost([a, b], ["자료구조"], weight=0.2)
    assert [c.job_id for c in result] == ["a", "b"]


def test_zero_weight_returns_input_unchanged() -> None:
    a = _candidate("a", score=0.5, tech_stacks=["python"], competency_tags=[])
    result = _apply_completed_course_boost([a], ["python"], weight=0.0)
    assert result[0].match_score == pytest.approx(0.5)


def test_empty_completed_courses_returns_input_unchanged() -> None:
    a = _candidate("a", score=0.5, tech_stacks=["python"], competency_tags=[])
    result = _apply_completed_course_boost([a], [], weight=0.2)
    assert result[0].match_score == pytest.approx(0.5)


def test_whitespace_only_completed_courses_yields_no_boost() -> None:
    a = _candidate("a", score=0.5, tech_stacks=["python"], competency_tags=[])
    result = _apply_completed_course_boost([a], ["   ", ""], weight=0.2)
    assert result[0].match_score == pytest.approx(0.5)
