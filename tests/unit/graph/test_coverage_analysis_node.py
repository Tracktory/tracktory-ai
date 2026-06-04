"""역량 커버리지 분석 노드 + 순수 계산(``compute_coverage``) 단위 테스트.

``compute_coverage`` 는 색인을 인자로 받는 순수 함수라 파일 I/O 없이 검증하고,
노드 진입점은 합성 ``courses.yaml`` 을 주입해 색인 로딩까지 포함한 계약을 검증한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tracktory.graph.models import JobCandidate
from tracktory.graph.nodes.coverage_analysis import CoverageAnalysisNode, compute_coverage


def _job(
    job_id: str,
    *,
    job_name: str | None = None,
    tech_stacks: list[str] | None = None,
    competency_tags: list[str] | None = None,
    match_score: float = 0.8,
) -> JobCandidate:
    return JobCandidate(
        job_id=job_id,
        job_name=job_name or job_id,
        tech_stacks=tech_stacks or [],
        competency_tags=competency_tags or [],
        match_score=match_score,
    )


# ---------------------------------------------------------------------------
# compute_coverage — 순수 계산
# ---------------------------------------------------------------------------


def test_current_and_expected_ratio_basic() -> None:
    """이수 과목이 현재 충족도를, 로드맵 과목이 예상 충족도를 채운다 (issue 조건 1)."""
    jobs = [_job("be", tech_stacks=["MySQL", "Docker"])]
    index = {"데이터베이스": ["MySQL"], "데브옵스": ["Docker"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],
        roadmap_courses=[("OPS", "데브옵스")],
        course_tech_index=index,
    )

    assert analysis.required_count == 2
    assert analysis.current_covered == 1
    assert analysis.expected_covered == 2
    assert analysis.current_ratio == pytest.approx(0.5)
    assert analysis.expected_ratio == pytest.approx(1.0)
    assert analysis.gap_tokens == []


def test_default_anchor_is_top_match_and_target_is_anchor_tokens_only() -> None:
    """기준 직무 미지정 시 매칭도 1순위 직무 토큰만 목표로 삼는다 (issue 조건 1).

    합집합이 아니라 단일 anchor 라, 비-anchor 직무(fe)의 토큰은 목표에서 빠진다.
    또 분모는 anchor 토큰 중 도달 가능한 것만이라, 어떤 과목도 안 가르치는
    Docker 는 분모에서 빠지고 gap 으로 보고된다.
    """
    jobs = [
        _job("be", job_name="백엔드", tech_stacks=["MySQL", "Docker"], match_score=0.9),
        _job("fe", job_name="프론트", tech_stacks=["React"], match_score=0.6),
    ]
    index = {"데이터베이스": ["MySQL"], "프론트개론": ["React"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],
        roadmap_courses=[("FE", "프론트개론")],
        course_tech_index=index,
    )

    assert analysis.anchor_job_id == "be"
    assert analysis.anchor_job_name == "백엔드"
    # 분모는 anchor(be) 토큰 {MySQL, Docker} 중 도달 가능한 {MySQL} 뿐.
    # Docker 는 어떤 과목도 안 가르쳐 분모에서 빠지고 gap 으로 보고된다.
    assert analysis.required_count == 1
    assert analysis.current_covered == 1  # MySQL
    assert analysis.current_ratio == pytest.approx(1.0)  # 도달 가능 분모 기준 완전 충족
    assert analysis.gap_tokens == ["Docker"]  # 도달 불가 — fe 의 React 는 anchor 무관


def test_default_anchor_respects_input_order_on_score_tie() -> None:
    """매칭도 동률이면 입력 순서(상위 노드 정렬)를 보존해 첫 직무를 anchor 로 둔다."""
    jobs = [
        _job("be", job_name="백엔드", tech_stacks=["MySQL"], match_score=0.8),
        _job("fe", job_name="프론트", tech_stacks=["React"], match_score=0.8),
    ]
    index = {"데이터베이스": ["MySQL"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],
        roadmap_courses=[],
        course_tech_index=index,
    )
    assert analysis.anchor_job_id == "be"


def test_explicit_anchor_recomputes_against_specified_job() -> None:
    """기준 직무를 지정하면 1순위가 아니어도 그 직무 기준으로 다시 산출한다 (issue 조건 2)."""
    jobs = [
        _job("be", job_name="백엔드", tech_stacks=["MySQL", "Docker"], match_score=0.9),
        _job("fe", job_name="프론트", tech_stacks=["React"], match_score=0.6),
    ]
    index = {"프론트개론": ["React"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["프론트개론"],
        roadmap_courses=[],
        course_tech_index=index,
        anchor_job_id="fe",
    )

    assert analysis.anchor_job_id == "fe"
    assert analysis.required_count == 1  # fe 토큰 {React} 뿐
    assert analysis.current_ratio == pytest.approx(1.0)  # React 이수 완료


def test_invalid_anchor_falls_back_to_default_top_match() -> None:
    """추천 직무에 없는 기준 직무 식별자는 매칭도 1순위로 graceful fallback 한다."""
    jobs = [
        _job("be", job_name="백엔드", tech_stacks=["MySQL"], match_score=0.9),
        _job("fe", job_name="프론트", tech_stacks=["React"], match_score=0.6),
    ]
    index = {"데이터베이스": ["MySQL"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],
        roadmap_courses=[],
        course_tech_index=index,
        anchor_job_id="nonexistent",
    )
    assert analysis.anchor_job_id == "be"


def test_anchor_bound_subobjects_align_to_anchor_job() -> None:
    """게이지·잔여 과목 기여도·다음 액션·gap 이 모두 같은 anchor 직무 토큰에 정렬된다.

    분야별 분석(``jobs``)은 anchor 와 무관한 별개 축이라 여기서 검증하지 않는다
    (``test_jobs_field_holds_all_recommended_jobs_independent_of_anchor`` 참조).
    """
    jobs = [
        _job("be", job_name="백엔드", tech_stacks=["MySQL", "Docker", "Kafka"], match_score=0.9),
        _job("fe", job_name="프론트", tech_stacks=["React", "TypeScript"], match_score=0.6),
    ]
    index = {
        "데이터베이스": ["MySQL"],
        "데브옵스": ["Docker", "Kafka"],
        "프론트개론": ["React"],  # anchor(be) 와 무관 — 기여 0 이어야
    }

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],
        roadmap_courses=[("OPS", "데브옵스"), ("FE", "프론트개론")],
        course_tech_index=index,
        anchor_job_id="be",
    )

    anchor_tokens = {"MySQL", "Docker", "Kafka"}

    # 게이지(목표 토큰 수)가 anchor 직무 토큰에 정렬된다.
    assert analysis.anchor_job_id == "be"
    assert analysis.required_count == len(anchor_tokens)

    # 잔여 과목 기여·다음 액션이 더하는 토큰은 모두 anchor 목표 토큰 안에 있다.
    for contribution in analysis.course_contributions:
        assert set(contribution.added_tokens) <= anchor_tokens
    next_action_ids = {action.course_id for action in analysis.next_actions}
    contribution_ids = {c.course_id for c in analysis.course_contributions}
    assert next_action_ids <= contribution_ids

    # anchor 와 무관한 과목(프론트개론/React)은 기여 0 — 다른 직무 토큰을 끌어오지 않는다.
    contrib_by_id = {c.course_id: c for c in analysis.course_contributions}
    assert contrib_by_id["FE"].contribution_ratio == pytest.approx(0.0)
    assert contrib_by_id["FE"].added_tokens == []

    # gap 토큰도 anchor 목표 토큰 안에서만 보고된다 (다른 직무 토큰 누출 없음).
    assert set(analysis.gap_tokens) <= anchor_tokens


def test_jobs_field_holds_all_recommended_jobs_independent_of_anchor() -> None:
    """분야별 분석(``jobs``)은 anchor 와 무관하게 전 추천 직무를 각자 토큰 기준으로 담는다.

    직무 간 비교 카드의 데이터원이라, 비-1순위 직무를 anchor 로 지정해도 jobs 에는
    전 직무가 남고 현재 충족률 내림차순으로 정렬된다.
    """
    jobs = [
        _job("be", job_name="백엔드", tech_stacks=["MySQL", "Docker"], match_score=0.9),
        _job("fe", job_name="프론트", tech_stacks=["React"], match_score=0.6),
    ]
    index = {"데이터베이스": ["MySQL"], "프론트개론": ["React"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["프론트개론"],  # fe 의 React 충족 → fe 현재 충족률 높음
        roadmap_courses=[],
        course_tech_index=index,
        anchor_job_id="fe",  # 비-1순위 anchor 여도 jobs 는 전 직무 유지
    )

    by_job = {coverage.job_id: coverage for coverage in analysis.jobs}
    assert set(by_job) == {"be", "fe"}  # anchor 와 무관하게 전 직무
    # 각 직무는 자기 토큰 기준으로 평가된다.
    assert by_job["be"].required_count == 2  # MySQL, Docker
    assert by_job["be"].current_covered == 0  # 프론트개론은 be 와 무관
    assert by_job["fe"].required_count == 1  # React
    assert by_job["fe"].current_covered == 1  # 프론트개론 → React
    # 현재 충족률 내림차순: fe(1.0) → be(0.0)
    assert [coverage.job_id for coverage in analysis.jobs] == ["fe", "be"]


def test_per_course_contribution_and_next_actions() -> None:
    """잔여 과목별 기여도(+%)와 기여 큰 순 다음 액션을 산출한다 (issue 조건 3·4)."""
    jobs = [_job("be", tech_stacks=["MySQL", "Docker", "Kafka"])]
    index = {
        "데브옵스": ["Docker", "Kafka"],  # 2개 기여
        "데이터베이스": ["MySQL"],  # 1개 기여
        "교양글쓰기": ["없는기술"],  # 0개 기여 (목표 토큰 아님)
    }

    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[("DB", "데이터베이스"), ("OPS", "데브옵스"), ("GE", "교양글쓰기")],
        course_tech_index=index,
    )

    contrib_by_id = {c.course_id: c for c in analysis.course_contributions}
    # 데브옵스: 2/3, 데이터베이스: 1/3, 교양: 0
    assert contrib_by_id["OPS"].contribution_ratio == pytest.approx(2 / 3)
    assert set(contrib_by_id["OPS"].added_tokens) == {"Docker", "Kafka"}
    assert contrib_by_id["DB"].contribution_ratio == pytest.approx(1 / 3)
    assert contrib_by_id["GE"].contribution_ratio == pytest.approx(0.0)
    assert contrib_by_id["GE"].added_tokens == []

    # 기여 큰 순 정렬 + 다음 액션은 기여>0 인 과목만, 데브옵스 우선
    assert [c.course_id for c in analysis.course_contributions] == ["OPS", "DB", "GE"]
    assert [a.course_id for a in analysis.next_actions] == ["OPS", "DB"]
    assert "67%" in analysis.next_actions[0].message  # round(2/3*100)=67


def test_next_actions_capped_at_three() -> None:
    """다음 액션은 기여 상위 3건으로 제한된다."""
    jobs = [_job("be", tech_stacks=["A", "B", "C", "D", "E"])]
    index = {
        "c1": ["A"],
        "c2": ["B"],
        "c3": ["C"],
        "c4": ["D"],
        "c5": ["E"],
    }
    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[(f"id{i}", f"c{i}") for i in range(1, 6)],
        course_tech_index=index,
    )
    assert len(analysis.next_actions) == 3


def test_token_alias_normalization_matches_across_notation() -> None:
    """직무·과목이 같은 기술을 다른 표기로 적어도 한 토큰으로 묶인다."""
    jobs = [_job("fe", tech_stacks=["ReactJS"])]
    index = {"프론트개론": ["React"]}  # 다른 표기

    analysis = compute_coverage(
        jobs,
        completed_course_names=["프론트개론"],
        roadmap_courses=[],
        course_tech_index=index,
    )
    assert analysis.required_count == 1
    assert analysis.current_covered == 1
    assert analysis.current_ratio == pytest.approx(1.0)


def test_overlapping_course_tokens_count_once_in_expected() -> None:
    """과목 토큰이 겹쳐도 예상 충족도는 합집합으로 한 번만 센다 (기여도는 독립)."""
    jobs = [_job("be", tech_stacks=["Docker", "MySQL"])]
    # 색인 키는 ``load_course_tech_index`` 가 만드는 casefold 정규화 형태여야 한다.
    index = {"수업a": ["Docker"], "수업b": ["Docker", "MySQL"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[("A", "수업A"), ("B", "수업B")],
        course_tech_index=index,
    )
    # 합집합 {Docker, MySQL} → 예상 2/2
    assert analysis.expected_covered == 2
    contrib = {c.course_id: c for c in analysis.course_contributions}
    # 독립 한계 기여: 두 과목 모두 Docker 기여 (현재 미충족 기준)
    assert "Docker" in contrib["A"].added_tokens
    assert "Docker" in contrib["B"].added_tokens
    # 기여도 합(0.5 + 1.0) > 예상-현재 증가분(1.0) — 독립 한계 기여의 의도된 성질
    assert contrib["B"].contribution_ratio == pytest.approx(1.0)


def test_already_completed_token_not_counted_as_contribution() -> None:
    """이미 이수로 덮은 토큰은 잔여 과목 기여에서 제외된다."""
    jobs = [_job("be", tech_stacks=["MySQL", "Docker"])]
    # 색인 키는 casefold 정규화 형태 (Latin 문자 포함 이름은 소문자).
    index = {"데이터베이스": ["MySQL"], "또다른db": ["MySQL"], "데브옵스": ["Docker"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],  # MySQL 이미 충족
        roadmap_courses=[("DB2", "또다른DB"), ("OPS", "데브옵스")],
        course_tech_index=index,
    )
    contrib = {c.course_id: c for c in analysis.course_contributions}
    assert contrib["DB2"].contribution_ratio == pytest.approx(0.0)  # MySQL 이미 덮음
    assert contrib["DB2"].added_tokens == []
    assert contrib["OPS"].contribution_ratio == pytest.approx(0.5)  # Docker 신규


def test_gap_tokens_lists_unreachable_targets() -> None:
    """이수·로드맵으로도 못 덮는 목표 토큰이 gap 으로 보고된다."""
    jobs = [_job("be", tech_stacks=["MySQL", "Kubernetes"])]
    index = {"데이터베이스": ["MySQL"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],
        roadmap_courses=[],
        course_tech_index=index,
    )
    assert analysis.expected_covered == 1
    assert analysis.gap_tokens == ["Kubernetes"]


def test_empty_target_returns_empty_analysis() -> None:
    """추천 직무가 없거나 직무 토큰이 비면 빈 분석을 반환한다."""
    analysis = compute_coverage(
        jobs=[_job("be", tech_stacks=[], competency_tags=[])],
        completed_course_names=["데이터베이스"],
        roadmap_courses=[("OPS", "데브옵스")],
        course_tech_index={"데이터베이스": ["MySQL"]},
    )
    assert analysis.required_count == 0
    assert analysis.current_ratio == pytest.approx(0.0)
    assert analysis.expected_ratio == pytest.approx(0.0)
    assert analysis.jobs == []
    assert analysis.course_contributions == []
    assert analysis.next_actions == []
    assert analysis.next_actions_covered == 0
    assert analysis.next_actions_ratio == pytest.approx(0.0)


def test_competency_tags_included_in_target() -> None:
    """역량 태그도 목표 토큰에 포함된다 (기술스택뿐 아니라 역량까지).

    분모는 도달 가능 토큰뿐이라, 로드맵이 가르치는 역량 태그(문제해결능력)는
    분모에 들고 어떤 과목도 안 가르치는 MySQL 은 gap 으로 빠진다.
    """
    jobs = [_job("be", tech_stacks=["MySQL"], competency_tags=["문제해결능력"])]
    index = {"문제해결입문": ["문제해결능력"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[("PS", "문제해결입문")],
        course_tech_index=index,
    )
    assert analysis.required_count == 1  # 도달 가능한 문제해결능력만 (MySQL 은 gap)
    assert analysis.expected_covered == 1  # 문제해결능력만 로드맵으로 도달
    assert analysis.gap_tokens == ["MySQL"]


def test_next_actions_ratio_is_union_not_sum_of_contributions() -> None:
    """다음 액션 과목을 모두 이수했을 때 도달 충족도는 합집합(중복 제거)이다.

    과목별 독립 기여의 합과 달리, 토큰이 겹치면 합집합이 작아 over-claim 을 막는다.
    """
    jobs = [_job("be", tech_stacks=["A", "B", "C", "D"])]
    index = {"c1": ["A", "B"], "c2": ["B", "C"], "c3": ["D"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[("ID1", "c1"), ("ID2", "c2"), ("ID3", "c3")],
        course_tech_index=index,
    )

    # 독립 기여 합 = 2/4 + 2/4 + 1/4 = 1.25 (>1) — 토큰 B 중복 때문
    assert sum(a.contribution_ratio for a in analysis.next_actions) > 1.0
    # 합집합 {A, B, C, D} → 4/4
    assert analysis.next_actions_covered == 4
    assert analysis.next_actions_ratio == pytest.approx(1.0)
    # 불변: current <= next_actions <= expected
    assert analysis.current_ratio <= analysis.next_actions_ratio <= analysis.expected_ratio


def test_next_actions_ratio_equals_current_when_no_contributing_courses() -> None:
    """기여 과목이 없으면 다음 액션 도달 충족도는 현재 충족도와 같다."""
    jobs = [_job("be", tech_stacks=["MySQL", "Docker"])]
    index = {"데이터베이스": ["MySQL"], "교양글쓰기": ["무관기술"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],  # 현재 MySQL 충족
        roadmap_courses=[("GE", "교양글쓰기")],  # 목표와 무관 → 기여 0
        course_tech_index=index,
    )

    assert analysis.next_actions == []
    assert analysis.next_actions_covered == analysis.current_covered == 1
    assert analysis.next_actions_ratio == pytest.approx(analysis.current_ratio)


# ---------------------------------------------------------------------------
# 도달 가능 토큰 분모 (issue #185)
# ---------------------------------------------------------------------------


def test_denominator_limited_to_reachable_tokens() -> None:
    """분모는 anchor 요구 토큰 중 도달 가능한 것만 — 도달 불가는 gap, expected=100%."""
    jobs = [_job("be", tech_stacks=["MySQL", "Docker", "Kubernetes"])]
    index = {"데이터베이스": ["MySQL"], "데브옵스": ["Docker"]}  # Kubernetes 가르치는 과목 없음

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],
        roadmap_courses=[("OPS", "데브옵스")],
        course_tech_index=index,
    )

    assert analysis.required_count == 2  # {MySQL, Docker} — Kubernetes 는 도달 불가 제외
    assert analysis.current_covered == 1  # MySQL
    assert analysis.expected_covered == 2  # 도달 가능 전부 → 로드맵 이수 후 100%
    assert analysis.expected_ratio == pytest.approx(1.0)
    assert analysis.gap_tokens == ["Kubernetes"]


def test_coverage_rises_when_completed_courses_reflected() -> None:
    """이수 과목을 반영하면 (도달 가능 분모 기준) 현재 충족도가 의미 있게 상승한다 (issue 조건 4)."""
    jobs = [_job("be", tech_stacks=["MySQL", "Docker"])]
    index = {"데이터베이스": ["MySQL"], "데브옵스": ["Docker"]}
    roadmap = [("DB", "데이터베이스"), ("OPS", "데브옵스")]

    before = compute_coverage(jobs, [], roadmap, index)
    after = compute_coverage(jobs, ["데이터베이스"], roadmap, index)

    assert before.required_count == after.required_count == 2  # 분모 안정
    assert before.current_ratio == pytest.approx(0.0)
    assert after.current_ratio == pytest.approx(0.5)  # 이수 반영 시 상승
    assert after.current_covered > before.current_covered


def test_three_tier_invariant_with_reachable_denominator() -> None:
    """도달 가능 분모에서도 3단 표기 불변 current <= next_actions <= expected 가 유지된다."""
    jobs = [_job("be", tech_stacks=["A", "B", "C", "Unreachable"])]
    index = {"c1": ["A"], "c2": ["B"], "c3": ["C"]}  # Unreachable 가르치는 과목 없음

    analysis = compute_coverage(
        jobs,
        completed_course_names=["c1"],
        roadmap_courses=[("ID2", "c2"), ("ID3", "c3")],
        course_tech_index=index,
    )

    assert analysis.required_count == 3  # A,B,C 도달 가능 / Unreachable 은 gap
    assert analysis.gap_tokens == ["Unreachable"]
    assert analysis.current_ratio <= analysis.next_actions_ratio <= analysis.expected_ratio
    assert analysis.expected_ratio == pytest.approx(1.0)


def test_reachable_target_capped_at_thirty_by_frequency() -> None:
    """도달 가능 토큰이 30 을 넘으면 학습 빈도 상위 30 개만 분모로 남긴다 (issue 조건 2)."""
    high = [f"HighTok{i:02d}" for i in range(30)]  # 두 과목이 가르침 → 빈도 2
    low = [f"LowTok{i:02d}" for i in range(5)]  # 한 과목만 가르침 → 빈도 1
    jobs = [_job("be", tech_stacks=high + low)]
    index = {"고빈도수업a": high, "고빈도수업b": high, "저빈도수업": low}

    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[("HA", "고빈도수업a"), ("HB", "고빈도수업b"), ("LO", "저빈도수업")],
        course_tech_index=index,
    )

    assert analysis.required_count == 30  # 35 도달 가능 → 상위 30 캡
    # 분모 토큰 재구성: completed 가 비어 모든 목표 토큰이 미충족분 = 기여 added_tokens 합집합.
    target_tokens: set[str] = set()
    for contribution in analysis.course_contributions:
        target_tokens |= set(contribution.added_tokens)
    assert target_tokens == set(high)  # 빈도 높은 30 개만 채택
    assert not (set(low) & target_tokens)  # 저빈도 5 개 제외


def test_reachable_target_not_capped_at_boundary() -> None:
    """도달 가능 토큰이 정확히 상한(30)이면 캡 없이 전부 분모가 된다."""
    tokens = [f"Tok{i:02d}" for i in range(30)]
    jobs = [_job("be", tech_stacks=tokens)]
    index = {"전부수업": tokens}

    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[("ALL", "전부수업")],
        course_tech_index=index,
    )
    assert analysis.required_count == 30


def test_completed_tokens_preserved_when_cap_triggers() -> None:
    """캡이 걸려도 이수로 충족한 토큰은 분모에 남아 현재 충족이 보존된다.

    이수 토큰은 학습 빈도가 낮아도 우선 보존한다. 보존하지 않으면 고빈도
    로드맵 토큰에 밀려 분모에서 빠지고 current 가 0 으로 과소집계된다.
    """
    completed_tokens = [f"CompTok{i:02d}" for i in range(3)]  # 이수 1 과목 → 빈도 1
    high_tokens = [f"HighTok{i:02d}" for i in range(32)]  # 로드맵 2 과목 → 빈도 2
    jobs = [_job("be", tech_stacks=completed_tokens + high_tokens)]
    index = {"이수수업": completed_tokens, "고빈도a": high_tokens, "고빈도b": high_tokens}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["이수수업"],
        roadmap_courses=[("HA", "고빈도a"), ("HB", "고빈도b")],
        course_tech_index=index,
    )

    # 35 도달 가능 → 30 캡. 이수 충족 3 개 우선 보존 + 나머지 27 은 고빈도로 채움.
    assert analysis.required_count == 30
    assert analysis.current_covered == 3  # 이수 토큰 보존 (보존 없으면 0)
    assert analysis.current_ratio == pytest.approx(0.1)


def test_anchor_with_no_reachable_tokens_keeps_anchor_and_gap() -> None:
    """기준 직무는 있으나 도달 가능 토큰이 없으면 required_count 0 + anchor·gap 보존."""
    jobs = [_job("be", job_name="백엔드", tech_stacks=["MySQL", "Docker"])]
    index = {"교양글쓰기": ["무관기술"]}  # anchor 토큰 가르치는 과목 없음

    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[("GE", "교양글쓰기")],
        course_tech_index=index,
    )

    assert analysis.anchor_job_id == "be"  # 추천 직무는 있으니 anchor 보존
    assert analysis.anchor_job_name == "백엔드"
    assert analysis.required_count == 0
    assert analysis.current_ratio == pytest.approx(0.0)
    assert analysis.expected_ratio == pytest.approx(0.0)  # 분모 0 → 0.0 (1.0 아님)
    assert set(analysis.gap_tokens) == {"MySQL", "Docker"}  # 전부 도달 불가
    assert analysis.jobs  # 분야별 비교 카드는 anchor 무관 — 여전히 산출


def test_jobs_field_uses_full_job_tokens_not_reachable_scoped() -> None:
    """분야별 비교 카드(jobs)는 도달 가능 축소 없이 각 직무 전체 토큰 기준이다.

    메인 게이지는 도달 가능 분모로 좁히지만, 직무 간 비교는 각자 전체 요구
    토큰 기준이라야 공정하다 (도달 가능 토큰이 적은 직무가 과대 충족으로 보이지 않게).
    """
    jobs = [_job("be", job_name="백엔드", tech_stacks=["MySQL", "Unreachable"])]
    index = {"데이터베이스": ["MySQL"]}

    analysis = compute_coverage(jobs, ["데이터베이스"], [], index)

    assert analysis.required_count == 1  # 메인 게이지 분모 = 도달 가능 {MySQL}
    by_job = {coverage.job_id: coverage for coverage in analysis.jobs}
    assert by_job["be"].required_count == 2  # 분야별 카드는 직무 전체 토큰 기준
    assert by_job["be"].current_covered == 1
    assert by_job["be"].missing_tokens == ["Unreachable"]


# ---------------------------------------------------------------------------
# CoverageAnalysisNode — 노드 진입점 (색인 로딩 포함)
# ---------------------------------------------------------------------------


def _write_catalog(tmp_path: Path, courses: list[dict[str, Any]]) -> Path:
    path = tmp_path / "courses.yaml"
    path.write_text(yaml.safe_dump({"courses": courses}, allow_unicode=True), encoding="utf-8")
    return path


def _roadmap_state(courses: list[dict[str, Any]]) -> dict[str, Any]:
    """학기 분산 1건만 채운 최소 로드맵 dict (노드는 semesters 만 사용)."""
    return {
        "stages": [
            {"stage": stage, "courses": []}
            for stage in ("foundation", "core", "application", "industry")
        ],
        "semesters": [
            {
                "semester": 3,
                "grade": 2,
                "courses": courses,
                "credits_total": sum(c.get("credits", 3) for c in courses),
                "cap_reached": False,
                "graduation_insufficient": False,
            }
        ],
        "derived_from_combo_key": "T_A::T_B",
    }


def _normalized_profile(completed_courses: list[str]) -> dict[str, Any]:
    return {
        "admission_year": 2025,
        "college": "C1",
        "department": "컴퓨터공학부",
        "current_tracks": [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": completed_courses,
    }


def test_node_returns_coverage_analysis_with_ok_trace(tmp_path: Path) -> None:
    catalog = _write_catalog(
        tmp_path,
        [
            {"course_id": "DB", "course_name": "데이터베이스", "tech_stacks": ["MySQL"]},
            {"course_id": "OPS", "course_name": "데브옵스", "tech_stacks": ["Docker"]},
        ],
    )
    node = CoverageAnalysisNode(course_catalog_path=catalog)

    state = {
        "recommended_jobs": [
            _job("be", job_name="백엔드", tech_stacks=["MySQL", "Docker"]).model_dump(mode="json")
        ],
        "roadmap": _roadmap_state(
            [
                {
                    "course_id": "OPS",
                    "course_name": "데브옵스",
                    "credits": 3,
                    "stage": "core",
                    "score": 0.5,
                }
            ]
        ),
        "normalized_profile": _normalized_profile(completed_courses=["데이터베이스"]),
    }

    result = node(state)

    assert result["trace"] == ["coverage_analysis:ok"]
    analysis = result["coverage_analysis"]
    assert analysis["required_count"] == 2
    assert analysis["current_covered"] == 1
    assert analysis["expected_covered"] == 2
    assert analysis["current_ratio"] == pytest.approx(0.5)
    assert analysis["expected_ratio"] == pytest.approx(1.0)
    assert analysis["course_contributions"][0]["course_id"] == "OPS"
    assert analysis["next_actions"][0]["course_id"] == "OPS"
    assert analysis["next_actions_covered"] == 2
    assert analysis["next_actions_ratio"] == pytest.approx(1.0)
    assert analysis["anchor_job_id"] == "be"
    assert analysis["anchor_job_name"] == "백엔드"
    assert analysis["jobs"][0]["job_id"] == "be"


def test_node_uses_anchor_job_id_from_state(tmp_path: Path) -> None:
    """state 의 anchor_job_id 가 기준 직무를 1순위가 아닌 지정 직무로 바꾼다."""
    catalog = _write_catalog(
        tmp_path,
        [{"course_id": "FE", "course_name": "프론트개론", "tech_stacks": ["React"]}],
    )
    node = CoverageAnalysisNode(course_catalog_path=catalog)

    state = {
        "recommended_jobs": [
            _job("be", job_name="백엔드", tech_stacks=["MySQL"], match_score=0.9).model_dump(
                mode="json"
            ),
            _job("fe", job_name="프론트", tech_stacks=["React"], match_score=0.6).model_dump(
                mode="json"
            ),
        ],
        "roadmap": _roadmap_state([]),
        "normalized_profile": _normalized_profile(completed_courses=["프론트개론"]),
        "anchor_job_id": "fe",
    }

    analysis = node(state)["coverage_analysis"]

    # 게이지는 anchor(fe) 에 정렬된다.
    assert analysis["anchor_job_id"] == "fe"
    assert analysis["required_count"] == 1  # fe 토큰 {React} 뿐
    assert analysis["current_covered"] == 1
    # 분야별 분석(jobs)은 anchor 와 무관하게 전 직무를 담는다.
    assert {coverage["job_id"] for coverage in analysis["jobs"]} == {"be", "fe"}


def test_node_empty_when_no_recommended_jobs(tmp_path: Path) -> None:
    catalog = _write_catalog(
        tmp_path, [{"course_id": "DB", "course_name": "데이터베이스", "tech_stacks": ["MySQL"]}]
    )
    node = CoverageAnalysisNode(course_catalog_path=catalog)

    result = node(
        {
            "recommended_jobs": [],
            "roadmap": _roadmap_state([]),
            "normalized_profile": _normalized_profile(completed_courses=["데이터베이스"]),
        }
    )

    assert result["trace"] == ["coverage_analysis:empty"]
    assert result["coverage_analysis"]["required_count"] == 0
    assert result["coverage_analysis"]["jobs"] == []


def test_node_handles_missing_roadmap_gracefully(tmp_path: Path) -> None:
    """로드맵이 비면 예상 충족도는 현재와 같다 (잔여 과목 기여 없음)."""
    catalog = _write_catalog(
        tmp_path, [{"course_id": "DB", "course_name": "데이터베이스", "tech_stacks": ["MySQL"]}]
    )
    node = CoverageAnalysisNode(course_catalog_path=catalog)

    result = node(
        {
            "recommended_jobs": [
                _job("be", tech_stacks=["MySQL", "Docker"]).model_dump(mode="json")
            ],
            "roadmap": None,
            "normalized_profile": _normalized_profile(completed_courses=["데이터베이스"]),
        }
    )

    analysis = result["coverage_analysis"]
    assert analysis["current_covered"] == 1
    assert analysis["expected_covered"] == 1  # 로드맵 없음 → 현재와 동일
    assert analysis["course_contributions"] == []
    assert analysis["next_actions"] == []


def test_node_emits_no_reachable_tokens_trace(tmp_path: Path) -> None:
    """기준 직무는 있으나 도달 가능 토큰이 없으면 전용 trace 를 남긴다 (empty 와 구분)."""
    catalog = _write_catalog(
        tmp_path, [{"course_id": "GE", "course_name": "교양글쓰기", "tech_stacks": ["무관기술"]}]
    )
    node = CoverageAnalysisNode(course_catalog_path=catalog)

    result = node(
        {
            "recommended_jobs": [
                _job("be", job_name="백엔드", tech_stacks=["MySQL", "Docker"]).model_dump(
                    mode="json"
                )
            ],
            "roadmap": _roadmap_state(
                [
                    {
                        "course_id": "GE",
                        "course_name": "교양글쓰기",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.5,
                    }
                ]
            ),
            "normalized_profile": _normalized_profile(completed_courses=[]),
        }
    )

    assert result["trace"] == ["coverage_analysis:no_reachable_tokens"]
    analysis = result["coverage_analysis"]
    assert analysis["anchor_job_id"] == "be"  # anchor 는 보존
    assert analysis["required_count"] == 0
    assert set(analysis["gap_tokens"]) == {"MySQL", "Docker"}
