"""역량 커버리지 분석 도메인 모델 검증 단위 테스트.

비율 필드 [0, 1] 강제·필수 식별자 비어있음 차단·리스트 기본값 등 모델 계약을
외부 의존성 없이 검증한다.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tracktory.graph.models import (
    CourseCoverageContribution,
    CoverageAnalysis,
    JobCoverage,
    NextActionSuggestion,
)


def test_coverage_analysis_accepts_valid_payload() -> None:
    analysis = CoverageAnalysis(
        anchor_job_id="be",
        anchor_job_name="백엔드 개발자",
        reachable_required_count=4,
        current_covered=1,
        expected_covered=3,
        current_ratio=0.25,
        expected_ratio=0.75,
        next_actions_covered=2,
        next_actions_ratio=0.5,
        jobs=[
            JobCoverage(
                job_id="be",
                job_name="백엔드 개발자",
                total_required_count=4,
                current_covered=1,
                expected_covered=3,
                current_ratio=0.25,
                expected_ratio=0.75,
                missing_tokens=["Docker"],
            )
        ],
        course_contributions=[
            CourseCoverageContribution(
                course_id="c1",
                course_name="데이터베이스",
                added_tokens=["MySQL"],
                contribution_ratio=0.25,
            )
        ],
        next_actions=[
            NextActionSuggestion(
                course_id="c1",
                course_name="데이터베이스",
                contribution_ratio=0.25,
                message="데이터베이스를 이수하면 역량 충족도가 약 25% 오릅니다.",
            )
        ],
        gap_tokens=["Docker"],
    )

    assert analysis.current_ratio == pytest.approx(0.25)
    assert analysis.anchor_job_id == "be"
    assert analysis.anchor_job_name == "백엔드 개발자"
    assert analysis.jobs[0].missing_tokens == ["Docker"]
    assert analysis.next_actions[0].course_id == "c1"
    assert analysis.next_actions_ratio == pytest.approx(0.5)


def test_coverage_analysis_defaults_lists_empty() -> None:
    """리스트 필드는 기본값으로 비어 있어 목표 토큰 부재 시 graceful 종료가 가능하다."""
    analysis = CoverageAnalysis(
        reachable_required_count=0,
        current_covered=0,
        expected_covered=0,
        current_ratio=0.0,
        expected_ratio=0.0,
    )
    assert analysis.anchor_job_id == ""
    assert analysis.anchor_job_name == ""
    assert analysis.jobs == []
    assert analysis.course_contributions == []
    assert analysis.next_actions == []
    assert analysis.gap_tokens == []
    assert analysis.next_actions_covered == 0
    assert analysis.next_actions_ratio == 0.0


@pytest.mark.parametrize("ratio", [-0.01, 1.01, 2.0])
def test_coverage_analysis_rejects_ratio_out_of_unit_interval(ratio: float) -> None:
    """비율 필드는 [0, 1] 정의역을 벗어나면 거부한다."""
    with pytest.raises(ValidationError):
        CoverageAnalysis(
            reachable_required_count=4,
            current_covered=1,
            expected_covered=3,
            current_ratio=ratio,
            expected_ratio=0.5,
        )


def test_job_coverage_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        JobCoverage(
            job_id="be",
            job_name="백엔드 개발자",
            total_required_count=-1,
            current_covered=0,
            expected_covered=0,
            current_ratio=0.0,
            expected_ratio=0.0,
        )


def test_course_contribution_rejects_ratio_above_one() -> None:
    with pytest.raises(ValidationError):
        CourseCoverageContribution(
            course_id="c1",
            course_name="데이터베이스",
            added_tokens=["MySQL"],
            contribution_ratio=1.5,
        )


def test_next_action_requires_nonempty_message() -> None:
    with pytest.raises(ValidationError):
        NextActionSuggestion(
            course_id="c1",
            course_name="데이터베이스",
            contribution_ratio=0.25,
            message="",
        )
