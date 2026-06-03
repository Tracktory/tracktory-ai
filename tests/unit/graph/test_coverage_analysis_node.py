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
) -> JobCandidate:
    return JobCandidate(
        job_id=job_id,
        job_name=job_name or job_id,
        tech_stacks=tech_stacks or [],
        competency_tags=competency_tags or [],
        match_score=0.8,
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


def test_per_field_job_levels_current_and_expected() -> None:
    """분야(직무)별 현재/예상 역량 수준을 제공한다 (issue 조건 2)."""
    jobs = [
        _job("be", job_name="백엔드", tech_stacks=["MySQL", "Docker"]),
        _job("fe", job_name="프론트", tech_stacks=["React"]),
    ]
    index = {"데이터베이스": ["MySQL"], "프론트개론": ["React"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=["데이터베이스"],
        roadmap_courses=[("FE", "프론트개론")],
        course_tech_index=index,
    )

    by_job = {job.job_id: job for job in analysis.jobs}
    assert by_job["be"].required_count == 2
    assert by_job["be"].current_covered == 1  # MySQL
    assert by_job["be"].expected_covered == 1  # Docker 미공급 → 그대로
    assert by_job["be"].missing_tokens == ["Docker"]
    assert by_job["fe"].required_count == 1
    assert by_job["fe"].current_covered == 0
    assert by_job["fe"].expected_covered == 1  # 로드맵 프론트개론 → React
    assert by_job["fe"].current_ratio == pytest.approx(0.0)
    assert by_job["fe"].expected_ratio == pytest.approx(1.0)


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
    """역량 태그도 목표 토큰에 포함된다 (기술스택뿐 아니라 역량까지)."""
    jobs = [_job("be", tech_stacks=["MySQL"], competency_tags=["문제해결능력"])]
    index = {"문제해결입문": ["문제해결능력"]}

    analysis = compute_coverage(
        jobs,
        completed_course_names=[],
        roadmap_courses=[("PS", "문제해결입문")],
        course_tech_index=index,
    )
    assert analysis.required_count == 2  # MySQL + 문제해결능력
    assert analysis.expected_covered == 1  # 문제해결능력만 로드맵으로 도달


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
    assert analysis["jobs"][0]["job_id"] == "be"


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
