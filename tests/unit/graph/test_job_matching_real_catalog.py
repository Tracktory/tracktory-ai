"""직무 매칭 노드가 *배포되는* 과목 카탈로그로 이수 과목 부스팅을 수행하는지 검증.

다른 노드 테스트는 합성 tmp 카탈로그로 부스팅 로직을 검증한다. 본 테스트는
기본 경로(``config/courses.yaml``)로 노드를 구성해, 실제 배포 카탈로그가
이수 과목 부스팅에 쓸 기술 토큰을 실제로 담고 있는지를 회귀 가드로 고정한다.
원래 버그(카탈로그에 토큰이 없어 부스팅이 항상 0)가 재발하면 본 테스트가
먼저 깨진다.

카탈로그 데이터에 결합하지 않도록, 토큰을 가진 과목을 카탈로그에서 직접 골라
그 토큰을 직무에 심는 self-validating 방식을 쓴다 — 특정 과목명·토큰을
하드코딩하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tracktory.graph.nodes.job_matching import (
    JobMatchingNode,
    _load_course_tech_index,
)
from tracktory.rag.job_search import RagSearchResult

_REAL_CATALOG_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "tracktory" / "config" / "courses.yaml"
)


class _SingleResultClient:
    """결과 1건을 그대로 돌려주는 직무 검색 boundary 대역."""

    def __init__(self, result: RagSearchResult) -> None:
        self._result = result

    def rag_search_jobs(self, query: str, top_k: int = 3) -> list[RagSearchResult]:
        return [self._result]


def _profile(completed_courses: list[str]) -> dict[str, object]:
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


def test_shipped_catalog_carries_boost_tokens() -> None:
    """배포 카탈로그가 부스팅에 쓸 기술 토큰을 1건 이상 담고 있어야 한다."""
    index = _load_course_tech_index(_REAL_CATALOG_PATH)
    courses_with_tokens = {name: toks for name, toks in index.items() if toks}
    assert courses_with_tokens, "배포 courses.yaml 에 tech_stacks 토큰이 전혀 없음 (#159 회귀)"


def test_real_completed_course_boosts_matching_job() -> None:
    """카탈로그에서 토큰을 가진 과목 1개를 골라, 그 토큰을 가진 직무가 부스팅되는지 검증."""
    index = _load_course_tech_index(_REAL_CATALOG_PATH)
    course_name, tokens = next((n, t) for n, t in index.items() if t)

    result = RagSearchResult(
        job_id="probe",
        job_name="probe",
        score=0.6,
        description="probe job",
        tech_stacks=[tokens[0]],
        competency_tags=[],
    )
    node = JobMatchingNode(
        job_search_client=_SingleResultClient(result),
        course_catalog_path=_REAL_CATALOG_PATH,
    )
    state = {"profile_text": "백엔드 개발에 관심 있는 학생"}

    without = node({**state, "normalized_profile": _profile(completed_courses=[])})
    with_course = node({**state, "normalized_profile": _profile(completed_courses=[course_name])})

    raw = without["recommended_jobs"][0]["match_score"]
    boosted = with_course["recommended_jobs"][0]["match_score"]
    assert raw == pytest.approx(0.6)
    assert boosted > raw
    # similarity 는 raw 검색 점수를 보존한다 (부스팅은 match_score 에만).
    assert with_course["recommended_jobs"][0]["similarity"] == pytest.approx(0.6)


def test_real_irrelevant_course_does_not_boost() -> None:
    """카탈로그에 없는 과목명은 토큰이 없어 점수를 바꾸지 않는다 (오반영 없음)."""
    result = RagSearchResult(
        job_id="probe",
        job_name="probe",
        score=0.6,
        description="probe job",
        tech_stacks=["Spring", "Java"],
        competency_tags=[],
    )
    node = JobMatchingNode(
        job_search_client=_SingleResultClient(result),
        course_catalog_path=_REAL_CATALOG_PATH,
    )
    state = {"profile_text": "백엔드 개발에 관심 있는 학생"}
    out = node(
        {**state, "normalized_profile": _profile(completed_courses=["존재하지않는과목명zzz"])}
    )
    assert out["recommended_jobs"][0]["match_score"] == pytest.approx(0.6)


def test_real_course_without_tokens_does_not_boost() -> None:
    """카탈로그에 *있지만* 기술 토큰이 없는 실제 과목은 직무 점수를 바꾸지 않는다.

    추출 오탐(예: 어떤 비전공 과목에 web 토큰이 새는 경우)이 재발하면 본 테스트가
    깨진다 — 미지의 과목명만 검증하는 위 케이스와 달리, 배포 카탈로그 안의 실제
    토큰-없는 과목으로 criterion 4(무관한 과목은 영향 없음)를 고정한다.
    """
    catalog = yaml.safe_load(_REAL_CATALOG_PATH.read_text(encoding="utf-8"))["courses"]
    index = _load_course_tech_index(_REAL_CATALOG_PATH)
    # 색인은 토큰 없는 과목을 버리므로(부스팅 lookup 위장 방지), 색인에 없는
    # 실제 과목명 = 토큰을 한 개도 기여하지 않는 배포 카탈로그 과목이다.
    empty_course = next(
        (
            str(c["course_name"])
            for c in catalog
            if c.get("course_name") and str(c["course_name"]).strip().casefold() not in index
        ),
        None,
    )
    if empty_course is None:
        pytest.skip("배포 카탈로그의 모든 과목이 토큰을 가짐 — 본 가드 불필요")

    result = RagSearchResult(
        job_id="probe",
        job_name="probe",
        score=0.6,
        description="probe job",
        tech_stacks=["Spring", "Java", "Python", "React"],
        competency_tags=[],
    )
    node = JobMatchingNode(
        job_search_client=_SingleResultClient(result),
        course_catalog_path=_REAL_CATALOG_PATH,
    )
    out = node(
        {
            "profile_text": "백엔드 개발에 관심 있는 학생",
            "normalized_profile": _profile(completed_courses=[empty_course]),
        }
    )
    assert out["recommended_jobs"][0]["match_score"] == pytest.approx(0.6)
