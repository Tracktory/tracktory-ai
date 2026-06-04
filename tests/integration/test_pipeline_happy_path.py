"""추천 파이프라인 그래프 조립 + 컴파일 진입점의 end-to-end 통합 테스트.

본 모듈은 6 노드를 순차 실행하는 그래프 빌더 (``build_recommendation_graph``)
와 1 회 컴파일 헬퍼 (``get_recommendation_graph``) 의 계약을 검증한다. 외부
의존성 4 종은 모두 ``MagicMock(spec=...)`` 로 격리되며, 실제 RAGFlow /
LLM / Repository 호출은 발생하지 않는다.

테스트 격리:
    모듈 전역 ``lru_cache`` 가 테스트 간 누출되지 않도록 autouse fixture 가
    매 테스트 진입·종료 시 ``cache_clear()`` 를 호출한다.

``@pytest.mark.integration`` 으로 표시 — CI 기본 실행에서 제외하려면
``pytest -m "not integration"`` 을 사용한다.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from tracktory.graph import pipeline as graph_pipeline
from tracktory.graph.models import Course, Explanation, Track
from tracktory.graph.nodes.llm_explanation import LLMClient
from tracktory.graph.nodes.roadmap import CourseRepository
from tracktory.graph.nodes.track_synergy import TrackRepository
from tracktory.graph.pipeline import (
    PipelineClients,
    PipelineConfig,
    build_recommendation_graph,
    get_recommendation_graph,
)
from tracktory.rag.job_search import JobSearchClient, RagSearchResult

pytestmark = pytest.mark.integration


_EMBED_DIM = 1536

# 모의 의존성만 쓰는 hermetic happy path 의 wall-clock 상한. 네트워크·실제
# 임베딩이 끼어들면 깨지는 회귀 가드로, 운영 SLO (30s) 보다 훨씬 엄격하다.
_FAST_PATH_SECONDS: float = 3.0


# ---------------------------------------------------------------------------
# Fixtures — autouse cache 격리
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_pipeline_cache() -> Iterator[None]:
    """모듈 전역 lru_cache 가 테스트 간 누출되지 않도록 자동 격리.

    ``MagicMock(spec=Protocol)`` 의 hash 는 id 기반이라 매번 새로 만든
    mock 인스턴스로 호출하면 캐시 미스가 정상이다. 하지만 1 회 컴파일
    테스트는 동일 인스턴스를 재사용해 캐시 hit 를 검증하므로, 다른
    테스트의 첫 호출이 이전 테스트의 monkeypatch 잔존을 먹는 일을 피한다.
    """
    get_recommendation_graph.cache_clear()
    yield
    get_recommendation_graph.cache_clear()


# ---------------------------------------------------------------------------
# 헬퍼 — Track / Course / 정규화 프로필 / 의존성 4 종
#
# tests/integration/test_track_synergy_flow.py 의 헬퍼 패턴을 그대로 가져왔다.
# tests/integration/ 에 ``__init__.py`` 가 없어 직접 import 가 보장되지 않으므로
# 의도적으로 본 파일 안에 복제했다.
# ---------------------------------------------------------------------------


def _meta_vector(seed: int) -> list[float]:
    """deterministic seed 기반 L2-normalized 트랙 메타 벡터."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(_EMBED_DIM)
    norm = float(np.linalg.norm(raw))
    return (raw / norm).tolist() if norm > 0 else []


def _track(
    track_id: str,
    *,
    college_id: str,
    department_id: str,
    course_ids: list[str],
    tech_stacks: list[str],
    competencies: list[str],
    meta_seed: int,
) -> Track:
    return Track(
        college_id=college_id,
        department_id=department_id,
        major_id=department_id,
        track_id=track_id,
        track_name=track_id,
        course_ids=course_ids,
        meta_text="",
        meta_vector=_meta_vector(meta_seed),
        competencies=competencies,
        tech_stacks=tech_stacks,
    )


def _course(
    course_id: str,
    *,
    track_id: str,
    stage: str,
    prereq_ids: list[str] | None = None,
    credits: int = 3,
    priority: int = 1,
) -> Course:
    return Course(
        course_id=course_id,
        course_name=f"{course_id} 강의",
        credits=credits,
        stage=stage,  # type: ignore[arg-type]
        prereq_ids=prereq_ids or [],
        track_ids=[track_id],
        priority=priority,
    )


def _build_tracks() -> list[Track]:
    """단과대 cross 후보가 존재하는 7 트랙 셋업.

    사용자 단과대 = ``C1``. 본 셋업은 (a) 같은 단과대 내 트랙 3 개 +
    (b) 다른 단과대 트랙 4 개로 구성되어 슬롯 3 (cross-college 예약) 이
    fallback 없이 정상 채워지도록 한다.
    """
    in_college = [
        _track(
            f"in{i}",
            college_id="C1",
            department_id="D1",
            course_ids=[f"in_co{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"in_c{i}"],
            meta_seed=i,
        )
        for i in range(3)
    ]
    out_college = [
        _track(
            f"out{i}",
            college_id="C2",
            department_id="D2",
            course_ids=[f"out_co{i}"],
            tech_stacks=["py", "sql"],
            competencies=[f"out_c{i}"],
            meta_seed=100 + i,
        )
        for i in range(4)
    ]
    return in_college + out_college


def _build_courses() -> list[Course]:
    """4 단계 (foundation/core/application/industry) 를 모두 덮는 6 과목.

    선수 관계는 ``foundation → core → application → industry`` 단조 증가로
    설정해 ``_validate_prereqs`` 가 모든 후보를 통과시킨다.
    """
    return [
        _course("CS101", track_id="in0", stage="foundation", priority=1),
        _course("CS102", track_id="in1", stage="foundation", priority=2),
        _course("CS201", track_id="in0", stage="core", prereq_ids=["CS101"], priority=1),
        _course("CS202", track_id="in1", stage="core", prereq_ids=["CS101", "CS102"], priority=2),
        _course("CS301", track_id="in0", stage="application", prereq_ids=["CS201"], priority=1),
        _course("CS401", track_id="in0", stage="industry", prereq_ids=["CS301"], priority=1),
    ]


def _build_clients(
    *,
    job_score: float = 0.85,
    explanation_text: str = "추천 결과 종합 설명입니다.",
) -> PipelineClients:
    """4 종 의존성을 ``MagicMock(spec=...)`` 로 채운 컨테이너 생성.

    Args:
        job_score: 직무 검색 boundary 가 반환하는 상위 결과 점수.
            ``min_job_similarity`` (0.3) 이상이면 정상 분기로 흐른다.
        explanation_text: LLM 클라이언트가 invoke 결과로 반환하는 본문.
    """
    job_search = MagicMock(spec=JobSearchClient)
    job_search.rag_search_jobs.return_value = [
        RagSearchResult(
            job_id="BE",
            job_name="백엔드 개발자",
            score=job_score,
            description="서버 사이드 시스템을 설계하고 운영합니다.",
            tech_stacks=["py", "sql"],
            competency_tags=["문제해결능력"],
        ),
    ]

    track_repo = MagicMock(spec=TrackRepository)
    track_repo.list_all.return_value = _build_tracks()

    course_repo = MagicMock(spec=CourseRepository)
    course_repo.list_for_tracks.return_value = _build_courses()

    llm_client = MagicMock(spec=LLMClient)
    llm_client.invoke.return_value = Explanation(text=explanation_text, sections=[])

    return PipelineClients(
        job_search_client=job_search,
        track_repository=track_repo,
        course_repository=course_repo,
        llm_client=llm_client,
    )


def _valid_onboarding_payload(
    *, college: str = "C1", current_tracks: list[str] | None = None
) -> dict[str, Any]:
    """입력 정규화 노드가 검증을 통과하는 유효한 온보딩 입력.

    Args:
        college: 사용자 소속 단과대 ID.
        current_tracks: 현재 선택 트랙. ``None`` (1학년 — 트랙 미선택) 이면 빈
            리스트로, 2학년+ 페르소나는 정확히 2 개를 넘긴다.
    """
    return {
        "admission_year": 2025,
        "college": college,
        "department": "컴퓨터공학부",
        "current_tracks": current_tracks or [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("persona", "current_tracks"),
    [
        ("1학년 트랙 미선택", []),
        ("2학년+ 트랙 선택 완료", ["in0", "in1"]),
    ],
)
def test_recommendation_graph_happy_path_runs_seven_nodes_in_order(
    persona: str, current_tracks: list[str]
) -> None:
    """유효 입력으로 그래프를 invoke 하면 7 노드가 순차 실행되어 모든 산출 키가 채워진다.

    1학년 (트랙 미선택) 과 2학년+ (트랙 선택 완료) 두 페르소나 모두 동일한
    7 부분 산출을 만들어내는지 검증한다. 트랙 선택 여부는 트랙 시너지 노드의
    주 추천 후보 풀 생성 분기만 가를 뿐, end-to-end 계약은 동일하다.
    """
    clients = _build_clients()
    graph = build_recommendation_graph(clients)

    start = time.perf_counter()
    result = graph.invoke(
        {
            "user_id": "u1",
            "raw_input": _valid_onboarding_payload(current_tracks=current_tracks),
        },
    )
    elapsed = time.perf_counter() - start

    # 정규화·직렬화·직무·트랙·로드맵·설명 — 6 영역 모두 키 채워짐
    assert result.get("normalized_profile") is not None
    assert result.get("profile_text")
    assert result.get("recommended_jobs")
    assert len(result["primary_combos"]) == 2
    assert len(result["secondary_combos"]) == 5
    assert result["roadmap"]["stages"]
    assert result["explanation"]["text"] == "추천 결과 종합 설명입니다."

    # 역량 커버리지 분석 — 분모는 도달 가능(이수·로드맵) 토큰으로 좁혀진다.
    # 본 해피패스는 기본 과목 카탈로그를 쓰고 합성 로드맵 과목명이 카탈로그에
    # 없어 도달 가능 토큰이 비므로 required_count 는 0 이다 (분모 산식 자체의
    # 검증은 test_coverage_analysis_flows_through_graph_with_real_signal 가 맡는다).
    coverage = result["coverage_analysis"]
    assert coverage is not None
    assert coverage["required_count"] == 0  # 도달 가능 토큰 없음 (합성 과목 미매핑)
    assert 0.0 <= coverage["current_ratio"] <= 1.0
    assert 0.0 <= coverage["expected_ratio"] <= 1.0

    # 7 노드 trace 흔적 (각 노드는 자체 trace prefix 를 남긴다)
    trace_prefixes = {token.split(":", 1)[0] for token in result["trace"]}
    assert {
        "input_normalize",
        "profile_embed",
        "job_matching",
        "track_synergy",
        "roadmap",
        "coverage_analysis",
        "llm_explanation",
    } <= trace_prefixes

    # 외부 boundary 가 실제로 호출됨 (계약 검증)
    clients.job_search_client.rag_search_jobs.assert_called_once()
    clients.track_repository.list_all.assert_called_once()
    clients.course_repository.list_for_tracks.assert_called_once()
    clients.llm_client.invoke.assert_called_once()

    # hermetic 경로 속도 회귀 가드 — 모의 환경에서 3 초 안에 끝나야 한다.
    assert elapsed < _FAST_PATH_SECONDS


def test_recommendation_graph_invalid_input_short_circuits_to_end() -> None:
    """입력 정규화 실패 시 ``end`` 분기로 안전 종료 — 후속 노드 미실행."""
    clients = _build_clients()
    graph = build_recommendation_graph(clients)

    result = graph.invoke(
        {"user_id": "u1", "raw_input": {"admission_year": "INVALID"}},
    )

    # 정규화 실패 — 후속 노드는 모두 미실행
    assert result.get("normalized_profile") is None
    assert result.get("profile_text") is None
    assert result.get("recommended_jobs") is None
    assert result.get("primary_combos") is None
    assert result.get("roadmap") is None
    assert result.get("coverage_analysis") is None
    assert result.get("explanation") is None

    # trace 에 input_normalize 실패만 흐름
    assert any(token.startswith("input_normalize:fail") for token in result["trace"])
    assert not any(token.startswith("profile_embed") for token in result["trace"])
    assert not any(token.startswith("job_matching") for token in result["trace"])

    # 안전 종료이므로 외부 boundary 도 호출되지 않음
    clients.job_search_client.rag_search_jobs.assert_not_called()
    clients.track_repository.list_all.assert_not_called()
    clients.llm_client.invoke.assert_not_called()


def test_get_recommendation_graph_compiles_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """동일 의존성에 대해 헬퍼를 2회 호출해도 ``build_recommendation_graph`` 는 1회만 호출된다."""
    wrapped = MagicMock(wraps=graph_pipeline.build_recommendation_graph)
    monkeypatch.setattr(graph_pipeline, "build_recommendation_graph", wrapped)

    clients = _build_clients()
    config = PipelineConfig()

    graph_first = get_recommendation_graph(clients, config)
    graph_second = get_recommendation_graph(clients, config)

    assert graph_first is graph_second  # 동일 인스턴스 재사용
    assert wrapped.call_count == 1


def test_get_recommendation_graph_normalizes_none_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``config=None`` 과 ``config=PipelineConfig()`` 는 동일 캐시 키로 정규화된다.

    wrapper 가 ``None`` 을 기본 인스턴스로 치환한 뒤 내부 ``lru_cache`` 헬퍼에
    전달하므로, 두 호출 형태가 의미상 동일함을 캐시 hit 으로 확인한다.
    """
    wrapped = MagicMock(wraps=graph_pipeline.build_recommendation_graph)
    monkeypatch.setattr(graph_pipeline, "build_recommendation_graph", wrapped)

    clients = _build_clients()

    graph_from_none = get_recommendation_graph(clients)
    graph_from_default = get_recommendation_graph(clients, PipelineConfig())

    assert graph_from_none is graph_from_default  # 캐시 키 정규화 → 동일 인스턴스
    assert wrapped.call_count == 1


def test_coverage_analysis_flows_through_graph_with_real_signal(tmp_path: Path) -> None:
    """완료 과목 + 로드맵 과목이 추천 직무 토큰을 덮어 충족도가 상승하는 흐름을 검증.

    합성 과목 카탈로그를 ``PipelineConfig`` 로 주입해, 커버리지 노드가 실제로
    완료 과목(현재 충족)과 로드맵 과목(예상 충족)을 직무 목표 토큰에 매칭하는
    데이터 경로를 end-to-end 로 고정한다. mock course_repo 가 돌려주는 과목명
    (``f"{course_id} 강의"``)을 그대로 토큰에 매핑한다.
    """
    catalog = {
        "courses": [
            {"course_id": "CS101", "course_name": "CS101 강의", "tech_stacks": ["py"]},
            {"course_id": "CS102", "course_name": "CS102 강의", "tech_stacks": ["sql"]},
        ]
    }
    catalog_path = tmp_path / "courses.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")

    clients = _build_clients()  # 직무 tech_stacks=["py","sql"], 역량=["문제해결능력"]
    config = PipelineConfig(coverage_course_catalog_path=catalog_path)
    graph = build_recommendation_graph(clients, config)

    payload = _valid_onboarding_payload(current_tracks=["in0", "in1"])
    payload["completed_courses"] = ["CS101 강의"]  # py 현재 충족

    result = graph.invoke({"user_id": "u1", "raw_input": payload})

    coverage = result["coverage_analysis"]
    # 분모는 도달 가능 토큰뿐 — py(완료)·sql(로드맵)만. 문제해결능력은 가르치는
    # 과목이 없어 분모에서 빠지고 gap 으로 보고된다.
    assert coverage["required_count"] == 2
    assert "문제해결능력" in coverage["gap_tokens"]
    assert coverage["current_covered"] == 1  # 완료 과목 → py
    # 로드맵의 CS102(sql) 가 예상 충족도를 끌어올린다
    assert coverage["expected_covered"] >= 2
    assert coverage["current_ratio"] < coverage["expected_ratio"]
    # 추천 기반 다음 액션 제안이 비어 있지 않다
    assert coverage["next_actions"]
    assert "%" in coverage["next_actions"][0]["message"]
