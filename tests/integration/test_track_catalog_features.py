"""실제 트랙 카탈로그(tracks.yaml)가 시너지 입력 신호를 공급하는지 검증한다.

트랙 시너지 점수는 트랙의 ``tech_stacks``·``competencies`` 와 직무 토큰의 교집합으로
계산된다. 카탈로그에 이 신호가 비면 모든 조합 점수가 0 으로 붕괴해 추천 트랙이
점수 변별 없이 정해진다. 본 테스트는 그 회귀(카탈로그 신호 공백 → 점수 평탄화)를
막는다 — 노드 로직이 아니라 *런타임 카탈로그 데이터*가 살아 있는지 확인한다.

``@pytest.mark.integration`` — 패키지 config 의 실제 tracks.yaml 을 로드하므로 단위
테스트가 아닌 통합 테스트로 분류한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tracktory.common.tech_keywords import canonical_tech_keys, canonical_tech_token
from tracktory.graph.models import JobCandidate, Track, WeightsConfig
from tracktory.graph.nodes.track_synergy import _generate_combos, _synergy_score
from tracktory.rag.yaml_track_repository import YamlTrackRepository

pytestmark = pytest.mark.integration

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "tracktory" / "config"

# 컴퓨터공학부 4 트랙 — MVP 추천 범위의 핵심.
_CS_TRACK_IDS = frozenset(
    [
        "빅데이터트랙",
        "모바일소프트웨어트랙",
        "웹공학트랙",
        "디지털콘텐츠·가상현실트랙",
    ]
)

# 프로그래밍·도구 기술을 가르치지 않는 인문·예술·디자인 트랙. 이 트랙들은 커리큘럼·
# 강의계획서에 기술 토큰이 없어 tech_stacks 가 정직하게 비어 있다(없는 기술을 지어내면
# 직무 커버율·상보성이 오염된다). 이들은 competencies(역량)로 시너지에 기여한다 —
# 직무가 매칭되는 역량 태그를 요구할 때 complementarity 가 competencies 로 발화한다.
_NON_TECH_TRACK_IDS = frozenset(
    [
        "뷰티디자인매니지먼트학과",
        "역사문화큐레이션트랙",
        "한국어교육트랙",
    ]
)

# 백엔드·데이터 직무를 대표하는 정합 표기 기술 토큰.
_BACKEND_JOB = JobCandidate(
    job_id="be",
    job_name="백엔드/데이터 개발",
    tech_stacks=["Python", "Java", "Spring Boot", "SQL", "TensorFlow", "Django"],
    competency_tags=[],
    match_score=0.8,
)

_DEFAULT_WEIGHTS = WeightsConfig(complementarity=0.3, coverage=0.5, redundancy=0.2)


def _all_tracks() -> list[Track]:
    tracks: list[Track] = YamlTrackRepository().list_all()
    return tracks


def _course_tech_by_id() -> dict[str, list[str]]:
    raw = yaml.safe_load((_CONFIG_DIR / "courses.yaml").read_text(encoding="utf-8"))
    return {
        str(c["course_id"]): list(c.get("tech_stacks") or [])
        for c in raw["courses"]
        if isinstance(c, dict) and c.get("course_id")
    }


def test_every_track_has_competencies() -> None:
    """44 개 트랙 모두 역량 신호를 공급받는다 (criterion 1 — 역량은 모든 트랙의 보편 신호)."""
    tracks = _all_tracks()
    assert len(tracks) == 44
    empty = [t.track_id for t in tracks if not t.competencies]
    assert not empty, f"competencies 가 빈 트랙: {empty}"


def test_tech_bearing_tracks_have_tech_stacks() -> None:
    """기술을 가르치는 과목을 가진 트랙은 tech_stacks 가 비지 않는다 (집계 파이프라인 회귀 가드).

    트랙의 소속 과목 중 하나라도 기술 토큰을 가지면, 그 토큰이 트랙 단위로 집계돼
    ``Track.tech_stacks`` 에 흘러야 한다. 본 테스트는 집계가 데이터를 떨어뜨리지
    않음을 보장한다. 기술 과목이 전무한 인문·예술 트랙은 정직하게 비는 것이 정상이라
    검사 대상이 아니다.
    """
    course_tech = _course_tech_by_id()
    for track in _all_tracks():
        has_tech_course = any(course_tech.get(cid) for cid in track.course_ids)
        if has_tech_course:
            assert track.tech_stacks, (
                f"{track.track_id} 의 과목에 기술 토큰이 있는데 트랙 tech_stacks 가 빔"
            )


def test_empty_tech_tracks_are_the_known_non_tech_set() -> None:
    """tech_stacks 가 빈 트랙은 알려진 비기술 트랙 집합과 정확히 일치한다.

    비기술 트랙(인문·예술·디자인)은 기술을 안 가르쳐 tech_stacks 가 정직하게 빈다.
    본 테스트는 그 집합을 못박아, 기술 트랙이 회귀로 tech 신호를 잃으면(빈 집합이
    늘면) 즉시 실패하게 한다 — 없는 기술을 지어내는 대신 빈 트랙 집합을 감시한다.
    """
    empty_tech = {t.track_id for t in _all_tracks() if not t.tech_stacks}
    assert empty_tech == _NON_TECH_TRACK_IDS, (
        f"빈 tech_stacks 트랙 집합이 알려진 비기술 집합과 불일치: "
        f"예상치 못한 빈 트랙 {empty_tech - _NON_TECH_TRACK_IDS}, "
        f"기대했으나 채워진 트랙 {_NON_TECH_TRACK_IDS - empty_tech}"
    )


def test_cs_tracks_have_tech_stacks() -> None:
    """컴퓨터공학부 트랙이 시너지 입력 기술스택을 충분히 공급받는다."""
    cs_tracks = [t for t in _all_tracks() if t.track_id in _CS_TRACK_IDS]
    assert len(cs_tracks) == len(_CS_TRACK_IDS), "CS 트랙이 카탈로그에 모두 존재해야 한다"
    for track in cs_tracks:
        assert len(track.tech_stacks) >= 3, (
            f"{track.track_id} 의 tech_stacks 가 비어/빈약함: {track.tech_stacks}"
        )


def test_all_track_tech_stacks_use_job_aligned_vocabulary() -> None:
    """모든 트랙의 저장 tech_stacks 가 직무 기술 정합 어휘의 고정점이다 (criterion 4).

    카탈로그 표기가 ``canonical_tech_token`` 의 고정점이면 직무 토큰과 같은 어휘
    위에서 비교돼 표기 차이로 커버율이 0 으로 떨어지지 않는다. 저장 시점에 직무
    어휘로 정규화했으므로 모든 트랙에서 성립해야 한다.
    """
    for track in _all_tracks():
        for tech in track.tech_stacks:
            assert canonical_tech_token(tech) == tech, (
                f"{track.track_id} 의 '{tech}' 가 직무 정합 표기의 고정점이 아님"
            )


def test_cs_track_tech_overlaps_backend_job() -> None:
    """CS 트랙 기술스택이 백엔드 직무 토큰과 실제로 교집합을 가진다 (커버율 > 0 보장)."""
    cs_tracks = [t for t in _all_tracks() if t.track_id in _CS_TRACK_IDS]
    job_keys = canonical_tech_keys(_BACKEND_JOB.tech_stacks)
    union_track_keys: set[str] = set()
    for track in cs_tracks:
        union_track_keys |= canonical_tech_keys(track.tech_stacks)
    assert job_keys & union_track_keys, "CS 트랙 기술스택이 백엔드 직무 토큰과 전혀 겹치지 않음"


def test_synergy_scores_nonzero_and_distinguishable() -> None:
    """실제 카탈로그 기반 조합 점수가 0 이 아니고 조합별로 구분된다 (criterion 2).

    1 학년 컴퓨터공학부 + 대표 직무로 전체 후보 조합을 채점해, 카탈로그가 비었을 때
    모든 조합이 0 으로 평탄해지던 버그를 직접 가드한다.
    """
    tracks = _all_tracks()
    combos = _generate_combos(tracks, user_college_id="IT공과대학", current_tracks=[])
    scores = [_synergy_score(c, [_BACKEND_JOB], _DEFAULT_WEIGHTS) for c in combos]

    assert any(s > 0.0 for s in scores), "모든 조합 점수가 0"
    distinct_nonzero = {round(s, 6) for s in scores if s > 0.0}
    assert len(distinct_nonzero) >= 5, (
        f"조합 점수 변별이 부족함(동률 평탄화): {sorted(distinct_nonzero)}"
    )


def test_primary_recommendations_sorted_by_score() -> None:
    """추천 주 트랙이 시너지 점수 내림차순으로 정렬된다 (criterion 3 — 점수 근거 정렬).

    카탈로그가 비어 모든 점수가 0 이던 시절에는 정렬 키가 점수가 아닌 조합 키로만
    갈려 순위가 임의였다. 신호가 공급된 지금은 주 추천이 점수로 정렬돼야 한다.
    """
    from unittest.mock import MagicMock

    from tracktory.graph.nodes.track_synergy import TrackRepository, TrackSynergyNode

    repo = MagicMock(spec=TrackRepository)
    repo.list_all.return_value = _all_tracks()
    node = TrackSynergyNode(track_repo=repo)
    state = {
        "recommended_jobs": [
            {
                "job_id": "be",
                "job_name": "백엔드",
                "tech_stacks": ["Python", "Java", "Spring Boot", "SQL", "Node.js"],
                "competency_tags": ["문제 해결 능력"],
                "match_score": 0.8,
            },
            {
                "job_id": "data",
                "job_name": "데이터분석",
                "tech_stacks": ["Python", "SQL", "R", "Pandas", "Machine Learning"],
                "competency_tags": ["데이터 분석 능력"],
                "match_score": 0.7,
            },
        ],
        "normalized_profile": {"college": "IT공과대학", "current_tracks": []},
    }
    result = node(state)
    primary_scores = [c["synergy_score"] for c in result["primary_combos"]]
    assert primary_scores == sorted(primary_scores, reverse=True)
    assert primary_scores[0] > 0.0, "최상위 주 추천 점수가 0 — 신호 공백 회귀"
