"""GraphState 출력 필드들이 선언과 round-trip 계약을 만족하는지 검증한다.

TypedDict 는 런타임에 dict 이므로 표준 json 모듈로 직렬화·역직렬화한다.
직무 매칭·학습 로드맵·LLM 설명 노드가 각자 Pydantic 모델의 ``model_dump``
결과 dict 를 그래프 state 에 흘려보내는 패턴을 시뮬레이션한다.
"""

import json

from tracktory.graph.state import GraphState


def _three_jobs() -> list[dict[str, object]]:
    return [
        {"job_id": "be_dev_001", "name": "백엔드 개발자", "score": 0.82},
        {"job_id": "data_eng_001", "name": "데이터 엔지니어", "score": 0.74},
        {"job_id": "ml_eng_001", "name": "머신러닝 엔지니어", "score": 0.69},
    ]


def _roadmap_payload() -> dict[str, object]:
    """Roadmap.model_dump(mode='json') 와 동일한 모양의 dict."""
    return {
        "stages": [
            {
                "stage": "foundation",
                "courses": [
                    {"course_id": "cs101", "course_name": "프로그래밍 기초", "priority": 1},
                ],
            },
            {
                "stage": "core",
                "courses": [
                    {"course_id": "cs201", "course_name": "자료구조", "priority": 1},
                    {"course_id": "cs202", "course_name": "알고리즘", "priority": 2},
                ],
            },
            {"stage": "application", "courses": []},
            {"stage": "industry", "courses": []},
        ],
    }


def _explanation_payload() -> dict[str, object]:
    """Explanation.model_dump(mode='json') 와 동일한 모양의 dict."""
    return {
        "text": "관심사와 흥미를 종합하면 백엔드 개발자가 가장 잘 맞아요.",
        "sections": [
            {"topic": "jobs", "body": "백엔드 개발자가 1순위로 추천된 이유는..."},
            {"topic": "tracks", "body": "빅데이터 + 모바일소프트웨어 조합이 시너지가 높아요."},
        ],
    }


def test_recommended_jobs_field_optional_default() -> None:
    """필드 미설정 시에도 GraphState 는 정상 생성·접근 가능해야 한다."""
    state: GraphState = {"user_id": "u1"}
    assert "recommended_jobs" not in state


def test_recommended_jobs_accepts_none() -> None:
    """명시적 None 할당이 가능해야 한다."""
    state: GraphState = {"user_id": "u1", "recommended_jobs": None}
    assert state["recommended_jobs"] is None


def test_recommended_jobs_accepts_empty_list() -> None:
    """빈 list 할당이 가능해야 한다."""
    state: GraphState = {"user_id": "u1", "recommended_jobs": []}
    assert state["recommended_jobs"] == []


def test_recommended_jobs_roundtrip_with_other_fields() -> None:
    """직무 dict 3 개 + 다른 필드와 함께 json round-trip 후 동일 dict 가 복원된다."""
    state: GraphState = {
        "user_id": "u1",
        "profile_text": "데이터 분석에 관심 있는 학생",
        "recommended_jobs": _three_jobs(),
        "errors": [],
        "trace": ["input_normalize:ok", "profile_embed:ok"],
    }

    serialized = json.dumps(state, ensure_ascii=False)
    restored = json.loads(serialized)

    assert restored == state
    assert len(restored["recommended_jobs"]) == 3
    assert restored["recommended_jobs"][0]["job_id"] == "be_dev_001"


def test_roadmap_field_optional_default() -> None:
    """roadmap 미설정 시에도 GraphState 는 정상 생성·접근 가능해야 한다."""
    state: GraphState = {"user_id": "u1"}
    assert "roadmap" not in state


def test_roadmap_accepts_none() -> None:
    """roadmap 명시적 None 할당이 가능해야 한다."""
    state: GraphState = {"user_id": "u1", "roadmap": None}
    assert state["roadmap"] is None


def test_roadmap_accepts_empty_stages() -> None:
    """4 단계 stage 가 비어 있어도 dict 자체는 state 에 들어갈 수 있다 (TypedDict 단계 검증 X)."""
    payload: dict[str, object] = {"stages": []}
    state: GraphState = {"user_id": "u1", "roadmap": payload}
    assert state["roadmap"] == payload


def test_roadmap_roundtrip_with_other_fields() -> None:
    """roadmap dict 가 다른 필드와 함께 json round-trip 후 동일 dict 가 복원된다."""
    state: GraphState = {
        "user_id": "u1",
        "recommended_jobs": _three_jobs(),
        "roadmap": _roadmap_payload(),
        "trace": ["job_matching:ok", "roadmap:ok"],
    }

    serialized = json.dumps(state, ensure_ascii=False)
    restored = json.loads(serialized)

    assert restored == state
    assert len(restored["roadmap"]["stages"]) == 4
    assert restored["roadmap"]["stages"][1]["stage"] == "core"


def test_explanation_field_optional_default() -> None:
    """explanation 미설정 시에도 GraphState 는 정상 생성·접근 가능해야 한다."""
    state: GraphState = {"user_id": "u1"}
    assert "explanation" not in state


def test_explanation_accepts_none() -> None:
    """explanation 명시적 None 할당이 가능해야 한다."""
    state: GraphState = {"user_id": "u1", "explanation": None}
    assert state["explanation"] is None


def test_explanation_accepts_minimal_text_only() -> None:
    """sections 가 비어 있는 minimal 설명도 state 에 들어갈 수 있다."""
    payload: dict[str, object] = {"text": "전체 요약입니다.", "sections": []}
    state: GraphState = {"user_id": "u1", "explanation": payload}
    assert state["explanation"] == payload


def test_explanation_roundtrip_with_other_fields() -> None:
    """explanation dict 가 다른 필드와 함께 json round-trip 후 동일 dict 가 복원된다."""
    state: GraphState = {
        "user_id": "u1",
        "recommended_jobs": _three_jobs(),
        "explanation": _explanation_payload(),
        "trace": ["explanation:ok"],
    }

    serialized = json.dumps(state, ensure_ascii=False)
    restored = json.loads(serialized)

    assert restored == state
    assert len(restored["explanation"]["sections"]) == 2
    assert restored["explanation"]["sections"][0]["topic"] == "jobs"


def test_full_state_roundtrip_all_output_fields() -> None:
    """모든 출력 필드를 채운 풀 cycle 직렬화가 동일 dict 를 복원한다.

    파이프라인이 끝까지 흘렀을 때 reviewer 가 보는 최종 state 모양을
    한 곳에서 lock-down 한다.
    """
    primary = [
        {
            "combo": {"track_a_id": "bigdata", "track_b_id": "mobile"},
            "synergy_score": 0.78,
            "slot_type": "primary",
            "rank": 1,
        },
    ]
    secondary = [
        {
            "combo": {"track_a_id": "bigdata", "track_b_id": "web"},
            "synergy_score": 0.62,
            "slot_type": "mmr",
            "rank": 3,
        },
    ]
    state: GraphState = {
        "user_id": "u-full",
        "raw_input": {"interests": ["IT"], "year": 2026},
        "normalized_profile": {"interest_categories": ["it"]},
        "profile_text": "IT 에 관심이 많은 학생",
        "recommended_jobs": _three_jobs(),
        "primary_combos": primary,
        "secondary_combos": secondary,
        "slot3_fallback_triggered": True,
        "slot3_fallback_level": "T2",
        "roadmap": _roadmap_payload(),
        "explanation": _explanation_payload(),
        "errors": [],
        "trace": [
            "input_normalize:ok",
            "profile_embed:ok",
            "job_matching:ok",
            "track_synergy:ok",
            "roadmap:ok",
            "explanation:ok",
        ],
    }

    serialized = json.dumps(state, ensure_ascii=False)
    restored = json.loads(serialized)

    assert restored == state
