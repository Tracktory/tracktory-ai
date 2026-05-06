"""GraphState 의 recommended_jobs 필드가 선언과 round-trip 계약을 만족하는지 검증한다.

TypedDict 는 런타임에 dict 이므로 표준 json 모듈로 직렬화·역직렬화한다.
"""

import json

from tracktory.graph.state import GraphState


def _three_jobs() -> list[dict[str, object]]:
    return [
        {"job_id": "be_dev_001", "name": "백엔드 개발자", "score": 0.82},
        {"job_id": "data_eng_001", "name": "데이터 엔지니어", "score": 0.74},
        {"job_id": "ml_eng_001", "name": "머신러닝 엔지니어", "score": 0.69},
    ]


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
        "profile_vector": [0.1, 0.2, 0.3],
        "recommended_jobs": _three_jobs(),
        "errors": [],
        "trace": ["input_normalize:ok", "profile_embed:ok"],
    }

    serialized = json.dumps(state, ensure_ascii=False)
    restored = json.loads(serialized)

    assert restored == state
    assert len(restored["recommended_jobs"]) == 3
    assert restored["recommended_jobs"][0]["job_id"] == "be_dev_001"
