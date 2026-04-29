"""N1 입력 정규화 노드 단위 테스트.

normalize_input 의 성공·실패 계약만 검증한다. Pydantic 내부 동작은 검증 대상 X.
"""

from tracktory.graph.nodes.n1_input import NormalizedProfile, normalize_input


def _valid_raw() -> dict[str, object]:
    return {
        "admission_year": 2025,
        "college": "IT공과대학",
        "department": "컴퓨터공학부",
        "current_tracks": [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


def test_normalize_returns_dict_when_valid() -> None:
    """유효한 입력에 대해 round-trip 등가성과 trace 를 검증한다."""
    raw = _valid_raw()
    result = normalize_input({"raw_input": raw})
    expected = NormalizedProfile.model_validate(raw).model_dump()
    assert result["normalized_profile"] == expected
    assert result["trace"] == ["N1:ok"]


def test_normalize_accepts_zero_tracks_for_freshman() -> None:
    """1학년 케이스: current_tracks=[] 는 D-08 degenerate 로 정상 통과한다."""
    raw = _valid_raw()
    raw["current_tracks"] = []
    result = normalize_input({"raw_input": raw})
    assert result["trace"] == ["N1:ok"]
    assert result["normalized_profile"]["current_tracks"] == []


def test_normalize_accepts_two_tracks_for_upperclass() -> None:
    """2학년+ 케이스: current_tracks 에 정확히 2 개일 때 정상 통과한다."""
    raw = _valid_raw()
    raw["current_tracks"] = ["빅데이터트랙", "한국어교육트랙"]
    result = normalize_input({"raw_input": raw})
    assert result["trace"] == ["N1:ok"]
    assert len(result["normalized_profile"]["current_tracks"]) == 2


def test_normalize_rejects_one_track() -> None:
    """current_tracks 가 1 개이면 _tracks_zero_or_two 위배로 실패 계약을 반환한다."""
    raw = _valid_raw()
    raw["current_tracks"] = ["빅데이터트랙"]
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["N1:fail"]
    assert len(result["errors"]) >= 1


def test_normalize_rejects_three_tracks() -> None:
    """current_tracks 가 3 개이면 _tracks_zero_or_two 위배로 실패 계약을 반환한다."""
    raw = _valid_raw()
    raw["current_tracks"] = ["A", "B", "C"]
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["N1:fail"]
    assert len(result["errors"]) >= 1


def test_normalize_rejects_missing_required_field() -> None:
    """필수 필드 interests 누락 시 Pydantic ValidationError 로 실패 계약을 반환한다."""
    raw = _valid_raw()
    del raw["interests"]
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["N1:fail"]
    assert len(result["errors"]) >= 1


def test_normalize_rejects_six_interests() -> None:
    """interests 가 6 개(max_length=5 초과)이면 ValidationError 로 실패 계약을 반환한다."""
    raw = _valid_raw()
    raw["interests"] = ["A", "B", "C", "D", "E", "F"]
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["N1:fail"]
    assert len(result["errors"]) >= 1
