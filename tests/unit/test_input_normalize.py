"""normalize_input 의 성공·실패 계약을 검증한다.

Pydantic 내부 동작은 검증 대상 X.
"""

from tracktory.graph.nodes.input_normalize import NormalizedProfile, normalize_input


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
    assert result["trace"] == ["input_normalize:ok"]


def test_normalize_accepts_zero_tracks_for_freshman() -> None:
    """1학년 케이스: current_tracks=[] (트랙 미선택) 는 정상 통과한다."""
    raw = _valid_raw()
    raw["current_tracks"] = []
    result = normalize_input({"raw_input": raw})
    assert result["trace"] == ["input_normalize:ok"]
    assert result["normalized_profile"]["current_tracks"] == []


def test_normalize_accepts_two_tracks_for_upperclass() -> None:
    """2학년+ 케이스: current_tracks 에 정확히 2 개일 때 정상 통과한다."""
    raw = _valid_raw()
    raw["current_tracks"] = ["빅데이터트랙", "한국어교육트랙"]
    result = normalize_input({"raw_input": raw})
    assert result["trace"] == ["input_normalize:ok"]
    assert len(result["normalized_profile"]["current_tracks"]) == 2


def test_normalize_rejects_one_track() -> None:
    """current_tracks 가 1 개이면 _tracks_zero_or_two 위배로 실패 계약을 반환한다."""
    raw = _valid_raw()
    raw["current_tracks"] = ["빅데이터트랙"]
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["input_normalize:fail"]
    assert len(result["errors"]) >= 1


def test_normalize_rejects_three_tracks() -> None:
    """current_tracks 가 3 개이면 _tracks_zero_or_two 위배로 실패 계약을 반환한다."""
    raw = _valid_raw()
    raw["current_tracks"] = ["A", "B", "C"]
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["input_normalize:fail"]
    assert len(result["errors"]) >= 1


def test_normalize_rejects_missing_required_field() -> None:
    """필수 필드 interests 누락 시 Pydantic ValidationError 로 실패 계약을 반환한다."""
    raw = _valid_raw()
    del raw["interests"]
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["input_normalize:fail"]
    assert len(result["errors"]) >= 1


def test_normalize_rejects_six_interests() -> None:
    """interests 가 6 개(max_length=5 초과)이면 ValidationError 로 실패 계약을 반환한다."""
    raw = _valid_raw()
    raw["interests"] = ["A", "B", "C", "D", "E", "F"]
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["input_normalize:fail"]
    assert len(result["errors"]) >= 1


# ---------------------------------------------------------------------------
# current_semester — 사용자 명시 입력 / 입학년도 기반 fallback 추정
# ---------------------------------------------------------------------------


def test_normalize_preserves_user_provided_current_semester() -> None:
    """사용자가 명시한 current_semester 는 그대로 보존되고 state 키로도 흐른다."""
    raw = _valid_raw()
    raw["current_semester"] = 4
    result = normalize_input({"raw_input": raw})
    assert result["trace"] == ["input_normalize:ok"]
    assert result["normalized_profile"]["current_semester"] == 4
    assert result["current_semester"] == 4


def test_normalize_infers_current_semester_from_admission_year_when_missing() -> None:
    """current_semester 미입력 시 입학년도 기반 추정으로 채워진다.

    추정 공식 ``(현재 연도 - admission_year) * 2 + 1`` 의 결과는 [1, 8] 로
    clamp 된다. 본 테스트는 추정값이 1~8 범위 안인지만 검증하여 기준 연도
    변화에 의존하지 않는다.
    """
    raw = _valid_raw()
    raw["admission_year"] = 2025
    raw.pop("current_semester", None)
    result = normalize_input({"raw_input": raw})
    assert result["trace"] == ["input_normalize:ok"]
    inferred = result["normalized_profile"]["current_semester"]
    assert isinstance(inferred, int)
    assert 1 <= inferred <= 8
    assert result["current_semester"] == inferred


def test_normalize_clamps_current_semester_for_very_old_admission_year() -> None:
    """입학이 너무 오래된 케이스도 결과는 [1, 8] 범위 안으로 clamp 된다."""
    raw = _valid_raw()
    raw["admission_year"] = 2000  # 매우 오래된 입학
    raw.pop("current_semester", None)
    result = normalize_input({"raw_input": raw})
    assert result["trace"] == ["input_normalize:ok"]
    assert result["normalized_profile"]["current_semester"] == 8


def test_normalize_rejects_current_semester_out_of_range() -> None:
    """범위 [1, 8] 밖의 명시 입력은 ValidationError 로 차단된다."""
    raw = _valid_raw()
    raw["current_semester"] = 9
    result = normalize_input({"raw_input": raw})
    assert "normalized_profile" not in result
    assert result["trace"] == ["input_normalize:fail"]
