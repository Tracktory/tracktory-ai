"""_split_prereq_raw / _normalize 의 파싱·정규화 동작을 검증한다.

선수과목 원문 → 조각 리스트 분리 규칙과 과목명 정규화 규칙은 데이터 매칭
정확도에 직결되므로 회귀 방지 테스트가 필수.
"""

import pytest

from tracktory.relation.prerequisite.build_prerequisites import (
    _normalize,
    _split_prereq_raw,
)

# --- _split_prereq_raw ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("자료구조", ["자료구조"]),
        ("자료구조, 알고리즘", ["자료구조", "알고리즘"]),
        ("자료구조/알고리즘", ["자료구조", "알고리즘"]),
        ("자료구조 또는 알고리즘", ["자료구조", "알고리즘"]),
        ("자료구조 및 알고리즘", ["자료구조", "알고리즘"]),
    ],
)
def test_split_by_separators(raw: str, expected: list[str]) -> None:
    assert _split_prereq_raw(raw) == expected


def test_split_removes_text_before_colon() -> None:
    """콜론 앞 설명은 제거되고 콜론 뒤 값만 분리 대상."""
    assert _split_prereq_raw("필수: 자료구조, 알고리즘") == ["자료구조", "알고리즘"]


def test_split_removes_paren_before_splitting() -> None:
    """괄호 부연설명을 제거한 뒤 구분자로 분리한다."""
    assert _split_prereq_raw("수학(미적), 영어(교양)") == ["수학", "영어"]


def test_split_strips_trailing_period() -> None:
    assert _split_prereq_raw("자료구조.") == ["자료구조"]


def test_split_empty_returns_empty_list() -> None:
    assert _split_prereq_raw("") == []


# --- _normalize ---


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("자료구조", "자료구조"),
        ("자료 구조", "자료구조"),  # 공백 제거
        ("Programming I", "programmingi"),  # 소문자 + 공백 제거
        ("수학(미적)", "수학"),  # 괄호 제거
        ("객체 지향 (실습)", "객체지향"),  # 공백 + 괄호 동시
        ("", ""),  # 빈 문자열
    ],
)
def test_normalize_course_name(name: str, expected: str) -> None:
    assert _normalize(name) == expected
