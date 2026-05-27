"""_normalize_prereq_value 의 null 판정·정상값 처리를 검증한다.

회귀 방지가 목적이며 _NULL_EXACT / _NULL_STARTS 의 60자 휴리스틱·괄호 제거
순서가 바뀌면 다음 케이스 중 일부가 깨진다.
"""

import pytest

from tracktory.relation.prerequisite.normalize_prereq import _normalize_prereq_value


@pytest.mark.parametrize(
    "raw",
    [
        "",  # 빈 문자열
        "  ",  # 공백만
        "없음",  # null 변형 1
        "해당없음",  # null 변형 2
        "없 음",  # null 변형 3 (공백 무시 일치)
        "선수과목없음",
        "무관",
        "n/a",
        "-",
    ],
)
def test_returns_null_for_empty_or_negation(raw: str) -> None:
    assert _normalize_prereq_value(raw) == "null"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("자료구조", "자료구조"),
        ("자료 구조", "자료 구조"),  # 공백은 유지 (정규화는 비교용일 뿐)
        ("자료구조 (필수)", "자료구조"),  # 괄호 부연설명 제거
        ("자료구조.", "자료구조"),  # 끝 마침표 제거
    ],
)
def test_returns_cleaned_value(raw: str, expected: str) -> None:
    assert _normalize_prereq_value(raw) == expected


def test_returns_null_when_paren_strip_leaves_negation() -> None:
    """괄호 부연설명을 떼면 '없음'으로 남는 경우 null 판정."""
    assert _normalize_prereq_value("없음 (단, 추천: 자료구조)") == "null"


def test_returns_null_for_long_free_text_without_separator() -> None:
    """구분자 없이 60자 초과면 자유서술로 판단해 null."""
    long_text = (
        "이 과목은 학생들이 사전에 관련 분야의 기초 지식을 갖춘 상태에서 "
        "수강하는 것이 학습 효과를 높일 수 있어 매우 권장됩니다"
    )
    assert _normalize_prereq_value(long_text) == "null"
