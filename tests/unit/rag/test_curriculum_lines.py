from __future__ import annotations

import pytest

from tracktory.rag.curriculum_lines import parse_course_line


# stage(학습 깊이)는 과목구분이 아니라 학년에서 도출되므로 본 파서는 더 이상
# 산출하지 않는다 — 여기서는 전공 판정(None 여부)·course_id·course_type 만 본다.
@pytest.mark.parametrize(
    ("line", "code", "course_type"),
    [
        ("  - [전공기초] 데이터리터러시 (CTE0029, 3학점)", "CTE0029", "전공선택"),
        ("  - [전공필수] 데이터공학 선형대수 (V076009, 3학점)", "V076009", "전공필수"),
        ("  - [전공선택] 공학프로그래밍 (V070044, 3학점)", "V070044", "전공선택"),
        ("  - [전공선택(상호인정)] 상호인정과목 (X100, 3학점)", "X100", "전공선택"),
    ],
)
def test_parses_major_course_lines(line: str, code: str, course_type: str) -> None:
    parsed = parse_course_line(line)
    assert parsed is not None
    assert parsed.course_id == code
    assert parsed.course_type == course_type


@pytest.mark.parametrize(
    "tag",
    ["교양필수", "선택필수교양", "일반교양"],
)
def test_liberal_arts_tags_are_excluded(tag: str) -> None:
    # 교양류는 추천 대상이 아니므로 None — 트랙/과목 저장소 양쪽에서 제외된다.
    assert parse_course_line(f"  - [{tag}] 글쓰기 (GEN0123, 3학점)") is None


def test_unknown_non_major_tag_is_excluded() -> None:
    # GEN prefix 가 아닌 비전공 태그(예: 교직)도 태그 기준으로 제외된다 —
    # prefix 가 아니라 과목구분이 단일 판정 기준임을 보장.
    assert parse_course_line("  - [교직] 교직과목 (REQ0001, 2학점)") is None


@pytest.mark.parametrize(
    "line",
    [
        "2학년 1학기",  # 섹션 헤더
        "[트랙: 응용산업데이터공학트랙 | 대학: IT공과대학]",  # 문서 헤더
        "",  # 빈 줄
        "  - [전공선택] 태그만 있고 괄호 없음",  # 코드/학점 누락
    ],
)
def test_non_course_lines_return_none(line: str) -> None:
    assert parse_course_line(line) is None


def test_paren_whitespace_variants_are_tolerated() -> None:
    # 괄호 안 화이트스페이스가 들쭉날쭉해도 동일하게 파싱된다.
    parsed = parse_course_line("  - [전공필수] 회계원론 ( ACC101 ,  3 학점 )")
    assert parsed is not None
    assert parsed.course_id == "ACC101"
    assert parsed.credits == 3
