"""apply_course_tech_stacks 의 코드 추출·분반 union·빈 처리 계약을 검증한다.

회귀 방지가 목적이다. 강의계획서 파일명에서 7자리 과목코드를 뽑는 규칙,
같은 과목의 여러 분반 토큰을 union 하는 규칙, 강의계획서가 없는 과목을 빈
리스트로 채우는 규칙이 데이터 정합의 핵심이라 직접 검증한다. 실제 데이터·
실제 기술 어휘에는 의존하지 않고, tmp_path 합성 파일 + stub 추출기로 순수
로직만 본다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "apply_course_tech_stacks.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """apply 스크립트를 파일 경로로 로드한다 (scripts/ 가 패키지 경로 밖이므로)."""
    spec = importlib.util.spec_from_file_location("_apply_course_tech_stacks", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_syllabus(directory: Path, filename: str, body: str) -> None:
    (directory / filename).write_text(body, encoding="utf-8")


def test_extract_course_id_pulls_seven_char_code(script: ModuleType) -> None:
    # 파일명 = 강의계획서_<11자리 등록번호><7자리 과목코드><1자리 분반>.txt
    assert script._extract_course_id("강의계획서_20261110065CTA0017B.txt") == "CTA0017"
    assert script._extract_course_id("강의계획서_20261110855W0800231.txt") == "W080023"


def test_extract_course_id_returns_none_on_malformed_name(script: ModuleType) -> None:
    assert script._extract_course_id("notes.txt") is None
    assert script._extract_course_id("강의계획서_123CTA0017B.txt") is None  # 등록번호 11자리 미만


def test_aggregate_unions_tokens_across_sections(script: ModuleType, tmp_path: Path) -> None:
    syllabi = tmp_path / "syllabi"
    syllabi.mkdir()
    # 같은 과목 (CTA0017) 의 두 분반 — 각기 다른 토큰을 담아 union 을 확인
    _write_syllabus(syllabi, "강의계획서_20261110065CTA0017A.txt", "SQL 데이터베이스")
    _write_syllabus(syllabi, "강의계획서_20261110065CTA0017B.txt", "Python 데이터 분석")

    def fake_extract(text: str) -> list[str]:
        mapping = {"SQL": "SQL", "Python": "Python"}
        return [token for keyword, token in mapping.items() if keyword in text]

    tokens, scanned, no_course = script._aggregate_syllabus_tokens(
        syllabi, {"CTA0017"}, fake_extract
    )

    assert tokens == {"CTA0017": {"SQL", "Python"}}
    assert scanned == 2
    assert no_course == 0


def test_aggregate_skips_files_without_catalog_match(script: ModuleType, tmp_path: Path) -> None:
    syllabi = tmp_path / "syllabi"
    syllabi.mkdir()
    _write_syllabus(syllabi, "강의계획서_20261110065CTA0017A.txt", "Python")
    # GEN0001 은 카탈로그에 없는 교양 과목 → 토큰 집계에서 제외, no_course 증가
    _write_syllabus(syllabi, "강의계획서_20261110855GEN0001A.txt", "Python")

    def fake_extract(text: str) -> list[str]:
        return ["Python"] if "Python" in text else []

    tokens, scanned, no_course = script._aggregate_syllabus_tokens(
        syllabi, {"CTA0017"}, fake_extract
    )

    assert set(tokens) == {"CTA0017"}
    assert scanned == 2
    assert no_course == 1


def test_apply_tokens_fills_empty_list_for_course_without_syllabus(script: ModuleType) -> None:
    courses: list[dict[str, object]] = [
        {"course_id": "CTA0017", "course_name": "데이터베이스"},
        {"course_id": "CTA9999", "course_name": "강의계획서 없는 과목"},
    ]

    stats = script._apply_tokens(courses, {"CTA0017": {"SQL", "Python"}})

    assert courses[0]["tech_stacks"] == ["Python", "SQL"]  # 정렬·중복 제거된 리스트
    assert courses[1]["tech_stacks"] == []  # 강의계획서 없는 과목은 빈 리스트
    assert stats == {"courses": 2, "courses_with_tokens": 1}


def test_catalog_ids_collects_non_empty_course_ids(script: ModuleType) -> None:
    courses: list[object] = [
        {"course_id": "CTA0017"},
        {"course_id": "  W080023  "},  # 공백은 strip
        {"course_id": ""},  # 빈 id 는 제외
        {"course_name": "id 없는 엔트리"},  # course_id 키 부재
        "not a dict",  # dict 가 아닌 엔트리는 무시
    ]

    assert script._catalog_ids(courses) == {"CTA0017", "W080023"}


@pytest.mark.parametrize(
    "noise",
    [
        "문의: prof@hanmail.net",
        "담당교수 이메일 abc@daum.net",
        "참고: https://wikidocs.net/book",
        "reference wikidocs.net/page",
    ],
)
def test_strip_contact_noise_removes_email_and_url(script: ModuleType, noise: str) -> None:
    """이메일·URL·도메인은 추출 전에 지워져 도메인 꼬리(.net)가 .NET 으로 새지 않는다."""
    cleaned = script._strip_contact_noise(noise)
    assert "@" not in cleaned
    assert ".net" not in cleaned.lower()


def test_strip_contact_noise_preserves_inline_dotnet(script: ModuleType) -> None:
    """본문의 정식 표기(ASP.NET)는 소문자 호스트가 앞서지 않아 보존된다."""
    text = "ASP.NET 으로 서버를 구현하고 .NET Core 를 사용한다"
    assert script._strip_contact_noise(text) == text
