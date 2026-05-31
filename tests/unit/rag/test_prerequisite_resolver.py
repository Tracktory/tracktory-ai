from __future__ import annotations

from tracktory.rag.prerequisite_resolver import normalize_course_name, resolve_prereq_ids


def test_normalize_strips_parens_spaces_and_lowercases() -> None:
    assert normalize_course_name("회계원론 (미시기초)") == "회계원론"
    assert normalize_course_name("AI 프로그래밍") == "ai프로그래밍"
    assert normalize_course_name("English Writing Skills") == "englishwritingskills"


def test_resolves_names_to_codes() -> None:
    prereqs = {"공학프로그래밍": ["데이터리터러시"], "데이터리터러시": []}
    courses = [("데이터리터러시", "C001"), ("공학프로그래밍", "C002")]

    result = resolve_prereq_ids(prereqs, courses)

    assert result.prereq_ids_by_course == {"C002": ["C001"]}
    assert result.dropped_edges == []
    assert result.total_resolved_edges == 1


def test_drops_prereq_not_in_catalog() -> None:
    # 카탈로그(전공)에 없는 선수(교양/타과)는 드롭되고 통계에 기록된다.
    prereqs = {"전자기학": ["대학수학Ⅱ", "회로이론"]}
    courses = [("전자기학", "E001"), ("회로이론", "E002")]

    result = resolve_prereq_ids(prereqs, courses)

    assert result.prereq_ids_by_course == {"E001": ["E002"]}
    assert result.dropped_edges == [("전자기학", "대학수학Ⅱ")]


def test_self_reference_is_dropped() -> None:
    prereqs = {"객체지향언어2": ["객체지향언어2", "객체지향언어1"]}
    courses = [("객체지향언어1", "O001"), ("객체지향언어2", "O002")]

    result = resolve_prereq_ids(prereqs, courses)

    # 자기참조(O002)는 빠지고 O001 만 남는다.
    assert result.prereq_ids_by_course == {"O002": ["O001"]}


def test_dedup_and_order_preserved() -> None:
    prereqs = {"종합설계": ["기초A", "기초B", "기초A"]}
    courses = [("기초A", "A1"), ("기초B", "B1"), ("종합설계", "D1")]

    result = resolve_prereq_ids(prereqs, courses)

    assert result.prereq_ids_by_course == {"D1": ["A1", "B1"]}


def test_paren_and_space_variants_match() -> None:
    # 카탈로그 이름과 선수 이름의 괄호·공백 차이는 정규화로 흡수된다.
    prereqs = {"공학프로그래밍": ["데이터 리터러시 (기초)"]}
    courses = [("데이터리터러시", "C001"), ("공학프로그래밍", "C002")]

    result = resolve_prereq_ids(prereqs, courses)

    assert result.prereq_ids_by_course == {"C002": ["C001"]}


def test_course_without_prereq_entry_is_absent() -> None:
    prereqs = {"공학프로그래밍": ["데이터리터러시"]}
    courses = [("데이터리터러시", "C001"), ("미등록과목", "C999")]

    result = resolve_prereq_ids(prereqs, courses)

    # prereq 맵에 없거나 선수가 비면 결과 dict 에 키가 없다(기본 [] 로 덤프됨).
    assert "C999" not in result.prereq_ids_by_course
