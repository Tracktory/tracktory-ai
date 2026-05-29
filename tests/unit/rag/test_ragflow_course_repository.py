from __future__ import annotations

import pytest

from tracktory.rag.ragflow_client import RagflowDocument, RagflowError
from tracktory.rag.ragflow_course_repository import (
    CourseRepositoryError,
    RagflowCourseRepository,
)

_CUR_A = RagflowDocument(
    document_id="curA",
    name="교육과정_트랙A.txt",
    meta_fields={"doc_type": "curriculum", "track_name": "트랙A"},
)
_CUR_B = RagflowDocument(
    document_id="curB",
    name="교육과정_트랙B.txt",
    meta_fields={"doc_type": "curriculum", "track_name": "트랙B"},
)

_BODY_A = (
    "[트랙: 트랙A | 대학: X]\n"
    " 1학년 1학기\n"
    "  - [전공기초] 기초과목 (C001, 3학점)\n"
    "  - [일반교양] 교양과목 (G001, 2학점)\n"
    " 2학년 1학기\n"
    "  - [전공선택] 선택과목 (C002, 3학점)\n"
    "  - [전공필수] 필수과목 (C003, 3학점)\n"
)
_BODY_B = (
    "[트랙: 트랙B | 대학: X]\n"
    " 1학년 1학기\n"
    "  - [전공선택] 기초과목 (C001, 3학점)\n"
    " 3학년 1학기\n"
    "  - [전공선택] 비공유과목 (C004, 3학점)\n"
)


class FakeClient:
    def __init__(
        self,
        docs: list[RagflowDocument],
        texts: dict[str, str],
        *,
        list_error: RagflowError | None = None,
    ) -> None:
        self._docs = docs
        self._texts = texts
        self._list_error = list_error

    def list_documents(
        self, *, name_keyword: str | None = None, page_size: int = 100
    ) -> list[RagflowDocument]:
        if self._list_error is not None:
            raise self._list_error
        return self._docs

    def fetch_document_text(self, document_id: str, *, page_size: int = 100) -> str:
        return self._texts.get(document_id, "")


def _repo(
    docs: list[RagflowDocument], texts: dict[str, str], **kwargs: object
) -> RagflowCourseRepository:
    return RagflowCourseRepository(FakeClient(docs, texts, **kwargs))  # type: ignore[arg-type]


def test_list_for_tracks_parses_courses_and_excludes_liberal_arts() -> None:
    repo = _repo([_CUR_A], {"curA": _BODY_A})

    courses = {c.course_id: c for c in repo.list_for_tracks(["트랙A"])}

    # 교양(G001)은 제외.
    assert set(courses) == {"C001", "C002", "C003"}
    assert courses["C001"].stage == "foundation"
    assert courses["C001"].course_type == "전공선택"  # 전공기초 → 전공선택
    assert courses["C001"].available_grades == [1]
    assert courses["C001"].available_semesters == [1]
    assert courses["C002"].stage == "application"  # 전공선택
    assert courses["C002"].available_grades == [2]
    assert courses["C002"].available_semesters == [3]
    assert courses["C003"].stage == "core"  # 전공필수
    assert courses["C003"].course_type == "전공필수"
    assert courses["C003"].available_semesters == [3]
    assert courses["C001"].track_ids == ["트랙A"]


def test_dedup_unions_track_ids_and_grades_across_tracks() -> None:
    repo = _repo([_CUR_A, _CUR_B], {"curA": _BODY_A, "curB": _BODY_B})

    courses = {c.course_id: c for c in repo.list_for_tracks(["트랙A", "트랙B"])}

    assert set(courses) == {"C001", "C002", "C003", "C004"}
    # C001 은 두 트랙에 등장 → track_ids union, stage 는 첫 등장(전공기초) 유지.
    assert courses["C001"].track_ids == ["트랙A", "트랙B"]
    assert courses["C001"].stage == "foundation"
    assert courses["C001"].available_grades == [1]
    assert courses["C001"].available_semesters == [1]
    assert courses["C004"].track_ids == ["트랙B"]
    assert courses["C004"].available_grades == [3]
    assert courses["C004"].available_semesters == [5]


def test_middle_dot_track_name_matches_curriculum() -> None:
    cur = RagflowDocument(
        document_id="cur1",
        name="교육과정_회계_재무경영트랙.txt",
        meta_fields={"doc_type": "curriculum", "track_name": "회계·재무경영트랙"},
    )
    body = " 2학년 1학기\n  - [전공필수] 회계원론 (ACC101, 3학점)\n"
    repo = _repo([cur], {"cur1": body})

    # 호출 측은 트랙소개 변형(ㆍ)으로 요청해도 매칭되어야 한다.
    courses = repo.list_for_tracks(["회계ㆍ재무경영트랙"])

    assert [c.course_id for c in courses] == ["ACC101"]


def test_non_curriculum_doc_type_is_ignored() -> None:
    syllabus = RagflowDocument(
        document_id="syl",
        name="강의계획서_교육과정의 이해_2026.txt",
        meta_fields={"doc_type": "syllabus", "track_name": "트랙A"},
    )
    repo = _repo([syllabus], {"syl": " 1학년 1학기\n  - [전공필수] 가짜 (X999, 3학점)\n"})

    assert repo.list_for_tracks(["트랙A"]) == []


def test_empty_track_ids_returns_empty() -> None:
    repo = _repo([_CUR_A], {"curA": _BODY_A})
    assert repo.list_for_tracks([]) == []


def test_list_error_is_converted_to_domain_error() -> None:
    repo = _repo([], {}, list_error=RagflowError(reason="network_retry_exhausted"))

    with pytest.raises(CourseRepositoryError):
        repo.list_for_tracks(["트랙A"])
