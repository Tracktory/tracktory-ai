from __future__ import annotations

from typing import Any

import pytest

from tracktory.rag.ragflow_client import RagflowDocument, RagflowError
from tracktory.rag.ragflow_track_repository import (
    RagflowTrackRepository,
    TrackRepositoryError,
)

_INTRO = RagflowDocument(
    document_id="intro1",
    name="트랙소개_응용산업데이터공학트랙_산업시스템공학부.txt",
    meta_fields={
        "doc_type": "track_intro",
        "track_name": "응용산업데이터공학트랙",
        "college": "IT공과대학",
        "department": "산업시스템공학부",
    },
)
_CURRICULUM = RagflowDocument(
    document_id="cur1",
    name="교육과정_응용산업데이터공학트랙_산업시스템공학부.txt",
    meta_fields={
        "doc_type": "curriculum",
        "track_name": "응용산업데이터공학트랙",
        "college": "IT공과대학",
        "department": "산업시스템공학부",
    },
)
# 같은 과목코드(CTE0029)가 두 번 등장 → dedup 검증.
_CURRICULUM_BODY = (
    "1학년 1학기\n"
    "  - [전공기초] 데이터리터러시 (CTE0029, 3학점)\n"
    "2학년 1학기\n"
    "  - [전공선택] 공학프로그래밍 (V070044, 3학점)\n"
    "  - [전공필수] 데이터공학 선형대수 (V076009, 3학점)\n"
    "2학년 2학기\n"
    "  - [전공기초] 데이터리터러시 (CTE0029, 3학점)\n"
)


class FakeClient:
    """list_documents / fetch_document_text 만 흉내내는 RagflowClient 대역."""

    def __init__(
        self,
        docs_by_keyword: dict[str | None, list[RagflowDocument]],
        texts: dict[str, str],
        *,
        list_error: RagflowError | None = None,
    ) -> None:
        self._docs = docs_by_keyword
        self._texts = texts
        self._list_error = list_error

    def list_documents(
        self, *, name_keyword: str | None = None, page_size: int = 100
    ) -> list[RagflowDocument]:
        if self._list_error is not None:
            raise self._list_error
        return self._docs.get(name_keyword, [])

    def fetch_document_text(self, document_id: str, *, page_size: int = 100) -> str:
        return self._texts.get(document_id, "")


def _repo(
    docs_by_keyword: dict[str | None, list[RagflowDocument]],
    texts: dict[str, str],
    **kwargs: Any,
) -> RagflowTrackRepository:
    return RagflowTrackRepository(FakeClient(docs_by_keyword, texts, **kwargs))  # type: ignore[arg-type]


def test_list_all_assembles_track_with_courses() -> None:
    repo = _repo(
        {"트랙소개": [_INTRO], "교육과정": [_CURRICULUM]},
        {"intro1": "AI, 통계 등 데이터 분석 인재 양성.", "cur1": _CURRICULUM_BODY},
    )

    tracks = repo.list_all()

    assert len(tracks) == 1
    track = tracks[0]
    assert track.track_id == "응용산업데이터공학트랙"
    assert track.track_name == "응용산업데이터공학트랙"
    assert track.college_id == "IT공과대학"
    assert track.department_id == "산업시스템공학부"
    # single-track degenerate: major_id == department_id.
    assert track.major_id == "산업시스템공학부"
    # 등장 순서 보존 + dedup.
    assert track.course_ids == ["CTE0029", "V070044", "V076009"]
    # meta_text 는 소비처가 없어 빈 문자열(소개 본문 fetch 생략).
    assert track.meta_text == ""
    # 메타 임베딩 보류.
    assert track.meta_vector == []
    assert track.competencies == []
    assert track.tech_stacks == []


def test_course_ids_exclude_liberal_arts_and_non_major() -> None:
    # 교양류·비전공 과목은 course_ids 에서 빠져야 한다 — 과목 카탈로그(전공만)와
    # 정합. 과목 저장소와 동일한 전공 판정 기준(parse_course_line)을 공유한다.
    body = (
        "1학년 1학기\n"
        "  - [전공필수] 전공과목 (MAJ001, 3학점)\n"
        "  - [일반교양] 교양과목 (GEN0123, 2학점)\n"
        "  - [선택필수교양] 글쓰기 (GEN0456, 3학점)\n"
        "  - [교직] 교직과목 (REQ0001, 2학점)\n"
        "  - [전공선택] 또다른전공 (MAJ002, 3학점)\n"
    )
    repo = _repo(
        {"트랙소개": [_INTRO], "교육과정": [_CURRICULUM]},
        {"intro1": "소개", "cur1": body},
    )

    tracks = repo.list_all()

    assert len(tracks) == 1
    # GEN0123 / GEN0456 / REQ0001 은 비전공 → 제외, 전공만 순서 보존.
    assert tracks[0].course_ids == ["MAJ001", "MAJ002"]


def test_track_without_matching_curriculum_has_empty_course_ids() -> None:
    repo = _repo(
        {"트랙소개": [_INTRO], "교육과정": []},
        {"intro1": "소개 텍스트."},
    )

    tracks = repo.list_all()

    assert len(tracks) == 1
    assert tracks[0].course_ids == []


def test_track_intro_missing_track_name_is_skipped() -> None:
    bad = RagflowDocument(
        document_id="bad",
        name="트랙소개_미상.txt",
        meta_fields={"doc_type": "track_intro", "college": "C", "department": "D"},
    )
    repo = _repo({"트랙소개": [bad], "교육과정": []}, {"bad": "x"})

    assert repo.list_all() == []


def test_track_missing_affiliation_is_skipped() -> None:
    bad = RagflowDocument(
        document_id="bad",
        name="트랙소개_무소속트랙.txt",
        meta_fields={"doc_type": "track_intro", "track_name": "무소속트랙"},
    )
    repo = _repo({"트랙소개": [bad], "교육과정": []}, {"bad": "x"})

    assert repo.list_all() == []


def test_find_by_track_ids_filters_and_ignores_unknown() -> None:
    other = RagflowDocument(
        document_id="intro2",
        name="트랙소개_빅데이터트랙_컴퓨터공학부.txt",
        meta_fields={
            "doc_type": "track_intro",
            "track_name": "빅데이터트랙",
            "college": "IT공과대학",
            "department": "컴퓨터공학부",
        },
    )
    repo = _repo(
        {"트랙소개": [_INTRO, other], "교육과정": []},
        {"intro1": "a", "intro2": "b"},
    )

    tracks = repo.find_by_track_ids(["빅데이터트랙", "존재하지않는트랙"])

    assert [t.track_id for t in tracks] == ["빅데이터트랙"]


def test_empty_track_ids_returns_empty() -> None:
    repo = _repo({"트랙소개": [_INTRO], "교육과정": []}, {"intro1": "a"})
    assert repo.find_by_track_ids([]) == []


def test_middle_dot_variants_still_match_curriculum() -> None:
    # 트랙소개는 한글 아래아(ㆍ), 교육과정은 가운뎃점(·) — 같은 트랙이지만
    # 가운뎃점 문자만 다르다. 정규화 후 짝지어져야 한다.
    intro = RagflowDocument(
        document_id="intro1",
        name="트랙소개_회계ㆍ재무경영트랙_사회과학부.txt",
        meta_fields={
            "doc_type": "track_intro",
            "track_name": "회계ㆍ재무경영트랙",
            "college": "사회과학대학",
            "department": "사회과학부",
        },
    )
    curriculum = RagflowDocument(
        document_id="cur1",
        name="교육과정_회계_재무경영트랙.txt",
        meta_fields={"doc_type": "curriculum", "track_name": "회계·재무경영트랙"},
    )
    repo = _repo(
        {"트랙소개": [intro], "교육과정": [curriculum]},
        {"intro1": "소개", "cur1": "  - [전공필수] 회계원론 (ACC101, 3학점)"},
    )

    tracks = repo.list_all()

    assert len(tracks) == 1
    # 표시용 이름은 원본(아래아) 유지.
    assert tracks[0].track_name == "회계ㆍ재무경영트랙"
    # 가운뎃점만 다른 교육과정과 짝지어져 과목이 채워짐.
    assert tracks[0].course_ids == ["ACC101"]


def test_curriculum_filter_ignores_non_curriculum_doc_types() -> None:
    # 파일명에 "교육과정" 이 들었지만 doc_type 은 syllabus 인 문서(과목)는
    # 교육과정으로 오인되지 않아야 한다.
    syllabus_lookalike = RagflowDocument(
        document_id="syl1",
        name="강의계획서_한국어 교육과정의 이해와 적용_2026.txt",
        meta_fields={
            "doc_type": "syllabus",
            "track_name": "응용산업데이터공학트랙",
        },
    )
    repo = _repo(
        {"트랙소개": [_INTRO], "교육과정": [syllabus_lookalike]},
        {"intro1": "소개", "syl1": "  - [전공필수] 가짜 (FAKE999, 3학점)"},
    )

    tracks = repo.list_all()

    assert len(tracks) == 1
    # syllabus 가 교육과정으로 오인됐다면 FAKE999 가 섞였을 것 → 비어야 정상.
    assert tracks[0].course_ids == []


def test_list_error_is_converted_to_domain_error() -> None:
    repo = _repo({}, {}, list_error=RagflowError(reason="network_retry_exhausted"))

    with pytest.raises(TrackRepositoryError):
        repo.list_all()
