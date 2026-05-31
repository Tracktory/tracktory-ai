"""embedding_doc 순수 함수 단위 테스트 (네트워크 없음)."""

from tracktory.rag.preprocessing.tracks.embedding_doc import (
    _BOILERPLATE_PHRASES,
    _clean_line,
    _clean_section,
    build_doc_from_txt,
    build_embedding_doc,
    parse_track_txt,
)

# 실제 데이터 모양을 본뜬 픽스처 — 가운뎃점(ㆍ) 트랙명 + 네비 노이즈 + 마케팅 문구 혼입.
_SAMPLE_TXT = """[트랙: AIㆍ소프트웨어학과 | 대학: 미래플러스대학 | 학부: 미래플러스대학]

■ 소개
교육과정 로드맵
졸업요건
AI·소프트웨어학과는 인공지능(AI)과 소프트웨어 개발 역량을 겸비한 융합형 실무 인재 양성을 목표로 합니다.
4차 산업혁명 시대를 선도할 실용적인 AI·SW 전문가가 되고 싶다면 여러분의 미래를 시작해보세요!

■ 졸업 후 진로
AI 시스템 분야: 삼성전자, LG전자, 하이닉스

■ 관련 홈페이지
https://example.ac.kr
"""


def test_parse_track_txt_extracts_name_and_sections() -> None:
    name, sections = parse_track_txt(_SAMPLE_TXT)

    # 헤더 트랙명은 가운뎃점(ㆍ) 보존 — tracks.yaml track_id 와 원문 정합.
    assert name == "AIㆍ소프트웨어학과"
    assert set(sections) == {"소개", "진로"}  # 홈페이지는 _LABEL_TO_KEY 미수록 → 스킵
    assert "삼성전자" in sections["진로"]


def test_clean_line_drops_nav_noise() -> None:
    assert _clean_line("교육과정 로드맵") is None
    assert _clean_line("졸업요건") is None
    assert _clean_line("   ") is None


def test_clean_line_strips_boilerplate_but_keeps_domain_tokens() -> None:
    line = "인공지능(AI)과 소프트웨어 개발 역량을 겸비한 융합형 실무 인재 양성을 목표로 합니다."
    cleaned = _clean_line(line)

    assert cleaned is not None
    # 도메인 토큰은 보존, 마케팅 문구는 제거.
    assert "인공지능" in cleaned
    assert "소프트웨어 개발" in cleaned
    assert not any(phrase in cleaned for phrase in _BOILERPLATE_PHRASES)


def test_clean_section_removes_noise_lines() -> None:
    section = "교육과정 로드맵\n졸업요건\n삼성전자, LG전자"
    assert _clean_section(section) == "삼성전자, LG전자"


def test_build_embedding_doc_leads_with_name_and_includes_signal() -> None:
    sections = {
        "양성인력": "AI 시스템 개발 전문가",
        "진로": "삼성전자, 네이버",
        "소개": "교육과정 로드맵\n4차 산업혁명 융합 인재",
    }
    doc = build_embedding_doc("AI응용학과", sections)
    lines = doc.splitlines()

    assert lines[0] == "AI응용학과"  # 트랙명이 머리줄
    assert "양성 인력: AI 시스템 개발 전문가" in doc
    assert "진로: 삼성전자, 네이버" in doc
    # 소개는 네비/마케팅만 남아 정제 후 비므로 섹션 자체가 생략됨.
    assert "소개:" not in doc


def test_build_embedding_doc_skips_empty_sections() -> None:
    assert build_embedding_doc("빈트랙", {}) == "빈트랙"


def test_build_doc_from_txt_end_to_end() -> None:
    name, doc = build_doc_from_txt(_SAMPLE_TXT)

    assert name == "AIㆍ소프트웨어학과"
    assert doc.splitlines()[0] == "AIㆍ소프트웨어학과"
    assert "삼성전자" in doc
    assert "4차 산업혁명" not in doc  # 마케팅 문구 제거 확인
    assert "교육과정 로드맵" not in doc  # 네비 노이즈 제거 확인
