"""한성대 학사 카탈로그(트랙소개·교육과정) 공통 유틸.

트랙소개와 교육과정이 같은 트랙을 가리키면서도 가운뎃점 문자만 다른 경우가
있다 — 트랙소개는 한글 아래아(ㆍ, U+318D), 교육과정은 가운뎃점(·, U+00B7).
트랙명을 짝지을 때 이 가운뎃점류를 한 글자로 통일해야 같은 트랙으로 매칭된다.
트랙·과목 두 repository 가 동일 규칙을 공유하도록 본 모듈에 모은다.
"""

from __future__ import annotations

# 가운뎃점류 통일 (한글 아래아 ㆍ, bullet • 등 → 가운뎃점 ·).
_MIDDLE_DOT_UNIFY = str.maketrans({"ㆍ": "·", "•": "·", "‧": "·", "・": "·"})

# 가운뎃점류를 권위 표기인 한글 아래아(ㆍ, U+318D)로 복원. RAG 본문은 GraphRAG
# 엔티티 추출 NaN 회피를 위해 가운뎃점(·)으로 치환되지만, 카탈로그 트랙명은
# 원본 학사 데이터·백엔드 표기(ㆍ)와 글자 단위로 일치해야 이름 기준 조인·표시가
# 어긋나지 않는다.
_MIDDLE_DOT_CANONICAL = str.maketrans({"·": "ㆍ", "•": "ㆍ", "‧": "ㆍ", "・": "ㆍ"})


def match_track_name(track_name: str) -> str:
    """트랙소개·교육과정 트랙명을 짝지을 때 쓰는 정규화 키(가운뎃점 통일 + strip)."""
    return track_name.translate(_MIDDLE_DOT_UNIFY).strip()


def canonical_track_name(track_name: str) -> str:
    """트랙명을 권위 표기(가운뎃점류 → 한글 아래아 ㆍ)로 정규화한다.

    카탈로그가 노출하는 트랙 식별자·표시명의 단일 형태를 보장하는 용도. RAG 본문
    헤더는 임베딩 안정성 때문에 가운뎃점(·)으로 치환된 채 남지만, 그 헤더에서
    역파싱된 트랙명은 본 함수로 원본 학사 데이터 표기(ㆍ)로 되돌려 백엔드·원본과
    문자 단위로 일치시킨다.
    """
    return track_name.translate(_MIDDLE_DOT_CANONICAL).strip()
