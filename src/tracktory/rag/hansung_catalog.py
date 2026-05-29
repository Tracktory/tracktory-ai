"""한성대 학사 카탈로그(트랙소개·교육과정) 공통 유틸.

트랙소개와 교육과정이 같은 트랙을 가리키면서도 가운뎃점 문자만 다른 경우가
있다 — 트랙소개는 한글 아래아(ㆍ, U+318D), 교육과정은 가운뎃점(·, U+00B7).
트랙명을 짝지을 때 이 가운뎃점류를 한 글자로 통일해야 같은 트랙으로 매칭된다.
트랙·과목 두 repository 가 동일 규칙을 공유하도록 본 모듈에 모은다.
"""

from __future__ import annotations

# 가운뎃점류 통일 (한글 아래아 ㆍ, bullet • 등 → 가운뎃점 ·).
_MIDDLE_DOT_UNIFY = str.maketrans({"ㆍ": "·", "•": "·", "‧": "·", "・": "·"})


def match_track_name(track_name: str) -> str:
    """트랙소개·교육과정 트랙명을 짝지을 때 쓰는 정규화 키(가운뎃점 통일 + strip)."""
    return track_name.translate(_MIDDLE_DOT_UNIFY).strip()
