"""``hansung_catalog`` 의 가운뎃점 정규화 함수 단위 테스트."""

from __future__ import annotations

import pytest

from tracktory.rag.hansung_catalog import canonical_track_name, match_track_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("영상·애니메이션디자인트랙", "영상ㆍ애니메이션디자인트랙"),  # 가운뎃점(U+00B7) → 아래아
        ("영상ㆍ애니메이션디자인트랙", "영상ㆍ애니메이션디자인트랙"),  # 이미 권위 표기면 그대로
        ("VMD•전시디자인트랙", "VMDㆍ전시디자인트랙"),  # bullet 변형도 흡수
        ("  회계·재무경영트랙  ", "회계ㆍ재무경영트랙"),  # strip
        ("빅데이터트랙", "빅데이터트랙"),  # 가운뎃점 없는 트랙은 무변화
    ],
)
def test_canonical_track_name_restores_araea(raw: str, expected: str) -> None:
    assert canonical_track_name(raw) == expected


def test_canonical_and_match_agree_on_same_track() -> None:
    """표기만 다른 두 트랙명이 권위 표기로 모이면 글자 단위로 동일해진다."""
    intro = "회계ㆍ재무경영트랙"  # 트랙소개 표기(아래아)
    curriculum = "회계·재무경영트랙"  # 교육과정 표기(가운뎃점)
    assert canonical_track_name(intro) == canonical_track_name(curriculum)
    # 조인 키(match_track_name)는 가운뎃점으로 통일하므로 여전히 같은 트랙으로 매칭된다.
    assert match_track_name(intro) == match_track_name(curriculum)
