"""ProfileEmbedNode 의 계약을 검증한다.

본 노드는 더 이상 임베딩 클라이언트를 호출하지 않으므로 외부 의존성 없이
순수 직렬화 출력만 비교한다.
"""

from tracktory.graph.nodes.profile_embed import ProfileEmbedNode


def _valid_normalized() -> dict[str, object]:
    return {
        "admission_year": 2025,
        "college": "IT공과대학",
        "department": "컴퓨터공학부",
        "current_tracks": [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


def test_embed_skips_when_normalized_profile_missing() -> None:
    """normalized_profile 이 없으면 skip 트레이스를 반환하고 텍스트도 채우지 않는다."""
    node = ProfileEmbedNode()
    result = node({})
    assert result["trace"] == ["profile_embed:skip"]
    assert len(result["errors"]) >= 1
    assert "profile_text" not in result


def test_embed_returns_profile_text_only() -> None:
    """유효한 normalized_profile 이 있을 때 profile_text 만 채워 반환한다."""
    node = ProfileEmbedNode()
    result = node({"normalized_profile": _valid_normalized()})
    assert isinstance(result["profile_text"], str) and result["profile_text"]
    assert result["trace"] == ["profile_embed:ok"]
    assert "profile_vector" not in result


def test_embed_uses_template_pattern_deterministically() -> None:
    """동일 입력에 대해 두 번 호출해도 profile_text 가 동일하다 (Template 직렬화 결정론성)."""
    node = ProfileEmbedNode()
    profile = _valid_normalized()
    r1 = node({"normalized_profile": profile})
    r2 = node({"normalized_profile": profile})
    assert r1["profile_text"] == r2["profile_text"]


def test_embed_excludes_completed_courses_from_text() -> None:
    """completed_courses 는 profile_text 에 포함되지 않는다 (이수 과목은 직렬화 대상 X)."""
    node = ProfileEmbedNode()
    profile = _valid_normalized()
    profile["completed_courses"] = ["자료구조", "알고리즘"]
    result = node({"normalized_profile": profile})
    assert "자료구조" not in result["profile_text"]
    assert "알고리즘" not in result["profile_text"]
