"""프로필 임베딩 노드 단위 테스트.

EmbeddingClient 를 MagicMock 으로 대체하여 네트워크 호출 없이 노드 계약을 검증한다.
"""

from unittest.mock import MagicMock

from tracktory.graph.nodes.profile_embed import EmbeddingClient, ProfileEmbedNode


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
    """normalized_profile 이 없으면 skip 트레이스를 반환하고 embed 를 호출하지 않는다."""
    client = MagicMock(spec=EmbeddingClient)
    client.embed.return_value = [0.0]
    node = ProfileEmbedNode(embedding_client=client)
    result = node({})
    assert result["trace"] == ["N2:skip"]
    assert len(result["errors"]) >= 1
    assert "profile_text" not in result
    assert "profile_vector" not in result
    client.embed.assert_not_called()


def test_embed_invokes_client_and_returns_vector() -> None:
    """유효한 normalized_profile 이 있을 때 embed 를 1 회 호출하고 벡터를 그대로 반환한다."""
    sentinel_vector = [0.1, 0.2, 0.3]
    client = MagicMock(spec=EmbeddingClient)
    client.embed.return_value = sentinel_vector
    node = ProfileEmbedNode(embedding_client=client)
    result = node({"normalized_profile": _valid_normalized()})
    assert isinstance(result["profile_text"], str) and result["profile_text"]
    assert result["profile_vector"] is sentinel_vector
    assert result["trace"] == ["N2:ok"]
    assert client.embed.call_count == 1
    assert client.embed.call_args.args[0] == result["profile_text"]


def test_embed_uses_template_pattern_deterministically() -> None:
    """동일 입력에 대해 두 번 호출해도 profile_text 가 동일하다 (Template 직렬화 결정론성)."""
    client = MagicMock(spec=EmbeddingClient)
    client.embed.return_value = [0.0]
    node = ProfileEmbedNode(embedding_client=client)
    profile = _valid_normalized()
    r1 = node({"normalized_profile": profile})
    r2 = node({"normalized_profile": profile})
    assert r1["profile_text"] == r2["profile_text"]
    assert client.embed.call_count == 2
    assert client.embed.call_args_list[0] == client.embed.call_args_list[1]


def test_embed_excludes_completed_courses_from_text() -> None:
    """completed_courses 는 profile_text 에 포함되지 않는다 (이수 과목은 임베딩 대상 X)."""
    client = MagicMock(spec=EmbeddingClient)
    client.embed.return_value = [0.0]
    node = ProfileEmbedNode(embedding_client=client)
    profile = _valid_normalized()
    profile["completed_courses"] = ["자료구조", "알고리즘"]
    result = node({"normalized_profile": profile})
    assert "자료구조" not in result["profile_text"]
    assert "알고리즘" not in result["profile_text"]
