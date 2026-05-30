"""ProfileEmbedNode 의 계약을 검증한다.

본 노드는 더 이상 임베딩 클라이언트를 호출하지 않으므로 외부 의존성 없이
순수 직렬화 출력만 비교한다.
"""

from pathlib import Path

from tracktory.graph.nodes.profile_embed import (
    ProfileEmbedNode,
    _expand_dev_interests,
)


def _template_only_node() -> ProfileEmbedNode:
    """흥미 분야 확장을 끈 노드 — 템플릿 직렬화 자체만 검증하기 위함.

    존재하지 않는 사전 경로를 주면 확장 매핑이 비어 치환이 일어나지 않는다
    (사전 부재 시 확장 skip 은 노드의 계약).
    """
    return ProfileEmbedNode(dev_keywords_path=Path("__no_such_keywords__.yaml"))


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


def test_embed_reflects_all_semantic_fields() -> None:
    """관심사·흥미·취업 가치·선호 회사 유형·공부해본 분야 값이 모두 문장에 반영된다."""
    node = ProfileEmbedNode()
    profile = _valid_normalized()
    profile["interests"] = ["IT/인터넷"]
    profile["dev_interests"] = ["AI"]
    profile["work_values"] = ["성장성", "워라벨"]
    profile["company_types"] = ["대기업", "스타트업"]
    profile["ncs_studied"] = ["정보기술"]
    text = node({"normalized_profile": profile})["profile_text"]
    for value in ["IT/인터넷", "AI", "성장성", "워라벨", "대기업", "스타트업", "정보기술"]:
        assert value in text, f"{value} 가 직렬화 문장에 누락됨"


def test_embed_skips_empty_optional_clauses() -> None:
    """선택 입력(취업 가치·공부해본 분야)이 비어 있으면 해당 절을 건너뛴다."""
    node = ProfileEmbedNode()
    profile = _valid_normalized()
    profile["work_values"] = []
    profile["ncs_studied"] = []
    text = node({"normalized_profile": profile})["profile_text"]
    assert "가치를 중시하는" not in text
    assert "공부한 경험이 있는" not in text
    # 필수 입력 절 + suffix 는 항상 남아 문장이 완결된다.
    assert text.endswith("학생입니다.")


def test_embed_output_matches_snapshot() -> None:
    """결정론적 직렬화 결과를 정확한 문자열로 고정한다 (회귀 방지).

    템플릿 규칙이 의도치 않게 바뀌면 임베딩 공간이 흔들리므로, 정해진 입력에
    대한 출력 문장을 스냅샷으로 잠근다. 템플릿 변경이 의도적이면 본 스냅샷도
    함께 갱신한다. 흥미 분야 확장은 별도 테스트가 담당하므로 여기서는 끈다.
    """
    node = _template_only_node()
    text = node({"normalized_profile": _valid_normalized()})["profile_text"]
    assert text == (
        "IT/인터넷 분야에 관심이 많은, AI 개발에 흥미가 있는, "
        "성장성 가치를 중시하는, 대기업 취업을 선호하는 학생입니다."
    )


def test_embed_multivalue_uses_distinct_value_separator() -> None:
    """한 절의 여러 값은 절 구분자(', ')와 다른 기호(' · ')로 이어 붙는다.

    값 경계와 절 경계가 같은 구분자면 읽을 때 섞이므로, 다값 케이스의
    정확한 출력을 스냅샷으로 잠근다. 흥미 분야 확장은 끄고 구분자만 본다.
    """
    node = _template_only_node()
    profile = _valid_normalized()
    profile["dev_interests"] = ["AI", "데이터"]
    profile["work_values"] = ["성장성", "워라벨"]
    text = node({"normalized_profile": profile})["profile_text"]
    assert text == (
        "IT/인터넷 분야에 관심이 많은, AI · 데이터 개발에 흥미가 있는, "
        "성장성 · 워라벨 가치를 중시하는, 대기업 취업을 선호하는 학생입니다."
    )


def test_embed_expands_known_dev_interest() -> None:
    """확장 사전에 정의된 흥미 분야는 직무 어휘 키워드로 치환되어 문장에 반영된다."""
    node = ProfileEmbedNode()
    text = node({"normalized_profile": _valid_normalized()})["profile_text"]
    # "AI" 가 직무 검색 어휘(머신러닝 등)를 포함한 키워드로 확장된다.
    assert "머신러닝" in text
    assert "AI 개발에 흥미가 있는" not in text


def test_embed_passes_through_unknown_dev_interest() -> None:
    """확장 사전에 없는 흥미 분야는 원본 값 그대로 문장에 남는다 (신규 값 graceful)."""
    node = ProfileEmbedNode()
    profile = _valid_normalized()
    profile["dev_interests"] = ["미정의분야"]
    text = node({"normalized_profile": profile})["profile_text"]
    assert "미정의분야 개발에 흥미가 있는" in text


def test_missing_keywords_file_disables_expansion() -> None:
    """확장 사전 파일이 없으면 치환 없이 원본 값으로 직렬화한다 (확장은 선택 기능)."""
    node = _template_only_node()
    text = node({"normalized_profile": _valid_normalized()})["profile_text"]
    assert "AI 개발에 흥미가 있는" in text
    assert "머신러닝" not in text


def test_expand_dev_interests_does_not_mutate_original() -> None:
    """확장은 사본에만 적용하고 원본 프로필 dict 은 건드리지 않는다.

    다른 노드가 같은 normalized_profile 을 원본 값으로 읽으므로, 흥미 분야
    치환이 원본을 변형하면 안 된다.
    """
    profile = _valid_normalized()
    _expand_dev_interests(profile, {"AI": "AI(머신러닝)"})
    assert profile["dev_interests"] == ["AI"]
