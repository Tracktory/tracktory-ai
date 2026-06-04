"""정합 토큰 사전 (canonical_tech_token / key / keys) 검증.

직무·트랙 양쪽이 같은 기술을 다르게 적어둔 표기 차이를 한 어휘로 통합하는지,
약어·고유 대소문자가 .title() 폴백으로 오변환되지 않는지를 검증한다.
"""

from __future__ import annotations

import pytest

from tracktory.common.tech_keywords import (
    canonical_tech_key,
    canonical_tech_keys,
    canonical_tech_token,
)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("ReactJS", "React"),
        ("react.js", "React"),
        ("SpringBoot", "Spring Boot"),
        ("자바", "Java"),
        ("golang", "Go"),
    ],
)
def test_canonical_token_maps_alias(variant: str, expected: str) -> None:
    """NORMALIZATION_MAP 에 등록된 변형은 직무 어휘 정규 이름으로 통합된다."""
    assert canonical_tech_token(variant) == expected


@pytest.mark.parametrize("tech", ["SQL", "iOS", "TensorFlow", "GPT", "Power BI", "AWS"])
def test_canonical_token_preserves_casing_on_miss(tech: str) -> None:
    """사전에 없는 약어·고유 대소문자는 원본을 유지한다 (.title() 오변환 금지)."""
    assert canonical_tech_token(tech) == tech


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_canonical_token_blank_returns_empty(blank: str) -> None:
    """공백뿐인 입력은 빈 문자열로 정규화된다."""
    assert canonical_tech_token(blank) == ""


def test_canonical_key_unifies_casing_and_alias() -> None:
    """비교 키는 대소문자·별칭 차이를 모두 흡수한다."""
    assert canonical_tech_key("SQL") == canonical_tech_key("sql")
    assert canonical_tech_key("ReactJS") == canonical_tech_key("react")
    assert canonical_tech_key("Spring Boot") == canonical_tech_key("springboot")


def test_canonical_keys_dedups_and_drops_blank() -> None:
    """키 집합은 표기 차이를 합치고 빈 값을 제외한다."""
    keys = canonical_tech_keys(["React", "reactjs", "SQL", "sql", "", "  "])
    assert keys == {"react", "sql"}


def test_canonical_keys_splits_compound_labels() -> None:
    """복합 표기 직무 라벨이 원자 키로 분해돼 트랙 토큰과 매칭된다.

    분해 없이 ``"AWS / GCP"`` 가 한 키로 남으면 트랙의 ``"AWS"`` 와 교집합이
    잡히지 않아 직무 커버율이 0 으로 붕괴한다.
    """
    keys = canonical_tech_keys(
        [
            "AWS / GCP",
            "Docker + Kubernetes",
            "dbt (data build tool)",
            "SQL (+ BigQuery / Snowflake)",
        ]
    )
    assert {"aws", "gcp", "docker", "kubernetes", "dbt", "sql"} <= keys


def test_canonical_keys_preserve_separator_in_single_token() -> None:
    """구분자가 토큰 일부인 단일 기술은 쪼개지지 않는다 (C++, A/B, CI/CD, VR/AR)."""
    assert canonical_tech_keys(["C++"]) == {"c++"}
    assert canonical_tech_keys(["C#"]) == {"c#"}
    assert canonical_tech_keys(["CI/CD"]) == {"ci/cd"}
    assert canonical_tech_keys(["A/B 테스팅"]) == {"a/b 테스팅"}
    assert "vr/ar sdk" in canonical_tech_keys(["VR/AR SDK (ARCore, ARKit)"])
