"""Explanation 도메인 모델 invariant 검증.

영역별 단락의 topic Literal, 빈 본문 차단, 전체 요약 본문의 빈 문자열 차단,
sections 가 비어 있어도 valid 한지, 직렬화 round-trip 이 동일 모델을
복원하는지 확인한다.
"""

import pytest
from pydantic import ValidationError

from tracktory.graph.models import Explanation, ExplanationSection


def test_explanation_accepts_text_only_without_sections() -> None:
    explanation = Explanation(text="전체 요약 본문입니다.")
    assert explanation.sections == []


def test_explanation_accepts_text_with_sections() -> None:
    explanation = Explanation(
        text="전체 요약",
        sections=[
            ExplanationSection(topic="jobs", body="직무 근거"),
            ExplanationSection(topic="tracks", body="트랙 근거"),
        ],
    )
    assert len(explanation.sections) == 2
    assert explanation.sections[0].topic == "jobs"


def test_explanation_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Explanation(text="")


def test_section_rejects_unknown_topic() -> None:
    with pytest.raises(ValidationError):
        ExplanationSection(topic="careers", body="본문")  # type: ignore[arg-type]


def test_section_rejects_empty_body() -> None:
    with pytest.raises(ValidationError):
        ExplanationSection(topic="jobs", body="")


def test_section_accepts_all_three_topics() -> None:
    for topic in ("jobs", "tracks", "roadmap"):
        section = ExplanationSection(topic=topic, body="본문")  # type: ignore[arg-type]
        assert section.topic == topic


def test_explanation_roundtrip_model_dump_validate() -> None:
    """model_dump(mode='json') → model_validate 가 원본과 동등한 객체를 복원한다."""
    original = Explanation(
        text="관심사와 흥미를 종합하면 백엔드 개발자가 가장 잘 맞아요.",
        sections=[
            ExplanationSection(topic="jobs", body="백엔드 개발자가 1순위인 이유는..."),
            ExplanationSection(topic="tracks", body="빅데이터 + 모바일소프트웨어 조합 추천."),
            ExplanationSection(topic="roadmap", body="2학기 자료구조부터 시작하세요."),
        ],
    )
    dumped = original.model_dump(mode="json")
    restored = Explanation.model_validate(dumped)

    assert restored == original
    assert len(restored.sections) == 3
    assert restored.sections[2].topic == "roadmap"
