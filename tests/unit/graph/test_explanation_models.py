"""Explanation 도메인 모델 invariant 검증.

영역별 단락의 topic Literal, 빈 본문 차단, 전체 요약 본문의 빈 문자열 차단,
sections 가 비어 있어도 valid 한지, 직렬화 round-trip 이 동일 모델을
복원하는지 확인한다.
"""

import pytest
from pydantic import ValidationError

from tracktory.graph.models import (
    CourseFlow,
    Explanation,
    ExplanationSection,
    JobRationale,
    SemesterSubtitle,
    TrackRationale,
)


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


def test_explanation_defaults_new_outputs_to_empty() -> None:
    """기존 호출부 (text + sections) 가 새 출력 필드 없이도 valid — 하위호환."""
    explanation = Explanation(text="전체 요약")
    assert explanation.semester_subtitles == []
    assert explanation.course_flows == []
    assert explanation.job_rationales == []
    assert explanation.track_rationales == []


def test_job_rationale_accepts_valid_payload() -> None:
    rationale = JobRationale(
        job_id="be_dev_001", rationale="백엔드 개발자는 당신의 관심사와 잘 맞아요."
    )
    assert rationale.job_id == "be_dev_001"
    assert rationale.rationale.startswith("백엔드")


def test_job_rationale_rejects_empty_fields() -> None:
    with pytest.raises(ValidationError):
        JobRationale(job_id="", rationale="본문")
    with pytest.raises(ValidationError):
        JobRationale(job_id="be_dev_001", rationale="")


def test_track_rationale_accepts_valid_payload() -> None:
    rationale = TrackRationale(
        combo_key="t_a::t_b",
        combo_rationale="두 트랙을 함께 들으면 시너지가 큽니다.",
        track_a_rationale="빅데이터 트랙은 분석 역량을 키워줍니다.",
        track_b_rationale="모바일 트랙은 앱 개발 역량을 키워줍니다.",
    )
    assert rationale.combo_key == "t_a::t_b"
    # 조합 전체 근거와 개별 트랙 근거가 서로 다른 필드로 구분된다.
    assert rationale.combo_rationale != rationale.track_a_rationale
    assert rationale.track_a_rationale != rationale.track_b_rationale


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "combo_key": "",
            "combo_rationale": "x",
            "track_a_rationale": "x",
            "track_b_rationale": "x",
        },
        {
            "combo_key": "k",
            "combo_rationale": "",
            "track_a_rationale": "x",
            "track_b_rationale": "x",
        },
        {
            "combo_key": "k",
            "combo_rationale": "x",
            "track_a_rationale": "",
            "track_b_rationale": "x",
        },
        {
            "combo_key": "k",
            "combo_rationale": "x",
            "track_a_rationale": "x",
            "track_b_rationale": "",
        },
    ],
)
def test_track_rationale_rejects_empty_fields(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        TrackRationale(**kwargs)


def test_explanation_roundtrip_with_item_rationales() -> None:
    """직무·트랙 항목별 근거까지 포함한 round-trip 동등성."""
    original = Explanation(
        text="요약",
        job_rationales=[
            JobRationale(job_id="be_dev_001", rationale="백엔드 개발자가 1순위인 이유."),
            JobRationale(job_id="data_001", rationale="데이터 분석가도 잘 맞는 이유."),
        ],
        track_rationales=[
            TrackRationale(
                combo_key="t_a::t_b",
                combo_rationale="조합 시너지 근거.",
                track_a_rationale="1트랙 근거.",
                track_b_rationale="2트랙 근거.",
            ),
        ],
    )
    restored = Explanation.model_validate(original.model_dump(mode="json"))

    assert restored == original
    assert len(restored.job_rationales) == 2
    assert restored.job_rationales[1].job_id == "data_001"
    assert len(restored.track_rationales) == 1
    assert restored.track_rationales[0].combo_key == "t_a::t_b"


def test_semester_subtitle_accepts_valid_payload() -> None:
    subtitle = SemesterSubtitle(semester=3, subtitle="이번 학기는 트랙 핵심 단계입니다")
    assert subtitle.semester == 3
    assert "핵심" in subtitle.subtitle


@pytest.mark.parametrize("semester", [0, 9])
def test_semester_subtitle_rejects_out_of_range_semester(semester: int) -> None:
    with pytest.raises(ValidationError):
        SemesterSubtitle(semester=semester, subtitle="본문")


def test_semester_subtitle_rejects_empty_subtitle() -> None:
    with pytest.raises(ValidationError):
        SemesterSubtitle(semester=1, subtitle="")


def test_course_flow_accepts_valid_payload() -> None:
    flow = CourseFlow(
        course_id="c1",
        flow="당신의 관심사 → 백엔드 개발자 직무 → 빅데이터 + 모바일소프트웨어 트랙 → 이 과목이 기초입니다",
    )
    assert flow.course_id == "c1"
    assert flow.flow.startswith("당신의 관심사")


def test_course_flow_rejects_empty_fields() -> None:
    with pytest.raises(ValidationError):
        CourseFlow(course_id="", flow="본문")
    with pytest.raises(ValidationError):
        CourseFlow(course_id="c1", flow="")


def test_explanation_roundtrip_with_new_outputs() -> None:
    """학기 부제·과목 인과 흐름까지 포함한 round-trip 동등성."""
    original = Explanation(
        text="요약",
        sections=[ExplanationSection(topic="roadmap", body="자료구조부터 시작하세요.")],
        semester_subtitles=[
            SemesterSubtitle(semester=1, subtitle="이번 학기는 트랙 기초 단계입니다"),
            SemesterSubtitle(semester=2, subtitle="이번 학기는 트랙 핵심 단계입니다"),
        ],
        course_flows=[
            CourseFlow(
                course_id="c1",
                flow="당신의 관심사 → 백엔드 개발자 직무 → 빅데이터 + 모바일소프트웨어 트랙 → 이 과목이 기초입니다",
            ),
        ],
    )
    restored = Explanation.model_validate(original.model_dump(mode="json"))

    assert restored == original
    assert len(restored.semester_subtitles) == 2
    assert restored.semester_subtitles[1].semester == 2
    assert len(restored.course_flows) == 1
    assert restored.course_flows[0].course_id == "c1"
