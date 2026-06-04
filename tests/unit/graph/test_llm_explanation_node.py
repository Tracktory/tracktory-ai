"""LLMExplanationNode 의 계약을 검증한다.

LLMClient 를 MagicMock 으로 대체하여 네트워크 호출 없이 검증한다.

검증 케이스:
    1. 세 영역 모두 채워진 정상 케이스 — LLM 1 회 호출 + 결과 직렬화.
    2. 세 영역 모두 비어 있는 케이스 — LLM 호출 skip + 정적 안내.
    3. jobs 만 있고 tracks/roadmap 이 비어 있는 부분 케이스 — 프롬프트가
       빈 영역을 명시했는지 messages 내용으로 검증.
    4. fallback_used 가 섞인 케이스 — caveat_required="yes" 가 메시지에
       포함되는지 검증.
    5. 결정론적 직렬화 — 동일 state 두 번 호출 시 invoke 의 인자 동일.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tracktory.graph.models import (
    CourseFlow,
    Explanation,
    ExplanationSection,
    JobRationale,
    SemesterSubtitle,
    TrackRationale,
)
from tracktory.graph.nodes.llm_explanation import LLMClient, LLMExplanationNode


def _job(
    job_id: str = "be_dev_001",
    *,
    job_name: str = "백엔드 개발자",
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "job_name": job_name,
        "tech_stacks": ["Spring Boot", "MySQL"],
        "competency_tags": ["문제해결능력"],
        "match_score": 0.7,
        "similarity": 0.7,
        "fallback_used": fallback_used,
    }


def _ranked_combo(
    track_a_name: str = "빅데이터",
    track_b_name: str = "모바일소프트웨어",
    *,
    slot_type: str = "primary",
    rank: int = 1,
    synergy_score: float = 0.6,
) -> dict[str, Any]:
    def _track(name: str, track_id: str) -> dict[str, Any]:
        return {
            "college_id": "C1",
            "department_id": "D1",
            "major_id": "D1",
            "track_id": track_id,
            "track_name": name,
            "course_ids": [],
            "meta_text": "",
            "meta_vector": [],
            "competencies": [],
            "tech_stacks": [],
        }

    return {
        "combo": {
            "track_a": _track(track_a_name, "t_a"),
            "track_b": _track(track_b_name, "t_b"),
            "combo_key": "t_a::t_b",
        },
        "synergy_score": synergy_score,
        "slot_type": slot_type,
        "rank": rank,
    }


def _roadmap_dict() -> dict[str, Any]:
    return {
        "stages": [
            {
                "stage": "foundation",
                "courses": [
                    {
                        "course_id": "c1",
                        "course_name": "자료구조",
                        "credits": 3,
                        "stage": "foundation",
                        "score": 0.6,
                    },
                ],
            },
            {"stage": "core", "courses": []},
            {"stage": "application", "courses": []},
            {"stage": "industry", "courses": []},
        ],
    }


def _empty_roadmap_dict() -> dict[str, Any]:
    return {
        "stages": [
            {"stage": "foundation", "courses": []},
            {"stage": "core", "courses": []},
            {"stage": "application", "courses": []},
            {"stage": "industry", "courses": []},
        ],
    }


def _roadmap_with_semesters() -> dict[str, Any]:
    """단계별 뷰 + 학기 분산 뷰를 모두 갖춘 로드맵.

    1학기 → 자료구조(기초), 2학기 → 객체지향프로그래밍(핵심) 으로 학기별
    대표 단계가 달라 학기 부제 치환을 구분 검증할 수 있다.
    """
    return {
        "stages": [
            {
                "stage": "foundation",
                "courses": [
                    {
                        "course_id": "c1",
                        "course_name": "자료구조",
                        "credits": 3,
                        "stage": "foundation",
                        "score": 0.6,
                    }
                ],
            },
            {
                "stage": "core",
                "courses": [
                    {
                        "course_id": "c2",
                        "course_name": "객체지향프로그래밍",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.6,
                    }
                ],
            },
            {"stage": "application", "courses": []},
            {"stage": "industry", "courses": []},
        ],
        "semesters": [
            {
                "semester": 1,
                "grade": 1,
                "courses": [
                    {
                        "course_id": "c1",
                        "course_name": "자료구조",
                        "credits": 3,
                        "stage": "foundation",
                        "score": 0.6,
                    }
                ],
                "credits_total": 3,
                "cap_reached": False,
                "graduation_insufficient": False,
            },
            {
                "semester": 2,
                "grade": 1,
                "courses": [
                    {
                        "course_id": "c2",
                        "course_name": "객체지향프로그래밍",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.6,
                    }
                ],
                "credits_total": 3,
                "cap_reached": False,
                "graduation_insufficient": False,
            },
        ],
    }


def _messages_concat(call_args: Any) -> str:
    """invoke 호출에 전달된 메시지 리스트의 ``content`` 를 한 줄로 합친다."""
    messages = call_args.args[0] if call_args.args else call_args.kwargs.get("messages") or []
    return "\n".join(getattr(m, "content", "") for m in messages)


def test_explanation_invokes_llm_when_all_three_areas_present() -> None:
    """세 영역 모두 채워진 정상 case — LLM 1 회 호출 + explanation dict 반환."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(
        text="관심사 기반 추천을 정리했어요.",
        sections=[
            ExplanationSection(topic="jobs", body="백엔드 개발자가 가장 잘 맞아요."),
            ExplanationSection(topic="tracks", body="빅데이터+모바일소프트웨어 조합 추천."),
            ExplanationSection(topic="roadmap", body="자료구조부터 시작하시면 좋아요."),
        ],
    )
    node = LLMExplanationNode(llm_client=client)

    state: dict[str, Any] = {
        "recommended_jobs": [_job()],
        "primary_combos": [_ranked_combo()],
        "secondary_combos": [],
        "roadmap": _roadmap_dict(),
    }
    result = node(state)

    assert client.invoke.call_count == 1
    assert result["trace"] == ["llm_explanation:ok"]
    assert isinstance(result["explanation"], dict)
    assert result["explanation"]["text"]
    assert len(result["explanation"]["sections"]) == 3


def test_explanation_skips_llm_when_all_three_areas_empty() -> None:
    """세 영역 모두 비어 있으면 LLM 을 호출하지 않고 정적 안내를 반환한다."""
    client = MagicMock(spec=LLMClient)
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [],
            "primary_combos": [],
            "secondary_combos": [],
            "roadmap": _empty_roadmap_dict(),
        }
    )

    client.invoke.assert_not_called()
    assert result["trace"] == ["llm_explanation:empty"]
    assert result["explanation"]["text"]
    assert result["explanation"]["sections"] == []


def test_explanation_marks_empty_areas_in_prompt_when_partial() -> None:
    """jobs 만 있고 tracks/roadmap 이 비어 있으면 메시지에 빈 영역 안내 토큰이 흐른다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약", sections=[])
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [],
            "secondary_combos": [],
            "roadmap": None,
        }
    )

    assert client.invoke.call_count == 1
    rendered = _messages_concat(client.invoke.call_args)
    # 직무 영역은 실제 직무명이 노출되어야 한다.
    assert "백엔드 개발자" in rendered
    # tracks / roadmap 영역은 빈 영역 토큰이 노출되어야 한다.
    assert rendered.count("데이터 없음") >= 2


def test_explanation_passes_caveat_flag_when_fallback_used() -> None:
    """직무 후보에 fallback_used=True 가 있으면 caveat_required='yes' 가 흐른다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약", sections=[])
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [_job(fallback_used=True)],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": _roadmap_dict(),
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    assert "[Caveat required]" in rendered
    assert "yes" in rendered


def test_explanation_does_not_attach_caveat_when_no_fallback() -> None:
    """fallback_used=False 만 있으면 caveat_required='no' 로 흐른다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약", sections=[])
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [_job(fallback_used=False)],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": _roadmap_dict(),
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    # caveat 플래그가 "no" 로 흐르고 "yes" 는 흐르지 않는다.
    assert "[Caveat required]" in rendered
    assert "\nno\n" in rendered or rendered.rstrip().endswith("no")


def test_explanation_serialization_is_deterministic() -> None:
    """동일 state 두 번 호출 시 invoke 의 인자가 완전히 동일하다 (직렬화 결정론성)."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약", sections=[])
    node = LLMExplanationNode(llm_client=client)

    state: dict[str, Any] = {
        "recommended_jobs": [_job()],
        "primary_combos": [_ranked_combo()],
        "secondary_combos": [],
        "roadmap": _roadmap_dict(),
    }
    node(state)
    node(state)

    assert client.invoke.call_count == 2
    rendered_first = _messages_concat(client.invoke.call_args_list[0])
    rendered_second = _messages_concat(client.invoke.call_args_list[1])
    assert rendered_first == rendered_second


def test_prompt_substitutes_real_values_for_semester_and_course_outputs() -> None:
    """학기 부제·과목 인과 흐름 입력에 실제 직무명·트랙 조합·단계명·과목 식별자가 흐른다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약")
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": _roadmap_with_semesters(),
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    # 학기 부제 입력: 학기별 대표 단계명이 한국어 라벨로 치환되어 흐른다.
    assert "1학기(1학년): 단계명=기초" in rendered
    assert "2학기(1학년): 단계명=핵심" in rendered
    # 과목 인과 흐름 입력: course_id + 한국어 단계명이 흐른다.
    assert "course_id=c1" in rendered
    assert "course_id=c2" in rendered
    # 인과 흐름 anchor: 실제 직무명 + 트랙 조합이 흐른다 (placeholder 가 아님).
    assert "직무명=백엔드 개발자" in rendered
    assert "트랙 조합=빅데이터 + 모바일소프트웨어" in rendered


def test_node_passes_through_semester_subtitles_and_course_flows() -> None:
    """LLM 이 생성한 두 종류 출력이 explanation dict 로 그대로 흘러나온다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(
        text="요약",
        semester_subtitles=[
            SemesterSubtitle(semester=1, subtitle="이번 학기는 트랙 기초 단계입니다"),
            SemesterSubtitle(semester=2, subtitle="이번 학기는 트랙 핵심 단계입니다"),
        ],
        course_flows=[
            CourseFlow(
                course_id="c1",
                flow=(
                    "당신의 관심사 → 백엔드 개발자 직무 → 빅데이터 + 모바일소프트웨어 트랙"
                    " → 이 과목이 기초입니다"
                ),
            ),
        ],
    )
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": _roadmap_with_semesters(),
        }
    )

    explanation = result["explanation"]
    assert explanation["semester_subtitles"][0]["semester"] == 1
    assert "기초" in explanation["semester_subtitles"][0]["subtitle"]
    assert explanation["semester_subtitles"][1]["semester"] == 2
    assert explanation["course_flows"][0]["course_id"] == "c1"
    assert "백엔드 개발자" in explanation["course_flows"][0]["flow"]


def test_semester_subtitle_uses_dominant_stage_on_mixed_semester() -> None:
    """한 학기에 단계가 섞이면 최빈 단계가 대표 단계명으로 흐른다."""
    roadmap = {
        "stages": [
            {
                "stage": "foundation",
                "courses": [
                    {
                        "course_id": "c1",
                        "course_name": "자료구조",
                        "credits": 3,
                        "stage": "foundation",
                        "score": 0.6,
                    }
                ],
            },
            {
                "stage": "core",
                "courses": [
                    {
                        "course_id": "c2",
                        "course_name": "객체지향프로그래밍",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.6,
                    },
                    {
                        "course_id": "c3",
                        "course_name": "알고리즘",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.4,
                    },
                ],
            },
            {"stage": "application", "courses": []},
            {"stage": "industry", "courses": []},
        ],
        "semesters": [
            {
                "semester": 3,
                "grade": 2,
                "courses": [
                    {
                        "course_id": "c1",
                        "course_name": "자료구조",
                        "credits": 3,
                        "stage": "foundation",
                        "score": 0.6,
                    },
                    {
                        "course_id": "c2",
                        "course_name": "객체지향프로그래밍",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.6,
                    },
                    {
                        "course_id": "c3",
                        "course_name": "알고리즘",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.4,
                    },
                ],
                "credits_total": 9,
                "cap_reached": False,
                "graduation_insufficient": False,
            },
        ],
    }
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약")
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": roadmap,
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    # 기초 1 + 핵심 2 → 대표 단계 = 핵심.
    assert "3학기(2학년): 단계명=핵심" in rendered


def test_semester_subtitle_tie_break_prefers_earlier_stage() -> None:
    """단계 동률이면 더 이른 단계가 대표 단계명으로 선택된다 (기초 1 + 핵심 1 → 기초)."""
    roadmap = {
        "stages": [
            {
                "stage": "foundation",
                "courses": [
                    {
                        "course_id": "c1",
                        "course_name": "자료구조",
                        "credits": 3,
                        "stage": "foundation",
                        "score": 0.6,
                    }
                ],
            },
            {
                "stage": "core",
                "courses": [
                    {
                        "course_id": "c2",
                        "course_name": "객체지향프로그래밍",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.6,
                    }
                ],
            },
            {"stage": "application", "courses": []},
            {"stage": "industry", "courses": []},
        ],
        "semesters": [
            {
                "semester": 2,
                "grade": 1,
                "courses": [
                    {
                        "course_id": "c1",
                        "course_name": "자료구조",
                        "credits": 3,
                        "stage": "foundation",
                        "score": 0.6,
                    },
                    {
                        "course_id": "c2",
                        "course_name": "객체지향프로그래밍",
                        "credits": 3,
                        "stage": "core",
                        "score": 0.6,
                    },
                ],
                "credits_total": 6,
                "cap_reached": False,
                "graduation_insufficient": False,
            },
        ],
    }
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약")
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": roadmap,
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    assert "2학기(1학년): 단계명=기초" in rendered


def test_empty_roadmap_yields_empty_semester_and_course_context() -> None:
    """로드맵이 없으면 학기·과목 컨텍스트가 모두 빈 영역 토큰으로 흐른다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약")
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": None,
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    assert "[Semester stages]" in rendered
    assert "[Course stages]" in rendered
    # roadmap_context + semesters_context + courses_context 세 영역 모두 빈 토큰.
    assert rendered.count("데이터 없음") >= 3


@pytest.mark.parametrize(
    ("job_name", "track_a", "track_b"),
    [
        ("백엔드 개발자", "빅데이터", "모바일소프트웨어"),
        ("데이터 분석가", "빅데이터", "웹공학"),
        ("게임 클라이언트 개발자", "디지털콘텐츠·가상현실", "모바일소프트웨어"),
    ],
)
def test_substitution_robust_across_personas(job_name: str, track_a: str, track_b: str) -> None:
    """페르소나별 입력이 달라도 실제 값이 두 출력 입력 컨텍스트에 정확히 치환된다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약")
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [_job(job_name=job_name)],
            "primary_combos": [_ranked_combo(track_a, track_b)],
            "secondary_combos": [],
            "roadmap": _roadmap_with_semesters(),
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    assert f"직무명={job_name}" in rendered
    assert f"트랙 조합={track_a} + {track_b}" in rendered
    assert "course_id=c1" in rendered
    assert "단계명=기초" in rendered


def test_jobs_context_exposes_job_id_for_each_job() -> None:
    """추천 직무가 여러 건이면 각 직무의 job_id 가 컨텍스트에 노출돼 항목별 근거를 binding 할 수 있다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약")
    node = LLMExplanationNode(llm_client=client)

    node(
        {
            "recommended_jobs": [
                _job(job_id="be_dev_001", job_name="백엔드 개발자"),
                _job(job_id="data_001", job_name="데이터 분석가"),
            ],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": _roadmap_dict(),
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    assert "job_id=be_dev_001" in rendered
    assert "job_id=data_001" in rendered


def test_tracks_context_exposes_combo_key_and_per_track_tags() -> None:
    """각 조합의 combo_key + 두 트랙의 개별 역량/기술스택이 노출돼 조합/트랙 근거를 grounding 할 수 있다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약")
    node = LLMExplanationNode(llm_client=client)

    combo = _ranked_combo()
    combo["combo"]["track_a"]["competencies"] = ["데이터분석"]
    combo["combo"]["track_a"]["tech_stacks"] = ["Python"]
    combo["combo"]["track_b"]["competencies"] = ["앱개발"]
    combo["combo"]["track_b"]["tech_stacks"] = ["Kotlin"]

    node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [combo],
            "secondary_combos": [],
            "roadmap": _roadmap_dict(),
        }
    )

    rendered = _messages_concat(client.invoke.call_args)
    assert "combo_key=t_a::t_b" in rendered
    # 개별 트랙 근거 grounding 용으로 두 트랙의 역량·기술스택이 분리되어 흐른다.
    assert "데이터분석" in rendered
    assert "Python" in rendered
    assert "앱개발" in rendered
    assert "Kotlin" in rendered


def test_node_passes_through_job_and_track_rationales() -> None:
    """LLM 이 생성한 항목별 근거가 explanation dict 로 그대로 흘러나온다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(
        text="요약",
        job_rationales=[
            JobRationale(job_id="be_dev_001", rationale="백엔드 개발자가 잘 맞는 이유."),
            JobRationale(job_id="data_001", rationale="데이터 분석가가 잘 맞는 이유."),
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
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [
                _job(job_id="be_dev_001"),
                _job(job_id="data_001", job_name="데이터 분석가"),
            ],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [],
            "roadmap": _roadmap_dict(),
        }
    )

    explanation = result["explanation"]
    assert len(explanation["job_rationales"]) == 2
    assert explanation["job_rationales"][0]["job_id"] == "be_dev_001"
    # 각 직무가 서로 다른 근거 문구를 갖는다.
    assert (
        explanation["job_rationales"][0]["rationale"]
        != explanation["job_rationales"][1]["rationale"]
    )
    assert explanation["track_rationales"][0]["combo_key"] == "t_a::t_b"
    # 조합 전체 근거와 개별 트랙 근거가 구분된다.
    track = explanation["track_rationales"][0]
    assert track["combo_rationale"] != track["track_a_rationale"]
    assert track["track_a_rationale"] != track["track_b_rationale"]


def _combo_with_key(
    track_a_name: str,
    track_b_name: str,
    combo_key: str,
    *,
    slot_type: str = "mmr",
    rank: int = 3,
) -> dict[str, Any]:
    """combo_key 와 트랙명을 명시한 보조 조합 fixture."""
    combo = _ranked_combo(track_a_name, track_b_name, slot_type=slot_type, rank=rank)
    combo["combo"]["combo_key"] = combo_key
    combo["combo"]["track_a"]["track_id"] = combo_key + "_a"
    combo["combo"]["track_b"]["track_id"] = combo_key + "_b"
    return combo


def test_secondary_combos_get_rationales_when_llm_omits_them() -> None:
    """LLM 이 주 추천 조합 근거만 내도 보조 조합 전부가 combo_key 근거로 채워진다."""
    client = MagicMock(spec=LLMClient)
    # LLM 은 최상위(주 추천) 조합만 근거를 생성하고 보조 2개를 누락.
    client.invoke.return_value = Explanation(
        text="요약",
        track_rationales=[
            TrackRationale(
                combo_key="t_a::t_b",
                combo_rationale="주 추천 조합 시너지 근거.",
                track_a_rationale="주 1트랙 근거.",
                track_b_rationale="주 2트랙 근거.",
            ),
        ],
    )
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [
                _combo_with_key("웹공학", "한국어교육", "t_web::t_kor", rank=3),
                _combo_with_key("디지털콘텐츠·가상현실", "역사문화큐레이션", "t_dc::t_his", rank=4),
            ],
            "roadmap": _roadmap_dict(),
        }
    )

    rationales = result["explanation"]["track_rationales"]
    keys = {r["combo_key"] for r in rationales}
    # 주 추천 1 + 보조 2 = 세 조합 모두 combo_key 근거가 존재.
    assert keys == {"t_a::t_b", "t_web::t_kor", "t_dc::t_his"}
    # 두 보조 조합 fallback 의 조합 근거 문구가 서로 다르다 (동일 폴백 중복 방지).
    by_key = {r["combo_key"]: r for r in rationales}
    assert by_key["t_web::t_kor"]["combo_rationale"] != by_key["t_dc::t_his"]["combo_rationale"]
    # 합성된 조합 근거 안에서도 세 필드가 서로 구분된다.
    web = by_key["t_web::t_kor"]
    assert web["combo_rationale"] != web["track_a_rationale"]
    assert web["track_a_rationale"] != web["track_b_rationale"]


def test_llm_track_rationales_are_preserved_when_complete() -> None:
    """LLM 이 모든 조합 근거를 생성하면 fallback 으로 덮어쓰지 않고 그대로 보존된다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(
        text="요약",
        track_rationales=[
            TrackRationale(
                combo_key="t_a::t_b",
                combo_rationale="주 추천 조합 시너지 (LLM).",
                track_a_rationale="주 1트랙 (LLM).",
                track_b_rationale="주 2트랙 (LLM).",
            ),
            TrackRationale(
                combo_key="t_web::t_kor",
                combo_rationale="보조 조합 시너지 (LLM).",
                track_a_rationale="보조 1트랙 (LLM).",
                track_b_rationale="보조 2트랙 (LLM).",
            ),
        ],
    )
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [_combo_with_key("웹공학", "한국어교육", "t_web::t_kor")],
            "roadmap": _roadmap_dict(),
        }
    )

    rationales = result["explanation"]["track_rationales"]
    assert len(rationales) == 2
    by_key = {r["combo_key"]: r for r in rationales}
    # LLM 원문이 보존된다 (fallback 문구로 치환되지 않음).
    assert by_key["t_web::t_kor"]["combo_rationale"] == "보조 조합 시너지 (LLM)."


def test_fallback_track_rationale_passes_validation_with_empty_tags() -> None:
    """역량·기술스택이 비어 있어도 fallback 근거가 min_length 검증을 통과한다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(text="요약")
    node = LLMExplanationNode(llm_client=client)

    # _ranked_combo / _combo_with_key 의 트랙은 competencies/tech_stacks 가 빈 리스트.
    result = node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [_combo_with_key("웹공학", "한국어교육", "t_web::t_kor")],
            "roadmap": _roadmap_dict(),
        }
    )

    rationales = result["explanation"]["track_rationales"]
    assert {r["combo_key"] for r in rationales} == {"t_a::t_b", "t_web::t_kor"}
    for r in rationales:
        assert r["combo_rationale"]
        assert r["track_a_rationale"]
        assert r["track_b_rationale"]


def test_llm_generation_failure_still_returns_covered_explanation() -> None:
    """LLM 호출이 예외로 실패해도 추천 응답은 살아 있고 모든 조합 근거가 채워진다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.side_effect = RuntimeError("LLM down")
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [
                _combo_with_key("웹공학", "한국어교육", "t_web::t_kor", rank=3),
                _combo_with_key("디지털콘텐츠·가상현실", "역사문화큐레이션", "t_dc::t_his", rank=4),
            ],
            "roadmap": _roadmap_dict(),
        }
    )

    explanation = result["explanation"]
    # 응답 구조 자체는 유지된다 (예외 전파로 500 이 되지 않음).
    assert result["trace"] == ["llm_explanation:fallback"]
    assert explanation["text"]
    # 트랙 근거는 폴백으로 모든 조합 combo_key 가 채워진다.
    keys = {r["combo_key"] for r in explanation["track_rationales"]}
    assert keys == {"t_a::t_b", "t_web::t_kor", "t_dc::t_his"}


def test_hallucinated_and_duplicate_combo_keys_are_sanitized() -> None:
    """추천 조합에 없는 키는 버리고, 중복 키는 첫 건만 남겨 조합과 1:1 로 맞춘다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(
        text="요약",
        track_rationales=[
            TrackRationale(
                combo_key="t_a::t_b",
                combo_rationale="첫 번째 근거.",
                track_a_rationale="1트랙 근거.",
                track_b_rationale="2트랙 근거.",
            ),
            # 같은 키 중복 — 버려져야 한다.
            TrackRationale(
                combo_key="t_a::t_b",
                combo_rationale="중복 근거.",
                track_a_rationale="중복 1.",
                track_b_rationale="중복 2.",
            ),
            # 추천 조합에 없는 hallucinated 키 — 버려져야 한다.
            TrackRationale(
                combo_key="ghost::combo",
                combo_rationale="유령 근거.",
                track_a_rationale="유령 1.",
                track_b_rationale="유령 2.",
            ),
        ],
    )
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [_ranked_combo()],
            "secondary_combos": [_combo_with_key("웹공학", "한국어교육", "t_web::t_kor")],
            "roadmap": _roadmap_dict(),
        }
    )

    rationales = result["explanation"]["track_rationales"]
    # 추천 조합과 정확히 1:1 — 중복·유령 키 제거 후 누락 보조 조합 보강.
    assert [r["combo_key"] for r in rationales] == ["t_a::t_b", "t_web::t_kor"]
    by_key = {r["combo_key"]: r for r in rationales}
    # 중복 중 첫 건만 보존.
    assert by_key["t_a::t_b"]["combo_rationale"] == "첫 번째 근거."


def test_combo_key_middle_dot_drift_is_matched_and_corrected() -> None:
    """LLM 이 가운뎃점을 다른 변형으로 옮겨도 원본 키로 교정하고 LLM 문구를 보존한다."""
    # 추천 조합 키는 한글 아래아(U+318D ㆍ) 를 쓴다 (학사 원본 트랙명).
    auth_key = "디지털콘텐츠ㆍ가상현실트랙::역사문화큐레이션트랙"
    # LLM 은 일반 가운뎃점(U+00B7 ·) 으로 옮겼다 — byte 매칭은 빗나간다.
    drifted_key = "디지털콘텐츠·가상현실트랙::역사문화큐레이션트랙"
    assert auth_key != drifted_key

    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(
        text="요약",
        track_rationales=[
            TrackRationale(
                combo_key=drifted_key,
                combo_rationale="LLM 이 생성한 보조 조합 시너지.",
                track_a_rationale="LLM 1트랙.",
                track_b_rationale="LLM 2트랙.",
            ),
        ],
    )
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [
                _combo_with_key(
                    "디지털콘텐츠ㆍ가상현실트랙",
                    "역사문화큐레이션트랙",
                    auth_key,
                    slot_type="primary",
                    rank=1,
                )
            ],
            "secondary_combos": [],
            "roadmap": _roadmap_dict(),
        }
    )

    rationales = result["explanation"]["track_rationales"]
    assert len(rationales) == 1
    # 출력 키는 원본(U+318D) 으로 교정된다 — 백엔드 byte 매칭이 맞는다.
    assert rationales[0]["combo_key"] == auth_key
    # LLM 이 생성한 문구는 폐기되지 않고 보존된다 (fallback 으로 치환 X).
    assert rationales[0]["combo_rationale"] == "LLM 이 생성한 보조 조합 시너지."


def test_no_combos_clears_dangling_track_rationales() -> None:
    """추천 조합이 없으면 LLM 이 만든 트랙 근거는 바인딩 대상이 없어 비운다."""
    client = MagicMock(spec=LLMClient)
    client.invoke.return_value = Explanation(
        text="요약",
        track_rationales=[
            TrackRationale(
                combo_key="ghost::combo",
                combo_rationale="유령 근거.",
                track_a_rationale="유령 1.",
                track_b_rationale="유령 2.",
            ),
        ],
    )
    node = LLMExplanationNode(llm_client=client)

    result = node(
        {
            "recommended_jobs": [_job()],
            "primary_combos": [],
            "secondary_combos": [],
            "roadmap": _roadmap_dict(),
        }
    )

    assert result["explanation"]["track_rationales"] == []
