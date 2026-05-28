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

from tracktory.graph.models import Explanation, ExplanationSection
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
                    {"course_id": "c1", "course_name": "자료구조", "priority": 1},
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
    assert "캐비잇 부착 여부" in rendered
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
    assert "캐비잇 부착 여부" in rendered
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
