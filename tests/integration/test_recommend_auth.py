"""추천 엔드포인트 내부 인증 통합 테스트.

``POST /api/v1/ai/recommend`` 의 내부 호출 게이트를 검증한다:
- ``X-Internal-Token`` 부재/불일치/서버 미설정 → 403
- 유효 토큰 + ``X-User-Id`` → 200, 사용자 식별자가 그래프 state 로 전달
- ``X-User-Id`` 부재 → 400 (필수 헤더 검증 실패의 표준 envelope)

그래프 파이프라인 의존성은 fake 로 override 하여 인증 경계만 격리 검증한다.

``@pytest.mark.integration`` 으로 표시 — ``pytest -m "not integration"`` 으로 제외 가능.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tracktory.api.dependencies import get_recommendation_pipeline
from tracktory.api.main import app
from tracktory.common.config import settings as app_settings

pytestmark = pytest.mark.integration


_RECOMMEND_PATH = "/api/v1/ai/recommend"
_TEST_TOKEN = "test-internal-token"


def _valid_payload() -> dict[str, Any]:
    return {
        "admission_year": 2025,
        "college": "C1",
        "department": "컴퓨터공학부",
        "current_tracks": [],
        "interests": ["IT/인터넷"],
        "dev_interests": ["AI"],
        "work_values": ["성장성"],
        "company_types": ["대기업"],
        "ncs_studied": [],
        "completed_courses": [],
    }


class _FakeGraph:
    """전달된 state 를 기록하고 완결된 final_state 를 돌려주는 그래프 stub."""

    def __init__(self) -> None:
        self.received_state: dict[str, Any] | None = None

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.received_state = state
        return {
            "recommended_jobs": [],
            "primary_combos": [],
            "secondary_combos": [],
            "roadmap": {
                "stages": [
                    {"stage": "foundation", "courses": []},
                    {"stage": "core", "courses": []},
                    {"stage": "application", "courses": []},
                    {"stage": "industry", "courses": []},
                ],
                "semesters": [],
            },
            "explanation": {"text": "설명", "sections": []},
            "errors": [],
        }


@pytest.fixture(autouse=True)
def _reset_dependency_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _set_internal_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_settings, "ai_internal_token", _TEST_TOKEN)


@pytest.fixture
def fake_graph() -> _FakeGraph:
    graph = _FakeGraph()
    app.dependency_overrides[get_recommendation_pipeline] = lambda: graph
    return graph


def test_valid_token_and_user_id_returns_200_and_propagates_user(fake_graph: _FakeGraph) -> None:
    with TestClient(app) as client:
        response = client.post(
            _RECOMMEND_PATH,
            json=_valid_payload(),
            headers={"X-Internal-Token": _TEST_TOKEN, "X-User-Id": "u-42"},
        )

    assert response.status_code == 200
    assert response.json()["is_success"] is True
    assert fake_graph.received_state is not None
    assert fake_graph.received_state["user_id"] == "u-42"


def test_missing_internal_token_returns_403(fake_graph: _FakeGraph) -> None:
    with TestClient(app) as client:
        response = client.post(
            _RECOMMEND_PATH,
            json=_valid_payload(),
            headers={"X-User-Id": "u-42"},
        )

    assert response.status_code == 403
    assert response.json()["is_success"] is False
    assert fake_graph.received_state is None


def test_wrong_internal_token_returns_403(fake_graph: _FakeGraph) -> None:
    with TestClient(app) as client:
        response = client.post(
            _RECOMMEND_PATH,
            json=_valid_payload(),
            headers={"X-Internal-Token": "wrong-token", "X-User-Id": "u-42"},
        )

    assert response.status_code == 403
    assert fake_graph.received_state is None


def test_server_token_unset_rejects_even_with_header(
    fake_graph: _FakeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """서버에 토큰이 설정되지 않으면 deny-by-default 로 403."""
    monkeypatch.setattr(app_settings, "ai_internal_token", "")

    with TestClient(app) as client:
        response = client.post(
            _RECOMMEND_PATH,
            json=_valid_payload(),
            headers={"X-Internal-Token": _TEST_TOKEN, "X-User-Id": "u-42"},
        )

    assert response.status_code == 403
    assert fake_graph.received_state is None


def test_missing_user_id_returns_400(fake_graph: _FakeGraph) -> None:
    """유효 토큰이라도 사용자 식별 헤더가 없으면 표준 400 envelope."""
    with TestClient(app) as client:
        response = client.post(
            _RECOMMEND_PATH,
            json=_valid_payload(),
            headers={"X-Internal-Token": _TEST_TOKEN},
        )

    assert response.status_code == 400
    assert response.json()["is_success"] is False
