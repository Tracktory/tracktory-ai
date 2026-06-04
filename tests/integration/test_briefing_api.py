"""직무 브리핑 엔드포인트 통합 테스트.

``POST /api/v1/ai/briefing`` 가 내부 인증을 통과한 요청에 대해 큐레이션 서비스를
주입받아 직무에 연결된 브리핑 카드를 표준 envelope 으로 반환하는 계약을 검증한다.
브리핑 서비스는 ``dependency_overrides`` 로 합성 카탈로그 인스턴스를 주입한다.

``@pytest.mark.integration`` 으로 표시 — CI 기본 실행에서 제외하려면
``pytest -m "not integration"`` 을 사용한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tracktory.api.dependencies import get_briefing_service
from tracktory.api.main import app
from tracktory.briefing.service import BriefingService
from tracktory.common.config import settings as app_settings

pytestmark = pytest.mark.integration

# X-Request-Id 는 미들웨어가 강제하는 필수 헤더(누락 시 400)라 모든 요청에 동봉한다.
_BRIEFING_PATH = "/api/v1/ai/briefing"
_TEST_TOKEN = "test-internal-token"
_AUTH_HEADERS = {
    "X-Internal-Token": _TEST_TOKEN,
    "X-User-Id": "u-1",
    "X-Request-Id": "req-1",
}

_CATALOG = """\
briefings:
  BE:
    job_name: 백엔드 개발자
    cards:
      - headline: 백엔드 트렌드
        summary: 요약
        skills: [Spring Boot]
        sources:
          - { title: 출처 A, url: https://example.com/a, published_at: "2024" }
  FE:
    job_name: 프론트엔드 개발자
    cards:
      - headline: 프론트 트렌드
        summary: 요약
        sources:
          - { title: 출처 B, url: https://example.com/b }
"""


@pytest.fixture(autouse=True)
def _reset_dependency_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _set_internal_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_settings, "ai_internal_token", _TEST_TOKEN)


def _override_service(tmp_path: Path) -> None:
    path = tmp_path / "job_briefings.yaml"
    path.write_text(_CATALOG, encoding="utf-8")
    service = BriefingService(catalog_path=path)
    app.dependency_overrides[get_briefing_service] = lambda: service


def test_briefing_happy_path_returns_linked_cards(tmp_path: Path) -> None:
    """유효 입력 + 합성 카탈로그로 200 + 직무에 연결된 카드 묶음을 반환한다."""
    _override_service(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            _BRIEFING_PATH, json={"job_ids": ["BE", "FE"]}, headers=_AUTH_HEADERS
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    cards = body["data"]["briefings"]
    assert [card["job_id"] for card in cards] == ["BE", "FE"]
    # 각 카드는 추천 직무에 연결되고 검증 가능한 출처를 동반한다.
    assert all(card["sources"] for card in cards)


def test_briefing_skips_uncurated_jobs(tmp_path: Path) -> None:
    """큐레이션 없는 직무는 제외되고, 매칭이 없으면 빈 리스트를 200 으로 반환한다."""
    _override_service(tmp_path)

    with TestClient(app) as client:
        partial = client.post(
            _BRIEFING_PATH, json={"job_ids": ["BE", "ZZZ"]}, headers=_AUTH_HEADERS
        )
        none = client.post(_BRIEFING_PATH, json={"job_ids": ["ZZZ"]}, headers=_AUTH_HEADERS)

    assert [c["job_id"] for c in partial.json()["data"]["briefings"]] == ["BE"]
    assert none.status_code == 200
    assert none.json()["data"]["briefings"] == []


def test_briefing_empty_job_ids_returns_422(tmp_path: Path) -> None:
    """빈 job_ids 는 요청 검증 단계에서 거부되어 표준 에러 envelope 을 반환한다."""
    _override_service(tmp_path)

    with TestClient(app) as client:
        response = client.post(_BRIEFING_PATH, json={"job_ids": []}, headers=_AUTH_HEADERS)

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_FAILED"


def test_briefing_blank_job_id_item_returns_422(tmp_path: Path) -> None:
    """공백뿐인 직무 코드는 요청 단계에서 거부된다(조용한 빈 결과로 위장 차단)."""
    _override_service(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            _BRIEFING_PATH, json={"job_ids": ["BE", "  "]}, headers=_AUTH_HEADERS
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"


def test_briefing_requires_internal_token(tmp_path: Path) -> None:
    """내부 토큰 없는 호출은 403 으로 거부된다."""
    _override_service(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            _BRIEFING_PATH,
            json={"job_ids": ["BE"]},
            headers={"X-User-Id": "u-1", "X-Request-Id": "req-1"},
        )

    assert response.status_code == 403
    assert response.json()["success"] is False
