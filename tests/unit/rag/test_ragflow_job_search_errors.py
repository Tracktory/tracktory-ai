from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from tracktory.rag.ragflow_job_search import (
    RagflowConfig,
    RagflowJobSearchClient,
    RagflowSearchError,
)


class FakeSession:
    def __init__(self, outcomes: list[requests.Response | requests.RequestException]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def post(self, *_args: Any, **_kwargs: Any) -> requests.Response:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, requests.RequestException):
            raise outcome
        return outcome


def _response(
    status_code: int,
    body: Any,
    *,
    request_id: str = "req-123",
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.headers["X-Request-ID"] = request_id
    response._content = json.dumps(body).encode("utf-8")
    return response


def _invalid_json_response(status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = b"not-json"
    return response


def _client(
    tmp_path: Path,
    outcomes: list[requests.Response | requests.RequestException],
    *,
    max_retries: int,
) -> tuple[RagflowJobSearchClient, FakeSession]:
    category_map_path = tmp_path / "category_to_job_type.yaml"
    category_map_path.write_text("{}", encoding="utf-8")
    session = FakeSession(outcomes)
    client = RagflowJobSearchClient(
        RagflowConfig(
            base_url="https://ragflow.example",
            api_key="secret",
            dataset_id="dataset",
            max_retries=max_retries,
            timeout=0.1,
        ),
        session=session,
        category_map_path=category_map_path,
    )
    return client, session


def test_4xx_raises_structured_error_without_retry(tmp_path: Path) -> None:
    client, session = _client(
        tmp_path,
        [_response(401, {"code": 1001}, request_id="auth-failed")],
        max_retries=2,
    )

    with pytest.raises(RagflowSearchError) as exc_info:
        client._post_retrieval({"question": "backend"})

    error = exc_info.value
    assert error.reason == "http_4xx"
    assert error.status_code == 401
    assert error.code == 1001
    assert error.request_id == "auth-failed"
    assert session.calls == 1


def test_5xx_retry_exhaustion_preserves_last_response_context(tmp_path: Path) -> None:
    client, session = _client(
        tmp_path,
        [
            _response(503, {"code": "busy"}, request_id="first"),
            _response(502, {"code": "gateway"}, request_id="last"),
        ],
        max_retries=1,
    )

    with pytest.raises(RagflowSearchError) as exc_info:
        client._post_retrieval({"question": "backend"})

    error = exc_info.value
    assert error.reason == "http_5xx_retry_exhausted"
    assert error.status_code == 502
    assert error.code == "gateway"
    assert error.request_id == "last"
    assert session.calls == 2


def test_network_retry_exhaustion_preserves_exception_type(tmp_path: Path) -> None:
    client, session = _client(
        tmp_path,
        [requests.Timeout("slow"), requests.ConnectionError("closed")],
        max_retries=1,
    )

    with pytest.raises(RagflowSearchError) as exc_info:
        client._post_retrieval({"question": "backend"})

    error = exc_info.value
    assert error.reason == "network_retry_exhausted"
    assert error.error_type == "ConnectionError"
    assert session.calls == 2


def test_invalid_json_response_is_structured_error() -> None:
    with pytest.raises(RagflowSearchError) as exc_info:
        RagflowJobSearchClient._parse_response_body(_invalid_json_response())

    assert exc_info.value.reason == "invalid_json"


def test_ragflow_nonzero_code_is_structured_error() -> None:
    with pytest.raises(RagflowSearchError) as exc_info:
        RagflowJobSearchClient._parse_response_body(_response(200, {"code": 42}))

    error = exc_info.value
    assert error.reason == "ragflow_code_error"
    assert error.status_code == 200
    assert error.code == 42


def test_invalid_chunks_is_structured_error(tmp_path: Path) -> None:
    client, _session = _client(tmp_path, [], max_retries=0)

    with pytest.raises(RagflowSearchError) as exc_info:
        client._parse_chunks({"chunks": {"not": "a list"}})

    assert exc_info.value.reason == "invalid_chunks"
