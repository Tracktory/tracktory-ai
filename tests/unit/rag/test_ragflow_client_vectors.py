"""RagflowClient.fetch_chunk_vectors 페이지네이션·파싱 테스트 (네트워크 없음)."""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from tracktory.rag.ragflow_client import RagflowClient, RagflowConfig, RagflowError


class FakeSession:
    def __init__(self, responses: list[requests.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"url": url, **kwargs})
        return self._responses.pop(0)


def _response(body: Any, status_code: int = 200) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = json.dumps(body).encode("utf-8")
    return resp


def _client(session: FakeSession) -> RagflowClient:
    return RagflowClient(
        RagflowConfig(
            base_url="https://ragflow.example",
            api_key="secret",
            dataset_id="default-ds",
            max_retries=0,
            timeout=0.1,
        ),
        session=session,
    )


def _page(chunks: list[dict[str, Any]], total: int) -> dict[str, Any]:
    return {"code": 0, "data": {"total": total, "dimension": 2, "chunks": chunks}}


def test_fetch_chunk_vectors_paginates_until_total() -> None:
    session = FakeSession(
        [
            _response(_page([{"doc_name": "a.txt", "vector": [0.1, 0.2]}], total=2)),
            _response(_page([{"doc_name": "b.txt", "vector": [0.3, 0.4]}], total=2)),
        ]
    )

    chunks = _client(session).fetch_chunk_vectors(dataset_id="track-kb", page_size=1)

    assert [c["doc_name"] for c in chunks] == ["a.txt", "b.txt"]
    # 명시한 dataset_id 가 경로에 반영되고 두 페이지를 순회해야 한다.
    assert session.calls[0]["url"].endswith("/datasets/track-kb/chunks/vectors")
    assert len(session.calls) == 2


def test_fetch_chunk_vectors_defaults_to_config_dataset() -> None:
    session = FakeSession(
        [_response(_page([{"doc_name": "a.txt", "vector": [0.1, 0.2]}], total=1))]
    )

    _client(session).fetch_chunk_vectors()

    assert session.calls[0]["url"].endswith("/datasets/default-ds/chunks/vectors")


def test_fetch_chunk_vectors_invalid_payload_raises() -> None:
    session = FakeSession([_response({"code": 0, "data": {"total": 1, "chunks": "nope"}})])

    with pytest.raises(RagflowError):
        _client(session).fetch_chunk_vectors()
