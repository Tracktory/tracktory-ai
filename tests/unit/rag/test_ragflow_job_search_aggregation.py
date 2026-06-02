"""``RagflowJobSearchClient`` 의 직무 타입 단위 집계 동작 테스트.

RAGFlow 는 공고 단위로 청크를 내려주므로, 같은 직무 타입(카테고리 → 직무
카탈로그 표준 코드)의 공고들이 한 건으로 dedup 되고 posting_count 로 출현
횟수가 보존되는지 검증한다. tech_stacks 는 카테고리 집계(대표 스택)를 쓰고,
competency_tags 는 그룹 공고들이 실제 언급한 기술(빈도 누적) 중 집계에 없는
것만 담는다. 외부 HTTP 는 ``FakeSession`` 으로 통제해 네트워크 없이 retrieval
응답을 주입한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from tracktory.rag.ragflow_job_search import RagflowConfig, RagflowJobSearchClient

_CATEGORY_MAP_YAML = """
백엔드: { job_id: BE, job_name: 백엔드 개발자, tech_stacks: [Java] }
프론트엔드: { job_id: FE, job_name: 프론트엔드 개발자, tech_stacks: [React] }
"AI/ML": { job_id: AI, job_name: "AI/ML 엔지니어", tech_stacks: [PyTorch] }
"""


class _FakeSession:
    """``POST`` 호출에 대해 고정된 retrieval 응답을 돌려주는 세션."""

    def __init__(self, response: requests.Response) -> None:
        self._response = response
        self.calls = 0

    def post(self, *_args: Any, **_kwargs: Any) -> requests.Response:
        self.calls += 1
        return self._response


def _chunk(category: str, similarity: float, tech_stack: list[str], content: str) -> dict[str, Any]:
    return {
        "content": content,
        "similarity": similarity,
        "document_metadata": {"category": category, "tech_stack": json.dumps(tech_stack)},
    }


def _retrieval_response(chunks: list[dict[str, Any]]) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"code": 0, "data": {"chunks": chunks}}).encode("utf-8")
    return response


def _client(tmp_path: Path, chunks: list[dict[str, Any]]) -> RagflowJobSearchClient:
    category_map_path = tmp_path / "category_to_job_type.yaml"
    category_map_path.write_text(_CATEGORY_MAP_YAML, encoding="utf-8")
    session = _FakeSession(_retrieval_response(chunks))
    return RagflowJobSearchClient(
        RagflowConfig(
            base_url="https://ragflow.example",
            api_key="secret",
            dataset_id="dataset",
            max_retries=0,
            timeout=0.1,
        ),
        session=session,  # type: ignore[arg-type]
        category_map_path=category_map_path,
    )


def test_same_job_type_postings_are_deduped(tmp_path: Path) -> None:
    """같은 직무 타입 공고 3건 → 직무 카드 1장으로 dedup."""
    client = _client(
        tmp_path,
        [
            _chunk("백엔드", 0.5, ["Java"], "공고1"),
            _chunk("백엔드", 0.4, ["Python"], "공고2"),
            _chunk("백엔드", 0.3, ["Java"], "공고3"),
        ],
    )

    results = client.rag_search_jobs("백엔드 개발", top_k=3)

    assert len(results) == 1
    assert results[0].job_id == "BE"
    assert results[0].posting_count == 3


def test_score_is_group_max(tmp_path: Path) -> None:
    """직무 타입 score 는 그룹 내 최댓값."""
    client = _client(
        tmp_path,
        [
            _chunk("백엔드", 0.42, ["Java"], "공고1"),
            _chunk("백엔드", 0.71, ["Python"], "공고2"),
        ],
    )

    results = client.rag_search_jobs("백엔드", top_k=3)

    assert results[0].score == 0.71


def test_tech_stacks_is_aggregate_and_competency_is_remainder(tmp_path: Path) -> None:
    """tech_stacks 는 카테고리 집계, competency_tags 는 누적 기술 중 집계 제외분."""
    client = _client(
        tmp_path,
        [
            _chunk("백엔드", 0.5, ["Java", "AWS"], "공고1"),
            _chunk("백엔드", 0.4, ["Java", "Spring"], "공고2"),
            _chunk("백엔드", 0.3, ["Java"], "공고3"),
        ],
    )

    results = client.rag_search_jobs("백엔드", top_k=3)

    # tech_stacks = 백엔드 카테고리 집계(대표 스택).
    assert results[0].tech_stacks == ["Java"]
    # 누적 기술 Java(3)·AWS(1)·Spring(1) 중 집계(Java) 제외 → 빈도·등장순(AWS 먼저).
    assert results[0].competency_tags == ["AWS", "Spring"]


def test_top_k_counts_job_types_not_postings(tmp_path: Path) -> None:
    """top_k 는 공고 수가 아니라 서로 다른 직무 카드 수다."""
    client = _client(
        tmp_path,
        [
            _chunk("백엔드", 0.6, ["Java"], "be1"),
            _chunk("백엔드", 0.5, ["Python"], "be2"),
            _chunk("프론트엔드", 0.55, ["React"], "fe1"),
            _chunk("AI/ML", 0.45, ["PyTorch"], "ai1"),
        ],
    )

    results = client.rag_search_jobs("개발", top_k=2)

    # 3개 직무 타입 중 score 상위 2개 (BE 0.6, FE 0.55).
    assert [r.job_id for r in results] == ["BE", "FE"]
    assert all(r.posting_count >= 1 for r in results)


def test_unmapped_category_chunks_are_skipped(tmp_path: Path) -> None:
    """매핑에 없는 카테고리(비-IT) 공고는 집계에서 제외된다."""
    client = _client(
        tmp_path,
        [
            _chunk("백엔드", 0.5, ["Java"], "be"),
            _chunk("마케팅", 0.9, [], "non-it"),
        ],
    )

    results = client.rag_search_jobs("개발", top_k=3)

    assert [r.job_id for r in results] == ["BE"]
