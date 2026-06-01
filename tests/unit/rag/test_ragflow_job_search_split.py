"""RagflowJobSearchClient 의 tech_stacks/competency_tags 분리 로직 단위 테스트.

직무 타입 단위 집계 시 tech_stacks 는 카테고리 집계(대표 스택)를 쓰고,
competency_tags 는 그룹 공고들이 실제 언급한 기술(누적) 중 집계에 없는 것만
담는지 검증한다. HTTP 호출 없이 ``_chunk_to_hit`` + ``_aggregate_by_job_type``
만 직접 호출한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from tracktory.rag.ragflow_job_search import (
    RagflowConfig,
    RagflowJobSearchClient,
    _aggregate_by_job_type,
    _PostingHit,
)

_CATEGORY_YAML = """\
"AI/ML":
  job_id: AI
  job_name: "AI/ML 엔지니어"
  tech_stacks: [Python, PyTorch, LLM]
"""


def _client(tmp_path: Path) -> RagflowJobSearchClient:
    path = tmp_path / "category_to_job_type.yaml"
    path.write_text(_CATEGORY_YAML, encoding="utf-8")
    return RagflowJobSearchClient(
        RagflowConfig(
            base_url="https://ragflow.example",
            api_key="secret",
            dataset_id="dataset",
        ),
        session=None,
        category_map_path=path,
    )


def _chunk(category: str, tech_stack: list[str], *, similarity: float = 0.8) -> dict:
    return {
        "content": "직무: AI 엔지니어\n카테고리: AI/ML",
        "similarity": similarity,
        "document_metadata": {"category": category, "tech_stack": json.dumps(tech_stack)},
    }


def _hit(tech: list[str], *, score: float = 0.8) -> _PostingHit:
    return _PostingHit(
        job_id="AI",
        job_name="AI/ML 엔지니어",
        score=score,
        description="d",
        tech_stacks=tech,
        aggregate=("Python", "PyTorch", "LLM"),
    )


def test_chunk_to_hit_carries_category_aggregate(tmp_path: Path) -> None:
    client = _client(tmp_path)

    hit = client._chunk_to_hit(_chunk("AI/ML", ["Python", "Rust"]))

    assert hit is not None
    assert hit.aggregate == ("Python", "PyTorch", "LLM")
    assert "Rust" in hit.tech_stacks


def test_unmapped_category_is_skipped(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client._chunk_to_hit(_chunk("블록체인", ["Solidity"])) is None


def test_tech_stacks_uses_category_aggregate() -> None:
    results = _aggregate_by_job_type([_hit(["Python", "PyTorch"])])

    assert results[0].tech_stacks == ["Python", "PyTorch", "LLM"]


def test_competency_tags_are_accumulated_posting_tech_not_in_aggregate() -> None:
    # 두 공고가 그룹으로 묶임: 누적 기술 Python·Rust·Go·PyTorch.
    results = _aggregate_by_job_type([_hit(["Python", "Rust"]), _hit(["Go", "PyTorch"])])

    assert len(results) == 1
    # 집계(Python·PyTorch·LLM) 중복 제거 → Rust·Go 만 역량 태그.
    assert set(results[0].competency_tags) == {"Rust", "Go"}
    assert results[0].posting_count == 2


def test_competency_tags_empty_when_subset_of_aggregate() -> None:
    results = _aggregate_by_job_type([_hit(["Python", "LLM"])])

    assert results[0].competency_tags == []
