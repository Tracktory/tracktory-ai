"""``InMemoryJobIndex`` 동작 단위 테스트.

실 구현체는 RAGFlow 등 외부 인덱스로 교체되므로 본 테스트는 인메모리 구현의
계약 (``list_all`` 이 외부 변경에 격리된 복사본을 반환) 만 검증한다.
"""

from __future__ import annotations

from tracktory.rag.job_index import InMemoryJobIndex, Job


def _make_job(job_id: str) -> Job:
    return Job(
        job_id=job_id,
        job_name=job_id,
        job_vector=[1.0, 0.0],
        tech_stacks=[f"{job_id}_stack"],
    )


def test_list_all_returns_empty_when_index_is_empty() -> None:
    index = InMemoryJobIndex(jobs=[])
    assert index.list_all() == []


def test_list_all_returns_all_jobs_in_order() -> None:
    jobs = [_make_job("a"), _make_job("b"), _make_job("c")]
    index = InMemoryJobIndex(jobs=jobs)
    result = index.list_all()
    assert [j.job_id for j in result] == ["a", "b", "c"]


def test_list_all_returns_copy_so_external_mutation_does_not_leak() -> None:
    """반환 리스트의 변경이 인덱스 내부 상태를 오염시키지 않아야 한다."""
    jobs = [_make_job("a"), _make_job("b")]
    index = InMemoryJobIndex(jobs=jobs)
    snapshot = index.list_all()
    snapshot.append(_make_job("c"))
    # 원본 인덱스에는 'c' 가 추가되지 않아야 한다
    assert [j.job_id for j in index.list_all()] == ["a", "b"]


def test_constructor_does_not_alias_input_list() -> None:
    """생성자에 전달한 리스트가 이후 mutate 되어도 인덱스는 영향 X."""
    jobs = [_make_job("a")]
    index = InMemoryJobIndex(jobs=jobs)
    jobs.append(_make_job("b"))
    assert [j.job_id for j in index.list_all()] == ["a"]
