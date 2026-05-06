"""직무 인덱스 추상화.

직무 매칭 노드는 본 모듈의 ``JobIndex`` Protocol 만 의존하며 구체 구현은
외부에서 주입받는다. 단위 테스트는 ``InMemoryJobIndex`` 로 검증하고,
production 은 RAGFlow 등 외부 인덱스 구현체로 교체한다.

단일 임베딩 boundary (ADR-0001) 준수:
    ``Job.job_vector`` 는 단일 ``embedding.yaml`` 모델로 사전 임베딩된
    L2 normalized 상태를 전제한다. 직무 매칭 노드의 cosine 계산이
    단순 내적으로 환원되도록 적재 시점에 normalize 가 보장되어야 한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Job(BaseModel):
    """직무 도메인 객체.

    Attributes:
        job_id: 직무 식별자.
        job_name: 사용자 표시용 직무명.
        job_vector: 단일 임베딩 모델로 사전 임베딩된 L2 normalized 직무 벡터.
            빈 리스트는 인덱스 적재 미완 상태를 의미하며, 매칭 단계에서
            cosine 0 으로 처리된다.
        tech_stacks: 채용공고 기술스택 합집합. 후속 트랙 시너지 계산의
            도달도 분모로 흐른다.
        competency_tags: 직무가 요구하는 역량 태그.
    """

    job_id: str = Field(..., min_length=1)
    job_name: str = Field(..., min_length=1)
    job_vector: list[float] = Field(default_factory=list)
    tech_stacks: list[str] = Field(default_factory=list)
    competency_tags: list[str] = Field(default_factory=list)


@runtime_checkable
class JobIndex(Protocol):
    """직무 인덱스 인터페이스.

    구현체는 외부 I/O (네트워크·파일·DB) 가 적재 시점에 한 번만 발생하도록
    보장하고, ``list_all`` 호출은 in-memory 조회로 환원되어야 한다.
    이는 직무 매칭 노드가 외부 I/O 진입점을 단 1곳으로 격리하기 위한 전제다.
    """

    def list_all(self) -> list[Job]:
        """전체 직무 목록을 반환한다."""
        ...


class InMemoryJobIndex:
    """리스트 한 번 보유 + ``list_all`` 그대로 반환하는 인메모리 구현.

    단위 테스트와 fixture 주입 경로에서 사용한다. 실 RAGFlow 연동은
    별도 구현체로 교체된다.
    """

    def __init__(self, jobs: list[Job]) -> None:
        self._jobs = list(jobs)

    def list_all(self) -> list[Job]:
        return list(self._jobs)
