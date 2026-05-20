"""직무 검색 boundary 추상화.

직무 매칭 노드는 본 모듈의 ``JobSearchClient`` Protocol 만 의존하며 구체
구현은 외부 검색 (RAGFlow) 어댑터로 분리된다. 자체 코드는 자연어 질의
텍스트만 boundary 너머로 넘기고, 임베딩·검색·재정렬은 외부 검색이 일괄
처리한다. ADR-0001 의 단일 임베딩 boundary 원칙을 호출 단위에서 호출
대상으로 흡수시키는 형태이다.

단위 테스트는 ``rag_search_jobs`` 를 반환값 리스트로 대체하는 fake 구현체로
검증한다. 운영용 RAGFlow wrap 구현체는 별도 모듈로 분리한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RagSearchResult(BaseModel):
    """외부 직무 검색 boundary 의 반환 단건.

    boundary 너머에서 직무 KB 검색 + hybrid retrieval + reranker 결합을 마친
    뒤 자체 코드로 넘어오는 형태다. ``score`` 는 검색 시스템의 결합 점수로
    ``[0, 1]`` 범위 단조 정렬을 보장한다.

    Attributes:
        job_id: 직무 식별자.
        job_name: 사용자 표시용 직무명.
        score: 검색 시스템의 hybrid + reranker 결합 점수 ``[0, 1]``.
            상위 1 개의 점수가 ``min_job_similarity`` 임계값 미만이면 직무
            매칭 노드는 카테고리 사전 매핑 fallback 으로 전환한다.
        description: 직무 설명. boundary 의 청크 텍스트로 dynamic 길이.
        tech_stacks: 채용공고 기술스택. 후속 트랙 시너지 계산의 직무
            도달도 분모로 흐른다.
        competency_tags: 직무가 요구하는 역량 태그. 후속 트랙 시너지의
            상호 보완성 계산에 사용한다.
    """

    job_id: str = Field(..., min_length=1)
    job_name: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    description: str = Field(..., min_length=1, max_length=10_000)
    tech_stacks: list[str] = Field(default_factory=list)
    competency_tags: list[str] = Field(default_factory=list)


@runtime_checkable
class JobSearchClient(Protocol):
    """직무 검색 boundary 인터페이스.

    자체 코드는 본 Protocol 만 의존하고 구체 구현은 인프라 레이어에서
    주입된다. 호출 단위는 자연어 질의 텍스트와 후보 수이며, 임베딩·검색·
    재정렬은 모두 boundary 안에서 끝난다.

    구현체 contract:
        1. 외부 검색 실패 (네트워크 오류 / 타임아웃 / HTTP 4xx·5xx / 응답
           파싱 실패) 는 모두 ``RagSearchError`` 로 변환하여 raise. raw
           예외 (httpx.HTTPError, ValueError 등) 전파 금지 — 직무 매칭
           노드의 fallback 분기가 작동하지 않는다.
        2. 외부 호출 타임아웃 ≤ 5 초, retry ≤ 2 회 권장. 사용자 요청 SLO
           (응답 생성 < 30 초) 를 보장한다.
        3. ``RagSearchError`` 메시지에 자격증명·API 키·요청 본문 전체를
           포함하지 않는다 (status code / request_id 정도만).
    """

    def rag_search_jobs(self, query: str, top_k: int = 3) -> list[RagSearchResult]:
        """자연어 질의에 대한 직무 KB 검색 결과 상위 ``top_k`` 건을 반환한다.

        실패 시 ``RagSearchError`` 를 raise 한다. 직무 매칭 노드는 이 예외를
        catch 하여 카테고리 사전 매핑 fallback 분기로 전환한다.
        """
        ...


class RagSearchError(Exception):
    """외부 직무 검색 호출이 실패했을 때 raise 한다.

    호출 측은 본 예외를 catch 하여 fallback 분기로 전환한다. 메시지 문자열은
    구현체에서 디버깅 컨텍스트 (status code / request_id 등) 를 실어 보내되,
    개인정보·자격증명·요청 본문 전체는 포함하지 않는다.
    """
