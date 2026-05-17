"""RAGFlow SDK 기반 챗봇 RAG 검색"""

from __future__ import annotations

from typing import Any, TypedDict

from tracktory.common.config import settings


class RetrievedChunk(TypedDict):
    """RAGFlow Chunk 에서 챗봇이 쓰는 필드만 추린 dict 스키마"""

    content: str
    document_name: str
    similarity: float
    id: str


class RagSearchError(Exception):
    """RAG 검색 실패 — SDK 예외를 wrap 해 노드 단일 catch"""


class RagFlowChatbotRetriever:
    """RAGFlow SDK 어댑터 — retrieve() 호출 + Chunk → RetrievedChunk 변환

    settings 에서 인증·URL·dataset_id 읽어옴
    실패 시 RagSearchError 로 wrap 해 raise
    """

    def __init__(self) -> None:
        # lazy import: ragflow_sdk 미설치 환경에서도 모듈 자체는 import 가능
        from ragflow_sdk import RAGFlow

        self._rag = RAGFlow(
            api_key=settings.ragflow_api_key,
            base_url=settings.ragflow_base_url,
        )

    def search(
        self,
        query: str,
    ) -> list[RetrievedChunk]:
        """RAGFlow 검색 — 현재 metadata_condition 미사용 (운영자 doc_type 메타 적재 대기 중)"""
        kwargs: dict[str, Any] = {
            "question": query,
            "dataset_ids": [settings.ragflow_dataset_id],
            "page": 1,
            "page_size": 10,
            "similarity_threshold": 0.2,
            "vector_similarity_weight": 0.3,
            "top_k": 32,  # rerank 전 후보 풀 크기
            "keyword": True,
            "use_kg": False,
        }
        if settings.ragflow_reranker_id:
            kwargs["rerank_id"] = settings.ragflow_reranker_id

        try:
            chunks = self._rag.retrieve(**kwargs)
        except Exception as exc:
            raise RagSearchError(str(exc)) from exc

        return [
            {
                "content": c.content,
                "document_name": c.document_name,
                "similarity": c.similarity,
                "id": c.id,
            }
            for c in chunks
        ]
