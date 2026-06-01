"""RAGFlow retrieval HTTP 호출 범용 코어.

``POST /api/v1/retrieval`` 호출·재시도·에러 변환·payload 조립·응답 파싱이라는
RAGFlow 접근의 공통 골격을 담는다. 직무 검색 / 트랙 / 과목 등 도메인별 어댑터는
본 모듈의 ``RagflowClient`` 를 상속하여 도메인 전용 청크 변환만 특수화한다.

본 모듈은 도메인 의미 (직무·트랙·과목) 를 모른다. 원시 청크 (본문 +
``document_metadata`` + 점수) 까지만 책임지며, 청크를 도메인 모델로 변환하는
일은 하위 어댑터 책임이다. 따라서 새 RAGFlow 데이터셋의 실측 (probe) 도 본
모듈의 ``retrieve`` 한 번으로 가능하다.

자격증명은 코드에 하드코딩하지 않고 ``RagflowConfig.from_env`` 로 환경변수에서
읽는다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

_RETRIEVAL_PATH = "/api/v1/retrieval"
_DOCUMENTS_PATH_TMPL = "/api/v1/datasets/{dataset_id}/documents"
_CHUNKS_PATH_TMPL = "/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks"
_DATASET_VECTORS_PATH_TMPL = "/api/v1/datasets/{dataset_id}/chunks/vectors"


def _env_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RagflowError(Exception):
    """RAGFlow retrieval 호출 실패의 범용 예외.

    문자열에는 status/code/request_id 정도의 안전한 컨텍스트만 담고, 자격증명·
    요청 본문 전체는 포함하지 않는다. 하위 도메인 어댑터는 본 예외를 상속해
    자신의 boundary contract 에 맞는 예외 타입으로 좁힌다 (예: 직무 검색의
    ``RagSearchError`` 계약).
    """

    reason: str = "response_error"
    status_code: int | None = None
    code: Any | None = None
    request_id: str | None = None
    error_type: str | None = None

    @classmethod
    def from_response(
        cls,
        resp: requests.Response,
        body: dict[str, Any] | None = None,
        reason: str = "response_error",
    ) -> RagflowError:
        if body is None:
            try:
                parsed = resp.json()
            except ValueError:
                parsed = None
            body = parsed if isinstance(parsed, dict) else None

        return cls(
            reason=reason,
            status_code=resp.status_code,
            code=body.get("code") if body else None,
            request_id=cls._extract_request_id(resp),
        )

    def __str__(self) -> str:
        parts = [self.reason]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.code is not None:
            parts.append(f"code={self.code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        if self.error_type:
            parts.append(f"error_type={self.error_type}")
        return f"RAGFlow retrieval 실패 ({', '.join(parts)})"

    @staticmethod
    def _extract_request_id(resp: requests.Response) -> str | None:
        for header in ("X-Request-ID", "X-Request-Id", "Request-ID", "X-Correlation-ID"):
            value: str | None = resp.headers.get(header)
            if value:
                return value[:100]
        return None


@dataclass(frozen=True)
class RagflowConfig:
    """RAGFlow 접속·검색 파라미터.

    자격증명은 코드/노트북에 하드코딩하지 않고 ``from_env`` 로 환경변수에서
    읽는다.
    """

    base_url: str
    api_key: str
    dataset_id: str
    rerank_id: str | None = None
    timeout: float = 5.0  # 외부 호출 타임아웃 ≤ 5초
    max_retries: int = 2  # retry ≤ 2회
    page_size: int = 100  # 여러 직무 타입(카테고리)을 확보하려면 공고를 넉넉히 retrieve
    similarity_threshold: float = 0.2
    vector_similarity_weight: float = 0.3
    # True 면 RAGFlow 가 질의 키워드 추출용 LLM(gpt-4o-mini)을 호출한다 →
    keyword: bool = False
    # RAGFlow Graph RAG option. Sent as retrieval payload field `use_kg`.
    use_kg: bool = False
    # metadata_condition 으로 거를 doc_type. 단일 데이터셋에 job_posting/syllabus 가
    # 혼재하므로 직무만 받으려면 필터가 필요. None 이면 미적용.
    doc_type: str | None = "job_posting"

    @classmethod
    def from_env(cls) -> RagflowConfig:
        """``RAGFLOW_*`` 환경변수에서 설정을 읽는다. 누락 시 즉시 실패한다."""
        import os

        try:
            return cls(
                base_url=os.environ["RAGFLOW_BASE_URL"],
                api_key=os.environ["RAGFLOW_API_KEY"],
                dataset_id=os.environ["RAGFLOW_DATASET_ID"],
                rerank_id=os.environ.get("RAGFLOW_RERANKER_ID"),
                use_kg=_env_bool(os.environ.get("RAGFLOW_USE_KG")),
            )
        except KeyError as exc:
            raise RuntimeError(f"RAGFlow 환경변수 누락: {exc}") from exc


@dataclass(frozen=True)
class RetrievedChunk:
    """retrieval 응답의 원시 청크 단건.

    도메인 가공 이전의 날것이다. ``document_metadata`` 는 ``include_metadata=true``
    요청 시 RAGFlow 가 청크에 실어 보내는 문서 메타필드 dict 다. 도메인 어댑터·
    실측(probe) 모두 본 형태에서 출발한다.
    """

    content: str
    similarity: float
    document_metadata: dict[str, Any]
    document_id: str | None = None


@dataclass(frozen=True)
class RagflowDocument:
    """문서목록 API (``GET /datasets/{id}/documents``) 응답의 문서 단건.

    의미 검색(retrieval)과 달리 질문 없이 데이터셋 문서를 **전수 나열**할 때
    쓴다. ``meta_fields`` 로 doc_type·트랙명 등 문서 메타를 그대로 받으며, 본문은
    목록 응답에 없으므로 ``fetch_document_text`` 로 별도 조회한다.
    """

    document_id: str
    name: str
    meta_fields: dict[str, Any]
    chunk_count: int = 0


class RagflowClient:
    """RAGFlow retrieval HTTP 코어. 도메인 어댑터의 기반 클래스.

    하위 어댑터는 본 클래스를 상속하여 ``_error_cls`` 를 자신의 boundary 예외로
    덮고, 원시 청크 → 도메인 모델 변환만 추가한다. 본 클래스 자체로 인스턴스화하면
    범용 retrieval 클라이언트로 동작하므로 새 데이터셋 실측에 그대로 쓸 수 있다.
    """

    # 하위 어댑터가 자신의 boundary contract 예외로 덮는다. 기반 클래스 기본값은
    # 범용 ``RagflowError`` 이며, 모든 실패는 본 타입(또는 그 하위)으로 변환된다.
    _error_cls: type[RagflowError] = RagflowError

    def __init__(
        self,
        config: RagflowConfig,
        session: requests.Session | None = None,
    ) -> None:
        self._cfg = config
        # 테스트에서 fake session 주입 가능 (부작용 격리·DI 원칙).
        self._session = session or requests.Session()

    def retrieve(
        self,
        query: str,
        *,
        doc_type: str | None = None,
        metadata_fields: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """질의에 대한 원시 청크를 similarity 내림차순으로 반환한다.

        도메인 의미 없이 청크만 돌려주므로 새 데이터셋 실측에 그대로 쓸 수 있다.
        ``doc_type`` 미지정 시 config 의 기본값(직무 검색용 ``job_posting``)을
        쓰며, 모든 doc_type 을 받으려면 ``doc_type=None`` 을 명시하고 config 의
        기본값도 ``None`` 인 클라이언트를 쓴다.

        실패는 모두 ``self._error_cls`` (기본 ``RagflowError``) 로 변환된다.
        """
        resolved_doc_type = doc_type if doc_type is not None else self._cfg.doc_type
        payload = self._build_payload(
            query, doc_type=resolved_doc_type, metadata_fields=metadata_fields
        )
        data = self._post_retrieval(payload)
        chunks = [self._to_chunk(c) for c in self._extract_raw_chunks(data)]
        results = [c for c in chunks if c is not None]
        results.sort(key=lambda c: c.similarity, reverse=True)
        return results[:top_k] if top_k is not None else results

    def list_documents(
        self, *, name_keyword: str | None = None, page_size: int = 100
    ) -> list[RagflowDocument]:
        """데이터셋 문서를 전수 나열한다 (질문 없는 카탈로그 조회).

        retrieval 은 의미 검색이라 doc_type 전수 수집이 불가능하다 (질문에 따라
        결과가 흔들림이 실측으로 확인됨). 트랙·과목 카탈로그 전수가 필요한
        Repository 는 본 메서드를 쓴다. ``name_keyword`` 로 파일명 부분일치
        필터(예: "트랙소개")를 걸 수 있으며, ``total`` 에 도달할 때까지
        페이지를 순회한다.
        """
        path = _DOCUMENTS_PATH_TMPL.format(dataset_id=self._cfg.dataset_id)
        documents: list[RagflowDocument] = []
        page = 1
        while True:
            params: dict[str, Any] = {"page": page, "page_size": page_size}
            if name_keyword:
                params["keywords"] = name_keyword
            data = self._get_data(path, params, label="documents")
            raw_docs = data.get("docs")
            if not isinstance(raw_docs, list):
                raise self._error_cls(reason="invalid_documents")
            documents.extend(doc for doc in map(self._to_document, raw_docs) if doc is not None)
            total = data.get("total")
            if not raw_docs or not isinstance(total, int) or len(documents) >= total:
                break
            page += 1
        return documents

    def fetch_document_text(self, document_id: str, *, page_size: int = 100) -> str:
        """문서의 모든 청크 본문을 순서대로 이어붙여 반환한다.

        문서목록 응답엔 본문이 없으므로 청크 API 로 별도 조회한다. 트랙 소개·
        교육과정처럼 본문 전체를 파싱해야 하는 Repository 가 쓴다.
        """
        path = _CHUNKS_PATH_TMPL.format(dataset_id=self._cfg.dataset_id, document_id=document_id)
        parts: list[str] = []
        seen = 0
        page = 1
        while True:
            data = self._get_data(path, {"page": page, "page_size": page_size}, label="chunks")
            chunks = data.get("chunks")
            if not isinstance(chunks, list):
                raise self._error_cls(reason="invalid_chunks")
            seen += len(chunks)
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                content = chunk.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content)
            total = data.get("total")
            if not chunks or not isinstance(total, int) or seen >= total:
                break
            page += 1
        return "\n".join(parts)

    def fetch_chunk_vectors(
        self, *, dataset_id: str | None = None, page_size: int = 50
    ) -> list[dict[str, Any]]:
        """데이터셋 전체 청크의 임베딩 벡터를 ``doc_name`` 과 함께 평탄 리스트로 반환한다.

        ``GET /datasets/{id}/chunks/vectors`` 를 ``total`` 까지 페이지네이션한다. 사전
        적재된 벡터를 그대로 받아 오므로(임베딩 재계산 없음) 트랙 meta_vector 카탈로그
        생성에 쓴다. ``dataset_id`` 미지정 시 config 의 기본 데이터셋을 쓰며, 트랙 벡터가
        별도 KB 에 있으면 명시한다. 본문(content)은 받지 않는다(벡터만 필요).

        Returns:
            ``{"id", "document_id", "doc_name", "vector"}`` 형태 청크 dict 리스트.
        """
        ds = dataset_id or self._cfg.dataset_id
        path = _DATASET_VECTORS_PATH_TMPL.format(dataset_id=ds)
        chunks: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self._get_data(path, {"page": page, "page_size": page_size}, label="vectors")
            page_chunks = data.get("chunks")
            if not isinstance(page_chunks, list):
                raise self._error_cls(reason="invalid_vectors")
            chunks.extend(c for c in page_chunks if isinstance(c, dict))
            total = data.get("total")
            if not page_chunks or not isinstance(total, int) or len(chunks) >= total:
                break
            page += 1
        return chunks

    @staticmethod
    def _to_document(doc: Any) -> RagflowDocument | None:
        if not isinstance(doc, dict):
            return None
        document_id = doc.get("id")
        name = doc.get("name")
        if not isinstance(document_id, str) or not isinstance(name, str):
            return None
        meta = doc.get("meta_fields")
        if not isinstance(meta, dict):
            meta = {}
        chunk_count = doc.get("chunk_count")
        return RagflowDocument(
            document_id=document_id,
            name=name,
            meta_fields=meta,
            chunk_count=chunk_count if isinstance(chunk_count, int) else 0,
        )

    def _build_payload(
        self,
        query: str,
        *,
        doc_type: str | None = None,
        metadata_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": query,
            "dataset_ids": [self._cfg.dataset_id],
            "page": 1,
            "page_size": self._cfg.page_size,
            "similarity_threshold": self._cfg.similarity_threshold,
            "vector_similarity_weight": self._cfg.vector_similarity_weight,
            "keyword": self._cfg.keyword,
            "highlight": False,
        }
        if self._cfg.use_kg:
            payload["use_kg"] = True
        if metadata_fields:
            # 청크에 document_metadata 를 붙여 받음. ragflow 측 코드 커스텀으로 같이 받아옴.
            payload["include_metadata"] = True
            payload["metadata_fields"] = metadata_fields
        if self._cfg.rerank_id:
            payload["rerank_id"] = self._cfg.rerank_id
        if doc_type:
            payload["metadata_condition"] = {
                "conditions": [
                    {
                        "name": "doc_type",
                        "comparison_operator": "is",
                        "value": doc_type,
                    }
                ]
            }
        return payload

    def _post_retrieval(self, payload: dict[str, Any]) -> dict[str, Any]:
        """retrieval 엔드포인트를 호출하고 ``data`` 블록을 반환한다."""
        url = f"{self._cfg.base_url.rstrip('/')}{_RETRIEVAL_PATH}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }
        return self._request_data(
            lambda: self._session.post(
                url, json=payload, headers=headers, timeout=self._cfg.timeout
            ),
            label="retrieval",
        )

    def _get_data(self, path: str, params: dict[str, Any], *, label: str) -> dict[str, Any]:
        """GET 엔드포인트(문서목록·청크)를 호출하고 ``data`` 블록을 반환한다."""
        url = f"{self._cfg.base_url.rstrip('/')}{path}"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}
        return self._request_data(
            lambda: self._session.get(
                url, params=params, headers=headers, timeout=self._cfg.timeout
            ),
            label=label,
        )

    def _request_data(self, send: Callable[[], requests.Response], *, label: str) -> dict[str, Any]:
        """HTTP 호출을 재시도 정책과 함께 보내고 ``data`` 블록을 반환한다.

        네트워크 오류·타임아웃·5xx 는 재시도, 4xx·파싱 실패는 즉시 중단하며,
        모든 실패는 ``self._error_cls`` 로 변환한다 (raw 예외 전파 금지,
        메시지에 자격증명/본문 미포함). retrieval(POST)·문서목록/청크(GET)가
        동일한 재시도·에러 변환 정책을 공유하도록 호출만 ``send`` 로 주입받는다.
        """
        error_cls = self._error_cls
        last_response_error: RagflowError | None = None
        last_network_error_type: str | None = None
        for attempt in range(self._cfg.max_retries + 1):
            if attempt > 0:
                time.sleep(0.2 * attempt)  # 가벼운 선형 backoff
            try:
                resp = send()
            except requests.RequestException as exc:
                last_network_error_type = type(exc).__name__
                last_response_error = None
                logger.warning(
                    "RAGFlow %s 호출 실패 (attempt %d/%d): %s",
                    label,
                    attempt + 1,
                    self._cfg.max_retries + 1,
                    type(exc).__name__,
                )
                continue

            if resp.status_code >= 500:
                last_response_error = self._response_error(resp, reason="http_5xx")
                last_network_error_type = None
                logger.warning(
                    "RAGFlow %s 5xx (attempt %d): %s", label, attempt + 1, resp.status_code
                )
                continue
            if resp.status_code >= 400:
                # 4xx 는 재시도 무의미 (인증/요청 오류) → 즉시 중단.
                raise self._response_error(resp, reason="http_4xx")

            return self._parse_response_body(resp)

        if last_response_error is not None:
            raise error_cls(
                reason="http_5xx_retry_exhausted",
                status_code=last_response_error.status_code,
                code=last_response_error.code,
                request_id=last_response_error.request_id,
            )
        if last_network_error_type is not None:
            raise error_cls(
                reason="network_retry_exhausted",
                error_type=last_network_error_type,
            )
        raise error_cls(reason="retry_exhausted")

    @classmethod
    def _parse_response_body(cls, resp: requests.Response) -> dict[str, Any]:
        try:
            body = resp.json()
        except ValueError as exc:
            raise cls._error_cls.from_response(resp, reason="invalid_json") from exc
        if not isinstance(body, dict):
            raise cls._error_cls.from_response(resp, reason="invalid_body")
        if body.get("code", 0) != 0:
            raise cls._response_error(resp, body, reason="ragflow_code_error")
        data = body.get("data")
        if not isinstance(data, dict):
            raise cls._error_cls.from_response(resp, body, reason="missing_data")
        return data

    @classmethod
    def _response_error(
        cls,
        resp: requests.Response,
        body: dict[str, Any] | None = None,
        reason: str = "response_error",
    ) -> RagflowError:
        """RAGFlow 실패 응답을 안전한 boundary 예외로 변환한다.

        응답 본문 전체나 요청 본문은 노출하지 않고 status/code/request_id 정도만
        남긴다.
        """
        return cls._error_cls.from_response(resp, body, reason=reason)

    def _extract_raw_chunks(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """``data`` 블록에서 청크 리스트를 꺼낸다. 형식 오류 청크는 스킵한다."""
        chunks = data.get("chunks") or []
        if not isinstance(chunks, list):
            raise self._error_cls(reason="invalid_chunks")
        return [chunk for chunk in chunks if isinstance(chunk, dict)]

    @staticmethod
    def _to_chunk(chunk: dict[str, Any]) -> RetrievedChunk | None:
        raw_content = chunk.get("content")
        if not isinstance(raw_content, str):
            return None
        content = raw_content.strip()
        if not content:
            return None
        meta = chunk.get("document_metadata")
        if not isinstance(meta, dict):
            meta = {}
        try:
            similarity = float(chunk.get("similarity", 0.0))
        except (TypeError, ValueError):
            similarity = 0.0
        document_id = chunk.get("document_id")
        return RetrievedChunk(
            content=content,
            similarity=similarity,
            document_metadata=meta,
            document_id=document_id if isinstance(document_id, str) else None,
        )
