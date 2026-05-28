"""RAGFlow 기반 ``JobSearchClient`` 구현체.

``job_search.JobSearchClient`` Protocol 의 운영 어댑터. 자연어 질의를 RAGFlow
retrieval HTTP API (``POST /api/v1/retrieval``) 로 넘겨 채용공고 청크 상위
``top_k`` 건을 받아와 ``RagSearchResult`` 형식으로 변환해 반환한다.

검색된 공고를 **per-posting** 으로 반환하되, ``job_id``/``job_name`` 은 공고
번호·제목이 아니라 **공고 카테고리를 직무 타입으로 매핑한 값**(``config/
category_to_job_type.yaml``)을 쓴다. — 카테고리별로 추리거나
합치는 것은 호출 측(다운스트림) 책임이다.

category·tech_stack 은 ``include_metadata=true`` 로 요청해 RAGFlow 가 청크에
붙여주는 ``document_metadata``에서 읽는다. ``competency_tags`` 는 RAGFlow 에
없어 빈 리스트로 둔다.

호출 측은 ``RagflowJobSearchClient`` 를 직접 import 하지 않고 ``JobSearchClient``
Protocol 타입으로만 주입받는다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import requests
import yaml

from tracktory.common.tech_keywords import extract_tech_keywords
from tracktory.rag.job_search import RagSearchError, RagSearchResult

logger = logging.getLogger(__name__)

_RETRIEVAL_PATH = "/api/v1/retrieval"

# document_metadata 로 받아올 meta_fields (RAGFlow include_metadata).
_METADATA_FIELDS = ["category", "tech_stack"]

_MAX_DESCRIPTION_LEN = 10_000

# 공고 카테고리 → 직무 타입 매핑 yaml (job_id 슬러그는 category_to_jobs.yaml 과 정합).
_DEFAULT_CATEGORY_MAP_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "category_to_job_type.yaml"
)


class _JobType(NamedTuple):
    """공고 카테고리에서 매핑된 직무 타입 식별자."""

    job_id: str
    job_name: str


@dataclass(frozen=True)
class RagflowSearchError(RagSearchError):
    """RAGFlow retrieval 실패를 나타내는 구현체 전용 예외.

    ``JobSearchClient`` contract 에 맞춰 호출 측에는 ``RagSearchError`` 로
    잡히며, 문자열에는 status/code/request_id 정도의 안전한 컨텍스트만 담는다.
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
    ) -> RagflowSearchError:
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
            value = resp.headers.get(header)
            if value:
                return value[:100]
        return None


def _load_category_to_job_type(path: Path) -> dict[str, _JobType]:
    """``category_to_job_type.yaml`` 에서 카테고리 → 직무 타입 매핑을 로드한다."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"category_to_job_type 파일은 매핑이어야 함: {path}")
    mapping: dict[str, _JobType] = {}
    for category, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        job_id = entry.get("job_id")
        job_name = entry.get("job_name")
        if job_id and job_name:
            mapping[str(category)] = _JobType(job_id=str(job_id), job_name=str(job_name))
    return mapping


@dataclass(frozen=True)
class RagflowConfig:
    """RAGFlow 접속·검색 파라미터.

    자격증명은 코드/노트북에 하드코딩하지 않고 ``from_env`` 로 환경변수에서
    읽는다 (py-test.ipynb 의 API_KEY 하드코딩 패턴 회피).
    """

    base_url: str
    api_key: str
    dataset_id: str
    rerank_id: str | None = None
    timeout: float = 5.0  # contract 2: 외부 호출 타임아웃 ≤ 5초
    max_retries: int = 2  # contract 2: retry ≤ 2회
    page_size: int = 100  # 여러 직무 타입(카테고리)을 확보하려면 공고를 넉넉히 retrieve
    similarity_threshold: float = 0.2
    vector_similarity_weight: float = 0.3
    # True 면 RAGFlow 가 질의 키워드 추출용 LLM(gpt-4o-mini)을 호출한다 →
    keyword: bool = False
    # metadata_condition 으로 거를 doc_type. 단일 데이터셋에 job_posting/syllabus 가
    # 혼재하므로 직무만 받으려면 필터가 필요. None 이면 미적용.
    doc_type: str | None = "job_posting"

    @classmethod
    def from_env(cls) -> RagflowConfig:
        """``RAGFLOW_*`` 환경변수에서 설정을 읽는다. 누락 시 즉시 실패한다."""
        try:
            return cls(
                base_url=os.environ["RAGFLOW_BASE_URL"],
                api_key=os.environ["RAGFLOW_API_KEY"],
                dataset_id=os.environ["RAGFLOW_DATASET_ID"],
                rerank_id=os.environ.get("RAGFLOW_RERANKER_ID"),
            )
        except KeyError as exc:
            raise RuntimeError(f"RAGFlow 환경변수 누락: {exc}") from exc


class RagflowJobSearchClient:
    """``JobSearchClient`` Protocol 의 RAGFlow retrieval 어댑터."""

    def __init__(
        self,
        config: RagflowConfig,
        session: requests.Session | None = None,
        category_map_path: Path | None = None,
    ) -> None:
        self._cfg = config
        # 테스트에서 fake session 주입 가능 (contract 의 부작용 격리·DI 원칙).
        self._session = session or requests.Session()
        # 공고 카테고리 → 직무 타입 매핑 (생성자에서 1회 로드).
        self._category_map = _load_category_to_job_type(
            category_map_path or _DEFAULT_CATEGORY_MAP_PATH
        )

    def rag_search_jobs(self, query: str, top_k: int = 3) -> list[RagSearchResult]:
        """자연어 질의에 대한 직무 KB 검색 결과 상위 ``top_k`` 건을 반환한다.

        순수 retrieval 경로 (``keyword=False``). 검색된 공고를
        per-posting 으로 변환하되 ``job_id``/``job_name`` 은 카테고리 기반 직무
        타입 값을 쓴다. dedup 없이 전부 반환하며(같은 직무 타입 중복 가능), 추리는
        것은 호출 측 책임이다. 실패는 모두 ``RagSearchError`` 로 변환된다.
        """
        payload = self._build_payload(query)
        data = self._post_retrieval(payload)
        results = self._parse_chunks(data)
        # contract 5: score 내림차순 정렬 후 상위 top_k.
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _build_payload(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": query,
            "dataset_ids": [self._cfg.dataset_id],
            "page": 1,
            "page_size": self._cfg.page_size,
            "similarity_threshold": self._cfg.similarity_threshold,
            "vector_similarity_weight": self._cfg.vector_similarity_weight,
            "keyword": self._cfg.keyword,
            "highlight": False,
            # 청크에 document_metadata(category/tech_stack) 를 붙여 받음, ragflow 측 코드 커스텀으로 같이 받아옴
            "include_metadata": True,
            "metadata_fields": _METADATA_FIELDS,
        }
        if self._cfg.rerank_id:
            payload["rerank_id"] = self._cfg.rerank_id
        if self._cfg.doc_type:
            payload["metadata_condition"] = {
                "conditions": [
                    {
                        "name": "doc_type",
                        "comparison_operator": "is",
                        "value": self._cfg.doc_type,
                    }
                ]
            }
        return payload

    def _post_retrieval(self, payload: dict[str, Any]) -> dict[str, Any]:
        """retrieval 엔드포인트를 호출하고 ``data`` 블록을 반환한다.

        네트워크 오류·타임아웃·5xx 는 재시도, 4xx·파싱 실패는 즉시 중단하며,
        모든 실패는 ``RagSearchError`` 로 변환한다 (contract 1·3: raw 예외
        전파 금지, 메시지에 자격증명/본문 미포함).
        """
        url = f"{self._cfg.base_url.rstrip('/')}{_RETRIEVAL_PATH}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }
        last_response_error: RagflowSearchError | None = None
        last_network_error_type: str | None = None
        for attempt in range(self._cfg.max_retries + 1):
            if attempt > 0:
                time.sleep(0.2 * attempt)  # 가벼운 선형 backoff
            try:
                resp = self._session.post(
                    url, json=payload, headers=headers, timeout=self._cfg.timeout
                )
            except requests.RequestException as exc:
                last_network_error_type = type(exc).__name__
                last_response_error = None
                logger.warning(
                    "RAGFlow retrieval 호출 실패 (attempt %d/%d): %s",
                    attempt + 1,
                    self._cfg.max_retries + 1,
                    type(exc).__name__,
                )
                continue

            if resp.status_code >= 500:
                last_response_error = self._response_error(resp, reason="http_5xx")
                last_network_error_type = None
                logger.warning("RAGFlow 5xx (attempt %d): %s", attempt + 1, resp.status_code)
                continue
            if resp.status_code >= 400:
                # 4xx 는 재시도 무의미 (인증/요청 오류) → 즉시 중단.
                raise self._response_error(resp, reason="http_4xx")

            return self._parse_response_body(resp)

        if last_response_error is not None:
            raise RagflowSearchError(
                reason="http_5xx_retry_exhausted",
                status_code=last_response_error.status_code,
                code=last_response_error.code,
                request_id=last_response_error.request_id,
            )
        if last_network_error_type is not None:
            raise RagflowSearchError(
                reason="network_retry_exhausted",
                error_type=last_network_error_type,
            )
        raise RagflowSearchError(reason="retry_exhausted")

    @staticmethod
    def _parse_response_body(resp: requests.Response) -> dict[str, Any]:
        try:
            body = resp.json()
        except ValueError as exc:
            raise RagflowSearchError.from_response(resp, reason="invalid_json") from exc
        if not isinstance(body, dict):
            raise RagflowSearchError.from_response(resp, reason="invalid_body")
        if body.get("code", 0) != 0:
            raise RagflowJobSearchClient._response_error(resp, body, reason="ragflow_code_error")
        data = body.get("data")
        if not isinstance(data, dict):
            raise RagflowSearchError.from_response(resp, body, reason="missing_data")
        return data

    @staticmethod
    def _response_error(
        resp: requests.Response,
        body: dict[str, Any] | None = None,
        reason: str = "response_error",
    ) -> RagflowSearchError:
        """RAGFlow 실패 응답을 안전한 boundary 예외로 변환한다.

        프로젝트 API 핸들러처럼 응답 본문 전체나 요청 본문은 노출하지 않고,
        ``job_search.py`` contract 에 맞춰 status/code/request_id 정도만 남긴다.
        """
        return RagflowSearchError.from_response(resp, body, reason=reason)

    def _parse_chunks(self, data: dict[str, Any]) -> list[RagSearchResult]:
        chunks = data.get("chunks") or []
        if not isinstance(chunks, list):
            raise RagflowSearchError(reason="invalid_chunks")
        # 공고를 per-posting 으로 전부 변환해 넘긴다. 같은 직무 타입(카테고리)이
        # 여러 건이어도 dedup 하지 않는다 — 카테고리별로 추리는 건 호출 측 책임.
        results: list[RagSearchResult] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                logger.debug("RAGFlow chunk 형식 오류 스킵: %s", type(chunk).__name__)
                continue
            result = self._chunk_to_result(chunk)
            if result is not None:
                results.append(result)
        return results

    def _chunk_to_result(self, chunk: dict[str, Any]) -> RagSearchResult | None:
        raw_content = chunk.get("content")
        if not isinstance(raw_content, str):
            return None
        content = raw_content.strip()
        if not content:
            return None
        meta = chunk.get("document_metadata") or {}
        if not isinstance(meta, dict):
            logger.debug(
                "RAGFlow chunk document_metadata 형식 오류 스킵: %s",
                type(meta).__name__,
            )
            return None
        category = meta.get("category")
        if not category:
            return None
        job_type = self._category_map.get(category)
        if job_type is None:
            # 매핑에 없는 카테고리(비-IT 등) → 스킵.
            logger.debug("미매핑 category 스킵: %s", category)
            return None
        try:
            score = self._clamp(float(chunk.get("similarity", 0.0)))
        except (TypeError, ValueError):
            logger.debug("RAGFlow chunk similarity 파싱 실패: %r", chunk.get("similarity"))
            score = 0.0
        return RagSearchResult(
            job_id=job_type.job_id,
            job_name=job_type.job_name,
            score=score,
            description=content[:_MAX_DESCRIPTION_LEN],
            tech_stacks=self._extract_tech_stacks(meta),
            competency_tags=[],  # RAGFlow 미보유 → 호출 측이 필요 시 별도 join.
        )

    @staticmethod
    def _extract_tech_stacks(meta: dict[str, Any]) -> list[str]:
        # document_metadata 의 tech_stack 은 JSON 문자열 → 파싱 후 공용 정규화
        # (표기 변형 통합 + 약어 보존 + dedup).
        raw = meta.get("tech_stack")
        if not raw:
            return []
        try:
            tags = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            return []
        if not isinstance(tags, list):
            return []
        return extract_tech_keywords(", ".join(str(t) for t in tags))

    @staticmethod
    def _clamp(value: float) -> float:
        # contract 4: score 는 [0, 1] 범위 보장. RAGFlow similarity 는 이미
        # [0, 1] hybrid 점수지만 방어적으로 clamp.
        return max(0.0, min(1.0, value))
