"""RAGFlow 기반 ``JobSearchClient`` 구현체.

``job_search.JobSearchClient`` Protocol 의 운영 어댑터. RAGFlow retrieval HTTP
코어(``ragflow_client.RagflowClient``)를 상속하여, 도메인 전용 부분 — 공고
카테고리 → 직무 타입 매핑, 청크 → ``RagSearchResult`` 변환 — 만 특수화한다.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import requests
import yaml

from tracktory.common.tech_keywords import extract_tech_keywords
from tracktory.rag.job_search import RagSearchError, RagSearchResult
from tracktory.rag.ragflow_client import RagflowClient, RagflowConfig, RagflowError

__all__ = ["RagflowConfig", "RagflowJobSearchClient", "RagflowSearchError"]

logger = logging.getLogger(__name__)

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
class RagflowSearchError(RagflowError, RagSearchError):
    """RAGFlow retrieval 실패를 나타내는 직무 검색 전용 예외.

    범용 ``RagflowError`` 의 안전 메시지·필드를 그대로 물려받으면서,
    ``JobSearchClient`` contract 의 ``RagSearchError`` 로도 잡히도록 두 base 를
    동시에 상속한다. 직무 매칭 노드의 fallback 분기가 본 예외를 catch 한다.
    """


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


class RagflowJobSearchClient(RagflowClient):
    """``JobSearchClient`` Protocol 의 RAGFlow retrieval 어댑터.

    HTTP 호출·재시도·에러 변환·응답 파싱은 ``RagflowClient`` 기반 클래스가
    담당하고, 본 클래스는 공고 카테고리 → 직무 타입 매핑과 청크 →
    ``RagSearchResult`` 변환만 특수화한다.
    """

    # 기반 클래스의 범용 예외를 직무 검색 boundary contract 예외로 좁힌다.
    _error_cls = RagflowSearchError

    def __init__(
        self,
        config: RagflowConfig,
        session: requests.Session | None = None,
        category_map_path: Path | None = None,
    ) -> None:
        super().__init__(config, session)
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
        payload = self._build_payload(
            query, doc_type=self._cfg.doc_type, metadata_fields=_METADATA_FIELDS
        )
        data = self._post_retrieval(payload)
        results = self._parse_chunks(data)
        # score 내림차순 정렬 후 상위 top_k.
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _parse_chunks(self, data: dict[str, Any]) -> list[RagSearchResult]:
        # 공고를 per-posting 으로 전부 변환해 넘긴다. 같은 직무 타입(카테고리)이
        # 여러 건이어도 dedup 하지 않는다 — 카테고리별로 추리는 건 호출 측 책임.
        results: list[RagSearchResult] = []
        for chunk in self._extract_raw_chunks(data):
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
        # score 는 [0, 1] 범위 보장. RAGFlow similarity 는 이미 [0, 1] hybrid
        # 점수지만 방어적으로 clamp.
        return max(0.0, min(1.0, value))
