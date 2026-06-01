"""RAGFlow 기반 ``JobSearchClient`` 구현체.

``job_search.JobSearchClient`` Protocol 의 운영 어댑터. RAGFlow retrieval HTTP
코어(``ragflow_client.RagflowClient``)를 상속하여, 도메인 전용 부분 — 공고
카테고리 → 직무 타입 매핑, 청크 → 직무 타입 단위 ``RagSearchResult`` 집계 — 만
특수화한다.

RAGFlow 는 **공고 단위(per-posting)** 로 청크를 내려주므로, 같은 직무 타입
(공고 카테고리를 ``config/category_to_job_type.yaml`` 로 매핑한 직무 카탈로그
표준 코드)에 속한 공고들을 boundary 안에서 **한 건으로 dedup·집계**한다. 이
집계가 ``JobSearchClient`` contract 6 — 직무 카드 중복 방지 — 을 충족한다::

    공고 풀 retrieve  →  job_id 별 그룹핑
      score          = 그룹 내 최댓값 (가장 강한 매칭 근거)
      tech_stacks    = 직무 카테고리별 빈도 상위 N개 집계 (category_to_job_type.yaml)
      competency_tags= 그룹 공고들이 실제 언급한 기술(빈도 누적) 중 집계에 없는 것
      posting_count  = 그룹 크기 (직무 타입 출현 횟수)
      description    = 최고 점수 공고의 청크 본문 (대표값)

category·tech_stack 은 ``include_metadata=true`` 로 요청해 RAGFlow 가 청크에
붙여주는 ``document_metadata``에서 읽는다. ``tech_stacks`` 는 단일 공고 값이 아니라
``category_to_job_type.yaml`` 의 카테고리별 집계(대표 스택)를 쓰고, ``competency_tags``
는 그룹 공고들이 실제 언급한 기술(누적) 중 그 집계에 없는 것만 분리해 채운다.

호출 측은 ``RagflowJobSearchClient`` 를 직접 import 하지 않고 ``JobSearchClient``
Protocol 타입으로만 주입받는다.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
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

# 공고 카테고리 → 직무 카탈로그 표준 코드 매핑 yaml. job_id 는 직무 카탈로그
# 표준 코드(job_tech_stacks.json 의 category_id)이며, fallback
# category_to_jobs.yaml 과 동일한 표준 코드 어휘를 공유한다.
_DEFAULT_CATEGORY_MAP_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "category_to_job_type.yaml"
)


class _JobType(NamedTuple):
    """공고 카테고리에서 매핑된 직무 타입 식별자와 대표 기술스택."""

    job_id: str
    job_name: str
    tech_stacks: tuple[str, ...]


class _PostingHit(NamedTuple):
    """집계 이전의 공고 단위 retrieval 적중 단건.

    같은 직무 타입의 ``_PostingHit`` 여러 건이 ``_aggregate_by_job_type`` 에서
    하나의 ``RagSearchResult`` 로 합쳐진다.
    """

    job_id: str
    job_name: str
    score: float
    description: str
    tech_stacks: list[str]
    aggregate: tuple[str, ...]  # 이 공고 카테고리의 대표 기술 집계 (yaml)


@dataclass(frozen=True)
class RagflowSearchError(RagflowError, RagSearchError):
    """RAGFlow retrieval 실패를 나타내는 직무 검색 전용 예외.

    범용 ``RagflowError`` 의 안전 메시지·필드를 그대로 물려받으면서,
    ``JobSearchClient`` contract 의 ``RagSearchError`` 로도 잡히도록 두 base 를
    동시에 상속한다. 직무 매칭 노드의 fallback 분기가 본 예외를 catch 한다.
    """


def _accumulate_tech_stacks(group: list[_PostingHit]) -> list[str]:
    """그룹 공고들의 기술스택을 공고 출현 빈도 내림차순으로 누적·정렬한다.

    한 공고 안의 중복 태그는 1회로 세어 (presence count) 공고 단위 빈도를
    구한다. 빈도가 같으면 처음 등장한 순서를 유지해 결정적(deterministic)
    이다 — RAGFlow score 순으로 공고를 입력받으므로 동점 시 상위 공고에서
    먼저 나온 스택이 앞선다.
    """
    counter: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for hit in group:
        for tech in dict.fromkeys(hit.tech_stacks):  # 공고 내 중복 제거 (순서 보존)
            counter[tech] += 1
            if tech not in first_seen:
                first_seen[tech] = len(first_seen)
    return sorted(counter, key=lambda tech: (-counter[tech], first_seen[tech]))


def _aggregate_by_job_type(hits: list[_PostingHit]) -> list[RagSearchResult]:
    """공고 단위 적중을 직무 타입(``job_id``) 단위 결과로 dedup·집계한다.

    같은 ``job_id`` 의 공고들을 한 건으로 묶어 ``score`` 는 그룹 최댓값,
    ``tech_stacks`` 는 카테고리 집계(대표 스택), ``competency_tags`` 는 그룹
    공고들이 실제 언급한 기술(빈도 누적) 중 집계에 없는 것, ``posting_count`` 는
    그룹 크기, ``description`` 은 최고 점수 공고 본문으로 채운다. 그룹 출현 순서를
    보존해, 정렬 전에도 입력 순서가 결정적으로 유지된다.

    같은 ``job_id`` 에 여러 카테고리가 매핑될 수 있으므로(예: 데이터분석·
    데이터사이언스 → DA) 집계는 그룹 공고들의 카테고리 집계를 순서 보존
    합집합으로 병합한다.
    """
    groups: dict[str, list[_PostingHit]] = {}
    for hit in hits:
        groups.setdefault(hit.job_id, []).append(hit)

    results: list[RagSearchResult] = []
    for job_id, group in groups.items():
        best = max(group, key=lambda hit: hit.score)
        aggregate = list(dict.fromkeys(tech for hit in group for tech in hit.aggregate))
        aggregate_lower = {tech.lower() for tech in aggregate}
        competency_tags = [
            tech for tech in _accumulate_tech_stacks(group) if tech.lower() not in aggregate_lower
        ]
        results.append(
            RagSearchResult(
                job_id=job_id,
                job_name=best.job_name,
                score=best.score,
                description=best.description,
                tech_stacks=aggregate,
                competency_tags=competency_tags,
                posting_count=len(group),
            )
        )
    return results


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
            raw_stacks = entry.get("tech_stacks") or []
            tech_stacks = (
                tuple(str(t) for t in raw_stacks if t) if isinstance(raw_stacks, list) else ()
            )
            mapping[str(category)] = _JobType(
                job_id=str(job_id), job_name=str(job_name), tech_stacks=tech_stacks
            )
    return mapping


class RagflowJobSearchClient(RagflowClient):
    """``JobSearchClient`` Protocol 의 RAGFlow retrieval 어댑터.

    HTTP 호출·재시도·에러 변환·응답 파싱은 ``RagflowClient`` 기반 클래스가
    담당하고, 본 클래스는 공고 카테고리 → 직무 타입 매핑과 공고 단위 청크 →
    직무 타입 단위 ``RagSearchResult`` 집계만 특수화한다.
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
        """자연어 질의에 대한 직무 KB 검색 결과 상위 ``top_k`` **직무 타입**을 반환한다.

        순수 retrieval 경로 (``keyword=False``). 공고 풀을 retrieve 한 뒤 ``job_id``
        (카테고리 → 직무 타입 매핑) 단위로 dedup·집계하므로, ``top_k`` 는 공고 수가
        아니라 서로 다른 직무 카드 수다. 같은 직무 타입의 공고들은 한 건으로 묶여
        ``tech_stacks`` 가 누적되고 ``posting_count`` 로 출현 횟수가 보존된다. 실패는
        모두 ``RagSearchError`` 로 변환된다.
        """
        payload = self._build_payload(
            query, doc_type=self._cfg.doc_type, metadata_fields=_METADATA_FIELDS
        )
        data = self._post_retrieval(payload)
        hits = self._parse_hits(data)
        results = _aggregate_by_job_type(hits)
        # 직무 타입 단위 score(그룹 최댓값) 내림차순 정렬 후 상위 top_k.
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def _parse_hits(self, data: dict[str, Any]) -> list[_PostingHit]:
        # 공고 단위 청크를 적중(_PostingHit)으로 변환한다. dedup·누적은
        # _aggregate_by_job_type 가 직무 타입 단위로 수행한다.
        hits: list[_PostingHit] = []
        for chunk in self._extract_raw_chunks(data):
            hit = self._chunk_to_hit(chunk)
            if hit is not None:
                hits.append(hit)
        return hits

    def _chunk_to_hit(self, chunk: dict[str, Any]) -> _PostingHit | None:
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
        # 공고 단위 적중. tech_stacks 는 공고가 실제 언급한 기술(competency 분리용),
        # aggregate 는 직무 타입 단위 집계 후 대표 tech_stacks 가 된다.
        return _PostingHit(
            job_id=job_type.job_id,
            job_name=job_type.job_name,
            score=score,
            description=content[:_MAX_DESCRIPTION_LEN],
            tech_stacks=self._extract_tech_stacks(meta),
            aggregate=job_type.tech_stacks,
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
