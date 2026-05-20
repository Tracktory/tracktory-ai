"""직무 검색 boundary 위임 + 카테고리 사전 매핑 fallback 으로 직무 후보를 산출한다.

자체 코드는 자연어 프로필 문장을 ``JobSearchClient.rag_search_jobs`` 로 넘기고,
임베딩·검색·재정렬은 외부 검색이 boundary 안에서 일괄 처리한다. 반환된 직무
검색 결과의 결합 점수 (hybrid + reranker) 상위 1 개가 ``min_job_similarity``
임계값 미만이면 사용자 1순위 관심사 카테고리에 사전 정의된 인기 직무로 안전
fallback 한다. 외부 검색 호출이 ``RagSearchError`` 로 실패한 경우에도 동일한
fallback 분기로 전환한다.

처리 흐름 (5 단계):
    1. 입력 검증 — ``profile_text`` 또는 ``normalized_profile`` 부재 시 skip.
    2. 직무 검색 boundary 호출 (외부 I/O — 진입점 단 1곳).
       호출 실패 시 fallback 분기로 전환.
    3. 상위 결과 점수 임계값 분기 — ≥ 임계값이면 정상, 미만이면 fallback.
    4. 정상 → ``RagSearchResult`` → ``JobCandidate`` 매핑.
    5. fallback → 카테고리 매핑에서 후보 구성 (boundary 호출 없음).

부작용 격리:
    - ``job_search_client`` 호출은 ``__call__`` 단 1곳.
    - ``logger.info`` / ``logger.warning`` 은 fallback 분기 단 1곳씩.
    - 모듈-level 헬퍼는 순수 함수 — mock 없이 테스트 가능.

단일 임베딩 boundary (ADR-0001):
    자체 코드는 임베딩 클라이언트를 호출하지 않는다. 직무 검색 boundary 가
    호출 단위에서 임베딩·검색·재정렬을 일괄 처리하므로, 임베딩 공간 정합은
    boundary 안쪽에서 자동 보장된다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from tracktory.graph.models import JobCandidate, JobMatchingConfig
from tracktory.graph.state import GraphState
from tracktory.rag.job_search import JobSearchClient, RagSearchError, RagSearchResult

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "synergy.yaml"
_DEFAULT_CATEGORY_MAPPING_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "category_to_jobs.yaml"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 순수 헬퍼 함수
# ---------------------------------------------------------------------------


def _load_category_mapping(path: Path) -> dict[str, list[dict[str, Any]]]:
    """yaml 파일에서 카테고리 → 직무 매핑을 로드한다.

    Args:
        path: ``category_to_jobs.yaml`` 의 경로.

    Returns:
        ``{카테고리명: [{job_id, job_name, rank}, ...]}`` 매핑. 빈 파일은 ``{}``.

    Raises:
        ValueError: yaml 의 최상위가 mapping 도 ``None`` 도 아닐 때.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Category mapping file {path} must define a top-level mapping")
    return raw


def _to_candidate(result: RagSearchResult) -> JobCandidate:
    """직무 검색 결과 단건을 직무 후보 모델로 변환한다.

    검색 결합 점수가 ``match_score`` 와 ``similarity`` 두 필드에 동일하게
    채워진다. 본 PR 로 점수 의미가 자체 코사인 → 외부 검색 hybrid + reranker
    결합 점수로 silent 변경됨에 유의 (PR 본문 회귀 캐비잇 참조).
    """
    return JobCandidate(
        job_id=result.job_id,
        job_name=result.job_name,
        tech_stacks=list(result.tech_stacks),
        competency_tags=list(result.competency_tags),
        match_score=result.score,
        similarity=result.score,
        fallback_used=False,
    )


def _build_fallback_candidates(
    primary_category: str,
    mapping: dict[str, list[dict[str, Any]]],
    k: int,
) -> list[JobCandidate]:
    """카테고리 매핑에서 fallback 직무 후보를 ``rank`` 오름차순 ``k`` 건 구성한다.

    매핑이 비어 있으면 빈 리스트를 반환한다. ``tech_stacks`` / ``competency_tags``
    는 빈 리스트로 두어 다운스트림 노드가 ``fallback_used=True`` 후보를 보수적으로
    처리하도록 한다. ``job_name`` 누락 시 ``job_id`` 로 대체하여 사용자 표시명이
    항상 비어 있지 않도록 보장한다.
    """
    entries = mapping.get(primary_category) or []
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda entry: entry.get("rank", 999))[:k]
    candidates: list[JobCandidate] = []
    for entry in sorted_entries:
        job_id = entry.get("job_id")
        if not job_id:
            continue
        candidates.append(
            JobCandidate(
                job_id=job_id,
                job_name=entry.get("job_name") or job_id,
                tech_stacks=[],
                competency_tags=[],
                match_score=0.0,
                similarity=0.0,
                fallback_used=True,
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# 노드 진입점
# ---------------------------------------------------------------------------


class JobMatchingNode:
    """직무 매칭 노드 — 외부 검색 boundary 위임 + 카테고리 사전 매핑 fallback.

    의존성을 ``__init__`` 으로 주입받으므로 단위 테스트에서 ``JobSearchClient``
    를 fake 구현체로 교체하면 외부 I/O 없이 검증 가능하다. config·매핑 yaml 은
    생성자에서 한 번만 로드한다.
    """

    def __init__(
        self,
        job_search_client: JobSearchClient,
        config_path: Path | None = None,
        category_mapping_path: Path | None = None,
    ) -> None:
        self._client = job_search_client
        self._config = JobMatchingConfig.load_from_yaml(config_path or _DEFAULT_CONFIG_PATH)
        self._category_mapping = _load_category_mapping(
            category_mapping_path or _DEFAULT_CATEGORY_MAPPING_PATH
        )

    def __call__(self, state: GraphState) -> dict[str, Any]:
        # 단계 1: 입력 검증
        profile_text: str | None = state.get("profile_text")
        normalized: dict[str, Any] | None = state.get("normalized_profile")
        if not profile_text or not normalized:
            return {
                "errors": ["job_matching skipped: profile_text or normalized_profile missing"],
                "trace": ["job_matching:skip"],
            }

        top_k = self._config.top_k.default
        threshold = self._config.min_job_similarity

        # 단계 2: 외부 검색 boundary 호출 (외부 I/O — 진입점 단 1곳)
        try:
            results = self._client.rag_search_jobs(query=profile_text, top_k=top_k)
        except RagSearchError as exc:
            logger.warning(
                "job_matching_rag_search_error",
                extra={"error": str(exc)},
            )
            return self._fallback_response(normalized, top_k, max_similarity=0.0, reason="error")

        # 단계 3: 임계값 분기
        max_similarity = results[0].score if results else 0.0
        if results and max_similarity >= threshold:
            candidates = [_to_candidate(result) for result in results]
            return {
                "recommended_jobs": [c.model_dump(mode="json") for c in candidates],
                "trace": ["job_matching:ok"],
            }

        return self._fallback_response(normalized, top_k, max_similarity, reason="low_score")

    def _fallback_response(
        self,
        normalized: dict[str, Any],
        top_k: int,
        max_similarity: float,
        reason: str,
    ) -> dict[str, Any]:
        """카테고리 사전 매핑으로 fallback 후보를 구성하여 state 부분 결과를 반환한다."""
        threshold = self._config.min_job_similarity
        interests = normalized.get("interests") or []
        primary_category = interests[0] if interests else ""
        fallback = _build_fallback_candidates(primary_category, self._category_mapping, top_k)

        if not fallback:
            logger.warning(
                "job_matching_category_unmapped",
                extra={
                    "primary_category": primary_category,
                    "max_similarity": max_similarity,
                    "threshold": threshold,
                    "reason": reason,
                },
            )
            return {
                "recommended_jobs": [],
                "trace": ["job_matching:fallback_unmapped"],
            }

        logger.info(
            "job_matching_category_fallback",
            extra={
                "primary_category": primary_category,
                "max_similarity": max_similarity,
                "threshold": threshold,
                "reason": reason,
            },
        )
        return {
            "recommended_jobs": [c.model_dump(mode="json") for c in fallback],
            "trace": ["job_matching:fallback_categorized"],
        }
