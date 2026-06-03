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
       임계값 비교는 raw 검색 점수 기준 (이수 과목 부스팅 이전).
    4. 정상 → ``RagSearchResult`` → ``JobCandidate`` 매핑 후 이수 과목 부스팅
       후처리 (점수 가산 + 재정렬, 임베딩 호출 없는 순수 집합 연산).
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

from tracktory.common.tech_keywords import canonical_tech_keys
from tracktory.graph.course_tech import (
    DEFAULT_COURSE_CATALOG_PATH,
    load_course_tech_index,
    resolve_course_tokens,
)
from tracktory.graph.models import JobCandidate, JobMatchingConfig
from tracktory.graph.state import GraphState
from tracktory.rag.job_search import JobSearchClient, RagSearchError, RagSearchResult

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "synergy.yaml"
_DEFAULT_CATEGORY_MAPPING_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "category_to_jobs.yaml"
)
# yaml entry 에 ``rank`` 가 누락된 경우 정렬 시 마지막으로 밀어내는 sentinel.
# 매직 넘버 회피용 상수 — yaml 스키마가 ``rank`` 를 필수화하면 제거 가능.
_RANK_SENTINEL = 999

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

    검색 직후에는 후처리 전이라 ``match_score`` 와 ``similarity`` 가 동일한
    검색 결합 점수에서 출발한다. 이수 과목 부스팅은 이후 ``match_score`` 에만
    가산되어 두 값이 갈라진다. ``result.score`` 는 외부 검색의 hybrid +
    reranker 결합 점수다.
    """
    return JobCandidate(
        job_id=result.job_id,
        job_name=result.job_name,
        tech_stacks=list(result.tech_stacks),
        competency_tags=list(result.competency_tags),
        match_score=result.score,
        similarity=result.score,
        fallback_used=False,
        posting_count=result.posting_count,
    )


def _apply_completed_course_boost(
    candidates: list[JobCandidate],
    completed_tokens: list[str],
    weight: float,
) -> list[JobCandidate]:
    """이수 과목 기술 토큰과 직무 토큰의 교집합 비율만큼 점수를 가산하고 재정렬한다.

    boost = weight · |completed ∩ job_tokens| / |job_tokens|
    new_score = min(base_score + boost, 1.0)

    두 번째 인자는 이수 과목 *이름* 이 아니라 카탈로그로 이미 환산된 기술
    *토큰* 이다 (``resolve_course_tokens`` 참조). 교집합은 직무 기술 어휘로
    계산되어야 의미가 있으므로, 양쪽 토큰을 모두 ``canonical_tech_keys`` 로
    정규화한 뒤 비교한다 — "자바" 와 "Java", "spring boot" 와 "Spring Boot" 처럼
    별칭·대소문자만 다른 표기를 같은 기술로 본다. 이 어휘 정합이 본 부스팅의
    핵심 계약이다 (직무 채용공고 어휘와 동일한 정합 사전을 재사용).

    job_tokens 는 직무의 ``tech_stacks`` 와 ``competency_tags`` 합집합이다. 직무
    토큰이 비어 있거나 이수 과목 토큰과 정규 키가 겹치지 않으면 가산은 0 이고
    점수·순서는 변하지 않는다 (무관 과목은 영향 없음). 가산 후 점수 내림차순으로
    안정 정렬하여 동점은 원래 검색 순서를 유지한다.

    이수 과목을 의미 임베딩 단계에 절대 투입하지 않는 정책을 보존하기 위해,
    본 신호는 외부 검색이 끝난 뒤 점수 후처리로만 반영한다 (집합 교집합 연산
    이며 임베딩 호출이 없다). ``weight = 0`` 이거나 이수 과목 토큰이 없으면
    입력을 그대로 반환한다.

    가산은 최종 적합도인 ``match_score`` 에만 반영하고, 검색 원시 점수인
    ``similarity`` 는 보존한다. 두 필드가 갈라지는 지점이 바로 이 후처리다 —
    부스팅이 0 이거나 검색 직후에는 두 값이 일치하지만, 가산이 발생하면
    ``match_score`` 만 올라간다.

    Args:
        candidates: 직무 검색 결과 후보들 (검색 점수 순).
        completed_tokens: 이수 과목에서 환산된 기술 토큰 목록 (이름이 아님).
        weight: 교집합 비율 1 일 때의 최대 가산량.

    Returns:
        가산·재정렬된 새 후보 리스트 (변경 없으면 입력 그대로).
    """
    if weight <= 0.0 or not completed_tokens:
        return candidates
    completed_keys = canonical_tech_keys(completed_tokens)
    if not completed_keys:
        return candidates

    boosted: list[JobCandidate] = []
    for cand in candidates:
        job_keys = canonical_tech_keys((*cand.tech_stacks, *cand.competency_tags))
        overlap_ratio = len(completed_keys & job_keys) / len(job_keys) if job_keys else 0.0
        new_score = min(cand.match_score + weight * overlap_ratio, 1.0)
        boosted.append(cand.model_copy(update={"match_score": new_score}))

    boosted.sort(key=lambda c: c.match_score, reverse=True)
    return boosted


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

    sorted_entries = sorted(entries, key=lambda entry: entry.get("rank", _RANK_SENTINEL))[:k]
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
        course_catalog_path: Path | None = None,
    ) -> None:
        self._client = job_search_client
        self._config = JobMatchingConfig.load_from_yaml(config_path or _DEFAULT_CONFIG_PATH)
        self._category_mapping = _load_category_mapping(
            category_mapping_path or _DEFAULT_CATEGORY_MAPPING_PATH
        )
        self._course_tech_index = load_course_tech_index(
            course_catalog_path or DEFAULT_COURSE_CATALOG_PATH
        )

    def __call__(self, state: GraphState) -> dict[str, Any]:
        """프로필 텍스트로 직무 후보를 산출하여 부분 state 를 반환한다.

        반환값 계약:
            정상: ``{"recommended_jobs": [...], "trace": ["job_matching:ok"]}``
                — ``fallback_used=False`` 인 후보 ``top_k`` 건.
            fallback (low score 또는 외부 호출 실패): trace 는
                ``job_matching:fallback_categorized``. 모든 후보가 ``fallback_used=True``.
            fallback unmapped (카테고리 매핑 부재): trace 는
                ``job_matching:fallback_unmapped``. ``recommended_jobs`` 는 빈 list.
            skip: trace 는 ``job_matching:skip``. ``recommended_jobs`` 키 부재.

        부작용:
            ``job_search_client.rag_search_jobs`` 호출 1 회 (외부 검색 boundary).
            ``logger.warning`` 은 RagSearchError 분기·미매핑 분기에서 발생,
            ``logger.info`` 는 카테고리 fallback 분기에서 발생한다.
        """
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
            # PII / 자격증명 누출 회피를 위해 메시지 전체가 아닌 예외 타입과
            # 축약 메시지만 로깅한다. 구현체는 ``RagSearchError`` 메시지에
            # status code / request_id 정도만 실어 보낼 책임이 있다.
            logger.warning(
                "job_matching_rag_search_error",
                extra={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:200],
                },
            )
            return self._fallback_response(normalized, top_k, max_similarity=0.0, reason="error")

        # 단계 3: 임계값 분기 — fallback 결정은 raw 검색 점수 기준.
        # 이수 과목 부스팅은 정상 후보 사이의 재정렬 신호이지, 의미가 약한
        # 매칭을 임계값 위로 끌어올려 fallback 을 우회하는 수단이 아니다.
        max_similarity = results[0].score if results else 0.0
        if results and max_similarity >= threshold:
            candidates = [_to_candidate(result) for result in results]
            completed_tokens = resolve_course_tokens(
                normalized.get("completed_courses") or [],
                self._course_tech_index,
            )
            candidates = _apply_completed_course_boost(
                candidates,
                completed_tokens,
                self._config.completed_course_boost.weight,
            )
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
