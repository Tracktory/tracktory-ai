"""사용자 프로필 임베딩과 직무 임베딩 간 코사인 유사도로 top-k 직무 후보를 산출한다.

max similarity 가 임계값 미만이면 사용자 1순위 관심사 카테고리에 사전 정의된
인기 직무로 안전 fallback 한다. 매핑이 없으면 빈 결과 + 경고 로그.

처리 흐름 (5단계):
    1. 입력 검증 — ``profile_vector`` 또는 ``normalized_profile`` 부재 시 skip.
    2. 직무 인덱스 전체 로드 (외부 I/O — 진입점 단 1곳).
    3. 모든 직무에 코사인 유사도 매기기 + top-k 추출.
    4. max similarity ≥ 임계값 → 정상 결과 (fallback_used=False).
       미만 → 카테고리 사전 매핑 fallback (fallback_used=True 또는 빈 결과).
    5. state 부분 반환.

부작용 격리:
    - ``job_index`` 호출은 ``__call__`` 단 1곳.
    - ``logger.info`` / ``logger.warning`` 은 fallback 분기 단 1곳씩.
    - 순수 계산 함수는 모두 module-level — mock 없이 테스트 가능.

단일 임베딩 boundary (ADR-0001) 준수:
    직무 임베딩 ``Job.job_vector`` 와 사용자 ``profile_vector`` 모두 단일 임베딩
    모델로 사전 임베딩된 L2 normalized 상태를 전제하므로 본 노드는 새로
    임베딩을 호출하지 않으며 cosine = 단순 내적으로 환원된다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tracktory.graph.models import JobCandidate, JobMatchingConfig
from tracktory.graph.state import GraphState
from tracktory.rag.job_index import Job, JobIndex

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "synergy.yaml"
_DEFAULT_CATEGORY_MAPPING_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "category_to_jobs.yaml"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 순수 계산 함수
# ---------------------------------------------------------------------------


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """두 L2 normalized 벡터 사이 코사인 유사도 = 단순 내적.

    단일 임베딩 boundary (ADR-0001) 가 사전 normalize 를 보장하므로 본 노드는
    내적만 계산한다. 빈 벡터·dim 불일치는 0.0 으로 안전 처리.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    arr_a = np.asarray(vec_a, dtype=float)
    arr_b = np.asarray(vec_b, dtype=float)
    return float(np.dot(arr_a, arr_b))


def _top_k_by_similarity(
    jobs: list[Job],
    profile_vector: list[float],
    k: int,
) -> list[tuple[Job, float]]:
    """모든 직무에 코사인 유사도를 매기고 상위 k 개 ``(job, score)`` 를 내림차순 반환.

    음수 cosine 은 ``[0, 1]`` 정의역 강제로 0.0 으로 clip 된다 — 직무 후보 모델의
    ``similarity`` 필드 제약을 만족시키기 위함이며, 의미상 "유사하지 않음" 으로
    해석되어 fallback 임계 비교에서도 자연스럽게 동일 분기를 탄다.

    동점 시 ``job_id`` 알파벳 순으로 deterministic 결정. ``k`` 가 직무 수보다
    크면 가용한 만큼만 반환한다.
    """
    scored = [(job, max(0.0, _cosine_similarity(profile_vector, job.job_vector))) for job in jobs]
    scored.sort(key=lambda pair: (-pair[1], pair[0].job_id))
    return scored[:k]


def _load_category_mapping(path: Path) -> dict[str, list[dict[str, Any]]]:
    """yaml 파일에서 카테고리 → 직무 매핑을 로드한다.

    Args:
        path: ``category_to_jobs.yaml`` 의 경로.

    Returns:
        ``{카테고리명: [{job_id, rank}, ...]}`` 매핑. 빈 파일은 ``{}``.

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


def _build_fallback_candidates(
    primary_category: str,
    mapping: dict[str, list[dict[str, Any]]],
    k: int,
    job_lookup: dict[str, Job],
) -> list[JobCandidate]:
    """카테고리 매핑 + 직무 인덱스 lookup 으로 fallback 후보 리스트 생성.

    ``rank`` 오름차순으로 상위 k 개를 채택하며, 매핑된 직무가 인덱스에 없으면
    조용히 건너뛴다 (가용한 만큼만 반환).
    """
    entries = mapping.get(primary_category) or []
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda entry: entry.get("rank", 999))[:k]
    candidates: list[JobCandidate] = []
    for entry in sorted_entries:
        job_id = entry.get("job_id")
        if not job_id or job_id not in job_lookup:
            continue
        job = job_lookup[job_id]
        candidates.append(
            JobCandidate(
                job_id=job.job_id,
                job_name=job.job_name,
                tech_stacks=list(job.tech_stacks),
                competency_tags=list(job.competency_tags),
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
    """직무 매칭 노드 — 코사인 매칭 + 카테고리 사전 매핑 fallback.

    의존성을 ``__init__`` 으로 주입받으므로 단위 테스트에서 ``JobIndex`` 를
    ``InMemoryJobIndex`` 로 교체하면 외부 I/O 없이 검증 가능하다. config·매핑
    yaml 은 생성자에서 한 번만 로드한다.
    """

    def __init__(
        self,
        job_index: JobIndex,
        config_path: Path | None = None,
        category_mapping_path: Path | None = None,
    ) -> None:
        self._index = job_index
        self._config = JobMatchingConfig.load_from_yaml(config_path or _DEFAULT_CONFIG_PATH)
        self._category_mapping = _load_category_mapping(
            category_mapping_path or _DEFAULT_CATEGORY_MAPPING_PATH
        )

    def __call__(self, state: GraphState) -> dict[str, Any]:
        # 단계 1: 입력 검증
        profile_vector: list[float] | None = state.get("profile_vector")
        normalized: dict[str, Any] | None = state.get("normalized_profile")
        if not profile_vector or not normalized:
            return {
                "errors": ["job_matching skipped: profile_vector or normalized_profile missing"],
                "trace": ["job_matching:skip"],
            }

        # 단계 2: 직무 인덱스 로드 (외부 I/O — 진입점 단 1곳)
        jobs = self._index.list_all()
        if not jobs:
            return {
                "errors": ["job_matching skipped: empty job index"],
                "trace": ["job_matching:skip"],
            }

        # 단계 3: top-k 코사인 매칭
        top_k = self._config.top_k.default
        top_pairs = _top_k_by_similarity(jobs, profile_vector, top_k)
        max_similarity = top_pairs[0][1] if top_pairs else 0.0
        threshold = self._config.min_job_similarity

        # 단계 4: 임계값 분기
        if max_similarity >= threshold:
            candidates = [
                JobCandidate(
                    job_id=job.job_id,
                    job_name=job.job_name,
                    tech_stacks=list(job.tech_stacks),
                    competency_tags=list(job.competency_tags),
                    match_score=score,
                    similarity=score,
                    fallback_used=False,
                )
                for job, score in top_pairs
            ]
            return {
                "recommended_jobs": [c.model_dump(mode="json") for c in candidates],
                "trace": ["job_matching:ok"],
            }

        # fallback: 카테고리 사전 매핑
        interests = normalized.get("interests") or []
        primary_category = interests[0] if interests else ""
        job_lookup = {job.job_id: job for job in jobs}
        fallback = _build_fallback_candidates(
            primary_category, self._category_mapping, top_k, job_lookup
        )

        if not fallback:
            logger.warning(
                "job_matching_category_unmapped",
                extra={
                    "primary_category": primary_category,
                    "max_similarity": max_similarity,
                    "threshold": threshold,
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
            },
        )
        return {
            "recommended_jobs": [c.model_dump(mode="json") for c in fallback],
            "trace": ["job_matching:fallback_categorized"],
        }
