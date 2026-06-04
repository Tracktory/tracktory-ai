"""직무 매칭 후보를 받아 주 추천 2 + 보조 추천 5 트랙 조합을 산출한다.

학과 경계를 넘는 이색 조합이 항상 노출되도록 슬롯 예약 (Hard Constraint) 으로
보장하고, 시너지 점수와 다양성의 균형은 MMR (Carbonell-Goldstein 1998) 로
조절한다.

처리 흐름 (6단계):
    1. 입력 검증 — ``recommended_jobs`` / ``normalized_profile`` 부재 시 skip.
    2. 트랙 전체 로드 (외부 I/O — 진입점 단 1곳).
    3. 후보 조합 생성 — 1트랙 주전공 제약 적용.
    4. 각 조합에 시너지 점수 매기기 — 외부화 가중치, ``[0, 1]`` clip.
    5. 주 추천 슬롯 1~2 선택 — 같은 단과대 조합 우선, 시너지 상위 k.
    6. cross-college 슬롯 (학과 경계) + 학부 cross-dept fallback +
       MMR 흘림 fallback. 감사 로그는 단일 위치에만 기록.
    7. MMR 으로 나머지 보조 추천 슬롯 채움.
    8. state 부분 반환.

부작용 격리:
    - ``track_repo`` 호출은 ``__call__`` 단 1곳.
    - ``logger.info`` 는 cross-college fallback 발생 위치 단 1곳.
    - 순수 계산 함수는 모두 module-level — mock 없이 테스트 가능.

단일 임베딩 boundary (ADR-0001) 준수:
    - 트랙 메타 cosine 은 사전 임베딩된 ``Track.meta_vector`` 만 사용한다.
      본 노드 안에서 새로 임베딩 호출하지 않는다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np

from tracktory.common.tech_keywords import canonical_tech_keys
from tracktory.graph.models import (
    JobCandidate,
    RankedCombo,
    SimilarityConfig,
    SynergyConfig,
    Track,
    TrackCombo,
    WeightsConfig,
)
from tracktory.graph.state import GraphState
from tracktory.rag.track_repository import TrackRepository

_DEFAULT_SYNERGY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "synergy.yaml"

logger = logging.getLogger(__name__)


class _ScoredCombo(NamedTuple):
    """후보 조합 + 시너지 점수의 중간 표현.

    ``RankedCombo`` 는 ``slot_type`` / ``rank`` 가 필수 필드라 슬롯 분류 이전 단계에서
    표현할 수 없다. 본 NamedTuple 은 슬롯 분류 직전까지의 가벼운 캐리어다.
    """

    combo: TrackCombo
    synergy_score: float


# ---------------------------------------------------------------------------
# 후보 생성
# ---------------------------------------------------------------------------


def _generate_combos(
    tracks: list[Track],
    user_college_id: str,
    current_tracks: list[str],
) -> list[TrackCombo]:
    """후보 트랙 조합을 생성한다.

    1트랙 주전공 제약:
        - ``current_tracks`` 가 비어있지 않으면 (2학년+): 사용자가 이미 선택한
          트랙을 1트랙 풀로 사용한다 (현재 트랙 기반 시너지 분석).
        - 비어있으면 (1학년): 사용자 단과대 소속 트랙을 1트랙 풀로 사용한다
          (조합 신규 추천).

    self-pair (``track_a == track_b``) 는 제외하며, 두 트랙 ID 의 정렬된 조합으로
    dedup 한다.
    """
    by_id = {track.track_id: track for track in tracks}

    if current_tracks:
        primary_pool: list[Track] = [by_id[tid] for tid in current_tracks if tid in by_id]
    else:
        primary_pool = [track for track in tracks if track.college_id == user_college_id]

    seen: set[str] = set()
    combos: list[TrackCombo] = []
    for primary_track in primary_pool:
        for partner in tracks:
            if primary_track.track_id == partner.track_id:
                continue
            ids_sorted = sorted([primary_track.track_id, partner.track_id])
            key = "::".join(ids_sorted)
            if key in seen:
                continue
            seen.add(key)
            combos.append(TrackCombo(track_a=primary_track, track_b=partner, combo_key=key))
    return combos


# ---------------------------------------------------------------------------
# 시너지 점수
# ---------------------------------------------------------------------------


def _complementarity(combo: TrackCombo, jobs: list[JobCandidate]) -> float:
    """두 트랙이 직무 요구 토큰을 분업하는 정도 (job-anchored division of labor).

    각 트랙의 기여 = 트랙 토큰(역량·기술스택)과 직무 토큰의 교집합. 직무가 요구하는
    토큰(jobs 의 기술스택·역량 태그)을 공유 기준으로 삼아, 두 트랙이 서로 다른 직무
    토큰을 공급할수록 높고, 한 트랙이 직무와 무관(기여 0)하거나 둘이 같은 토큰만
    덮으면(중복) 낮다.

    분모는 두 트랙 기여의 합집합이 아니라 **전체 직무 토큰 집합**이다. 합집합을 분모로
    쓰면 두 기여가 서로소이기만 하면 크기와 무관하게 항상 1.0 이 된다 — 직무 토큰이
    아무리 많아도 단 1 개를 공급하는 트랙이 6 개를 공급하는 트랙과 짝지으면 그 1 개가
    완전 분업(1.0)으로 둔갑해, 직무를 거의 못 덮는 무관 트랙이 상위 추천을 점령한다.
    전체 직무 토큰을 분모로 삼으면 comp 가 "직무의 몇 %를 둘이 겹침 없이 분담하는가"
    가 되어 크기에 민감해진다. 직무 전체를 둘이 비겹침으로 완전히 나눠 가질 때만
    1.0 에 도달하고, 일부만 분담하면 그만큼 작아진다.

    토큰 비교는 직무 커버율과 같은 정합 키로 통일한다 — 직무·트랙이 같은 기술을 다른
    표기로 적어도(예: "ReactJS" vs "React") 한 토큰으로 묶여야 분업·중복 판정이 표기
    차이에 흔들리지 않는다. 사전에 없는 역량 구문은 대소문자만 정규화돼 그대로 비교된다.

    Returns:
        ``[0, 1]``. 직무 토큰이 없거나 한 트랙이라도 직무 기여가 0 이면 0.0(분업의
        전제는 양쪽 모두의 기여). 둘의 직무 기여가 동일하면 0.0(중복). 둘이 직무 토큰을
        겹침 없이 나눠 공급할수록 커지며, 직무 전체를 완전 분담할 때 1.0.
    """
    job_tokens: set[str] = set()
    for job in jobs:
        job_tokens |= canonical_tech_keys(job.tech_stacks)
        job_tokens |= canonical_tech_keys(job.competency_tags)
    if not job_tokens:
        return 0.0

    contrib_a = canonical_tech_keys([*combo.track_a.competencies, *combo.track_a.tech_stacks])
    contrib_a &= job_tokens
    contrib_b = canonical_tech_keys([*combo.track_b.competencies, *combo.track_b.tech_stacks])
    contrib_b &= job_tokens
    if not contrib_a or not contrib_b:
        return 0.0

    return len(contrib_a ^ contrib_b) / len(job_tokens)


def _job_coverage(combo: TrackCombo, jobs: list[JobCandidate]) -> float:
    """직무 후보의 채용공고 기술스택 중 두 트랙 합집합으로 커버되는 비율.

    직무와 트랙은 같은 기술을 서로 다른 표기로 적어둘 수 있어 (예: "spring boot"
    vs "Spring Boot", "ReactJS" vs "React"), 양쪽을 정합 토큰 키로 통합한 뒤
    교집합을 센다. 통합 없이 원본 표기로 비교하면 표기 불일치로 커버율이 0 에
    수렴해 트랙 간 시너지 점수가 평탄해진다.
    """
    job_keys = canonical_tech_keys(stack for job in jobs for stack in job.tech_stacks)
    if not job_keys:
        return 0.0
    track_keys = canonical_tech_keys(combo.track_a.tech_stacks) | canonical_tech_keys(
        combo.track_b.tech_stacks
    )
    return len(job_keys & track_keys) / len(job_keys)


def _redundancy(combo: TrackCombo) -> float:
    """두 트랙의 과목 중복도 (Jaccard similarity)."""
    courses_a = set(combo.track_a.course_ids)
    courses_b = set(combo.track_b.course_ids)
    union = courses_a | courses_b
    if not union:
        return 0.0
    return len(courses_a & courses_b) / len(union)


def _synergy_score(
    combo: TrackCombo,
    jobs: list[JobCandidate],
    weights: WeightsConfig,
) -> float:
    """가중 합산 후 ``[0, 1]`` 로 clip 한 시너지 점수."""
    raw = (
        weights.complementarity * _complementarity(combo, jobs)
        + weights.coverage * _job_coverage(combo, jobs)
        - weights.redundancy * _redundancy(combo)
    )
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Slot Reservation — primary / cross-college / fallback
# ---------------------------------------------------------------------------


def _select_primary(scored: list[_ScoredCombo], k: int) -> list[RankedCombo]:
    """주 추천 k 개 — 같은 단과대 조합을 우선하고 시너지 상위로 채운다.

    주 추천은 사용자 단과대 내부의 직무 적합 조합이어야 한다. 학과 경계를 넘는 이색
    조합은 보조 추천의 예약 슬롯이 따로 보장하므로, 주 추천까지 cross-college 후보를
    허용하면 직무와 거의 무관한 타 단과대 트랙이 상위를 점령하고 그로부터 파생되는
    학습 로드맵까지 무관 과목으로 채워진다. 후보 생성 단계는 1트랙만 사용자 단과대로
    제약하고 2트랙(partner)은 전 트랙을 허용하므로, 여기서 두 트랙이 같은 단과대인
    조합으로 한 번 더 좁혀야 주 추천이 단과대 내부로 닫힌다.

    같은 단과대 조합이 k 개에 못 미치면(예: 단일 트랙 단과대) 나머지를 전체 후보의
    시너지 상위로 backfill 하여 항상 가능한 만큼 채운다. backfill 로 채워진 자리는
    cross-college 조합일 수 있다 — 같은 단과대 후보가 바닥났을 때는 빈 주 추천보다
    낫기 때문이며, 이때만 주 추천이 단과대 내부로 닫힌다는 보장이 풀린다. 동점은
    ``combo_key`` 알파벳 순으로 deterministic 결정.
    """

    def _rank_key(cand: _ScoredCombo) -> tuple[float, str]:
        return (-cand.synergy_score, cand.combo.combo_key)

    same_college = [
        cand for cand in scored if cand.combo.track_a.college_id == cand.combo.track_b.college_id
    ]
    ordered = sorted(same_college, key=_rank_key)
    if len(ordered) < k:
        chosen_keys = {cand.combo.combo_key for cand in ordered}
        backfill = sorted(
            (cand for cand in scored if cand.combo.combo_key not in chosen_keys),
            key=_rank_key,
        )
        ordered = [*ordered, *backfill]

    return [
        RankedCombo(
            combo=cand.combo,
            synergy_score=cand.synergy_score,
            slot_type="primary",
            rank=i + 1,
        )
        for i, cand in enumerate(ordered[:k])
    ]


def _is_cross_college(combo: TrackCombo, primary: list[RankedCombo]) -> bool:
    """combo 의 두 트랙 중 최소 한 개의 단과대가 primary 에 없는 경우 True."""
    primary_colleges = {
        track.college_id
        for ranked in primary
        for track in (ranked.combo.track_a, ranked.combo.track_b)
    }
    combo_colleges = {combo.track_a.college_id, combo.track_b.college_id}
    return bool(combo_colleges - primary_colleges)


def _is_cross_department(combo: TrackCombo, primary: list[RankedCombo]) -> bool:
    """학부 단위 (T2) cross 판정 — T1 fallback 시 사용."""
    primary_depts = {
        track.department_id
        for ranked in primary
        for track in (ranked.combo.track_a, ranked.combo.track_b)
    }
    combo_depts = {combo.track_a.department_id, combo.track_b.department_id}
    return bool(combo_depts - primary_depts)


def _select_cross_college_slot(
    scored: list[_ScoredCombo],
    primary: list[RankedCombo],
    min_cross_synergy: float,
) -> tuple[RankedCombo | None, Literal["T2", "MMR"] | None]:
    """학과 경계 슬롯 + T1 → T2 → MMR 흘림 fallback.

    Returns:
        ``(슬롯3 RankedCombo 또는 None, fallback level)``.
        fallback level == ``None``: 단과대 단위 cross 후보 정상 채택.
        fallback level == ``"T2"``: 단과대 cross 후보 0개 → 학부 cross-dept 채택.
        fallback level == ``"MMR"``: 학부 cross 도 0개 → 슬롯3 = ``None``,
            MMR 후보군에 흘려보냄.
    """
    primary_keys = {r.combo.combo_key for r in primary}
    eligible = [
        cand
        for cand in scored
        if cand.combo.combo_key not in primary_keys and cand.synergy_score >= min_cross_synergy
    ]

    t1 = [cand for cand in eligible if _is_cross_college(cand.combo, primary)]
    if t1:
        winner = sorted(t1, key=lambda cand: (-cand.synergy_score, cand.combo.combo_key))[0]
        return (
            RankedCombo(
                combo=winner.combo,
                synergy_score=winner.synergy_score,
                slot_type="cross_college",
                rank=3,
            ),
            None,
        )

    t2 = [cand for cand in eligible if _is_cross_department(cand.combo, primary)]
    if t2:
        winner = sorted(t2, key=lambda cand: (-cand.synergy_score, cand.combo.combo_key))[0]
        return (
            RankedCombo(
                combo=winner.combo,
                synergy_score=winner.synergy_score,
                slot_type="cross_college",
                rank=3,
            ),
            "T2",
        )

    return (None, "MMR")


# ---------------------------------------------------------------------------
# sim_4tier + MMR
# ---------------------------------------------------------------------------


def _has_shared_attr(combo_a: TrackCombo, combo_b: TrackCombo, attr: str) -> float:
    """두 조합의 트랙 attr 집합에 공통 원소가 있으면 1.0, 없으면 0.0."""
    attrs_a = {getattr(combo_a.track_a, attr), getattr(combo_a.track_b, attr)}
    attrs_b = {getattr(combo_b.track_a, attr), getattr(combo_b.track_b, attr)}
    return 1.0 if attrs_a & attrs_b else 0.0


def _course_overlap_ratio(combo_a: TrackCombo, combo_b: TrackCombo) -> float:
    """두 조합의 모든 과목 합집합 대비 교집합 비율."""
    courses_a = set(combo_a.track_a.course_ids) | set(combo_a.track_b.course_ids)
    courses_b = set(combo_b.track_a.course_ids) | set(combo_b.track_b.course_ids)
    union = courses_a | courses_b
    if not union:
        return 0.0
    return len(courses_a & courses_b) / len(union)


def _combo_meta_vector(combo: TrackCombo) -> np.ndarray | None:
    """조합 대표 메타 벡터 — 두 트랙 메타 벡터의 평균.

    벡터 부재·dim 불일치 시 ``None`` 반환 (cosine 0 으로 처리).
    """
    raw_a = combo.track_a.meta_vector
    raw_b = combo.track_b.meta_vector
    if not raw_a or not raw_b:
        return None
    vec_a = np.asarray(raw_a, dtype=float)
    vec_b = np.asarray(raw_b, dtype=float)
    if vec_a.shape != vec_b.shape:
        return None
    mean: np.ndarray = (vec_a + vec_b) / 2.0
    return mean


def _meta_cosine(combo_a: TrackCombo, combo_b: TrackCombo) -> float:
    """두 조합의 대표 메타 벡터 사이 코사인 유사도.

    트랙 메타 벡터는 단일 임베딩 boundary (ADR-0001) 에서 L2 normalized 상태로
    적재된다. 평균 후에는 norm 이 변할 수 있어 안전하게 재 normalize 한다.
    """
    vec_a = _combo_meta_vector(combo_a)
    vec_b = _combo_meta_vector(combo_b)
    if vec_a is None or vec_b is None:
        return 0.0
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _sim_4tier(combo_a: TrackCombo, combo_b: TrackCombo, sim_cfg: SimilarityConfig) -> float:
    """4-tier hierarchy 일치도 + 과목 중복도 + 메타 cos 의 5항 가중합."""
    return (
        sim_cfg.w_college * _has_shared_attr(combo_a, combo_b, "college_id")
        + sim_cfg.w_department * _has_shared_attr(combo_a, combo_b, "department_id")
        + sim_cfg.w_track * _has_shared_attr(combo_a, combo_b, "major_id")
        + sim_cfg.w_course_overlap * _course_overlap_ratio(combo_a, combo_b)
        + sim_cfg.w_meta * _meta_cosine(combo_a, combo_b)
    )


def _mmr_select(
    candidates: list[_ScoredCombo],
    selected: list[RankedCombo],
    n: int,
    lambda_: float,
    sim_cfg: SimilarityConfig,
    start_rank: int,
) -> list[RankedCombo]:
    """Carbonell-Goldstein 1998 MMR 으로 n 개 슬롯을 채운다.

    매 단계마다::

        next = argmax_{i in candidates - selected}
            [ lambda * synergy(i) - (1 - lambda) * max_{j in selected} sim(i, j) ]

    동점 시 ``combo_key`` 알파벳 순으로 deterministic 선택.
    """
    selected_keys = {r.combo.combo_key for r in selected}
    pool = [cand for cand in candidates if cand.combo.combo_key not in selected_keys]
    selected_combos: list[TrackCombo] = [r.combo for r in selected]

    chosen: list[RankedCombo] = []
    rank_counter = start_rank

    while pool and len(chosen) < n:

        def _mmr_score(
            cand: _ScoredCombo,
            previously_selected: list[TrackCombo] = selected_combos,
        ) -> float:
            if not previously_selected:
                return lambda_ * cand.synergy_score
            max_sim = max(_sim_4tier(cand.combo, prev, sim_cfg) for prev in previously_selected)
            return lambda_ * cand.synergy_score - (1.0 - lambda_) * max_sim

        winner = sorted(pool, key=lambda cand: (-_mmr_score(cand), cand.combo.combo_key))[0]
        chosen.append(
            RankedCombo(
                combo=winner.combo,
                synergy_score=winner.synergy_score,
                slot_type="mmr",
                rank=rank_counter,
            )
        )
        selected_combos.append(winner.combo)
        pool.remove(winner)
        rank_counter += 1

    return chosen


# ---------------------------------------------------------------------------
# 노드 진입점
# ---------------------------------------------------------------------------


class TrackSynergyNode:
    """트랙 시너지 노드 — 직무 매칭 결과로부터 트랙 조합 7 슬롯 산출.

    의존성을 ``__init__`` 으로 주입받으므로 단위 테스트에서 ``TrackRepository`` 를
    mock 으로 교체하면 외부 I/O 없이 검증 가능하다.
    """

    def __init__(
        self,
        track_repo: TrackRepository,
        config_path: Path | None = None,
    ) -> None:
        self._repo = track_repo
        self._config = SynergyConfig.load_from_yaml(config_path or _DEFAULT_SYNERGY_CONFIG_PATH)

    def __call__(self, state: GraphState) -> dict[str, Any]:
        # 단계 1: 입력 검증
        recommended_jobs_raw: list[dict[str, Any]] | None = state.get("recommended_jobs")
        normalized: dict[str, Any] | None = state.get("normalized_profile")
        if not recommended_jobs_raw or not normalized:
            return {
                "errors": ["track_synergy skipped: recommended_jobs or normalized_profile missing"],
                "trace": ["track_synergy:skip"],
            }

        college_id = normalized.get("college")
        if not college_id:
            return {
                "errors": ["track_synergy skipped: college missing in normalized_profile"],
                "trace": ["track_synergy:skip"],
            }

        current_tracks = normalized.get("current_tracks") or []
        jobs = [JobCandidate.model_validate(j) for j in recommended_jobs_raw]

        # 단계 2: 트랙 전체 로드 (외부 I/O — 진입점 단 1곳)
        tracks = self._repo.list_all()

        # 단계 3: 후보 조합 생성
        combos = _generate_combos(tracks, college_id, current_tracks)
        if not combos:
            return {
                "errors": ["track_synergy skipped: no candidate combos generated"],
                "trace": ["track_synergy:skip"],
            }

        # 단계 4: 시너지 점수
        scored = [
            _ScoredCombo(
                combo=c,
                synergy_score=_synergy_score(c, jobs, self._config.weights),
            )
            for c in combos
        ]

        # 단계 5: 주 추천 슬롯
        primary = _select_primary(scored, self._config.slots.primary_count)

        # 단계 6: cross-college 슬롯 + fallback
        slot3, fallback_level = _select_cross_college_slot(
            scored, primary, self._config.slots.min_cross_synergy
        )
        if fallback_level is not None:
            logger.info(
                "track_synergy_cross_college_fallback",
                extra={
                    "fallback_level": fallback_level,
                    "primary_combo_keys": [r.combo.combo_key for r in primary],
                    "min_cross_synergy": self._config.slots.min_cross_synergy,
                },
            )

        # 단계 7: MMR 으로 나머지 보조 슬롯
        secondary_count = self._config.slots.secondary_count
        if slot3 is not None:
            mmr_slots = _mmr_select(
                candidates=scored,
                selected=[*primary, slot3],
                n=secondary_count - 1,
                lambda_=self._config.mmr.lambda_value,
                sim_cfg=self._config.similarity,
                start_rank=4,
            )
            secondary = [slot3, *mmr_slots]
        else:
            mmr_slots = _mmr_select(
                candidates=scored,
                selected=primary,
                n=secondary_count,
                lambda_=self._config.mmr.lambda_value,
                sim_cfg=self._config.similarity,
                start_rank=3,
            )
            secondary = mmr_slots

        # 단계 8: state 부분 반환
        trace_tokens = ["track_synergy:ok"]
        if fallback_level is not None:
            trace_tokens.append(f"track_synergy:cross_college_fallback:{fallback_level}")

        return {
            "primary_combos": [r.model_dump(mode="json") for r in primary],
            "secondary_combos": [r.model_dump(mode="json") for r in secondary],
            "slot3_fallback_triggered": fallback_level is not None,
            "slot3_fallback_level": fallback_level,
            "trace": trace_tokens,
        }


__all__ = ["TrackRepository", "TrackSynergyNode"]
