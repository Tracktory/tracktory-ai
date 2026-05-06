"""트랙 시너지 노드 + 직무 매칭 노드의 공통 도메인 모델.

7 종 Pydantic 모델:

- ``Track`` — 한성대 단일 트랙 메타데이터 (4-tier hierarchy + 메타 텍스트·벡터).
- ``JobCandidate`` — 직무 매칭 노드의 결과 단건.
- ``TrackCombo`` — 두 트랙의 조합 단위.
- ``RankedCombo`` — ``TrackCombo`` 에 시너지 점수·슬롯 분류·순위가 부착된 단위.
- ``SynergyConfig`` — ``synergy.yaml`` 의 시너지 외부화 설정 (4 nested config + 단조 제약).
- ``JobMatchingConfig`` — ``synergy.yaml`` 의 ``job_matching`` 섹션 외부화 매핑.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Track(BaseModel):
    """한성대 단일 트랙 메타데이터.

    4-tier hierarchy (단과대 → 학부 → 전공 → 트랙) 식별자로 학사 구조를 표현한다.
    단일 트랙만 운영하는 학과는 ``major_id`` 를 ``department_id`` 와 동일하게 두어
    동일 모델로 degenerate 표현이 가능하다.

    Attributes:
        college_id: 단과대 식별자 (T1).
        department_id: 학부 식별자 (T2).
        major_id: 전공 식별자 (T3). single-track 학과는 ``department_id`` 와 동일.
        track_id: 트랙 식별자 (T4 단위).
        track_name: 사용자 표시용 트랙명.
        course_ids: 트랙 커리큘럼 과목 식별자. 두 트랙 간 과목 중복도 계산에 사용.
        meta_text: 학과 소개·트랙 핵심 가치·졸업 후 진로 결합 텍스트 (디버깅용 보존).
        meta_vector: 사전 임베딩된 트랙 메타 벡터. 단일 임베딩 boundary (ADR-0001) 가
            보장하는 동일 임베딩 공간의 벡터로, L2 normalized 상태를 전제한다.
        competencies: 트랙이 양성하는 역량 태그. 두 트랙의 상호 보완성 계산에 사용.
        tech_stacks: 트랙 커리큘럼 기반 기술 스택. 직무 채용공고 기술스택과의
            합집합 커버율 계산에 사용.
    """

    college_id: str = Field(..., min_length=1)
    department_id: str = Field(..., min_length=1)
    major_id: str = Field(..., min_length=1)
    track_id: str = Field(..., min_length=1)
    track_name: str = Field(..., min_length=1)
    course_ids: list[str] = Field(default_factory=list)
    meta_text: str = ""
    meta_vector: list[float] = Field(default_factory=list)
    competencies: list[str] = Field(default_factory=list)
    tech_stacks: list[str] = Field(default_factory=list)


class JobCandidate(BaseModel):
    """직무 매칭 노드의 결과 단건.

    ``tech_stacks`` 는 후속 트랙 시너지 계산의 직무 도달도 (job coverage)
    분모로 흐른다. ``similarity`` 와 ``match_score`` 는 같은 값 (코사인 유사도
    또는 fallback 시 0.0) 으로 채워지며, 두 필드 동시 보존은 다운스트림
    노드가 어느 키를 참조해도 동일하게 동작하도록 보장한다.

    Attributes:
        job_id: 직무 식별자.
        job_name: 사용자 표시용 직무명.
        tech_stacks: 채용공고 기술스택.
        competency_tags: 직무가 요구하는 역량 태그.
        match_score: 사용자-직무 적합도 ([0, 1]). 다운스트림 트랙 시너지
            노드와의 호환을 유지한다.
        similarity: 직무 매칭 노드가 채우는 코사인 유사도 또는 fallback 시
            0.0 ([0, 1]).
        fallback_used: True 이면 카테고리 사전 매핑 fallback 으로 채택된
            후보. LLM 설명 단계가 사용자에게 캐비잇 메시지를 추가할 때
            본 플래그를 본다.
    """

    job_id: str = Field(..., min_length=1)
    job_name: str = Field(..., min_length=1)
    tech_stacks: list[str] = Field(default_factory=list)
    competency_tags: list[str] = Field(default_factory=list)
    match_score: float = Field(..., ge=0.0, le=1.0)
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    fallback_used: bool = False


class TrackCombo(BaseModel):
    """두 트랙의 조합 단위.

    조합 식별자 ``combo_key`` 는 두 ``track_id`` 의 정렬된 튜플 기반 deterministic
    문자열로, ``(a, b)`` 와 ``(b, a)`` 를 동일하게 dedup 처리하기 위한 키다.

    Attributes:
        track_a: 1트랙. 후보 생성 단계에서 사용자 단과대 소속 트랙으로 제약된다.
        track_b: 2트랙. 자유 선택 (전체 트랙 대상).
        combo_key: 정렬된 deterministic 조합 식별자.
    """

    model_config = ConfigDict(frozen=True)

    track_a: Track
    track_b: Track
    combo_key: str = Field(..., min_length=1)


class RankedCombo(BaseModel):
    """``TrackCombo`` 에 시너지 점수·슬롯 분류·순위가 부착된 단위.

    트랙 시너지 노드의 출력 단위로, 그래프 state 에는 ``model_dump()`` 결과 dict
    형태로 보관된다.

    Attributes:
        combo: 트랙 조합.
        synergy_score: 가중 합산 후 [0, 1] 로 clip 된 시너지 점수.
        slot_type: 슬롯 분류 — ``primary`` (주 추천 1~2 위) / ``cross_college``
            (보조 추천 중 학과 경계를 넘는 예약 슬롯) / ``mmr`` (보조 추천 중
            다양성 균형 슬롯).
        rank: 1~7 사이 순위. 주 추천 2 + 보조 추천 5 합 7 슬롯 중 한 자리.
    """

    combo: TrackCombo
    synergy_score: float = Field(..., ge=0.0, le=1.0)
    slot_type: Literal["primary", "cross_college", "mmr"]
    rank: int = Field(..., ge=1, le=7)


# ---------------------------------------------------------------------------
# SynergyConfig — synergy.yaml 외부화 매핑
# ---------------------------------------------------------------------------


class WeightsConfig(BaseModel):
    """시너지 점수 가중치.

    ``synergy = clip(complementarity · comp + coverage · cov - redundancy · red, 0, 1)``.

    Attributes:
        complementarity: 두 트랙의 상호 보완성 가중치.
        coverage: 직무 채용공고 기술스택 도달도 가중치.
        redundancy: 두 트랙의 학습 내용 중복도 페널티 가중치.
    """

    complementarity: float = Field(..., ge=0.0, le=1.0)
    coverage: float = Field(..., ge=0.0, le=1.0)
    redundancy: float = Field(..., ge=0.0, le=1.0)


class SimilarityConfig(BaseModel):
    """``sim_4tier`` 5항 가중치.

    ``sim_4tier = w_college · same_T1 + w_department · same_T2 + w_track · same_T3``
    ``+ w_course_overlap · overlap_ratio + w_meta · cos(meta_a, meta_b)``.

    상위 tier 가 더 강한 신호여야 하므로 ``w_college >= w_department >= w_track`` 의
    단조 제약을 모델 검증으로 강제한다. ``w_meta`` 는 tier 외 별도 축
    (도메인·서사 거리) 이라 단조 제약 대상이 아니다.

    Attributes:
        w_college: T1 가중치.
        w_department: T2 가중치.
        w_track: T3 가중치.
        w_course_overlap: T4 과목 중복도 가중치.
        w_meta: 트랙 메타 텍스트 임베딩 cosine 유사도 가중치.
    """

    w_college: float = Field(..., ge=0.0, le=1.0)
    w_department: float = Field(..., ge=0.0, le=1.0)
    w_track: float = Field(..., ge=0.0, le=1.0)
    w_course_overlap: float = Field(..., ge=0.0, le=1.0)
    w_meta: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _enforce_tier_monotone(self) -> Self:
        if not (self.w_college >= self.w_department >= self.w_track):
            raise ValueError(
                "tier weights must be monotone non-increasing: "
                f"w_college ({self.w_college}) >= w_department ({self.w_department}) "
                f">= w_track ({self.w_track})"
            )
        return self


class MmrConfig(BaseModel):
    """MMR 파라미터.

    ``next = argmax_i [ λ · synergy(i) - (1 - λ) · max sim_4tier(i, j) ]``.

    yaml 의 키 ``lambda`` 는 Python 예약어이므로 ``lambda_value`` 로 alias 한다.

    Attributes:
        lambda_value: synergy 와 다양성의 균형 ([0, 1]). 1 이면 순수 시너지,
            0 이면 순수 다양성.
    """

    model_config = ConfigDict(populate_by_name=True)

    lambda_value: float = Field(..., alias="lambda", ge=0.0, le=1.0)


class SlotsConfig(BaseModel):
    """슬롯 예약 설정.

    Attributes:
        primary_count: 주 추천 슬롯 수.
        secondary_count: 보조 추천 슬롯 수.
        cross_college_reserved: 학과 경계를 넘는 예약 슬롯 수 (보조 추천 안에서).
        min_cross_synergy: cross-college 슬롯 후보의 최소 시너지 임계값.
            학부 cross-dept fallback 시에도 동일 임계값을 적용한다.
    """

    primary_count: int = Field(..., ge=1)
    secondary_count: int = Field(..., ge=1)
    cross_college_reserved: int = Field(..., ge=0)
    min_cross_synergy: float = Field(..., ge=0.0, le=1.0)


class SynergyConfig(BaseModel):
    """``synergy.yaml`` 의 외부화 설정 1:1 매핑.

    yaml 로딩은 ``load_from_yaml`` classmethod 에 격리하여 노드 호출 경로에서
    파일 I/O 를 단일 위치로 모은다.
    """

    weights: WeightsConfig
    similarity: SimilarityConfig
    mmr: MmrConfig
    slots: SlotsConfig

    @classmethod
    def load_from_yaml(cls, path: Path) -> Self:
        """yaml 파일에서 ``SynergyConfig`` 를 로드한다.

        Args:
            path: ``synergy.yaml`` 의 경로.

        Returns:
            검증된 ``SynergyConfig`` 인스턴스.

        Raises:
            ValueError: yaml 이 mapping 이 아니거나 단조 제약을 위배할 때.
        """
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Synergy config file {path} must define a top-level mapping")
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# JobMatchingConfig — synergy.yaml 의 job_matching 섹션 외부화 매핑
# ---------------------------------------------------------------------------


class JobMatchingTopKConfig(BaseModel):
    """직무 매칭 노드의 top-k 정책.

    Attributes:
        default: 기본 응답 후보 수 (첫 화면 노출용).
        expanded: 사용자가 "더 보기" 를 요청했을 때의 후보 수.
    """

    default: int = Field(..., ge=1)
    expanded: int = Field(..., ge=1)


class JobMatchingConfig(BaseModel):
    """``synergy.yaml`` 의 ``job_matching`` 섹션 1:1 매핑.

    트랙 시너지와 같은 yaml 파일을 공유하여 가중치·임계값 외부화 단위를
    단순화한다. 본 클래스는 그 중 ``job_matching`` 매핑만 추출한다.

    Attributes:
        top_k: 응답 후보 수 정책.
        min_job_similarity: 코사인 유사도 하위 컷. max similarity 가 이 값
            미만이면 카테고리 사전 매핑 fallback 으로 전환된다.
    """

    top_k: JobMatchingTopKConfig
    min_job_similarity: float = Field(..., ge=0.0, le=1.0)

    @classmethod
    def load_from_yaml(cls, path: Path) -> Self:
        """yaml 파일에서 ``JobMatchingConfig`` 를 로드한다.

        Args:
            path: ``synergy.yaml`` 의 경로.

        Returns:
            검증된 ``JobMatchingConfig`` 인스턴스.

        Raises:
            ValueError: yaml 이 mapping 이 아니거나 ``job_matching`` 키가
                없거나 mapping 이 아닐 때.
        """
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Synergy config file {path} must define a top-level mapping")
        section = raw.get("job_matching")
        if not isinstance(section, dict):
            raise ValueError(f"Synergy config file {path} must contain a 'job_matching' mapping")
        return cls.model_validate(section)
