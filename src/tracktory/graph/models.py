"""추천 파이프라인 노드들의 공통 도메인 모델.

Pydantic 모델 목록:

- ``Track`` — 한성대 단일 트랙 메타데이터 (4-tier hierarchy + 메타 텍스트·벡터).
- ``JobCandidate`` — 직무 매칭 노드의 결과 단건.
- ``TrackCombo`` — 두 트랙의 조합 단위.
- ``RankedCombo`` — ``TrackCombo`` 에 시너지 점수·슬롯 분류·순위가 부착된 단위.
- ``Course`` — 학습 로드맵 노드의 입력 단위 (Repository 가 채워 반환).
- ``RoadmapCourse`` — 학습 로드맵 안의 추천 과목 단위.
- ``RoadmapStage`` — 학습 로드맵의 한 학습 깊이 단계 (기초·핵심·응용·산학).
- ``SemesterPlan`` — 학생의 잔여 학기 한 학기 단위 추천 과목 계획.
- ``Roadmap`` — 학습 깊이 라벨 + 학기 분산 plan 의 이중 출력 단위.
- ``ExplanationSection`` — LLM 자연어 설명의 주제별 단락.
- ``SemesterSubtitle`` — 학기 카드 헤더용 학기 단위 부제.
- ``CourseFlow`` — 과목 상세 모달용 과목 단위 인과 흐름.
- ``JobRationale`` — 직무 한 건 단위 개별 근거.
- ``TrackRationale`` — 트랙 조합 한 건 단위 개별 근거 (조합 전체 + 트랙별).
- ``Explanation`` — LLM 자연어 설명 전체.
- ``SynergyConfig`` — 시너지 외부화 설정 (4 nested config + 단조 제약).
- ``CompletedCourseBoostConfig`` — 이수 과목 부스팅 강도 정책.
- ``JobMatchingConfig`` — 직무 매칭 노드의 외부화 매핑.
- ``RoadmapConfig`` — 학습 로드맵 외부화 설정 (학기 용량 + 졸업 요건 + 학년별 학점 범위).
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
    분모로 흐른다. 두 점수 필드는 역할이 분리된다 — ``similarity`` 는 외부
    검색 단계의 원시 점수이고, ``match_score`` 는 이수 과목 부스팅 같은
    후처리가 반영된 최종 적합도다. 검색 직후나 부스팅이 0 일 때는 두 값이
    일치하지만, 항상 같다는 보장은 없다.

    Attributes:
        job_id: 직무 식별자.
        job_name: 사용자 표시용 직무명.
        tech_stacks: 채용공고 기술스택.
        competency_tags: 직무가 요구하는 역량 태그.
        match_score: 사용자-직무 적합도 ([0, 1]). 검색 점수에서 출발해 이수
            과목 부스팅이 가산된 최종 값. 후보 정렬·표시의 기준.
        similarity: 외부 검색 단계의 원시 결합 점수 (hybrid + reranker)
            ([0, 1]). 이수 과목 부스팅 등 후처리의 영향을 받지 않는다.
            fallback 시 0.0.
        fallback_used: True 이면 카테고리 사전 매핑 fallback 으로 채택된
            후보. LLM 설명 단계가 사용자에게 캐비잇 메시지를 추가할 때
            본 플래그를 본다.
        posting_count: 직무 검색에서 본 직무 타입으로 집계된 공고 수
            (출현 횟수). 검색 결과 내 직무 타입의 등장 빈도 신호다. fallback
            후보는 공고 검색을 거치지 않으므로 0.
    """

    job_id: str = Field(..., min_length=1)
    job_name: str = Field(..., min_length=1)
    tech_stacks: list[str] = Field(default_factory=list)
    competency_tags: list[str] = Field(default_factory=list)
    match_score: float = Field(..., ge=0.0, le=1.0)
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    fallback_used: bool = False
    posting_count: int = Field(default=0, ge=0)


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
# Roadmap / Explanation — 학습 로드맵 + LLM 자연어 설명 도메인 모델
# ---------------------------------------------------------------------------


class Course(BaseModel):
    """학습 로드맵 노드의 입력 단위 — Repository 가 채워 반환한다.

    Repository 책임:
        - ``stage`` 분류 — 강의계획서·커리큘럼 메타로부터 학습 깊이 4 단계
          (foundation / core / application / industry) 중 하나를 도출한다.
        - ``prereq_ids`` 추출 — 선수과목 관계 그래프에서 본 과목이 의존하는
          선수 ``course_id`` 리스트를 정규화하여 채운다.
        - ``priority`` 할당 — 낮은 숫자가 우선. 학년·필수 여부·선수 깊이
          등으로 도출되며, 본 노드는 Repository 가 부여한 값을 그대로 사용한다.
        - ``available_grades`` / ``available_semesters`` 채움 — 교육과정 메타로부터
          본 과목을 이수할 수 있는 학년·학기 리스트를 채운다. 메타 부재 시
          학년은 ``[1, 2, 3, 4]``, 학기는 ``[1, ..., 8]`` 로 보수적 fallback.
        - ``course_type`` 분류 — 학사 커리큘럼 메타로부터 전공필수 / 전공선택 /
          교양 중 하나를 도출한다. 학습 로드맵 추천 대상은 전공 (필수 + 선택)
          만이며, 교양은 추천 전 stream 진입 단계에서 필터링된다.

    Attributes:
        course_id: 과목 식별자.
        course_name: 사용자 표시용 과목명.
        credits: 학점 (학기·졸업 학점 cap 의 단위).
        stage: 카탈로그가 부여한 4 단계 학습 깊이 분류 (입력 메타). 학습
            로드맵 출력의 단계 라벨은 본 값이 아니라 과목이 배치된 학기의
            학년에서 재도출한다 (산학 단계 공백 방지).
        prereq_ids: 선수과목의 정규화된 ``course_id`` 리스트.
        track_ids: 본 과목이 권장되는 트랙 식별자 리스트.
        priority: 추천 정렬의 보조 tie-break 신호 (1 이 최우선). 1 차 정렬은
            추천 점수가 담당하며, 점수가 같을 때 본 값으로 순서를 가린다.
        available_grades: 본 과목을 이수할 수 있는 학년 리스트. 1 학년 전용
            기초 과목은 ``[1]``, 학년 무관 과목은 ``[1, 2, 3, 4]``.
        available_semesters: 본 과목을 이수할 수 있는 절대 학기 번호 리스트.
            1 학년 1 학기 = 1, 4 학년 2 학기 = 8. 학기 메타 부재 시
            ``[1, ..., 8]``.
        course_type: 학사 커리큘럼 분류. 본 시스템 추천 대상은 전공필수 /
            전공선택만이며, 교양은 사용자 자율 구성 영역으로 추천에서 제외한다.
        tech_stacks: 강의계획서 산문에서 추출된 기술 토큰 (직무 기술 어휘와 정합).
            이수 과목 부스팅 시 직무 토큰과의 교집합 계산에 사용.
    """

    course_id: str = Field(..., min_length=1)
    course_name: str = Field(..., min_length=1)
    credits: int = Field(..., ge=1)
    stage: Literal["foundation", "core", "application", "industry"]
    prereq_ids: list[str] = Field(default_factory=list)
    track_ids: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1)
    available_grades: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    available_semesters: list[int] = Field(default_factory=lambda: list(range(1, 9)))
    course_type: Literal["전공필수", "전공선택", "교양"] = "전공선택"
    tech_stacks: list[str] = Field(default_factory=list)


class RoadmapCourse(BaseModel):
    """학습 로드맵에 노출되는 추천 과목 — 학점·단계·정렬 점수를 함께 싣는다.

    ``stage`` 는 학사 카탈로그가 부여한 분류가 아니라 과목이 실제로 배치된
    학기의 학년에서 도출한다 (1→foundation, 2→core, 3→application,
    4→industry). 카탈로그 단계만으로는 산학(industry) 과목 데이터가 비어
    산학 단계가 항상 공백이 되므로, 배치 학년을 단계 라벨의 단일 진실원으로
    삼아 모든 단계가 학생의 잔여 학기에 맞춰 채워지도록 한다.

    Attributes:
        course_id: 과목 식별자.
        course_name: 사용자 표시용 과목명.
        credits: 학점. 사용자 화면의 과목 단위 학점 표기에 매핑된다.
        stage: 배치 학년에서 도출한 학습 깊이 단계.
        score: 추천 정렬·표시 점수 (0~1). 전공 필수/선택 분류와 두 트랙
            동시 권장 여부를 합산한 값으로, 고정 순위가 아닌 실제 정렬 근거다.
    """

    course_id: str = Field(..., min_length=1)
    course_name: str = Field(..., min_length=1)
    credits: int = Field(..., ge=1)
    stage: Literal["foundation", "core", "application", "industry"]
    score: float = Field(..., ge=0.0, le=1.0)


class RoadmapStage(BaseModel):
    """학습 로드맵의 한 단계.

    학습 깊이를 단조 증가시키는 4 단계 분류 — foundation (기초)
    → core (핵심) → application (응용) → industry (산학). 산학은
    캡스톤·인턴십·기업 협업 과목을 포함한다.

    Attributes:
        stage: 단계 식별자.
        courses: 본 단계에서 추천하는 과목들. 비어 있어도 valid 하다
            (예: 1학년 사용자의 산학 단계 빈 상태).
    """

    stage: Literal["foundation", "core", "application", "industry"]
    courses: list[RoadmapCourse] = Field(default_factory=list)


class SemesterPlan(BaseModel):
    """한 학기 단위의 추천 과목 계획.

    학습 깊이 단계와는 별개의 축으로, 학생의 잔여 학기에 전공 과목을 분산
    배치하기 위해 사용한다. 한 학기에는 여러 단계의 과목이 섞일 수 있다
    (예: 1 학년 후반 학기에 기초 마지막 과목 + 핵심 첫 과목).

    학기당 학점은 학사 학기 cap (한성대 일반 학기 18 학점) 안에 있어야 하며,
    누적 학점이 졸업 전공 학점 (78) 에 도달한 마지막 학기에 ``cap_reached``
    가 켜진다. 잔여 학기가 너무 짧아 졸업 학점에 미달하면 마지막 학기에
    ``graduation_insufficient`` 가 켜져 학사 상담 권유 UI 가 노출된다.

    Attributes:
        semester: 학기 번호 (1 학년 1 학기 = 1, 4 학년 2 학기 = 8).
        grade: 학년 번호. ``(semester + 1) // 2`` 로 도출하지만 데이터로
            명시 보존하여 UI 그룹 헤더와 학년별 학점 검증 모두에 동일 값을
            참조한다.
        courses: 본 학기 추천 과목 (학습 깊이 라벨 혼재 가능).
        credits_total: 본 학기 누적 추천 학점.
        cap_reached: 누적 추천 학점이 졸업 전공 학점에 도달한 마지막 학기
            마커. UI 의 졸업 도달 라벨 노출 조건.
        graduation_insufficient: 잔여 학기가 너무 짧아 졸업 학점에 미달한
            마지막 학기 마커. 학사 상담 권유 UI 노출 조건.
    """

    semester: int = Field(..., ge=1, le=8)
    grade: int = Field(..., ge=1, le=4)
    courses: list[RoadmapCourse] = Field(default_factory=list)
    credits_total: int = Field(default=0, ge=0)
    cap_reached: bool = False
    graduation_insufficient: bool = False


class Roadmap(BaseModel):
    """학습 로드맵 전체 — 학습 깊이 라벨 + 학기 분산 plan 의 이중 출력.

    학습 깊이 4 단계 (foundation / core / application / industry) 라벨은
    자연어 설명 생성에 사용되고, 학기 단위 분산 (semesters) 은 사용자
    화면의 학기 카드 row 에 직접 매핑된다. 두 축은 같은 데이터의 다른
    뷰이며, 같은 과목이 양쪽에 동시에 등장한다.

    모델 검증은 4 단계가 정확히 ``foundation → core → application →
    industry`` 순서로 한 번씩 등장하도록 강제한다 (단계 누락 / 순서 뒤바뀜은
    의미가 없으며, 각 단계의 과목 리스트는 비어 있을 수 있다). 학기 분산은
    별도 검증 없이 학기 분산 알고리즘이 invariant 를 보장한다.

    Attributes:
        stages: 학습 깊이 4 단계. 자연어 설명 노드가 단계별 과목 그룹화에
            사용한다.
        semesters: 학생의 잔여 학기에 분산된 추천 과목 plan. 비어 있을 수
            있다 (안전 종료 경로).
        derived_from_combo_key: 본 로드맵이 파생된 트랙 조합 식별자.
            후속 자연어 설명 노드가 어느 조합과의 binding 인지 추적할 때
            사용한다. 안전 종료 (조합 부재) 시 ``None``.
    """

    stages: list[RoadmapStage] = Field(...)
    semesters: list[SemesterPlan] = Field(default_factory=list)
    derived_from_combo_key: str | None = Field(default=None)

    @model_validator(mode="after")
    def _enforce_four_stages_in_order(self) -> Self:
        expected = ["foundation", "core", "application", "industry"]
        actual = [s.stage for s in self.stages]
        if actual != expected:
            raise ValueError(f"roadmap stages must be exactly {expected} in order, got {actual}")
        return self


class ExplanationSection(BaseModel):
    """LLM 자연어 설명의 영역별 단락.

    하나의 설명을 직무·트랙·로드맵 세 영역으로 분리하면 사용자가 어떤
    영역의 근거를 보고 있는지 시각적으로 구분 가능하다.

    Attributes:
        topic: 단락이 다루는 영역.
        body: 단락 본문. 빈 문자열은 의미가 없으므로 검증으로 차단한다.
    """

    topic: Literal["jobs", "tracks", "roadmap"]
    body: str = Field(..., min_length=1)


class SemesterSubtitle(BaseModel):
    """학기 카드 헤더에 노출되는 학기 단위 부제.

    학습 로드맵 화면은 학기별 카드로 구성되며, 각 카드 헤더에 그 학기가
    어떤 학습 깊이 단계인지 한 줄로 요약해 노출한다. 사용자가 카드를 펼치기
    전에도 "이 학기가 어느 단계인지" 를 인지할 수 있게 하는 정보 위계의
    상위 레벨이다.

    Attributes:
        semester: 부제가 매핑되는 학기 번호 (1 학년 1 학기 = 1).
        subtitle: 학기 카드 헤더 노출 문구. 학습 깊이 단계명을 포함한다.
    """

    semester: int = Field(..., ge=1, le=8)
    subtitle: str = Field(..., min_length=1)


class CourseFlow(BaseModel):
    """과목 상세 모달에 노출되는 과목 단위 인과 흐름.

    학기 카드의 과목 row 를 탭하면 열리는 상세 모달에서, 사용자가
    "관심사 → 직무 → 트랙 조합 → 이 과목" 의 추천 인과를 한 화면에서
    납득할 수 있도록 과목 단위로 인과 사슬을 노출한다. 정보 위계의 하위
    레벨로, 학기 부제보다 더 구체적인 근거를 담는다.

    Attributes:
        course_id: 흐름이 매핑되는 과목 식별자. 모달이 어떤 과목에 부착할지
            결정하는 키다.
        flow: 인과 흐름 문구. 관심사·직무·트랙 조합·학습 깊이 단계를 잇는다.
    """

    course_id: str = Field(..., min_length=1)
    flow: str = Field(..., min_length=1)


class JobRationale(BaseModel):
    """추천 직무 한 건의 개별 근거 문구.

    직무 영역 전체 단락(``ExplanationSection`` topic=jobs)은 직무군을 아우르는
    한 문단이라, 그것만으로는 추천된 직무 3~5 건이 같은 문구를 공유하게 된다.
    본 모델은 직무 한 건 단위로 "왜 이 직무가 당신에게 맞는지"를 따로 담아,
    각 직무가 자기 기술스택·역량·유사도에 근거한 서로 다른 근거를 갖게 한다.

    Attributes:
        job_id: 근거가 매핑되는 직무 식별자. 추천 직무 리스트의 해당 항목에
            binding 하는 키다.
        rationale: 해당 직무 단위 근거 문구.
    """

    job_id: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)


class TrackRationale(BaseModel):
    """추천 트랙 조합 한 건의 개별 근거 문구 — 조합 전체 + 트랙별로 분리한다.

    트랙 영역 전체 단락만으로는 주/보조 추천 조합이 같은 문구를 공유하므로,
    조합 한 건 단위로 근거를 담는다. 한 조합 안에서도 두 트랙의 시너지(조합
    전체)와 각 트랙 자체의 가치는 다른 층위라, 조합 전체 근거와 개별 트랙
    근거를 별도 필드로 구분한다.

    Attributes:
        combo_key: 근거가 매핑되는 조합 식별자. 주/보조 추천 조합 리스트의
            해당 항목에 binding 하는 키다.
        combo_rationale: 두 트랙을 함께 선택했을 때의 조합 단위 근거 (시너지).
        track_a_rationale: 1트랙 자체의 개별 근거.
        track_b_rationale: 2트랙 자체의 개별 근거.
    """

    combo_key: str = Field(..., min_length=1)
    combo_rationale: str = Field(..., min_length=1)
    track_a_rationale: str = Field(..., min_length=1)
    track_b_rationale: str = Field(..., min_length=1)


class Explanation(BaseModel):
    """LLM 자연어 설명 전체.

    영역별 단락 (``sections``) 은 추천 결과 영역별 (직무/트랙/로드맵) 의 요약
    근거를, 항목별 근거 (``job_rationales`` / ``track_rationales``) 는 직무 한 건·
    트랙 조합 한 건 단위의 개별 근거를 담는다. 영역 단락은 직무군·트랙군을
    아우르는 한 문단이라 같은 영역의 여러 항목이 동일 문구를 공유하므로, 항목별
    근거를 분리해 각 항목이 자기 데이터에 근거한 서로 다른 문구를 갖게 한다.
    학기 부제 (``semester_subtitles``) 와 과목 인과 흐름 (``course_flows``) 은
    학습 로드맵 화면의 정보 위계 (학기 카드 헤더 / 과목 상세 모달) 에 직접
    매핑된다. 리스트 필드는 모두 비어 있어도 valid 하다 (전체 요약만 ``text``
    로 채워진 상태, 또는 해당 영역이 비어 항목별 출력이 없는 상태).

    Attributes:
        text: 전체 요약 본문.
        sections: 영역별 요약 단락. 비어 있을 수 있다.
        job_rationales: 직무 한 건 단위 개별 근거. 직무 영역이 비면 빈 리스트.
        track_rationales: 트랙 조합 한 건 단위 개별 근거. 트랙 영역이 비면
            빈 리스트.
        semester_subtitles: 학기 카드 헤더용 부제. 로드맵이 비면 빈 리스트.
        course_flows: 과목 상세 모달용 인과 흐름. 로드맵이 비면 빈 리스트.
    """

    text: str = Field(..., min_length=1)
    sections: list[ExplanationSection] = Field(default_factory=list)
    job_rationales: list[JobRationale] = Field(default_factory=list)
    track_rationales: list[TrackRationale] = Field(default_factory=list)
    semester_subtitles: list[SemesterSubtitle] = Field(default_factory=list)
    course_flows: list[CourseFlow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CoverageAnalysis — 역량 커버리지 분석 도메인 모델
# ---------------------------------------------------------------------------


class JobCoverage(BaseModel):
    """추천 직무 한 건에 대한 역량 충족도 — 현재 / 예상.

    각 추천 직무를 하나의 목표 분야로 보고, 그 직무가 요구하는 기술·역량 토큰을
    학생이 현재(이수 과목) 얼마나 덮는지와 추천 로드맵을 모두 이수했을 때 얼마나
    덮게 되는지를 분리해 보여준다.

    Attributes:
        job_id: 직무 식별자.
        job_name: 사용자 표시용 직무명.
        required_count: 직무가 요구하는 정규화 토큰 수 (표기 정합 후 중복 제거).
        current_covered: 현재(이수 과목) 덮는 토큰 수.
        expected_covered: 추천 로드맵 이수 후 덮게 되는 토큰 수.
        current_ratio: ``current_covered / required_count`` ([0, 1]).
            ``required_count == 0`` 이면 0.0.
        expected_ratio: ``expected_covered / required_count`` ([0, 1]).
            ``required_count == 0`` 이면 0.0.
        missing_tokens: 추천 로드맵 이수 후에도 못 덮는 토큰 (사용자 표시 표기).
    """

    job_id: str = Field(..., min_length=1)
    job_name: str = Field(..., min_length=1)
    required_count: int = Field(..., ge=0)
    current_covered: int = Field(..., ge=0)
    expected_covered: int = Field(..., ge=0)
    current_ratio: float = Field(..., ge=0.0, le=1.0)
    expected_ratio: float = Field(..., ge=0.0, le=1.0)
    missing_tokens: list[str] = Field(default_factory=list)


class CourseCoverageContribution(BaseModel):
    """추천 로드맵의 잔여 과목 한 건이 전체 역량 충족도에 더하는 기여.

    기여도는 *현재 충족도 기준* 독립 한계 기여다 — 이 과목 하나만 이수했을 때
    새로 덮는 목표 토큰의 비율이다. 과목 간 토큰이 겹칠 수 있어 모든 과목의
    기여도 합은 (예상 - 현재) 충족도 증가분을 넘을 수 있다. 사용자 화면에서
    과목별 "+X%" 를 직관적으로 보여주기 위한 정의다.

    Attributes:
        course_id: 과목 식별자.
        course_name: 사용자 표시용 과목명.
        added_tokens: 이 과목이 새로 덮는 목표 토큰 (현재 미충족분 중, 표시 표기).
        contribution_ratio: ``len(added_tokens) / required_count`` ([0, 1]).
            목표 토큰이 없으면 0.0.
    """

    course_id: str = Field(..., min_length=1)
    course_name: str = Field(..., min_length=1)
    added_tokens: list[str] = Field(default_factory=list)
    contribution_ratio: float = Field(..., ge=0.0, le=1.0)


class NextActionSuggestion(BaseModel):
    """추천 기반 다음 액션 — 충족도를 가장 많이 올리는 과목 제안.

    잔여 과목 중 현재 충족도를 가장 크게 끌어올리는 과목을 우선 제안해, 학생이
    "다음에 무엇을 들어야 효과가 큰지" 를 한눈에 알 수 있게 한다.

    Attributes:
        course_id: 제안 과목 식별자.
        course_name: 사용자 표시용 과목명.
        contribution_ratio: 이 과목을 *단독으로* 이수했을 때의 독립 충족도 증가분 ([0, 1]).
            토큰이 겹치면 여러 과목의 값이 같은 토큰을 각자 세므로 합이 실제 누적
            증가분을 넘을 수 있다. 표시 shortlist 전체를 이수했을 때의 누적 도달은
            ``CoverageAnalysis.next_actions_ratio`` 를 쓴다 (합집합으로 중복 제거).
        message: 사용자 표시용 제안 문구.
    """

    course_id: str = Field(..., min_length=1)
    course_name: str = Field(..., min_length=1)
    contribution_ratio: float = Field(..., ge=0.0, le=1.0)
    message: str = Field(..., min_length=1)


class CoverageAnalysis(BaseModel):
    """단일 기준 직무(anchor) 요구 역량 대비 현재 → 예상 충족도 분석 전체.

    추천 결과(직무·트랙·로드맵)를 일회성 결과가 아닌 채워가는 지도로 만들기
    위해, **하나의 기준 직무**가 요구하는 기술·역량 토큰을 목표로 삼아 학생의
    현재 충족도와 추천 로드맵을 모두 이수했을 때의 예상 충족도를 산출한다.

    기준 직무를 하나로 고정하는 이유:
        추천 직무가 여럿일 때 충족도를 합집합·평균으로 섞으면 "무엇의 몇 %인지"
        가 모호해진다. 게이지·잔여 과목 기여도(``course_contributions``)·다음
        액션(``next_actions``)·gap 토큰을 모두 같은 기준 직무에 정렬해, 사용자가
        "어떤 직무를 목표로 한 충족도인지" 를 분명히 알 수 있게 한다. 단 분야별
        분석(``jobs``)은 직무 간 비교 카드라 anchor 와 무관하게 전 추천 직무를 담는다.

    기준 직무 선택:
        기본값은 매칭도 1순위 직무이며, 호출자가 ``anchor_job_id`` 로 다른
        추천 직무를 지정하면 그 직무로 다시 산출한다. ``anchor_job_id`` /
        ``anchor_job_name`` 으로 어느 직무를 기준으로 했는지 함께 싣는다
        (커버리지 모달의 "○○ 직무 기준" 라벨).

    비율 필드는 모두 [0, 1] 이며 사용자 화면에서 백분율로 렌더링한다. 목표 토큰이
    하나도 없으면(추천 직무 부재 또는 기준 직무 토큰 미보유) 모든 비율은 0.0,
    리스트는 비고 ``anchor_job_id`` 는 빈 문자열로 graceful 종료한다.

    Attributes:
        anchor_job_id: 충족도 산출의 기준이 된 직무 식별자. 목표 토큰 부재 시
            빈 문자열(``anchor_job_id == "" ⟺ required_count == 0``).
        anchor_job_name: 기준 직무명 (사용자 표시용 "○○ 직무 기준" 라벨).
        required_count: 기준 직무 목표 토큰 수 (표기 정합 후 중복 제거).
        current_covered: 현재(이수 과목) 덮는 목표 토큰 수.
        expected_covered: 추천 로드맵 이수 후 덮게 되는 목표 토큰 수.
        current_ratio: ``current_covered / required_count`` ([0, 1]). 분모 0 이면 0.0.
        expected_ratio: ``expected_covered / required_count`` ([0, 1]). 분모 0 이면 0.0.
        next_actions_covered: 다음 액션(``next_actions`` 에 노출된 shortlist) 과목까지
            이수했을 때 덮는 목표 토큰 수. 합집합으로 계산해 ``current_covered <=
            next_actions_covered <= expected_covered`` 를 만족한다.
        next_actions_ratio: ``next_actions_covered / required_count`` ([0, 1]). 분모 0 이면 0.0.
            "다음 N개 과목 이수 시 도달 충족도" 표기의 근거. 과목별 ``contribution_ratio``
            의 합과 달리 토큰 중복을 합집합으로 제거해 over-claim 을 막는다.
        jobs: 분야별 분석 — 전 추천 직무의 per-job 현재/예상 충족도 (직무 비교 카드용,
            anchor 와 무관). 각 직무를 자기 토큰 기준으로 평가하며 현재 충족률
            내림차순으로 정렬한다. 목표 토큰 부재 시 빈 리스트.
        course_contributions: 잔여(추천) 과목별 기준 직무 충족도 기여도.
        next_actions: 추천 기반 다음 액션 (기여도 상위 과목).
        gap_tokens: 추천 로드맵 이수 후에도 못 덮는 기준 직무 목표 토큰 (사용자 표시 표기).
    """

    anchor_job_id: str = Field(default="")
    anchor_job_name: str = Field(default="")
    required_count: int = Field(..., ge=0)
    current_covered: int = Field(..., ge=0)
    expected_covered: int = Field(..., ge=0)
    current_ratio: float = Field(..., ge=0.0, le=1.0)
    expected_ratio: float = Field(..., ge=0.0, le=1.0)
    next_actions_covered: int = Field(default=0, ge=0)
    next_actions_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    jobs: list[JobCoverage] = Field(default_factory=list)
    course_contributions: list[CourseCoverageContribution] = Field(default_factory=list)
    next_actions: list[NextActionSuggestion] = Field(default_factory=list)
    gap_tokens: list[str] = Field(default_factory=list)


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


class CompletedCourseBoostConfig(BaseModel):
    """이수 과목 부스팅 강도 정책.

    부스팅은 직무 검색 결과의 점수에 ``weight · overlap_ratio`` 만큼 가산하는
    후처리 신호로, ``weight`` 가 ``overlap_ratio = 1`` (이수 과목이 직무 토큰
    전부를 덮음) 일 때의 최대 가산량이다. 직관 할당값이며 가중치 ablation
    대상이다. ``weight = 0`` 이면 부스팅이 완전히 비활성화된다.

    Attributes:
        weight: 최대 가산량 ([0, 1]). 가산 후 점수는 ``[0, 1]`` 로 clip 된다.
    """

    weight: float = Field(default=0.15, ge=0.0, le=1.0)


class JobMatchingConfig(BaseModel):
    """``synergy.yaml`` 의 ``job_matching`` 섹션 1:1 매핑.

    트랙 시너지와 같은 yaml 파일을 공유하여 가중치·임계값 외부화 단위를
    단순화한다. 본 클래스는 그 중 ``job_matching`` 매핑만 추출한다.

    Attributes:
        top_k: 응답 후보 수 정책.
        min_job_similarity: 코사인 유사도 하위 컷. max similarity 가 이 값
            미만이면 카테고리 사전 매핑 fallback 으로 전환된다.
        completed_course_boost: 이수 과목 부스팅 강도 정책. yaml 에 섹션이
            없으면 기본값을 사용한다.
    """

    top_k: JobMatchingTopKConfig
    min_job_similarity: float = Field(..., ge=0.0, le=1.0)
    completed_course_boost: CompletedCourseBoostConfig = Field(
        default_factory=CompletedCourseBoostConfig
    )

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


# ---------------------------------------------------------------------------
# RoadmapConfig — roadmap.yaml 외부화 매핑
# ---------------------------------------------------------------------------


class CapacityConfig(BaseModel):
    """학기 용량 정책.

    한성대 일반 학기 제도를 기반으로 한 hard cap 으로, 가중치 ablation 대상이
    아니다. 사용자 입력에 직전 학기 평점 필드가 추가되기 전까지는 노드 본체가
    ``max_credits_per_semester_default`` 만 사용한다.

    Attributes:
        max_credits_per_semester_default: 학기당 기본 최대 학점.
        max_credits_per_semester_high_gpa: 직전 학기 평점이 우수 기준선 이상일 때
            허용하는 확장 학점. 현재는 보존만 하며 노드 본체는 미사용.
    """

    max_credits_per_semester_default: int = Field(..., ge=1)
    max_credits_per_semester_high_gpa: int = Field(..., ge=1)


class GradeCreditsRange(BaseModel):
    """학년별 전공 학점 범위 (min / max).

    학사 운영의 학년 단위 권장 학점 분포를 표현한다. 학습 로드맵 분산은
    학년별 ``max`` 를 hard ceiling 으로 강제하며, ``min`` 은 사용자 안내
    영역 (학년 그룹 헤더의 "권장 N ~ M 학점") 에서만 노출한다.

    Attributes:
        min: 학년 권장 최소 학점.
        max: 학년 권장 최대 학점 (분산 알고리즘의 hard ceiling).
    """

    model_config = ConfigDict(populate_by_name=True)

    min_credits: int = Field(..., alias="min", ge=0)
    max_credits: int = Field(..., alias="max", ge=0)

    @model_validator(mode="after")
    def _enforce_min_le_max(self) -> Self:
        if self.min_credits > self.max_credits:
            raise ValueError(
                f"grade credits range must have min <= max, "
                f"got min={self.min_credits}, max={self.max_credits}"
            )
        return self


class GraduationConfig(BaseModel):
    """졸업 요건 hard constraint.

    Attributes:
        total_credits_two_tracks: 1 트랙 + 2 트랙 합산 전공 필수 학점 총합.
            추천 분기 hint 로 보존하며 분산 알고리즘의 종료 조건으로는
            ``total_credits_major`` 를 사용한다.
        total_credits_major: 전공 (필수 + 선택) 합산 졸업 학점 총합. 학습
            로드맵 분산은 누적 추천 학점이 이 값에 도달하면 stream 을
            종료하고 해당 학기에 도달 마커를 설정한다.
    """

    total_credits_two_tracks: int = Field(..., ge=1)
    total_credits_major: int = Field(..., ge=1)


class RoadmapConfig(BaseModel):
    """``roadmap.yaml`` 의 외부화 설정 1:1 매핑.

    yaml 로딩은 ``load_from_yaml`` classmethod 에 격리하여 노드 호출 경로에서
    파일 I/O 를 단일 위치로 모은다. 본 설정은 한성대 학사 제도 기반이라
    실험적 가중치 변경 대상이 아니며, 시너지 가중치 ablation 과 변경 이유가
    독립적이라 ``synergy.yaml`` 과 분리된 파일로 운영한다.

    Attributes:
        capacity: 학기 용량 정책 (학기당 최대 학점).
        graduation: 졸업 요건 (전공 필수 / 전공 합산).
        grade_credits_range: 학년 번호 (1~4) 별 전공 학점 범위. 학년별 hard
            ceiling 강제 + 사용자 안내 영역의 권장 범위 노출에 동시 사용.
    """

    capacity: CapacityConfig
    graduation: GraduationConfig
    grade_credits_range: dict[int, GradeCreditsRange]

    @model_validator(mode="after")
    def _enforce_grade_keys(self) -> Self:
        expected = {1, 2, 3, 4}
        actual = set(self.grade_credits_range.keys())
        if actual != expected:
            raise ValueError(
                f"grade_credits_range must have keys exactly {sorted(expected)}, "
                f"got {sorted(actual)}"
            )
        return self

    @classmethod
    def load_from_yaml(cls, path: Path) -> Self:
        """yaml 파일에서 ``RoadmapConfig`` 를 로드한다.

        Args:
            path: ``roadmap.yaml`` 의 경로.

        Returns:
            검증된 ``RoadmapConfig`` 인스턴스.

        Raises:
            ValueError: yaml 이 mapping 이 아닐 때.
        """
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Roadmap config file {path} must define a top-level mapping")
        return cls.model_validate(raw)
