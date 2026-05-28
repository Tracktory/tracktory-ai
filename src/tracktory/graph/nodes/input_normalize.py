"""사용자 온보딩 원시 입력을 검증하고 정규화한다.

온보딩 화면(입학년도·소속·트랙·관심사·흥미 개발 분야·취업 선호도 등)의 원시 입력을
후속 노드가 공통으로 사용하는 정규화 프로필로 변환한다.
**단일 책임**: 입력 검증 + 정규화. 외부 I/O 없음.

설계 메모:
    - 백엔드 서비스가 전달하는 스키마와 1:1로 맞춘다.
    - ``completed_courses`` 는 의미 임베딩 입력에 포함하지 않고 정규화만 수행한다.
      이수 과목은 선수과목 충족 여부 확인 등 집합 연산에만 사용되므로 임베딩
      공간에 넣으면 관심사·흥미 의미를 희석한다.
    - 1학년(트랙 미선택)은 ``current_tracks = []`` 로 표현된다.
    - ``current_semester`` 는 학습 로드맵 노드의 잔여 학기 분산 시작점이다.
      사용자 입력이 없으면 입학년도 기반 fallback 추정으로 채운다.
"""

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from tracktory.graph.state import GraphState

# 입학년도로부터 현재 학기를 추정할 때 사용하는 기준 연도. 학기 매핑 (3 월
# 상반기 / 9 월 하반기) 은 미적용하고 연도 차이 * 2 + 1 의 단순 추정식만
# 쓴다. 학사 운영 유연성 (휴학 / 조기졸업 / 학기 reset) 흡수는 사용자 자율
# 수정 (마이페이지) 에 위임한다.
_FALLBACK_REFERENCE_YEAR = datetime.now().year

CompanyType = Literal[
    "대기업", "중견기업", "중소기업", "공기업", "스타트업", "프리랜서", "상관없음"
]
WorkValue = Literal["돈", "워라벨", "복지", "명예", "안정성", "성장성"]


class NormalizedProfile(BaseModel):
    """정규화된 사용자 프로필 — ``normalize_input`` 의 출력 스키마.

    Attributes:
        admission_year: 입학년도.
        college: 단과대명 (예: "IT공과대학").
        department: 학부명 (예: "컴퓨터공학부").
        current_tracks: 현재 선택 트랙. 1학년은 빈 리스트, 2학년+ 는 정확히 2 개 (주전공 트랙 · 보조 트랙).
        current_semester: 학생의 현재 학기 (1 ~ 8). 사용자가 명시하지 않으면
            입학년도 기반 fallback 추정으로 채워진다. 학습 로드맵 노드의
            잔여 학기 분산 시작점.
        interests: 관심사 (14 개 카테고리 중 1 ~ 5 개).
        dev_interests: 흥미 개발 분야 (1 ~ 3 개).
        work_values: 취업 시 중요 가치 (≤ 3 개).
        company_types: 선호 회사 유형 (≥ 1 개).
        ncs_studied: 공부해본 분야의 NCS 분류 코드 리스트 (선택).
        completed_courses: 이수 과목명. 의미 임베딩 입력에 포함하지 않으며
            선수과목 필터 단계에서만 사용한다.
    """

    admission_year: int = Field(..., ge=2000, le=2030)
    college: str = Field(..., min_length=1)
    department: str = Field(..., min_length=1)
    current_tracks: list[str] = Field(default_factory=list)
    current_semester: int | None = Field(default=None, ge=1, le=8)
    interests: list[str] = Field(..., min_length=1, max_length=5)
    dev_interests: list[str] = Field(..., min_length=1, max_length=3)
    work_values: list[WorkValue] = Field(..., max_length=3)
    company_types: list[CompanyType] = Field(..., min_length=1)
    ncs_studied: list[str] = Field(default_factory=list)
    completed_courses: list[str] = Field(default_factory=list)

    @field_validator("current_tracks")
    @classmethod
    def _tracks_zero_or_two(cls, tracks: list[str]) -> list[str]:
        if len(tracks) not in (0, 2):
            raise ValueError("current_tracks must be empty (1학년) or exactly 2 (2학년+)")
        return tracks

    @model_validator(mode="after")
    def _infer_current_semester_if_missing(self) -> Self:
        """입학년도 기반 fallback 추정.

        공식: ``(현재 연도 - admission_year) * 2 + 1``. 학기 매핑 (상/하반기)
        은 미적용하는 단순 추정으로, 학기 reset / 휴학 / 조기졸업 같은 학사
        운영 유연성은 사용자가 마이페이지에서 직접 수정해 흡수한다. 결과는
        [1, 8] 로 clamp 한다 — 입학년도가 5 년 이상 지난 케이스는 졸업 임박
        학기 (8) 로 처리되어 졸업 학점 미충족 경고 경로로 흐른다.
        """
        if self.current_semester is not None:
            return self
        estimated = (_FALLBACK_REFERENCE_YEAR - self.admission_year) * 2 + 1
        self.current_semester = max(1, min(8, estimated))
        return self


def normalize_input(state: GraphState) -> dict[str, Any]:
    """외부 비신뢰 입력 검증을 그래프 입구 한 곳에 격리하기 위한 노드.

    검증 실패는 그래프를 깨뜨리지 않고 ``errors`` 로 흘려, 후속 노드가
    ``normalized_profile`` 부재만 확인하면 안전하게 동작한다.

    ``current_semester`` 는 ``normalized_profile`` 안에도 보존되고 state 의
    top-level 키로도 함께 노출된다. 학습 로드맵 노드가 명시 state 키로
    접근해 분산 시작점을 잡을 수 있게 하기 위함이다 (다른 노드는 정규화
    프로필 dict 안의 값만 보면 충분).

    Args:
        state: ``raw_input`` 키가 온보딩 화면의 원시 입력을 담아야 한다.

    Returns:
        성공 시 ``{"normalized_profile": <dict>, "current_semester": <int>,
        "trace": ["input_normalize:ok"]}``. Pydantic 검증 실패 시
        ``{"errors": [...], "trace": ["input_normalize:fail"]}`` —
        ``normalized_profile`` 은 미설정 (이후 노드가 ``None`` 으로 인지).
    """
    raw = state.get("raw_input") or {}
    try:
        profile = NormalizedProfile.model_validate(raw)
    except ValidationError as e:
        return {
            "errors": [f"input normalization failed: {e}"],
            "trace": ["input_normalize:fail"],
        }

    return {
        "normalized_profile": profile.model_dump(),
        "current_semester": profile.current_semester,
        "trace": ["input_normalize:ok"],
    }
