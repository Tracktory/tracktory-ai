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
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from tracktory.graph.state import GraphState

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


def normalize_input(state: GraphState) -> dict[str, Any]:
    """외부 비신뢰 입력 검증을 그래프 입구 한 곳에 격리하기 위한 노드.

    검증 실패는 그래프를 깨뜨리지 않고 ``errors`` 로 흘려, 후속 노드가
    ``normalized_profile`` 부재만 확인하면 안전하게 동작한다.

    Args:
        state: ``raw_input`` 키가 온보딩 화면의 원시 입력을 담아야 한다.

    Returns:
        성공 시 ``{"normalized_profile": <dict>, "trace": ["N1:ok"]}``.
        Pydantic 검증 실패 시 ``{"errors": [...], "trace": ["N1:fail"]}``
        — ``normalized_profile`` 은 미설정 (이후 노드가 ``None`` 으로 인지).
    """
    raw = state.get("raw_input") or {}
    try:
        profile = NormalizedProfile.model_validate(raw)
    except ValidationError as e:
        return {"errors": [f"N1 normalization failed: {e}"], "trace": ["N1:fail"]}

    return {"normalized_profile": profile.model_dump(), "trace": ["N1:ok"]}
