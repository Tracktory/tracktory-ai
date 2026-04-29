"""N1 — 사용자 온보딩 입력 정규화 노드.

기능명세 ON-001 ~ ON-012 의 원시 입력을 후속 노드 (N2 ~ N5) 가 사용하는
정규화 프로필로 변환한다. **단일 책임**: 입력 검증 + 정규화. I/O 없음.

관련 결정:
    D-02: 백엔드 스키마 = functional-spec 권위.
    D-06: ``completed_courses`` 는 N2 임베딩 입력에 포함하지 않고 본 단계에서
          정규화만 수행. 이후 N5 선수과목 필터에서만 사용.
    D-08 degenerate: 1학년 (트랙 미선택) 은 ``current_tracks = []`` 로
          표현되어 N4 가 "조합 신규 추천 모드" 로 분기한다.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from tracktory.graph.state import GraphState

CompanyType = Literal[
    "대기업", "중견기업", "중소기업", "공기업", "스타트업", "프리랜서", "상관없음"
]
WorkValue = Literal["돈", "워라벨", "복지", "명예", "안정성", "성장성"]


class NormalizedProfile(BaseModel):
    """정규화된 사용자 프로필 (N1 의 출력 스키마).

    Attributes:
        admission_year: ON-001 입학년도.
        college: ON-003 / 005 단과대명 (예: "IT공과대학").
        department: ON-003 / 005 학부명 (예: "컴퓨터공학부").
        current_tracks: ON-006 현재 선택 트랙. 1학년은 빈 리스트, 2학년+ 는
            정확히 2 개 (1트랙 · 2트랙).
        interests: ON-007 관심사 (14 개 카테고리 중 1 ~ 5 개).
        dev_interests: ON-011 흥미 개발 분야 (1 ~ 3 개).
        work_values: ON-009 취업 시 중요 가치 (≤ 3 개).
        company_types: ON-008 회사 유형 (≥ 1 개).
        ncs_studied: ON-011-1 공부해본 분야의 NCS 분류 코드 리스트 (선택).
        completed_courses: ON-012 이수 과목명. **N2 임베딩 입력 X (D-06)**,
            N5 선수과목 필터에서만 사용.
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
    def _tracks_zero_or_two(cls, v: list[str]) -> list[str]:
        # ON-006: 1학년은 0 개, 2학년+ 는 정확히 2 개 (1트랙 · 2트랙).
        if len(v) not in (0, 2):
            raise ValueError("current_tracks must be empty (1학년) or exactly 2 (2학년+)")
        return v


def normalize_input(state: GraphState) -> dict[str, Any]:
    """``raw_input`` 을 ``NormalizedProfile`` 로 정규화한다.

    Args:
        state: ``raw_input`` 키가 ON-001 ~ ON-012 의 원시 입력을 담아야 한다.

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
