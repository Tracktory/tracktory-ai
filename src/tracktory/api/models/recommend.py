from pydantic import BaseModel, Field, field_validator

from tracktory.graph.models import Explanation, JobCandidate, RankedCombo, Roadmap
from tracktory.graph.nodes.input_normalize import CompanyType, WorkValue


class RecommendRequest(BaseModel):
    """사용자 온보딩 입력 — 그래프 입구의 `NormalizedProfile` 11 필드를 mirror 한다. drift 방지를 위해 동일 검증 규칙을 그대로 박는다."""

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


class RecommendResponse(BaseModel):
    """추천 그래프의 4 부분 묶음 응답. 도메인 모델 (`Job`, `RankedCombo`, `Roadmap`, `Explanation`) 을 직접 재사용하여 state ↔ response drift 를 차단한다."""

    jobs: list[JobCandidate]
    primary_combos: list[RankedCombo]
    secondary_combos: list[RankedCombo]
    roadmap: Roadmap
    explanation: Explanation
