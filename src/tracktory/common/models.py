"""
통합 데이터 모델 - 사람인/원티드 공통 스키마
"""
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class JobPosting(BaseModel):
    """채용 공고 단건 데이터 모델.

    사람인과 원티드에서 수집한 채용 공고를 통합된 스키마로 표현합니다.
    source_id 는 각 플랫폼 내부 고유 식별자입니다.
    extra 필드에는 플랫폼별 추가 정보(예: 원티드의 주요업무/자격요건/우대사항)를 저장합니다.
    """

    source: str
    """데이터 출처. 'saramin' 또는 'wanted'."""

    source_id: str
    """출처 플랫폼의 고유 공고 ID."""

    company: str
    """회사명."""

    title: str
    """공고 제목."""

    category: str
    """직무 카테고리."""

    tech_stacks: list[str]
    """기술 스택 목록."""

    location: str
    """근무지."""

    salary: str
    """급여 정보."""

    url: str
    """공고 URL."""

    experience: str
    """경력 요건."""

    collected_at: datetime = Field(default_factory=datetime.now)
    """수집 시각. 기본값은 현재 시각."""

    extra: dict[str, Any] = Field(default_factory=dict)
    """플랫폼별 추가 필드.

    예시 (원티드):
        {
            "responsibilities": "주요업무 텍스트",
            "requirements": "자격요건 텍스트",
            "preferred": "우대사항 텍스트",
        }
    """

    def to_csv_row(self) -> dict[str, str]:
        """CSV 행으로 변환합니다.

        unified CSV 스키마에 맞춘 평탄한 딕셔너리를 반환합니다.
        tech_stacks 는 '|' 구분자로 이어 붙인 문자열로 직렬화됩니다.

        Returns:
            dict[str, str]: CSV 컬럼명을 키로 갖는 평탄한 딕셔너리.
            컬럼 순서: source, source_id, company, title, category,
                       tech_stacks, location, salary, url, experience, collected_at
        """
        row = {
            "source": self.source,
            "source_id": self.source_id,
            "company": self.company,
            "title": self.title,
            "category": self.category,
            "tech_stacks": "|".join(self.tech_stacks),
            "location": self.location,
            "salary": self.salary,
            "url": self.url,
            "experience": self.experience,
            "collected_at": self.collected_at.isoformat(),
            "responsibilities": self.extra.get("responsibilities", ""),
            "requirements": self.extra.get("requirements", ""),
            "preferred": self.extra.get("preferred", ""),
            "benefits": self.extra.get("benefits", ""),
        }
        return row

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 딕셔너리로 변환합니다.

        collected_at 은 ISO 8601 문자열로 변환되어 반환됩니다.
        extra 필드도 포함됩니다.

        Returns:
            dict[str, Any]: 모든 필드를 포함한 딕셔너리.
        """
        data = self.model_dump()
        data["collected_at"] = self.collected_at.isoformat()
        return data


class CrawlResult(BaseModel):
    """크롤링 실행 결과 요약 모델.

    단일 플랫폼 크롤링 작업의 수행 결과와 수집된 공고 목록을 담습니다.
    """

    source: str
    """크롤링 대상 플랫폼. 'saramin' 또는 'wanted'."""

    total_collected: int
    """정상적으로 수집된 공고 수."""

    total_skipped: int
    """중복 등의 이유로 건너뛴 공고 수."""

    total_failed: int
    """오류로 인해 수집에 실패한 공고 수."""

    duration_seconds: float
    """크롤링 소요 시간(초)."""

    jobs: list[JobPosting]
    """수집된 채용 공고 목록."""

    def summary(self) -> str:
        """크롤링 결과 요약 문자열을 반환합니다.

        터미널 출력 또는 로그 기록에 사용합니다.

        Returns:
            str: 플랫폼, 수집/건너뜀/실패 수, 소요 시간을 포함한 요약 문자열.
        """
        return (
            f"[{self.source}] 크롤링 완료 | "
            f"수집: {self.total_collected}건 | "
            f"건너뜀: {self.total_skipped}건 | "
            f"실패: {self.total_failed}건 | "
            f"소요 시간: {self.duration_seconds:.1f}초"
        )
