"""직무 브리핑 도메인 모델.

브리핑은 추천 직무 한 건에 명시적으로 연결된 트렌드 카드다. 일반 뉴스가 아니라
특정 직무에 묶여야 하므로 ``job_id`` 가 카드의 1급 식별자이며, 사전 검증을 위해
모든 카드는 출처(``sources``)를 동반한다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class BriefingSource(BaseModel):
    """브리핑 근거 출처 — 사전 검증 가능성을 위해 카드마다 1개 이상 동반한다.

    헤드라인·요약이 어디서 왔는지 추적 가능해야 추천 보조 정보로서 신뢰된다.
    검증은 ``url`` 로 수행하므로 ``HttpUrl`` 로 형태를 강제해, 형식이 깨진 출처가
    큐레이션 카탈로그에 섞여 들어오는 것을 적재 시점에 차단한다 (사전 검증
    가능성 보장). ``published_at`` 은 트렌드의 시점 신뢰도를 가늠하는 보조
    신호다 (시점 미상 큐레이션은 생략 가능).

    Attributes:
        title: 출처 표시명 (예: "Stack Overflow Developer Survey 2024").
        url: 검증 가능한 출처 URL (http/https 형태 강제). 응답에서는 문자열로 직렬화된다.
        published_at: 발행 시점 (``YYYY`` 또는 ``YYYY-MM``). 미상이면 None.
    """

    title: str = Field(..., min_length=1)
    url: HttpUrl
    published_at: str | None = None


class JobBriefing(BaseModel):
    """추천 직무 한 건에 연결된 트렌드 브리핑 카드.

    ``job_id`` 는 추천 직무 후보의 식별자와 같은 표준 코드 어휘를 써서, 메인
    백엔드가 별칭 매핑 없이 추천 직무 ↔ 브리핑을 조인할 수 있게 한다. 카드는
    헤드라인(한 줄 트렌드) + 요약(맥락) + 필수 역량 + 출처로 구성되며, 출처
    부재 카드는 검증 불가하므로 모델 수준에서 차단한다.

    Attributes:
        job_id: 직무 카탈로그 표준 코드 (추천 직무 후보 식별자와 정합).
        job_name: 사용자 표시용 직무명 (카탈로그 표시명과 정합).
        headline: 한 줄 트렌드 헤드라인.
        summary: 트렌드 맥락 요약.
        skills: 카드가 강조하는 필수 역량·기술 키워드. 비어 있어도 valid.
        sources: 검증 가능한 출처. 최소 1개를 강제한다.
    """

    job_id: str = Field(..., min_length=1)
    job_name: str = Field(..., min_length=1)
    headline: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    skills: list[str] = Field(default_factory=list)
    sources: list[BriefingSource] = Field(..., min_length=1)


__all__ = ["BriefingSource", "JobBriefing"]
