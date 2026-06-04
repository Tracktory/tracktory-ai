"""직무 브리핑 엔드포인트 요청·응답 스키마.

브리핑 시트는 추천 결과와 별도 표면이라 추천 입력 전체가 아니라 직무 식별자만
받는다. 메인 백엔드가 직전 추천 응답에서 직무 코드를 들고 있으므로, 최소 계약은
``job_ids`` 다. 응답은 도메인 모델 ``JobBriefing`` 을 직접 재사용해 서비스 ↔
응답 drift 를 차단한다.
"""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from tracktory.briefing.models import JobBriefing

# 빈/공백 직무 코드는 요청 단계에서 거부한다. 그대로 통과시키면 조회가 조용히
# 빈 결과를 내 "큐레이션 없음" 과 "잘못된 중계 데이터" 가 구분되지 않는다.
JobId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class BriefingRequest(BaseModel):
    """브리핑 요청 — 추천 직무 식별자 목록.

    Attributes:
        job_ids: 추천 직무 표준 코드 목록. 최소 1개이며 각 항목은 공백 제거 후
            1자 이상 (빈 요청·빈 코드는 의미 없음).
    """

    job_ids: list[JobId] = Field(..., min_length=1)


class BriefingResponse(BaseModel):
    """브리핑 응답 — 직무에 연결된 트렌드 카드 묶음.

    Attributes:
        briefings: ``JobBriefing`` 리스트. 요청 직무 중 큐레이션이 없는 직무는
            제외되며, 매칭이 하나도 없으면 빈 리스트.
    """

    briefings: list[JobBriefing]
