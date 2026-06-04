"""직무 브리핑 엔드포인트 POST /api/v1/ai/briefing — Spring 내부 호출 전용.

홈 브리핑 시트의 중계 API 와 짝을 이룬다. 추천 직무 식별자를 받아 큐레이션된
트렌드 카드를 반환한다. 추천 그래프를 거치지 않고 정적 카탈로그만 조회하므로
다른 엔드포인트와 달리 boundary 묶음이 아닌 브리핑 서비스 싱글톤에 의존한다.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from tracktory.api.auth import verify_internal_token
from tracktory.api.dependencies import get_briefing_service
from tracktory.api.models.briefing import BriefingRequest, BriefingResponse
from tracktory.api.response.base import ApiResponse
from tracktory.briefing.service import BriefingService

router = APIRouter(
    prefix="/api/v1/ai/briefing",
    tags=["briefing"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post("", response_model=ApiResponse[BriefingResponse])
def briefing(
    request: BriefingRequest,
    service: Annotated[BriefingService, Depends(get_briefing_service)],
) -> ApiResponse[BriefingResponse]:
    """요청 직무들의 큐레이션 브리핑 카드를 반환한다.

    조회는 메모리 색인 lookup 이라 블로킹 I/O 가 없어 sync 핸들러로 둔다.
    큐레이션이 없는 직무는 빈 결과로 graceful 하게 흡수된다 (부분 충족 허용).
    """
    briefings = service.get_briefings(request.job_ids)
    return ApiResponse.ok(data=BriefingResponse(briefings=briefings))
