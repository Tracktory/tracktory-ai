from fastapi import APIRouter

from tracktory.api.models.recommend import RecommendReq, RecommendRes
from tracktory.api.response.success import SuccessResponse

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.post("", response_model=SuccessResponse[RecommendRes])
async def recommend(request: RecommendReq) -> SuccessResponse[RecommendRes]:
    data = RecommendRes(test_str="요청: " + request.test_str)
    return SuccessResponse.ok(data=data)
