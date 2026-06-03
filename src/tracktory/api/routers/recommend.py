"""추천 API — 온보딩 입력을 받아 컴파일된 추천 그래프를 실행한다.

그래프 실행은 lru_cache 가 보장하는 단일 컴파일 결과를 재사용하며,
state 의 errors 누적은 내부 계약 위반으로 분류하여 500 으로 매핑한다.
요청 검증 실패 (FastAPI 단계) 는 별도 422 envelope 으로 흐른다.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from langgraph.graph.state import CompiledStateGraph

from tracktory.api.auth import get_user_id, verify_internal_token
from tracktory.api.dependencies import get_recommendation_pipeline
from tracktory.api.models.recommend import RecommendRequest, RecommendResponse
from tracktory.api.response.base import ApiResponse

router = APIRouter(
    prefix="/api/v1/ai/recommend",
    tags=["recommend"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post("", response_model=ApiResponse[RecommendResponse])
async def recommend(
    request: RecommendRequest,
    graph: Annotated[CompiledStateGraph, Depends(get_recommendation_pipeline)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> ApiResponse[RecommendResponse]:
    """추천 그래프를 실행하고 4 부분 묶음 응답을 반환한다.

    그래프 state 의 errors 누적은 RecommendRequest pydantic 검증을 통과한
    입력이 그래프 내부 정규화 단계에서 거부되는 케이스로, 사용자 입력
    오류가 아닌 contract drift 이므로 500 으로 분류한다.
    """
    final_state = await graph.ainvoke(
        {
            "user_id": user_id,
            "raw_input": request.model_dump(),
        }
    )

    if final_state.get("errors"):
        raise HTTPException(status_code=500, detail=final_state["errors"])

    roadmap = final_state.get("roadmap")
    coverage_analysis = final_state.get("coverage_analysis")
    explanation = final_state.get("explanation")
    if roadmap is None or coverage_analysis is None or explanation is None:
        missing = [
            name
            for name, value in (
                ("roadmap", roadmap),
                ("coverage_analysis", coverage_analysis),
                ("explanation", explanation),
            )
            if value is None
        ]
        raise HTTPException(
            status_code=500,
            detail=[f"graph state missing required field(s): {', '.join(missing)}"],
        )

    data = RecommendResponse(
        jobs=final_state.get("recommended_jobs") or [],
        primary_combos=final_state.get("primary_combos") or [],
        secondary_combos=final_state.get("secondary_combos") or [],
        roadmap=roadmap,
        coverage_analysis=coverage_analysis,
        explanation=explanation,
    )
    return ApiResponse.ok(data=data)
