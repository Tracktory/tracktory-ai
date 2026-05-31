"""챗봇 엔드포인트 POST /api/v1/ai/chat — Spring 내부 호출 전용 (X-Internal-Token)"""

from fastapi import APIRouter, Depends, Request
from langgraph.graph.state import CompiledStateGraph

from tracktory.api.auth import verify_internal_token
from tracktory.api.models.chat import ChatReq, ChatRes
from tracktory.api.response.base import ApiResponse
from tracktory.chatbot.runner import run_chat_turn

router = APIRouter(
    prefix="/api/v1/ai/chat",
    tags=["chat"],
    dependencies=[Depends(verify_internal_token)],
)


def _get_graph(request: Request) -> CompiledStateGraph:
    """테스트에서 dependency_overrides 로 교체 가능한 lifespan 그래프"""
    return request.app.state.chatbot_graph


@router.post("", response_model=ApiResponse[ChatRes])
def chat(
    body: ChatReq,
    graph: CompiledStateGraph = Depends(_get_graph),
) -> ApiResponse[ChatRes]:
    """그래프 한 턴 실행 (sync — 블로킹 LLM/RAG 안전)"""
    response, choices, _ = run_chat_turn(
        graph,
        message=body.message,
        user_context=body.user_context.model_dump(exclude_none=True),
        thread_id=body.thread_id,
    )
    return ApiResponse.ok(data=ChatRes(thread_id=body.thread_id, message=response, choices=choices))
