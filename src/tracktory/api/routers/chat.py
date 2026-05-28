"""챗봇 엔드포인트.

현재는 더미 응답 — endpoint shape·계약을 먼저 확정해서 Spring 측 통합을
unblock 한다. 챗봇 LangGraph 가 완성되면 dummy 응답 구간을 graph.invoke(...)
호출로 교체.
"""

import uuid

from fastapi import APIRouter

from tracktory.api.models.chat import ChatReq, ChatRes
from tracktory.api.response.success import SuccessResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=SuccessResponse[ChatRes])
async def chat(request: ChatReq) -> SuccessResponse[ChatRes]:
    """챗봇 질의 응답.

    TODO: 챗봇 graph 완성 후 아래 더미 응답 블록을 graph.invoke({
        "message": request.message,
        "history": request.history,
        "user_context": request.user_context,
    }) 결과로 교체.
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # --- 더미 응답 (graph 완성 시 제거) ---
    data = ChatRes(
        message=f"(dummy) 받은 질문: {request.message}",
        conversation_id=conversation_id,
    )
    # --- 더미 응답 끝 ---

    return SuccessResponse.ok(data=data)
