"""X-Request-Id 미들웨어 — Spring 발급 ID 받아 ContextVar 보관 + 응답 헤더 echo

누락 시 400 (FastAPI 자체 발급하면 Spring 로그와 ID 가 갈라져 트레이스 끊김)
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from fastapi import Request
from starlette.responses import JSONResponse, Response

from tracktory.api.response.base import ApiResponse
from tracktory.api.response.codes import ErrorCode

REQUEST_ID_HEADER = "X-Request-Id"

logger = logging.getLogger("api.request_id")

# 요청 스코프 — 미들웨어 set, 로그 필터 get. default="-" 는 부팅 로그 등 컨텍스트 밖 안전망
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """X-Request-Id 필수 (Spring 발급) — 누락 시 400, 있으면 ContextVar set + 응답 echo"""
    request_id = request.headers.get(REQUEST_ID_HEADER)
    if not request_id:
        logger.warning("X-Request-Id 누락 — Spring propagate 확인 (path=%s)", request.url.path)
        payload: ApiResponse[None] = ApiResponse.fail(
            ErrorCode.BAD_REQUEST,
            message="X-Request-Id 헤더는 필수입니다",
        )
        return JSONResponse(status_code=400, content=payload.model_dump())

    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        request_id_var.reset(token)


class RequestIdLogFilter(logging.Filter):
    """LogRecord 에 request_id 박아 포매터 `%(request_id)s` 로 노출 (Spring `%X{requestId}` 대응)"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True
