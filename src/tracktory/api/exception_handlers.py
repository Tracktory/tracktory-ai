"""전역 예외 핸들러

FastAPI 기본 에러 응답을 명세 표준 envelope `ApiResponse`(success/data/error)로 변환
성공·실패가 동일 shape 를 갖도록 보장해 Spring 통합을 단순화
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tracktory.api.response.base import ApiResponse, ErrorDetail
from tracktory.api.response.codes import ErrorCode

# Pydantic v2 error type → 명세 reason 표기 매핑 (미정의 시 type 그대로)
_REASON_ALIASES = {"missing": "required"}


def _json(
    code: ErrorCode,
    *,
    message: str | None = None,
    details: list[ErrorDetail] | None = None,
) -> JSONResponse:
    payload: ApiResponse[None] = ApiResponse.fail(code, message=message, details=details)
    return JSONResponse(status_code=code.http_status, content=payload.model_dump())


def _to_details(exc: RequestValidationError) -> list[ErrorDetail]:
    """Pydantic 검증 오류를 {field, reason} 목록으로 — loc 의 'body' 접두는 제거"""
    details: list[ErrorDetail] = []
    for err in exc.errors():
        loc = [str(part) for part in err["loc"] if part != "body"]
        reason = _REASON_ALIASES.get(err["type"], err["type"])
        details.append(ErrorDetail(field=".".join(loc), reason=reason))
    return details


def register_exception_handlers(app: FastAPI) -> None:
    """앱에 표준 예외 핸들러를 등록"""

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _json(ErrorCode.VALIDATION_FAILED, details=_to_details(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 403:
            return _json(ErrorCode.FORBIDDEN_ERROR)
        if exc.status_code == 404:
            return _json(ErrorCode.NOT_FOUND)
        if exc.status_code == 405:
            return _json(ErrorCode.METHOD_NOT_ALLOWED)
        if exc.status_code >= 500:
            return _json(ErrorCode.INTERNAL_SERVER_ERROR)
        message = exc.detail if isinstance(exc.detail, str) else None
        return _json(ErrorCode.BAD_REQUEST, message=message)

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        # 예측 못 한 예외 — 500, traceback 은 uvicorn 로그로
        return _json(ErrorCode.INTERNAL_SERVER_ERROR)
