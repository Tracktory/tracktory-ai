"""전역 예외 핸들러.

FastAPI 기본 응답 포맷(예: 422 RequestValidationError)을 프로젝트 표준
`ErrorResponse` (BaseResponse 기반) 로 변환한다. Spring 통합 시 성공·실패
응답이 동일한 shape (is_success / http_status / message / timestamp / data)
을 유지하도록 보장.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tracktory.api.response.codes import ErrorResponseCode
from tracktory.api.response.error import ErrorResponse


def _error_response(code: ErrorResponseCode) -> JSONResponse:
    payload = ErrorResponse.from_code(code)
    return JSONResponse(status_code=payload.http_status, content=payload.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """앱에 표준 예외 핸들러를 등록한다."""

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(ErrorResponseCode.BAD_REQUEST_ERROR)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        if exc.status_code == 404:
            return _error_response(ErrorResponseCode.NOT_FOUND_ENDPOINT)
        if exc.status_code == 405:
            return _error_response(ErrorResponseCode.UNSUPPORTED_HTTP_METHOD)
        return _error_response(ErrorResponseCode.BAD_REQUEST_ERROR)

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        # 예측하지 못한 예외 — 500. 실제 traceback 은 uvicorn 로그로.
        return _error_response(ErrorResponseCode.SERVER_ERROR)
