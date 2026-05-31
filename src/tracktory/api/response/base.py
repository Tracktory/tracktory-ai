"""API 응답 표준 envelope

성공·실패가 동일한 shape `{success, data, error}` 를 갖도록 통일
Spring 측 ApiResponse 와 1:1 대응시켜 통합을 단순화
"""

from pydantic import BaseModel

from tracktory.api.response.codes import ErrorCode


class ErrorDetail(BaseModel):
    """검증 실패 등 필드 단위 오류 사유 (주로 422 에서 사용)"""

    field: str
    reason: str


class ApiError(BaseModel):
    code: str  # ErrorCode 의 멤버명 (예: "AUTH_REQUIRED")
    message: str
    details: list[ErrorDetail] | None = None


class ApiResponse[T](BaseModel):
    success: bool
    data: T | None = None
    error: ApiError | None = None

    @classmethod
    def ok(cls, data: T) -> "ApiResponse[T]":
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(
        cls,
        code: ErrorCode,
        *,
        message: str | None = None,
        details: list[ErrorDetail] | None = None,
    ) -> "ApiResponse[T]":
        return cls(
            success=False,
            data=None,
            error=ApiError(code=code.name, message=message or code.message, details=details),
        )
