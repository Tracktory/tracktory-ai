from tracktory.api.response.base import BaseResponse
from tracktory.api.response.codes import ErrorResponseCode


class ErrorResponse[T](BaseResponse):
    http_status: int
    data: T | None = None

    @classmethod
    def from_code(cls, code: ErrorResponseCode) -> "ErrorResponse[None]":
        return cls(is_success=False, http_status=code.http_status, message=code.message, data=None)  # type: ignore[return-value]

    @classmethod
    def of(
        cls,
        code: ErrorResponseCode,
        message: str | None = None,
        data: T | None = None,
    ) -> "ErrorResponse[T]":
        return cls(
            is_success=False,
            http_status=code.http_status,
            message=message if message is not None else code.message,
            data=data,
        )
