from tracktory.api.response.base import BaseResponse
from tracktory.api.response.codes import SuccessResponseCode


class SuccessResponse[T](BaseResponse):
    http_status: int
    data: T | None = None

    @classmethod
    def ok(cls, data: T) -> "SuccessResponse[T]":
        c = SuccessResponseCode.SUCCESS_OK
        return cls(is_success=True, http_status=c.http_status, message=c.message, data=data)

    @classmethod
    def created(cls, data: T) -> "SuccessResponse[T]":
        c = SuccessResponseCode.SUCCESS_CREATED
        return cls(is_success=True, http_status=c.http_status, message=c.message, data=data)

    @classmethod
    def empty(cls) -> "SuccessResponse[None]":
        c = SuccessResponseCode.SUCCESS_OK
        return cls(is_success=True, http_status=c.http_status, message=c.message, data=None)  # type: ignore[return-value]
