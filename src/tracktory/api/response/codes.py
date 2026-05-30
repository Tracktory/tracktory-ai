from enum import Enum


class SuccessResponseCode(Enum):
    SUCCESS_OK = (200, "호출에 성공했습니다.")
    SUCCESS_CREATED = (201, "생성에 성공했습니다.")

    def __init__(self, http_status: int, message: str) -> None:
        self.http_status = http_status
        self.message = message


class ErrorResponseCode(Enum):
    BAD_REQUEST_ERROR = (400, "잘못된 요청입니다.")
    INVALID_HTTP_MESSAGE_BODY = (400, "HTTP 요청 바디의 형식이 잘못되었습니다.")
    INVALID_HTTP_MESSAGE_PARAMETER = (400, "HTTP 요청 파라미터 형식이 잘못되었습니다.")
    FORBIDDEN_ERROR = (403, "내부 인증에 실패했습니다.")
    NOT_FOUND_ENDPOINT = (404, "존재하지 않는 앤드포인트입니다. 요청 URL을 확인해주세요.")
    UNSUPPORTED_HTTP_METHOD = (405, "지원하지 않는 HTTP 메소드입니다.")
    SERVER_ERROR = (500, "서버 내부에서 알 수 없는 에러가 발생했습니다.")

    def __init__(self, http_status: int, message: str) -> None:
        self.http_status = http_status
        self.message = message
