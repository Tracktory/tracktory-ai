from enum import Enum


class ErrorCode(Enum):
    """오류 코드 — `(http_status, 기본 message)`, `code` 문자열은 멤버명을 그대로 사용"""

    BAD_REQUEST = (400, "잘못된 요청입니다.")
    FORBIDDEN_ERROR = (403, "내부 인증에 실패했습니다.")
    NOT_FOUND = (404, "존재하지 않는 엔드포인트입니다. 요청 URL을 확인해주세요.")
    METHOD_NOT_ALLOWED = (405, "지원하지 않는 HTTP 메소드입니다.")
    VALIDATION_FAILED = (422, "요청 형식이 올바르지 않습니다.")
    INTERNAL_SERVER_ERROR = (500, "예기치 못한 서버 오류가 발생했습니다.")

    def __init__(self, http_status: int, message: str) -> None:
        self.http_status = http_status
        self.message = message
