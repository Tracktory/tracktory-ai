"""챗봇 엔드포인트 요청·응답 스키마.

설계 원칙:
- FastAPI 는 stateless. user 정보·대화 히스토리는 매 요청에 Spring 이 동봉.
- 멀티턴은 conversation_id + history 로 표현 — 첫 턴은 history=[], 새 대화면 conversation_id=None.
- 응답에 conversation_id 를 반드시 포함 (신규 발급 시 클라이언트가 다음 턴에 재사용).
"""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """대화 한 턴."""

    role: Literal["user", "assistant"]
    content: str


class UserContext(BaseModel):
    """챗봇 응답 개인화에 사용하는 사용자 정보.

    Spring Boot 가 DB 에서 조회한 결과를 매 요청에 담아 보낸다.
    FastAPI 는 DB 에 직접 접근하지 않으므로 이 필드가 유일한 사용자 컨텍스트 소스.

    TODO: 백엔드(Spring) 와 필드 확정 후 채워넣기. 현재는 placeholder.
    예상 필드: admission_year, department, current_tracks, interests,
              completed_courses, work_values, ncs_studied 등 (온보딩 데이터 일부).
    """

    user_id: int


class ChatReq(BaseModel):
    """챗봇 질의 요청"""

    message: str = Field(..., min_length=1, description="사용자 질문")
    conversation_id: str | None = Field(
        None, description="멀티턴 대화 식별자 — 새 대화면 None, 서버가 신규 발급"
    )
    history: list[Message] = Field(
        default_factory=list, # 인스턴스마다 새 빈 리스트 생성
        description="이전 대화 턴 — 첫 질문이면 빈 리스트",
    )
    user_context: UserContext = Field(
        ...,
        description="개인화용 사용자 정보",
    )


class ChatRes(BaseModel):
    """챗봇 응답"""

    message: str = Field(..., description="LLM 응답 텍스트")
    conversation_id: str = Field(..., description="대화 식별자 (재사용 또는 신규)")
