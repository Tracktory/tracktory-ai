"""챗봇 엔드포인트 요청·응답 스키마

설계 원칙:
- FastAPI 는 stateless, 대화 히스토리는 SQLite checkpointer 가 `thread_id` 단위로 영속
- `thread_id` 는 프론트엔드가 보내는 값, Spring 은 생성·관리하지 않고 그대로 FastAPI 로 전달(relay)만 함
- `user_context` 는 매 요청에 동봉 (Spring 이 DB 에서 조회)
- 같은 `thread_id` 로 재요청 시 직전 히스토리가 자동 복원
- 응답에 `thread_id` 를 그대로 echo 하고 후속 질문 후보(`choices`)를 함께 반환
"""

from pydantic import BaseModel, Field


class CompletedSubject(BaseModel):
    """이수 과목 — 다음 학기 추천 시 중복 회피에 사용"""

    name: str
    year: int = Field(..., ge=1, le=4, description="이수 학년")
    semester: int = Field(..., ge=1, le=2, description="이수 학기")


class UserContext(BaseModel):
    """챗봇 응답 개인화에 쓰는 사용자 온보딩·학적 정보

    Spring Boot 가 DB 에서 조회해 매 요청에 담아 보낸다
    FastAPI 는 DB 에 직접 접근하지 않으므로 이 필드가 유일한 사용자 컨텍스트 소스
    선택 필드는 미보유 시 누락(None) — 그래프는 `.get()` 으로 안전하게 읽는다
    """

    # --- 필수 ---
    user_id: int
    name: str
    entry_year: int = Field(..., description="입학년도 (YYYY)")
    grade: int = Field(..., ge=1, le=4, description="현재 학년")
    college: str
    department: str

    # --- 선택 ---
    tracks: list[str] | None = None
    interests: list[str] | None = None
    study_fields: list[str] | None = None
    tech_stacks: list[str] | None = None
    company_types: list[str] | None = None
    work_values: list[str] | None = None
    completed_subjects: list[CompletedSubject] | None = None


class ChatReq(BaseModel):
    """챗봇 질의 요청"""

    thread_id: str = Field(
        ..., min_length=1, description="대화 세션 ID — 같은 값이면 히스토리 복원"
    )
    message: str = Field(..., min_length=1, description="사용자 질문 본문")
    user_context: UserContext


class ChatRes(BaseModel):
    """챗봇 응답"""

    thread_id: str = Field(..., description="요청과 동일한 대화 세션 ID")
    message: str = Field(..., description="챗봇 응답 본문 — 평문 3~6 문장")
    choices: list[str] = Field(default_factory=list, description="후속 질문 후보 1~3개")
