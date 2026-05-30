"""내부 호출 인증 의존성.

이 AI 중계 서버는 외부에 직접 노출되지 않고 메인 백엔드(Spring)만 호출한다.
호출자 검증은 두 헤더로 이뤄진다:

- ``X-Internal-Token``: 메인 백엔드와 공유하는 preshared 토큰. 호출 주체를 게이트.
- ``X-User-Id``: 메인 백엔드가 JWT 검증 후 추출한 사용자 식별자. AI 서버는
  인증/DB 에 직접 접근하지 않고 이 헤더를 신뢰한다 (유일한 사용자 식별 소스).

토큰 검증 실패는 모두 403 으로 거부한다.
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

from fastapi import Header, HTTPException

from tracktory.common.config import settings

logger = logging.getLogger(__name__)


def verify_internal_token(
    x_internal_token: Annotated[str | None, Header()] = None,
) -> None:
    """내부 공유 토큰 헤더를 timing-safe 하게 검증한다.

    토큰 미설정(서버 환경변수 부재)·헤더 부재·불일치를 모두 403 으로 거부한다
    (deny-by-default). 토큰이 설정되지 않은 서버는 인증을 강제할 수 없으므로
    misconfiguration 을 열린 상태로 두지 않고 거부한다.

    비교는 timing attack 을 막기 위해 ``==`` 대신 ``secrets.compare_digest`` 를
    사용한다 — 일치 길이에 따라 응답 시간이 달라지지 않는다.

    Raises:
        HTTPException: 토큰 미설정/부재/불일치 시 403.
    """
    expected = settings.ai_internal_token
    if not expected:
        logger.warning("AI_INTERNAL_TOKEN 미설정 — 내부 호출을 거부한다")
        raise HTTPException(status_code=403, detail="internal authentication not configured")
    # compare_digest 는 비-ASCII str 에 TypeError 를 던지므로 bytes 로 인코딩해
    # 비교한다 — 어떤 토큰 값이 와도 일관되게 403 으로 거부하기 위함.
    if x_internal_token is None or not secrets.compare_digest(
        x_internal_token.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=403, detail="invalid internal token")


def get_user_id(x_user_id: Annotated[str, Header(min_length=1)]) -> str:
    """메인 백엔드가 전파한 사용자 식별자를 추출한다.

    필수 헤더 — 부재 시 FastAPI 요청 검증 단계에서 거부된다. 내부 토큰이
    유효하다는 것은 신뢰하는 메인 백엔드의 호출임을 뜻하고, 그 호출은 항상
    사용자 식별자를 동봉한다는 계약에 기반한다.
    """
    return x_user_id
