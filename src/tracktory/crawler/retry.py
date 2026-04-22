"""
재시도 데코레이터 - 지수 백오프 + 지터
동기/비동기 모두 지원
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("crawling.retry")


def retry(
    func: Callable | None = None,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """동기 함수에 지수 백오프 재시도 로직을 적용하는 데코레이터.

    지연 시간 = min(base_delay * 2^시도횟수 + random(0, 1), max_delay)
    최대 시도 횟수 초과 시 마지막 예외를 그대로 다시 발생시킵니다.

    Args:
        func: 데코레이터를 직접 적용할 때 전달되는 함수 (생략 가능).
        max_attempts: 최대 시도 횟수. 기본값 3.
        base_delay: 첫 번째 재시도의 기본 지연 시간(초). 기본값 1.0.
        max_delay: 지연 시간의 상한선(초). 기본값 30.0.
        exceptions: 재시도를 트리거할 예외 타입 튜플. 기본값 (Exception,).

    Returns:
        데코레이터 또는 래핑된 함수.

    Examples:
        @retry
        def fetch(): ...

        @retry(max_attempts=5, base_delay=2.0)
        def fetch(): ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        logger.error(
                            "[retry] '%s' 최대 재시도 횟수(%d) 초과. 마지막 오류: %s",
                            fn.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = min(base_delay * (2**attempt) + random.random(), max_delay)
                    logger.warning(
                        "[retry] '%s' 시도 %d/%d 실패 | 오류: %s | %.2f초 후 재시도",
                        fn.__name__,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            # 이 지점에 도달하는 경우는 없지만 타입 검사를 위해 재발생
            raise last_exc  # type: ignore[misc]

        return wrapper

    # @retry 형태 (인자 없이 직접 적용)
    if func is not None:
        return decorator(func)

    # @retry(...) 형태 (인자와 함께 적용)
    return decorator


def async_retry(
    func: Callable | None = None,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """비동기 함수에 지수 백오프 재시도 로직을 적용하는 데코레이터.

    retry 와 동일한 백오프 전략을 사용하며, 대기 시 asyncio.sleep 을 사용해
    이벤트 루프를 블로킹하지 않습니다.

    Args:
        func: 데코레이터를 직접 적용할 때 전달되는 함수 (생략 가능).
        max_attempts: 최대 시도 횟수. 기본값 3.
        base_delay: 첫 번째 재시도의 기본 지연 시간(초). 기본값 1.0.
        max_delay: 지연 시간의 상한선(초). 기본값 30.0.
        exceptions: 재시도를 트리거할 예외 타입 튜플. 기본값 (Exception,).

    Returns:
        데코레이터 또는 래핑된 코루틴 함수.

    Examples:
        @async_retry
        async def fetch(): ...

        @async_retry(max_attempts=5, exceptions=(aiohttp.ClientError,))
        async def fetch(): ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        logger.error(
                            "[async_retry] '%s' 최대 재시도 횟수(%d) 초과. 마지막 오류: %s",
                            fn.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = min(base_delay * (2**attempt) + random.random(), max_delay)
                    logger.warning(
                        "[async_retry] '%s' 시도 %d/%d 실패 | 오류: %s | %.2f초 후 재시도",
                        fn.__name__,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
