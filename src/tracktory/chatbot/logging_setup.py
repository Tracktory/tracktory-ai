"""챗봇 로깅 설정 — 콘솔 + 파일 핸들러 (UTF-8)

진입점(예: FastAPI startup, pytest fixture)에서 한 번 호출
이후 코드는 ``logging.getLogger("chatbot")`` 으로 동일 로거 사용
"""

import logging
import sys
from datetime import datetime

from tracktory.common.config import config


def setup_logging(
    name: str = "chatbot",
    verbose: bool = False,
    file_name: str | None = None,
) -> logging.Logger:
    """챗봇 로거 설정 및 반환

    Args:
        name: 로거 이름
        verbose: True 면 DEBUG, False 면 INFO
        file_name: 로그 파일 prefix (지정 시 이 값, 미지정 시 name)
            — 콘솔/API 진입점별로 파일 분리할 때 사용

    Returns:
        설정된 Logger 인스턴스
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    # logger 는 항상 DEBUG 까지 받아 handler 별로 필터링
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러 — verbose 면 DEBUG, 아니면 INFO 만
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 파일 핸들러 — 항상 DEBUG 전부 캡처
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"{file_name or name}_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
