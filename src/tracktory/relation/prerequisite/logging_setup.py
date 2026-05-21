"""
선수과목 파이프라인 로깅 설정
- 콘솔 + 파일 핸들러
- --verbose 모드 지원
"""

import logging
import sys
from datetime import datetime

from tracktory.common.config import config


def setup_logging(name: str = "prereq", verbose: bool = False) -> logging.Logger:
    """로거 설정 및 반환

    Args:
        name: 로거 이름
        verbose: True면 DEBUG, False면 INFO 레벨

    Returns:
        설정된 Logger 인스턴스
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"prereq_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
