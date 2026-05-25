"""챗봇 SQLite checkpointer 내용 조회 유틸 — 결과는 logs/ 에 파일로 기록.

사용:
    # 전체 thread 목록 + 각 thread 의 최신 state 요약
    uv run python scripts/inspect_chatbot_checkpoints.py

    # 특정 thread 만 자세히
    uv run python scripts/inspect_chatbot_checkpoints.py --thread-id test1
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from tracktory.common.config import config as common_config

_LOGGER_NAME = "checkpoint_inspect"
_DB_PATH = common_config.PROJECT_ROOT / "data" / "checkpoints" / "chatbot.sqlite"


def _setup_file_logger() -> tuple[logging.Logger, Path]:
    """logs/checkpoint_inspect_YYYYMMDD_HHMMSS.log 에 기록하는 logger 반환.

    매 실행마다 새 파일 — 시점별 스냅샷 비교 용이.
    콘솔 핸들러는 안 붙임 (사용자 요구상 콘솔 침묵).
    """
    log_dir = common_config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{_LOGGER_NAME}_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 재실행 시 중복 핸들러 방지

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def _list_threads(db_path: Path) -> list[tuple[str, int]]:
    """모든 thread_id + 각 thread 의 체크포인트 개수"""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
    ).fetchall()
    conn.close()
    return rows


def _log_thread(logger: logging.Logger, db_path: Path, thread_id: str) -> None:
    """특정 thread 의 최신 state 의 messages 를 logger 로 기록"""
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = saver.get_tuple(cfg)
        if snapshot is None:
            logger.info("thread_id=%r 의 체크포인트 없음", thread_id)
            return

        state = snapshot.checkpoint["channel_values"]
        messages = state.get("messages", [])
        intent = state.get("intent")
        response = state.get("response")

        logger.info("=" * 70)
        logger.info("thread_id=%r", thread_id)
        logger.info("  intent: %s", intent)
        logger.info("  response: %r", response)
        logger.info("  messages (%d개):", len(messages))
        for i, msg in enumerate(messages):
            cls = type(msg).__name__
            content = msg.content if hasattr(msg, "content") else str(msg)
            # 로그 파일이라 잘라낼 필요 없음 — 전체 그대로
            logger.info("    %d. %s: %s", i, cls, content)


def main() -> None:
    parser = argparse.ArgumentParser(description="챗봇 SQLite checkpointer 조회")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="특정 thread 의 messages 만 출력 (미지정 시 전체 thread 요약)",
    )
    args = parser.parse_args()

    logger, log_file = _setup_file_logger()

    if not _DB_PATH.exists():
        logger.error("DB 파일 없음: %s", _DB_PATH)
        print(f"DB 파일 없음: {_DB_PATH}", file=sys.stderr)
        print(f"로그: {log_file}")
        sys.exit(1)

    logger.info("DB: %s", _DB_PATH)

    threads = _list_threads(_DB_PATH)
    if not threads:
        logger.info("저장된 thread 없음")
        print(f"로그: {log_file}")
        return

    logger.info("[전체 thread %d개]", len(threads))
    for thread_id, count in threads:
        logger.info("  %s: 체크포인트 %d개", thread_id, count)

    if args.thread_id:
        _log_thread(logger, _DB_PATH, args.thread_id)
    else:
        for thread_id, _ in threads:
            _log_thread(logger, _DB_PATH, thread_id)

    # 콘솔에는 저장 위치만
    print(f"로그: {log_file}")


if __name__ == "__main__":
    main()
