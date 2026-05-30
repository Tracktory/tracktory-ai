"""챗봇 콘솔 인터랙티브 실행

질문 입력 → 그래프 stream 실행 → 각 노드 진행 로그 → 응답 + 후속 선택지 + 경과 시간

실행:
    uv run python -m tracktory.chatbot.main
    uv run python -m tracktory.chatbot.main --verbose            # DEBUG 로그까지
    uv run python -m tracktory.chatbot.main --thread-id <id>     # 대화 세션 분리

로그 파일:
    logs/chatbot_YYYYMMDD.log (UTF-8)

영속 저장:
    data/checkpoints/chatbot.sqlite — 같은 thread_id 면 프로세스 재시작 후에도 히스토리 복원
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from tracktory.chatbot.factory import CHECKPOINT_DB_PATH, build_default_chatbot_graph
from tracktory.chatbot.logging_setup import setup_logging
from tracktory.chatbot.runner import run_chat_turn

_EXIT_COMMANDS = {"exit", "quit"}


def _run_turn(
    graph: CompiledStateGraph,
    query: str,
    thread_id: str,
    logger: logging.Logger,
    user_context: dict[str, Any],
) -> tuple[str, list[str], float] | None:
    """공용 runner 호출 + 콘솔 후처리 (에러·빈 응답 폴백, 빈 줄 구분)"""
    try:
        response, choices, elapsed = run_chat_turn(
            graph, message=query, user_context=user_context, thread_id=thread_id
        )
    except Exception as exc:
        logger.exception("그래프 실행 중 에러")
        print(f"\n[에러] {exc}\n")
        return None

    if not response:
        response = "(응답 없음)"
    _log_blank_lines(logger, count=2)
    return response, choices, elapsed


def _log_blank_lines(logger: logging.Logger, count: int = 2) -> None:
    """logger 의 모든 handler 에 timestamp 없는 raw 빈 줄 N 개 — turn 구분용"""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            for _ in range(count):
                handler.stream.write("\n")
            handler.flush()


def _wrap_preserving_lines(text: str, width: int = 80) -> str:
    """줄바꿈·불릿 구조 유지하면서 긴 줄만 wrap"""
    wrapped_lines: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            wrapped_lines.append("")
        elif line.lstrip().startswith(("-", "*")):
            # 불릿 — 들여쓰기 유지하면서 wrap
            wrapped_lines.append(textwrap.fill(line, width=width, subsequent_indent="  "))
        else:
            wrapped_lines.append(textwrap.fill(line, width=width))
    return "\n".join(wrapped_lines)


def _print_response(response: str, choices: list[str], elapsed: float) -> None:
    """응답 + 후속 선택지 + 경과 시간 console 출력 (불릿·줄바꿈 보존하며 wrap)"""
    print(f"\n[응답] ({elapsed:.2f}s)")
    print(_wrap_preserving_lines(response, width=80))
    print("\n[후속 질문]")
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}. {choice}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="챗봇 콘솔 실행")
    parser.add_argument("--verbose", action="store_true", help="DEBUG 로그까지 출력")
    parser.add_argument(
        "--thread-id",
        default="console-session-1",
        help="대화 세션 ID (같은 ID 면 히스토리 유지)",
    )
    parser.add_argument(
        "--user-context-file",
        type=Path,
        default=None,
        help="user_context JSON fixture 경로 (예: src/tracktory/chatbot/fixtures/user_context_sample_year2.json). "
        "미지정 시 빈 dict — 개인화 없는 일반 응답",
    )
    return parser.parse_args()


def _load_user_context(path: Path | None, logger: logging.Logger) -> dict[str, Any]:
    """fixture 파일 로드 — 미지정/없음/파싱실패 시 빈 dict 로 안전 폴백"""
    if path is None:
        return {}
    if not path.exists():
        logger.warning("user_context_file 없음 — 빈 dict 로 진행: %s", path)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("user_context_file 파싱 실패 — 빈 dict 로 진행: %s (%s)", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("user_context_file 이 dict 아님 — 빈 dict 로 진행: %s", path)
        return {}
    return data


def main() -> None:
    args = _parse_args()
    logger = setup_logging(verbose=args.verbose)
    logger.info(
        "챗봇 시작 (thread_id=%s, verbose=%s, db=%s, user_context_file=%s)",
        args.thread_id,
        args.verbose,
        CHECKPOINT_DB_PATH,
        args.user_context_file,
    )

    user_context = _load_user_context(args.user_context_file, logger)
    if user_context:
        logger.info("user_context 로드 완료 — 키 %d개", len(user_context))

    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        graph = build_default_chatbot_graph(checkpointer=saver)

        print("\n챗봇 시작. 'exit' 또는 Ctrl+C 로 종료\n")

        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료")
                break

            if not query:
                continue
            if query.lower() in _EXIT_COMMANDS:
                print("\n종료")
                break

            if result := _run_turn(graph, query, args.thread_id, logger, user_context):
                _print_response(*result)


if __name__ == "__main__":
    main()
