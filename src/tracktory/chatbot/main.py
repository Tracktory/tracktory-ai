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
import logging
import textwrap
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from tracktory.chatbot.graph import build_chatbot_graph
from tracktory.chatbot.logging_setup import setup_logging
from tracktory.chatbot.rag.ragflow import RagFlowChatbotRetriever
from tracktory.common.config import config as common_config
from tracktory.prompts.chatbot.general_advice import GENERAL_ADVICE_PROMPT
from tracktory.prompts.chatbot.intent import (
    INTENT_CLASSIFIER_PROMPT,
    IntentClassification,
)
from tracktory.prompts.chatbot.rag_response import (
    RAG_RESPONSE_PROMPT,
    ChatbotResponse,
)

CHECKPOINT_DB_PATH = common_config.PROJECT_ROOT / "data" / "checkpoints" / "chatbot.sqlite"

_EXIT_COMMANDS = {"exit", "quit"}

# 노드 출력 dict key 별 한 줄 요약 포맷터
_FIELD_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "messages": lambda v: f"messages=+{len(v)}",
    "retrieved_docs": lambda v: f"retrieved_docs={len(v)}건",
    "search_keywords": lambda v: f"keywords={v}",
    "response": lambda v: f"response=({len(v)}자)",
    "response_choices": lambda v: f"choices={len(v)}개",
}


def _build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """4 종 의존성 + checkpointer 주입해서 그래프 컴파일"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return build_chatbot_graph(
        checkpointer=checkpointer,
        classifier=INTENT_CLASSIFIER_PROMPT | llm.with_structured_output(IntentClassification),
        retriever=RagFlowChatbotRetriever(),
        rag_response_chain=RAG_RESPONSE_PROMPT | llm.with_structured_output(ChatbotResponse),
        general_advice_chain=GENERAL_ADVICE_PROMPT | llm.with_structured_output(ChatbotResponse),
    )


def _summarize(node_output: Any) -> str:
    """노드 출력 dict 를 한 줄 로그 요약으로 (response 본문은 별도 DEBUG 로깅)"""
    if not isinstance(node_output, dict):
        return repr(node_output)
    parts: list[str] = []
    for k, v in node_output.items():
        if formatter := _FIELD_FORMATTERS.get(k):
            parts.append(formatter(v))
        elif isinstance(v, str) and len(v) > 60:
            parts.append(f"{k}={v[:60]!r}...")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _run_turn(
    graph: CompiledStateGraph,
    query: str,
    config: RunnableConfig,
    logger: logging.Logger,
) -> tuple[str, list[str], float] | None:
    """한 턴 실행 — (response, choices, elapsed) 반환. 에러 시 None"""
    logger.info("질문: %s", query)
    start = time.perf_counter()

    input_state = {
        "user_context": {},  # TODO: 백엔드 연동 시 실제 온보딩 정보
        "messages": [HumanMessage(content=query)],
    }

    try:
        for event in graph.stream(input_state, config=config):
            for node_name, node_output in event.items():
                logger.info("[%s] %s", node_name, _summarize(node_output))
    except Exception as exc:
        logger.exception("그래프 실행 중 에러")
        print(f"\n[에러] {exc}\n")
        return None

    elapsed = time.perf_counter() - start
    state = graph.get_state(config).values
    response = state.get("response") or "(응답 없음)"
    choices = state.get("response_choices") or []

    logger.info("응답 완료 (%.2fs, choices=%d개)", elapsed, len(choices))
    # 응답 본문·선택지는 DEBUG — 파일엔 저장, 콘솔엔 verbose 일 때만 (중복 방지)
    logger.debug("응답 본문:\n%s", response)
    logger.debug("후속 질문: %s", choices)
    _log_blank_lines(logger, count=2)  # turn 사이 시각적 구분

    return response, choices, elapsed


def _log_blank_lines(logger: logging.Logger, count: int = 2) -> None:
    """logger 의 모든 handler 에 timestamp 없는 raw 빈 줄 N 개 — turn 구분용"""
    for handler in logger.handlers:
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger = setup_logging(verbose=args.verbose)
    logger.info(
        "챗봇 시작 (thread_id=%s, verbose=%s, db=%s)",
        args.thread_id,
        args.verbose,
        CHECKPOINT_DB_PATH,
    )

    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        graph = _build_graph(checkpointer=saver)
        config: RunnableConfig = {"configurable": {"thread_id": args.thread_id}}

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

            if result := _run_turn(graph, query, config, logger):
                _print_response(*result)


if __name__ == "__main__":
    main()
