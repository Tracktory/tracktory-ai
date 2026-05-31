"""챗봇 그래프 한 턴 실행 — 콘솔·API 공용 진입점"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger("chatbot")

# stream 진행 로그에서 긴 본문·리스트는 길이만 표시
_FIELD_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "messages": lambda v: f"messages=+{len(v)}",
    "retrieved_docs": lambda v: f"retrieved_docs={len(v)}건",
    "search_keywords": lambda v: f"keywords={v}",
    "response": lambda v: f"response=({len(v)}자)",
    "response_choices": lambda v: f"choices={len(v)}개",
}


def _summarize(node_output: Any) -> str:
    """노드 출력 dict → 한 줄 요약"""
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


def run_chat_turn(
    graph: CompiledStateGraph,
    *,
    message: str,
    user_context: dict[str, Any],
    thread_id: str,
) -> tuple[str, list[str], float]:
    """한 턴 stream 실행 — 질문·노드별·완료 로그, (response, choices, elapsed) 반환"""
    logger.info("질문: %s", message)
    start = time.perf_counter()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    input_state = {"user_context": user_context, "messages": [HumanMessage(content=message)]}

    for event in graph.stream(input_state, config=config):
        for node_name, node_output in event.items():
            logger.info("[%s] %s", node_name, _summarize(node_output))

    elapsed = time.perf_counter() - start
    state = graph.get_state(config).values
    response = state.get("response") or ""
    choices = state.get("response_choices") or []
    logger.info("응답 완료 (%.2fs, choices=%d개)", elapsed, len(choices))
    logger.debug("응답 본문:\n%s", response)
    logger.debug("후속 질문: %s", choices)
    return response, choices, elapsed
