"""tracktory-ai API 진입점 — startup 에서 챗봇 로거·추천 그래프·챗봇 그래프 1회 구성"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.sqlite import SqliteSaver

from tracktory.api.dependencies import (
    get_pipeline_clients,
    get_pipeline_config,
    reset_pipeline_clients,
)
from tracktory.api.exception_handlers import register_exception_handlers
from tracktory.api.request_id import RequestIdLogFilter, request_id_middleware
from tracktory.api.routers import chat, recommend
from tracktory.chatbot.factory import CHECKPOINT_DB_PATH, build_default_chatbot_graph
from tracktory.chatbot.logging_setup import setup_logging
from tracktory.graph.pipeline import get_recommendation_graph

logger = logging.getLogger(__name__)


def _setup_chatbot_logger() -> None:
    """chatbot 로거 → logs/api_chatbot_YYYYMMDD.log + request_id 박힌 포매터·필터 부착"""
    chatbot_logger = setup_logging(name="chatbot", file_name="api_chatbot")
    fmt = logging.Formatter(
        "%(asctime)s [%(request_id)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    filt = RequestIdLogFilter()
    for h in chatbot_logger.handlers:
        h.setFormatter(fmt)
        h.addFilter(filt)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """startup: 챗봇 로거 + 추천 warm-up + 챗봇 그래프(`app.state.chatbot_graph`) 구성"""
    _setup_chatbot_logger()

    try:
        get_recommendation_graph(get_pipeline_clients(), get_pipeline_config())
        logger.info("recommendation graph warmed up")
    except NotImplementedError:
        logger.warning("skipping recommendation graph warm-up: boundary clients not configured")
    except Exception:
        # boundary 주입 후 compile/config 실패는 첫 요청에서 또 터지므로 stack 남김
        logger.exception("recommendation graph warm-up failed; continuing startup")

    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        app.state.chatbot_graph = build_default_chatbot_graph(checkpointer=saver)
        yield

    get_recommendation_graph.cache_clear()  # type: ignore[attr-defined]
    reset_pipeline_clients()


app = FastAPI(title="Tracktory AI API", lifespan=lifespan)

# 라우트·인증보다 먼저 — 인증 실패 응답에도 X-Request-Id 박히도록
app.middleware("http")(request_id_middleware)

register_exception_handlers(app)

app.include_router(recommend.router)
app.include_router(chat.router)
