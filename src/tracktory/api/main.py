"""tracktory-ai API 진입점.

앱 시작 시 추천 그래프를 1 회 컴파일하여 첫 요청 cold start latency 를
사용자 경계 밖으로 옮긴다. 운영 boundary 가 아직 조립되지 않은 환경에서는
warm-up 을 graceful 하게 skip 한다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tracktory.api.dependencies import (
    get_pipeline_clients,
    get_pipeline_config,
    reset_pipeline_clients,
)
from tracktory.api.exception_handlers import register_exception_handlers
from tracktory.api.routers import chat, recommend
from tracktory.graph.pipeline import get_recommendation_graph

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """운영 boundary 묶음 조립 + 추천 그래프 warm-up 을 startup 에서 1 회 수행한다.

    boundary 조립과 컴파일은 best-effort 다. 자격증명(``RAGFLOW_*`` /
    ``OPENAI_API_KEY``)이나 카탈로그 YAML 이 없는 환경에서는 warm-up 을 skip
    하고 startup 을 막지 않는다 — 첫 요청에서 동일 의존성 주입이 다시 시도되며
    거기서 표준 envelope 으로 에러가 표면화된다. boundary 가 정상 주입되면 첫
    요청 cold start latency 가 startup 으로 옮겨진다 (동일 lru_cache 재사용).
    """
    try:
        clients = get_pipeline_clients()
        config = get_pipeline_config()
        get_recommendation_graph(clients, config)
        logger.info("recommendation graph warmed up")
    except Exception:
        # 미설정(자격증명/카탈로그 부재) 또는 compile/import 실패 모두 startup 을
        # 막지 않는다. 첫 요청에서 재시도되므로 stack 을 남겨 디버깅 비용만 줄인다.
        logger.exception("recommendation graph warm-up skipped; continuing startup")
    yield
    get_recommendation_graph.cache_clear()  # type: ignore[attr-defined]
    reset_pipeline_clients()


app = FastAPI(title="Tracktory AI API", lifespan=lifespan)

register_exception_handlers(app)

app.include_router(recommend.router)
app.include_router(chat.router)
