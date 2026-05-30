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

from tracktory.api.dependencies import get_pipeline_clients, get_pipeline_config
from tracktory.api.exception_handlers import register_exception_handlers
from tracktory.api.routers import chat, recommend
from tracktory.graph.pipeline import get_recommendation_graph

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """추천 그래프 warm-up 을 startup 에서 1 회 수행한다.

    stub boundary 환경에서는 NotImplementedError 가 raise 되어 warm-up 을
    skip 한다. 실제 호출 시점에서도 동일 lru_cache 가 사용되므로 boundary
    가 주입되면 첫 요청에서 1 회만 컴파일된다.
    """
    try:
        clients = get_pipeline_clients()
        config = get_pipeline_config()
        get_recommendation_graph(clients, config)
        logger.info("recommendation graph warmed up")
    except NotImplementedError:
        logger.warning("skipping recommendation graph warm-up: boundary clients not configured")
    except Exception:
        # boundary 가 주입된 후의 compile / config / import 실패는 startup 을 통과해도
        # 첫 요청에서 의존성 주입이 다시 실패하므로, stack 을 남겨 운영 디버깅 비용을 줄인다.
        logger.exception("recommendation graph warm-up failed; continuing startup")
    yield
    get_recommendation_graph.cache_clear()  # type: ignore[attr-defined]


app = FastAPI(title="Tracktory AI API", lifespan=lifespan)

register_exception_handlers(app)

app.include_router(recommend.router)
app.include_router(chat.router)
