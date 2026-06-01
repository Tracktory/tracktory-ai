# syntax=docker/dockerfile:1

# uv 공식 바이너리를 슬림 베이스에 복사 — pip install uv 보다 빠르고 버전 고정
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# 1) 의존성 레이어 — lock 안 바뀌면 캐시 재사용 (소스 변경에도 재설치 X)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) 소스 복사 후 프로젝트 자체 설치
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Spring(8080)과 충돌 방지 — FastAPI 는 8000
CMD ["uv", "run", "uvicorn", "tracktory.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
