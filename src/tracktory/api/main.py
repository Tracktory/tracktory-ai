from fastapi import FastAPI

from tracktory.api.exception_handlers import register_exception_handlers
from tracktory.api.routers import chat, recommend

app = FastAPI(title="Tracktory AI API")

register_exception_handlers(app)

app.include_router(recommend.router)
app.include_router(chat.router)
