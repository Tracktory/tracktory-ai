from fastapi import FastAPI

from tracktory.api.routers import chat, recommend

app = FastAPI(title="Tracktory AI API")

app.include_router(recommend.router)
app.include_router(chat.router)
