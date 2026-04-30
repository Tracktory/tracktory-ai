from fastapi import FastAPI

from tracktory.api.routers import recommend

app = FastAPI(title="Tracktory AI API")

app.include_router(recommend.router)
