from datetime import datetime

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    is_success: bool
    http_status: int
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
