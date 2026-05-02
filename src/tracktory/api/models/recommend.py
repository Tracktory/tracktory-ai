from pydantic import BaseModel


class RecommendReq(BaseModel):
    test_str: str  # 테스트용


class RecommendRes(BaseModel):
    test_str: str  # 테스트용
