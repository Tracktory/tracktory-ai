"""
프로젝트 공통 설정 모듈
- .env에서 환경변수 로드
- 프로젝트 경로 해석
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = 4 levels up: common -> tracktory -> src -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class CommonConfig:
    """프로젝트 공통 설정"""

    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    def validate(self) -> None:
        """필수 디렉터리 생성"""
        self.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    ragflow_api_key: str
    ragflow_base_url: str
    ragflow_dataset_id: str
    ragflow_reranker_id: str

    openai_api_key: str | None = None

    # 메인 백엔드(Spring)와 공유하는 내부 호출 토큰. 미설정(빈 문자열)이면
    # 내부 인증 의존성이 deny-by-default 로 모든 호출을 거부한다.
    ai_internal_token: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


config = CommonConfig()
settings = Settings()
