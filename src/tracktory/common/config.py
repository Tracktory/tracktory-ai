"""
프로젝트 공통 설정 모듈
- .env에서 환경변수 로드
- 프로젝트 경로 해석
"""

from pathlib import Path

from dotenv import load_dotenv

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


config = CommonConfig()
