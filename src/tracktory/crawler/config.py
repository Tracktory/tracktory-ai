"""
크롤링 시스템 설정 모듈
- 크롤러 전용 설정 (API 키, 딜레이, URL 등)
- 공통 설정은 tracktory.common.config에서 가져옴
"""

import os
from pathlib import Path

from tracktory.common.categories import IT_JOB_CATEGORIES, WANTED_CATEGORY_TAGS
from tracktory.common.config import config as common_config


class CrawlingConfig:
    """크롤링 설정 싱글톤"""

    # Paths (from common config)
    PROJECT_ROOT: Path = common_config.PROJECT_ROOT
    DATA_RAW_DIR: Path = common_config.DATA_RAW_DIR
    DATA_PROCESSED_DIR: Path = common_config.DATA_PROCESSED_DIR
    LOG_DIR: Path = common_config.LOG_DIR

    # API Keys
    SARAMIN_API_KEY: str = os.getenv("SARAMIN_API_KEY", "")

    # Saramin settings
    SARAMIN_API_URL: str = "https://oapi.saramin.co.kr/job-search"
    SARAMIN_MAX_DAILY_CALLS: int = 500
    SARAMIN_REQUEST_DELAY: float = 1.0
    SARAMIN_DETAIL_DELAY: float = 1.5
    SARAMIN_RESULTS_PER_PAGE: int = 110

    # Wanted settings
    WANTED_BASE_URL: str = "https://www.wanted.co.kr"
    WANTED_JOB_LIST_URL: str = "https://www.wanted.co.kr/wdlist/518?country=kr&job_sort=job.latest_order&years=-1&locations=all"
    WANTED_DELAY_MIN: float = 2.0
    WANTED_DELAY_MAX: float = 3.5
    WANTED_VIEWPORT_WIDTH: int = 1280
    WANTED_VIEWPORT_HEIGHT: int = 800
    WANTED_BATCH_SAVE_SIZE: int = 10

    # Wanted category settings (from common categories)
    WANTED_CATEGORY_TAGS: dict[str, list[int]] = WANTED_CATEGORY_TAGS
    WANTED_CATEGORY_LIST_URL_TEMPLATE: str = (
        "https://www.wanted.co.kr/wdlist/518/{tag_id}"
        "?country=kr&job_sort=job.latest_order&years=-1&locations=all"
    )
    WANTED_DEFAULT_PER_CATEGORY: int = 200

    # Retry settings
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BASE_DELAY: float = 1.0
    RETRY_MAX_DELAY: float = 30.0

    # IT job categories (from common categories)
    IT_JOB_CATEGORIES: dict[str, list[str]] = IT_JOB_CATEGORIES

    def validate(self) -> None:
        """필수 설정값 검증"""
        common_config.validate()

    def require_saramin_key(self) -> str:
        """사람인 API 키 필수 확인"""
        if not self.SARAMIN_API_KEY:
            raise ValueError(
                "SARAMIN_API_KEY가 설정되지 않았습니다.\n"
                "1. https://oapi.saramin.co.kr 에서 API 키를 발급받으세요.\n"
                "2. 프로젝트 루트의 .env 파일에 SARAMIN_API_KEY=발급받은키 를 추가하세요."
            )
        return self.SARAMIN_API_KEY


config = CrawlingConfig()
