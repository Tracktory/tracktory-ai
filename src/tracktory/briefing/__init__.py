"""AI 직무 브리핑 — 추천 직무에 연결된 트렌드·역량 카드.

추천 결과를 보조하는 별도 표면(홈 브리핑 시트)으로, 추천 파이프라인과
독립적으로 직무 식별자 → 큐레이션 브리핑을 조회한다. 데모는 사전 큐레이션된
고정 출력을 사용하므로 외부 검색·LLM 호출 없이 정적 카탈로그만 읽는다.
"""

from tracktory.briefing.models import BriefingSource, JobBriefing
from tracktory.briefing.service import (
    DEFAULT_BRIEFING_CATALOG_PATH,
    BriefingCatalogError,
    BriefingService,
)

__all__ = [
    "DEFAULT_BRIEFING_CATALOG_PATH",
    "BriefingCatalogError",
    "BriefingService",
    "BriefingSource",
    "JobBriefing",
]
