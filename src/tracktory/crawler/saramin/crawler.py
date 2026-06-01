"""
사람인 채용정보 API 크롤러
==========================

사람인 Open API를 호출하여 IT 직군 채용공고를 수집하고,
각 공고 상세 페이지에서 기술 스택 키워드를 추출한다.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

from tracktory.common.models import CrawlResult, JobPosting
from tracktory.common.tech_keywords import extract_tech_keywords
from tracktory.crawler.config import config
from tracktory.crawler.logging_setup import setup_logging
from tracktory.crawler.retry import retry
from tracktory.crawler.storage import (
    get_output_path,
    load_collected_ids,
    save_jobs_csv,
)

from .parser import get_total_count, parse_job_list

logger = logging.getLogger("crawling.saramin.crawler")

# 상세 페이지 요청에 사용할 User-Agent
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# 사람인 채용공고 본문 영역 CSS 선택자 (우선순위 순)
_DETAIL_CONTENT_SELECTORS = [
    ".job_detail_info",
    ".job_description",
    "#job_info",
    ".jd_contents",
    "article",
]

# API 일일 호출 한도 경고 임계치 (80%)
_API_WARN_THRESHOLD_RATIO = 0.8

# 배치 저장 크기
_BATCH_SAVE_SIZE = 20


class SaraminCrawler:
    """사람인 Open API 기반 채용공고 크롤러.

    API를 호출하여 채용공고 목록을 수집하고, 선택적으로 각 공고의
    상세 페이지를 방문하여 기술 스택 키워드를 추출한다.

    Attributes:
        logger: 크롤러 전용 로거.
        api_key: 사람인 API 액세스 키.
        api_call_count: 현재 세션의 API 호출 횟수.
    """

    def __init__(self, verbose: bool = False) -> None:
        """크롤러를 초기화한다.

        Args:
            verbose: True이면 DEBUG 레벨 로깅을 활성화한다.
        """
        self.logger = setup_logging("crawling.saramin", verbose=verbose)
        config.validate()
        self.api_key: str = ""
        self.api_call_count: int = 0
        self._warn_threshold: int = int(config.SARAMIN_MAX_DAILY_CALLS * _API_WARN_THRESHOLD_RATIO)

    def crawl(
        self,
        keywords: list[str] | None = None,
        max_jobs: int = 110,
        dry_run: bool = False,
        skip_detail: bool = False,
    ) -> CrawlResult:
        """채용공고 수집을 실행한다.

        Args:
            keywords: 검색 키워드 리스트. None이면 IT_JOB_CATEGORIES의
                      키 전체를 사용한다.
            max_jobs: 키워드당 최대 수집 공고 수.
            dry_run: True이면 API 호출 없이 더미 데이터를 반환한다.
            skip_detail: True이면 상세 페이지 방문을 생략한다.

        Returns:
            CrawlResult: 수집 결과 요약 및 공고 목록.
        """
        start_time = time.time()

        if dry_run:
            self.logger.info("드라이런 모드: 더미 데이터를 반환합니다.")
            dummy_jobs = self._create_dummy_jobs()
            elapsed = time.time() - start_time
            return CrawlResult(
                source="saramin",
                total_collected=len(dummy_jobs),
                total_skipped=0,
                total_failed=0,
                duration_seconds=elapsed,
                jobs=dummy_jobs,
            )

        # API 키 확인
        self.api_key = config.require_saramin_key()

        # 검색 키워드 결정
        if keywords is None:
            keywords = list(config.IT_JOB_CATEGORIES.keys())

        # 기수집 ID 로드 (재개 지원)
        csv_path = get_output_path("saramin")
        collected_ids = load_collected_ids("saramin", csv_path)
        self.logger.info("기수집 공고 %d건 로드 완료 (중복 건너뜀 대상)", len(collected_ids))

        all_jobs: list[JobPosting] = []
        total_skipped: int = 0
        total_failed: int = 0
        unsaved_jobs: list[JobPosting] = []

        for keyword in keywords:
            if self._is_api_exhausted():
                self.logger.warning("일일 API 호출 한도 도달, 수집을 중단합니다.")
                break

            self.logger.info("키워드 [%s] 수집 시작", keyword)
            keyword_jobs: list[JobPosting] = []
            start = 0

            while len(keyword_jobs) < max_jobs:
                if self._is_api_exhausted():
                    break

                # API 호출
                xml_root = self._call_api(keyword, start=start)
                if xml_root is None:
                    self.logger.warning(
                        "[%s] start=%d API 응답 없음, 다음 키워드로 이동",
                        keyword,
                        start,
                    )
                    total_failed += 1
                    break

                total = get_total_count(xml_root)
                page_jobs = parse_job_list(xml_root, keyword)

                if not page_jobs:
                    self.logger.info("[%s] 더 이상 결과 없음 (start=%d)", keyword, start)
                    break

                self.logger.info(
                    "[%s] start=%d -> %d건 파싱 (전체 %d건)",
                    keyword,
                    start,
                    len(page_jobs),
                    total,
                )

                # 중복 필터링
                for job in page_jobs:
                    if job.source_id in collected_ids:
                        total_skipped += 1
                        continue

                    collected_ids.add(job.source_id)
                    keyword_jobs.append(job)
                    unsaved_jobs.append(job)

                    if len(keyword_jobs) >= max_jobs:
                        break

                # 배치 저장
                if len(unsaved_jobs) >= _BATCH_SAVE_SIZE:
                    saved = save_jobs_csv(unsaved_jobs, csv_path, append=True)
                    self.logger.info("중간 저장: %d건 -> %s", saved, csv_path)
                    unsaved_jobs = []

                # 다음 페이지
                start += config.SARAMIN_RESULTS_PER_PAGE
                if start >= total:
                    break

            self.logger.info("키워드 [%s] 수집 완료: %d건", keyword, len(keyword_jobs))
            all_jobs.extend(keyword_jobs)

        # 상세 페이지에서 기술 스택 추출
        if not skip_detail and all_jobs:
            self.logger.info("상세 페이지 기술 스택 추출 시작 (%d건)", len(all_jobs))
            for idx, job in enumerate(all_jobs, 1):
                if not job.url:
                    continue
                tech_stacks = self._fetch_detail_tech_stacks(job.url)
                job.tech_stacks = tech_stacks
                if idx % 10 == 0:
                    self.logger.info("상세 페이지 진행: %d/%d", idx, len(all_jobs))

        # 미저장분 최종 저장
        if unsaved_jobs:
            saved = save_jobs_csv(unsaved_jobs, csv_path, append=True)
            self.logger.info("최종 저장: %d건 -> %s", saved, csv_path)

        # 상세 페이지 방문 후 tech_stacks가 갱신되었으면 전체 다시 저장
        if not skip_detail and all_jobs:
            save_jobs_csv(all_jobs, csv_path, append=False)
            self.logger.info("기술 스택 포함 전체 저장: %d건 -> %s", len(all_jobs), csv_path)

        elapsed = time.time() - start_time
        result = CrawlResult(
            source="saramin",
            total_collected=len(all_jobs),
            total_skipped=total_skipped,
            total_failed=total_failed,
            duration_seconds=elapsed,
            jobs=all_jobs,
        )
        self.logger.info(result.summary())
        return result

    @retry(
        max_attempts=config.RETRY_MAX_ATTEMPTS,
        base_delay=config.RETRY_BASE_DELAY,
        max_delay=config.RETRY_MAX_DELAY,
        exceptions=(requests.RequestException, ET.ParseError),
    )
    def _call_api(self, keyword: str, start: int = 0) -> ET.Element | None:
        """사람인 채용공고 검색 API를 호출한다.

        Args:
            keyword: 검색 키워드.
            start: 페이지네이션 오프셋 (0부터 시작).

        Returns:
            파싱된 XML 루트 엘리먼트. 실패 시 None.
        """
        params: dict[str, str | int] = {
            "access-key": self.api_key,
            "keywords": keyword,
            "count": config.SARAMIN_RESULTS_PER_PAGE,
            "start": start,
        }

        self.logger.debug(
            "API 요청: keyword=%s, start=%d, count=%d",
            keyword,
            start,
            config.SARAMIN_RESULTS_PER_PAGE,
        )

        response = requests.get(config.SARAMIN_API_URL, params=params, timeout=15)
        response.raise_for_status()

        # API 호출 카운터 갱신
        self.api_call_count += 1
        remaining = config.SARAMIN_MAX_DAILY_CALLS - self.api_call_count
        if remaining % 50 == 0 and remaining > 0:
            self.logger.info(
                "API 호출 잔여 횟수: %d/%d",
                remaining,
                config.SARAMIN_MAX_DAILY_CALLS,
            )
        if self.api_call_count >= self._warn_threshold:
            self.logger.warning(
                "API 호출 횟수 경고: %d/%d (%.0f%% 사용)",
                self.api_call_count,
                config.SARAMIN_MAX_DAILY_CALLS,
                (self.api_call_count / config.SARAMIN_MAX_DAILY_CALLS) * 100,
            )

        # 요청 간 딜레이
        time.sleep(config.SARAMIN_REQUEST_DELAY)

        root = ET.fromstring(response.content)
        return root

    def _fetch_detail_tech_stacks(self, url: str) -> list[str]:
        """채용공고 상세 페이지를 방문하여 기술 스택 키워드를 추출한다.

        Args:
            url: 채용공고 상세 URL.

        Returns:
            추출된 기술 키워드 리스트. 실패 시 빈 리스트.
        """
        try:
            headers = {"User-Agent": _USER_AGENT}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # 본문 영역 추출 (여러 선택자 순차 시도)
            text = ""
            for selector in _DETAIL_CONTENT_SELECTORS:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(separator=" ", strip=True)
                    break

            # 선택자로 추출 실패 시 전체 body 텍스트 사용
            if not text:
                body = soup.find("body")
                text = body.get_text(separator=" ", strip=True) if body else ""

            tech_stacks = extract_tech_keywords(text)

            self.logger.debug(
                "기술 스택 추출: %s -> %s",
                url[:80],
                tech_stacks or "(없음)",
            )

            # 상세 페이지 요청 간 딜레이
            time.sleep(config.SARAMIN_DETAIL_DELAY)

            return tech_stacks

        except requests.RequestException as exc:
            self.logger.warning("상세 페이지 접근 실패 (%s): %s", url[:80], exc)
            return []
        except Exception as exc:
            self.logger.warning("상세 페이지 파싱 실패 (%s): %s", url[:80], exc)
            return []

    def _create_dummy_jobs(self) -> list[JobPosting]:
        """드라이런용 더미 JobPosting 객체 3건을 생성한다.

        Returns:
            더미 JobPosting 리스트 (3건).
        """
        dummy_data = [
            {
                "source_id": "dry_run_001",
                "company": "테스트컴퍼니A",
                "title": "[드라이런] 프론트엔드 개발자",
                "category": "프론트엔드",
                "tech_stacks": ["React", "TypeScript", "Next.js"],
                "location": "서울 강남구",
                "salary": "협의 후 결정",
                "url": "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=dry001",
                "experience": "경력 3년 이상",
            },
            {
                "source_id": "dry_run_002",
                "company": "테스트컴퍼니B",
                "title": "[드라이런] 백엔드 엔지니어",
                "category": "백엔드",
                "tech_stacks": ["Java", "Spring Boot", "MySQL"],
                "location": "서울 서초구",
                "salary": "5,000만원 이상",
                "url": "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=dry002",
                "experience": "경력 5년 이상",
            },
            {
                "source_id": "dry_run_003",
                "company": "테스트컴퍼니C",
                "title": "[드라이런] 데이터 엔지니어",
                "category": "데이터엔지니어",
                "tech_stacks": ["Python", "Spark", "Airflow"],
                "location": "서울 송파구",
                "salary": "4,500만원 이상",
                "url": "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=dry003",
                "experience": "경력 2년 이상",
            },
        ]

        return [JobPosting(source="saramin", **data) for data in dummy_data]

    def _is_api_exhausted(self) -> bool:
        """일일 API 호출 한도에 도달했는지 확인한다.

        Returns:
            True이면 한도 도달.
        """
        return self.api_call_count >= config.SARAMIN_MAX_DAILY_CALLS
