"""
원티드(Wanted) 채용공고 Playwright 비동기 크롤러
================================================

Playwright 기반으로 React SPA인 원티드에서 구조화된 기술스택 태그를 수집합니다.
common/ 모듈의 모델, 설정, 저장소, 재시도, 로깅을 공용으로 사용합니다.

사전 준비:
    pip install playwright
    playwright install chromium

robots.txt 준수 안내:
    https://www.wanted.co.kr/robots.txt 를 확인하면 /wdlist/ 경로에 대한
    크롤링 제한은 없으나, 서버 부하를 최소화하기 위해 요청 간 2~3초 딜레이를 적용합니다.
    상업적 목적이 아닌 학술 연구 목적으로만 사용하세요.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

from playwright.async_api import (
    Page,
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from tracktory.common.models import CrawlResult, JobPosting
from tracktory.common.tech_keywords import (
    classify_job_category,
    extract_tech_keywords,
    normalize_tech_tags,
)
from tracktory.crawler.config import config
from tracktory.crawler.logging_setup import setup_logging
from tracktory.crawler.retry import async_retry
from tracktory.crawler.storage import (
    get_category_output_path,
    get_output_path,
    load_category_progress,
    load_collected_ids,
    save_category_progress,
    save_jobs_csv,
    save_jobs_json,
)

from .selectors import (
    extract_company_info,
    extract_company_name,
    extract_deadline,
    extract_experience,
    extract_job_title,
    extract_location,
    extract_sections,
    extract_skill_tags,
)

logger = logging.getLogger("crawling.wanted")

# 페이지 로드 대기 최대 시간 (ms)
_PAGE_TIMEOUT: int = 60_000


class WantedCrawler:
    """원티드 채용공고 Playwright 비동기 크롤러.

    무한 스크롤 기반의 원티드 채용 목록 페이지에서 공고 URL을 수집한 뒤,
    각 상세 페이지에서 기술스택 태그와 직무 정보를 추출합니다.

    Attributes:
        verbose: 상세 로그 출력 여부.
        headless: 브라우저를 헤드리스 모드로 실행할지 여부.
    """

    def __init__(self, verbose: bool = False, headless: bool = True) -> None:
        """크롤러를 초기화한다.

        Args:
            verbose: True이면 DEBUG 레벨 로그를 출력합니다.
            headless: True이면 브라우저를 헤드리스 모드로 실행합니다.
        """
        self.verbose = verbose
        self.headless = headless
        self._logger = setup_logging("crawling.wanted", verbose=verbose)
        config.validate()

    async def crawl(
        self,
        max_jobs: int = 50,
        dry_run: bool = False,
    ) -> CrawlResult:
        """원티드 채용공고를 크롤링한다.

        목록 페이지에서 무한 스크롤로 공고 URL을 수집한 뒤,
        각 상세 페이지를 순차적으로 방문하여 데이터를 추출합니다.
        10건 단위로 CSV에 중간 저장하여 중단 시에도 데이터를 보존합니다.

        Args:
            max_jobs: 수집할 최대 채용공고 수. 기본값 50.
            dry_run: True이면 실제 크롤링 없이 더미 데이터 3건을 반환합니다.

        Returns:
            CrawlResult: 크롤링 결과 요약 및 수집된 공고 목록.
        """
        start_time = time.time()

        # 드라이런 모드
        if dry_run:
            self._logger.info("드라이런 모드: 더미 데이터 3건 생성")
            dummy_jobs = self._create_dummy_jobs()
            duration = time.time() - start_time
            return CrawlResult(
                source="wanted",
                total_collected=len(dummy_jobs),
                total_skipped=0,
                total_failed=0,
                duration_seconds=duration,
                jobs=dummy_jobs,
            )

        # 출력 경로 준비
        csv_path = get_output_path("wanted", "csv")
        json_path = get_output_path("wanted", "json")

        # 재개 지원: 이미 수집된 ID 로드
        collected_ids = load_collected_ids("wanted", csv_path)
        if collected_ids:
            self._logger.info(
                "이미 수집된 공고 %d건 발견 -- 해당 공고는 건너뜁니다.",
                len(collected_ids),
            )

        all_jobs: list[JobPosting] = []
        total_skipped = 0
        total_failed = 0

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self.headless,
            )

            context = await browser.new_context(
                viewport={
                    "width": config.WANTED_VIEWPORT_WIDTH,
                    "height": config.WANTED_VIEWPORT_HEIGHT,
                },
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                locale="ko-KR",
                timezone_id="Asia/Seoul",
            )

            try:
                # 1단계: 채용공고 URL 수집
                list_page = await context.new_page()
                all_urls = await self._collect_job_urls(list_page, max_jobs)
                await list_page.close()

                # 이미 수집된 공고 필터링
                pending_urls: list[str] = []
                for url in all_urls:
                    source_id = self._extract_source_id(url)
                    if source_id in collected_ids:
                        total_skipped += 1
                    else:
                        pending_urls.append(url)

                if total_skipped > 0:
                    self._logger.info("%d개 URL 재크롤링 스킵 (이미 수집됨)", total_skipped)
                self._logger.info("크롤링 예정: %d개", len(pending_urls))

                if not pending_urls:
                    self._logger.info("크롤링할 새 URL이 없습니다.")
                    duration = time.time() - start_time
                    return CrawlResult(
                        source="wanted",
                        total_collected=0,
                        total_skipped=total_skipped,
                        total_failed=0,
                        duration_seconds=duration,
                        jobs=[],
                    )

                # 2단계: 상세 페이지 순차 크롤링
                detail_page = await context.new_page()
                batch: list[JobPosting] = []

                for idx, url in enumerate(pending_urls, start=1):
                    self._logger.info("[%d/%d] 크롤링 중: %s", idx, len(pending_urls), url)

                    job = await self._extract_job_detail(detail_page, url)

                    if job:
                        batch.append(job)
                        all_jobs.append(job)
                        self._logger.info(
                            "  완료: [%s] %s -- %s (태그 %d개)",
                            job.category,
                            job.company,
                            job.title,
                            len(job.tech_stacks),
                        )
                    else:
                        total_failed += 1
                        self._logger.warning("  실패: %s", url)

                    # 배치 저장 (10건 단위)
                    if len(batch) >= config.WANTED_BATCH_SAVE_SIZE:
                        saved = save_jobs_csv(batch, csv_path, append=True)
                        self._logger.info(
                            "  >> %d건 중간 저장 완료 (누적 성공: %d건)",
                            saved,
                            len(all_jobs),
                        )
                        batch.clear()

                    # 서버 부하 방지 딜레이
                    if idx < len(pending_urls):
                        delay = random.uniform(config.WANTED_DELAY_MIN, config.WANTED_DELAY_MAX)
                        self._logger.debug("  딜레이 %.1f초 대기 중...", delay)
                        await asyncio.sleep(delay)

                # 남은 배치 저장
                if batch:
                    save_jobs_csv(batch, csv_path, append=True)
                    batch.clear()

                # JSON 전체 저장
                if all_jobs:
                    save_jobs_json(all_jobs, json_path)

                await detail_page.close()

            finally:
                await context.close()
                await browser.close()

        duration = time.time() - start_time

        result = CrawlResult(
            source="wanted",
            total_collected=len(all_jobs),
            total_skipped=total_skipped,
            total_failed=total_failed,
            duration_seconds=duration,
            jobs=all_jobs,
        )

        self._logger.info(result.summary())
        return result

    async def crawl_urls(
        self,
        urls: list[str],
        output_prefix: str = "wanted_recrawl",
    ) -> CrawlResult:
        """미리 지정된 URL 목록으로 재크롤링한다.

        기존 crawl()과 달리 목록 페이지 스크롤을 건너뛰고,
        주어진 URL을 직접 방문하여 데이터를 추출합니다.
        resume 로직을 사용하지 않으므로 모든 URL을 재크롤링합니다.

        Args:
            urls: 크롤링할 채용공고 URL 리스트.
            output_prefix: 출력 파일 접두사. 기본값 'wanted_recrawl'.

        Returns:
            CrawlResult: 크롤링 결과 요약 및 수집된 공고 목록.
        """
        start_time = time.time()

        # 출력 경로 준비
        csv_path = get_output_path(output_prefix, "csv")
        json_path = get_output_path(output_prefix, "json")

        all_jobs: list[JobPosting] = []
        total_failed = 0

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self.headless,
            )

            context = await browser.new_context(
                viewport={
                    "width": config.WANTED_VIEWPORT_WIDTH,
                    "height": config.WANTED_VIEWPORT_HEIGHT,
                },
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                locale="ko-KR",
                timezone_id="Asia/Seoul",
            )

            try:
                detail_page = await context.new_page()
                batch: list[JobPosting] = []

                for idx, url in enumerate(urls, start=1):
                    self._logger.info("[%d/%d] 재크롤링 중: %s", idx, len(urls), url)

                    job = await self._extract_job_detail(detail_page, url)

                    if job:
                        batch.append(job)
                        all_jobs.append(job)
                        self._logger.info(
                            "  완료: [%s] %s -- %s (태그 %d개)",
                            job.category,
                            job.company,
                            job.title,
                            len(job.tech_stacks),
                        )
                    else:
                        total_failed += 1
                        self._logger.warning("  실패: %s", url)

                    # 배치 저장 (10건 단위)
                    if len(batch) >= config.WANTED_BATCH_SAVE_SIZE:
                        saved = save_jobs_csv(batch, csv_path, append=True)
                        self._logger.info(
                            "  >> %d건 중간 저장 완료 (누적 성공: %d건)",
                            saved,
                            len(all_jobs),
                        )
                        batch.clear()

                    # 서버 부하 방지 딜레이
                    if idx < len(urls):
                        delay = random.uniform(config.WANTED_DELAY_MIN, config.WANTED_DELAY_MAX)
                        self._logger.debug("  딜레이 %.1f초 대기 중...", delay)
                        await asyncio.sleep(delay)

                # 남은 배치 저장
                if batch:
                    save_jobs_csv(batch, csv_path, append=True)
                    batch.clear()

                # JSON 전체 저장
                if all_jobs:
                    save_jobs_json(all_jobs, json_path)

                await detail_page.close()

            finally:
                await context.close()
                await browser.close()

        duration = time.time() - start_time

        result = CrawlResult(
            source="wanted",
            total_collected=len(all_jobs),
            total_skipped=0,
            total_failed=total_failed,
            duration_seconds=duration,
            jobs=all_jobs,
        )

        self._logger.info(result.summary())
        return result

    async def crawl_by_category(
        self,
        categories: list[str] | None = None,
        max_per_category: int = 200,
        dry_run: bool = False,
        progress_path: Path | None = None,
    ) -> dict[str, CrawlResult]:
        """카테고리별로 채용공고를 수집한다.

        WANTED_CATEGORY_TAGS에 정의된 카테고리별 목록 URL을 사용하여
        각 카테고리에서 무한 스크롤로 URL을 수집한 뒤 상세 페이지를 크롤링합니다.
        카테고리 단위로 resume이 가능합니다.

        Args:
            categories: 수집할 카테고리 목록. None이면 전체 카테고리.
            max_per_category: 카테고리당 최대 수집 건수. 기본값 200.
            dry_run: True이면 URL 수집만 하고 상세 크롤링은 스킵.
            progress_path: resume 상태 파일 경로. None이면 기본 경로 사용.

        Returns:
            카테고리별 CrawlResult를 담은 딕셔너리.
        """
        start_time = time.time()

        # 카테고리 목록 결정
        category_tags = config.WANTED_CATEGORY_TAGS
        if categories:
            # Validate requested categories
            invalid = [c for c in categories if c not in category_tags]
            if invalid:
                self._logger.warning("알 수 없는 카테고리 (건너뜀): %s", ", ".join(invalid))
            target_categories = [c for c in categories if c in category_tags]
        else:
            target_categories = list(category_tags.keys())

        if not target_categories:
            self._logger.warning("수집할 카테고리가 없습니다.")
            return {}

        # Progress 파일 경로
        if progress_path is None:
            progress_path = config.DATA_RAW_DIR / "wanted_category_progress.json"

        progress = load_category_progress(progress_path)

        # 기존 데이터의 source_id 수집 (글로벌 중복 방지)
        # Scan all existing wanted CSV files in data/raw/
        global_collected_ids: set[str] = set()
        for csv_file in config.DATA_RAW_DIR.glob("wanted_*.csv"):
            ids = load_collected_ids("wanted", csv_file)
            global_collected_ids |= ids
        if global_collected_ids:
            self._logger.info(
                "기존 수집된 공고 %d건 발견 (글로벌 중복 방지)",
                len(global_collected_ids),
            )

        results: dict[str, CrawlResult] = {}

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                viewport={
                    "width": config.WANTED_VIEWPORT_WIDTH,
                    "height": config.WANTED_VIEWPORT_HEIGHT,
                },
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                locale="ko-KR",
                timezone_id="Asia/Seoul",
            )

            try:
                list_page = await context.new_page()
                detail_page = await context.new_page()

                for cat_idx, category in enumerate(target_categories, start=1):
                    cat_start = time.time()
                    separator = "=" * 60
                    self._logger.info(
                        "\n%s\n[카테고리 %d/%d] %s 수집 시작\n%s",
                        separator,
                        cat_idx,
                        len(target_categories),
                        category,
                        separator,
                    )

                    # Resume check
                    cat_progress = progress.get(category, {})
                    if cat_progress.get("status") == "done":
                        self._logger.info(
                            "  %s: 이미 완료됨 (수집 %d건). 건너뜁니다.",
                            category,
                            cat_progress.get("collected", 0),
                        )
                        results[category] = CrawlResult(
                            source="wanted",
                            total_collected=cat_progress.get("collected", 0),
                            total_skipped=0,
                            total_failed=0,
                            duration_seconds=0,
                            jobs=[],
                        )
                        continue

                    # Output paths for this category
                    csv_path = get_category_output_path(category, "csv")
                    json_path = get_category_output_path(category, "json")

                    # Collect URLs from all tag_ids for this category
                    tag_ids = category_tags[category]
                    all_urls: list[str] = []
                    seen_urls: set[str] = set()

                    for tag_id in tag_ids:
                        list_url = config.WANTED_CATEGORY_LIST_URL_TEMPLATE.format(tag_id=tag_id)
                        urls = await self._collect_job_urls_from(
                            list_page, list_url, max_per_category
                        )
                        for url in urls:
                            if url not in seen_urls:
                                seen_urls.add(url)
                                all_urls.append(url)

                    self._logger.info("  %s: URL %d개 수집 완료", category, len(all_urls))

                    # Filter already collected
                    pending_urls = [
                        url
                        for url in all_urls
                        if self._extract_source_id(url) not in global_collected_ids
                    ]
                    skipped = len(all_urls) - len(pending_urls)
                    if skipped > 0:
                        self._logger.info(
                            "  %s: %d개 중복 제거, %d개 크롤링 예정",
                            category,
                            skipped,
                            len(pending_urls),
                        )

                    # Update progress
                    progress[category] = {
                        "status": "in_progress",
                        "collected": cat_progress.get("collected", 0),
                        "target": max_per_category,
                        "urls_found": len(all_urls),
                    }
                    save_category_progress(progress, progress_path)

                    # Dry run: just report and continue
                    if dry_run:
                        self._logger.info(
                            "  [DRY RUN] %s: URL %d개 (중복 제거 후 %d개)",
                            category,
                            len(all_urls),
                            len(pending_urls),
                        )
                        progress[category]["status"] = "dry_run"
                        save_category_progress(progress, progress_path)
                        results[category] = CrawlResult(
                            source="wanted",
                            total_collected=0,
                            total_skipped=skipped,
                            total_failed=0,
                            duration_seconds=time.time() - cat_start,
                            jobs=[],
                        )
                        continue

                    # Crawl detail pages
                    cat_jobs: list[JobPosting] = []
                    cat_failed = 0
                    batch: list[JobPosting] = []

                    for idx, url in enumerate(pending_urls, start=1):
                        self._logger.info(
                            "  [%s %d/%d] 크롤링 중: %s",
                            category,
                            idx,
                            len(pending_urls),
                            url,
                        )

                        job = await self._extract_job_detail(detail_page, url)

                        if job:
                            batch.append(job)
                            cat_jobs.append(job)
                            global_collected_ids.add(job.source_id)
                            self._logger.info(
                                "    완료: [%s] %s -- %s (태그 %d개)",
                                job.category,
                                job.company,
                                job.title,
                                len(job.tech_stacks),
                            )
                        else:
                            cat_failed += 1
                            self._logger.warning("    실패: %s", url)

                        # Batch save (every 10)
                        if len(batch) >= config.WANTED_BATCH_SAVE_SIZE:
                            save_jobs_csv(batch, csv_path, append=True)
                            self._logger.info(
                                "    >> %d건 중간 저장 (누적: %d건)",
                                len(batch),
                                len(cat_jobs),
                            )
                            batch.clear()

                            # Update progress
                            progress[category]["collected"] = len(cat_jobs)
                            save_category_progress(progress, progress_path)

                        # Delay between requests
                        if idx < len(pending_urls):
                            delay = random.uniform(config.WANTED_DELAY_MIN, config.WANTED_DELAY_MAX)
                            await asyncio.sleep(delay)

                    # Save remaining batch
                    if batch:
                        save_jobs_csv(batch, csv_path, append=True)
                        batch.clear()

                    # Save full JSON
                    if cat_jobs:
                        save_jobs_json(cat_jobs, json_path)

                    # Mark category as done
                    cat_duration = time.time() - cat_start
                    progress[category] = {
                        "status": "done",
                        "collected": len(cat_jobs),
                        "target": max_per_category,
                        "urls_found": len(all_urls),
                    }
                    save_category_progress(progress, progress_path)

                    results[category] = CrawlResult(
                        source="wanted",
                        total_collected=len(cat_jobs),
                        total_skipped=skipped,
                        total_failed=cat_failed,
                        duration_seconds=cat_duration,
                        jobs=cat_jobs,
                    )

                    self._logger.info(
                        "  %s 완료: 수집 %d건, 실패 %d건, 소요 %.1f초",
                        category,
                        len(cat_jobs),
                        cat_failed,
                        cat_duration,
                    )

                await list_page.close()
                await detail_page.close()

            finally:
                await context.close()
                await browser.close()

        # Print summary
        total_duration = time.time() - start_time
        total_collected = sum(r.total_collected for r in results.values())
        total_failed = sum(r.total_failed for r in results.values())
        self._logger.info(
            "\n=== 카테고리별 크롤링 완료 ===\n"
            "  총 카테고리: %d개\n  총 수집: %d건\n  총 실패: %d건\n"
            "  총 소요시간: %.1f초",
            len(results),
            total_collected,
            total_failed,
            total_duration,
        )

        return results

    async def _collect_job_urls(self, page: Page, max_jobs: int) -> list[str]:
        """목록 페이지에서 무한 스크롤을 통해 채용공고 URL을 수집한다.

        원티드는 스크롤 시 XHR로 추가 카드를 렌더링하는 React SPA이므로
        스크롤 후 DOM 변화를 감지하는 방식을 사용합니다.

        Args:
            page: Playwright Page 인스턴스.
            max_jobs: 수집할 최대 URL 수.

        Returns:
            수집된 채용공고 URL 리스트.
        """
        self._logger.info("목록 페이지 로딩 중: %s", config.WANTED_JOB_LIST_URL)
        await self._navigate_with_retry(page, config.WANTED_JOB_LIST_URL)
        await page.wait_for_timeout(3000)

        job_urls: list[str] = []
        seen_urls: set[str] = set()
        no_new_count = 0

        self._logger.info("무한 스크롤 시작 (목표: %d개 URL 수집)", max_jobs)

        while len(job_urls) < max_jobs:
            # 현재 렌더링된 채용 카드 링크 수집
            card_links: list[str] = await page.evaluate("""
                () => {
                    const anchors = document.querySelectorAll('a[href*="/wd/"]');
                    return Array.from(anchors)
                        .map(a => a.href)
                        .filter(href => /\\/wd\\/\\d+/.test(href));
                }
            """)

            new_count = 0
            for href in card_links:
                clean_url = href.split("?")[0].rstrip("/")
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    job_urls.append(clean_url)
                    new_count += 1
                    if len(job_urls) >= max_jobs:
                        break

            self._logger.info(
                "  현재 수집된 URL: %d개 (이번 스크롤에서 신규: %d개)",
                len(job_urls),
                new_count,
            )

            if len(job_urls) >= max_jobs:
                break

            if new_count == 0:
                no_new_count += 1
                if no_new_count >= 3:
                    self._logger.info("  더 이상 새 카드가 없습니다. 스크롤 종료.")
                    break
            else:
                no_new_count = 0

            # 페이지 하단으로 스크롤
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2500)

        self._logger.info("총 %d개 채용공고 URL 수집 완료", len(job_urls))
        return job_urls[:max_jobs]

    async def _collect_job_urls_from(self, page: Page, list_url: str, max_jobs: int) -> list[str]:
        """주어진 목록 URL에서 무한 스크롤로 채용공고 URL을 수집한다.

        Args:
            page: Playwright Page 인스턴스.
            list_url: 채용공고 목록 페이지 URL.
            max_jobs: 수집할 최대 URL 수.

        Returns:
            수집된 채용공고 URL 리스트.
        """
        self._logger.info("목록 페이지 로딩 중: %s", list_url)
        await self._navigate_with_retry(page, list_url)
        await page.wait_for_timeout(3000)

        job_urls: list[str] = []
        seen_urls: set[str] = set()
        no_new_count = 0

        self._logger.info("무한 스크롤 시작 (목표: %d개 URL 수집)", max_jobs)

        while len(job_urls) < max_jobs:
            card_links: list[str] = await page.evaluate("""
                () => {
                    const anchors = document.querySelectorAll('a[href*="/wd/"]');
                    return Array.from(anchors)
                        .map(a => a.href)
                        .filter(href => /\\/wd\\/\\d+/.test(href));
                }
            """)

            new_count = 0
            for href in card_links:
                clean_url = href.split("?")[0].rstrip("/")
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    job_urls.append(clean_url)
                    new_count += 1
                    if len(job_urls) >= max_jobs:
                        break

            self._logger.info(
                "  현재 수집된 URL: %d개 (이번 스크롤에서 신규: %d개)",
                len(job_urls),
                new_count,
            )

            if len(job_urls) >= max_jobs:
                break

            if new_count == 0:
                no_new_count += 1
                if no_new_count >= 3:
                    self._logger.info("  더 이상 새 카드가 없습니다. 스크롤 종료.")
                    break
            else:
                no_new_count = 0

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2500)

        self._logger.info("총 %d개 채용공고 URL 수집 완료", len(job_urls))
        return job_urls[:max_jobs]

    async def _extract_job_detail(self, page: Page, url: str) -> JobPosting | None:
        """채용공고 상세 페이지에서 구조화된 정보를 추출한다.

        페이지 네비게이션, 데이터 추출, JobPosting 모델 생성까지
        단일 상세 페이지 처리의 전체 흐름을 담당합니다.

        Args:
            page: Playwright Page 인스턴스.
            url: 채용공고 상세 페이지 URL.

        Returns:
            추출 성공 시 JobPosting 인스턴스, 실패 시 None.
        """
        try:
            await self._navigate_with_retry(page, url)
            await page.wait_for_timeout(2000)
        except PlaywrightTimeoutError:
            self._logger.warning("  타임아웃: %s", url)
            return None
        except Exception as exc:
            self._logger.warning("  페이지 로드 실패 (%s): %s", url, exc)
            return None

        # 기본 정보 추출
        company = await extract_company_name(page)
        title = await extract_job_title(page)
        location = await extract_location(page)
        deadline = await extract_deadline(page)
        experience = await extract_experience(page)
        company_info = await extract_company_info(page)

        # 기술스택 태그 추출 및 정규화
        raw_tags = await extract_skill_tags(page)
        tech_stacks = normalize_tech_tags(raw_tags)

        # 텍스트 기반 추가 기술 키워드 추출 (섹션 본문에서)
        sections = await extract_sections(page)
        section_text = " ".join(sections.values())
        if section_text.strip():
            text_keywords = extract_tech_keywords(section_text)
            # 태그에서 추출한 것과 텍스트에서 추출한 것을 합침
            merged = set(tech_stacks) | set(text_keywords)
            tech_stacks = sorted(merged)

        # 직무 카테고리 분류
        category = classify_job_category(title, section_text)

        # source_id 추출
        source_id = self._extract_source_id(url)

        return JobPosting(
            source="wanted",
            source_id=source_id,
            company=company,
            title=title,
            category=category,
            tech_stacks=tech_stacks,
            location=location,
            salary="",
            url=url,
            experience=experience,
            extra={
                "responsibilities": sections.get("responsibilities", ""),
                "requirements": sections.get("requirements", ""),
                "preferred": sections.get("preferred", ""),
                "benefits": sections.get("benefits", ""),
                "deadline": deadline,
                "company_id": company_info.get("company_id", ""),
                "industry_name": company_info.get("industry_name", ""),
            },
        )

    def _create_dummy_jobs(self) -> list[JobPosting]:
        """드라이런용 더미 JobPosting 3건을 생성한다.

        Returns:
            더미 JobPosting 인스턴스 3개를 담은 리스트.
        """
        now = datetime.now()
        return [
            JobPosting(
                source="wanted",
                source_id="dummy-1",
                company="테스트 회사 A",
                title="주니어 백엔드 개발자",
                category="백엔드",
                tech_stacks=["Django", "PostgreSQL", "Python"],
                location="서울",
                salary="",
                url="https://www.wanted.co.kr/wd/dummy-1",
                experience="",
                collected_at=now,
                extra={
                    "responsibilities": "API 개발 및 유지보수",
                    "requirements": "Python 경험 1년 이상",
                    "preferred": "Django 경험",
                    "deadline": "2026-04-30",
                },
            ),
            JobPosting(
                source="wanted",
                source_id="dummy-2",
                company="테스트 회사 B",
                title="프론트엔드 엔지니어",
                category="프론트엔드",
                tech_stacks=["React", "Tailwind CSS", "TypeScript"],
                location="서울",
                salary="",
                url="https://www.wanted.co.kr/wd/dummy-2",
                experience="",
                collected_at=now,
                extra={
                    "responsibilities": "UI 컴포넌트 개발",
                    "requirements": "React 경험 2년 이상",
                    "preferred": "TypeScript 경험",
                    "deadline": "2026-05-15",
                },
            ),
            JobPosting(
                source="wanted",
                source_id="dummy-3",
                company="테스트 회사 C",
                title="DevOps 엔지니어",
                category="DevOps/인프라",
                tech_stacks=["AWS", "Docker", "Kubernetes"],
                location="서울",
                salary="",
                url="https://www.wanted.co.kr/wd/dummy-3",
                experience="",
                collected_at=now,
                extra={
                    "responsibilities": "인프라 운영 및 자동화",
                    "requirements": "쿠버네티스 경험 1년 이상",
                    "preferred": "AWS 경험",
                    "deadline": "2026-06-01",
                },
            ),
        ]

    @async_retry(
        max_attempts=config.RETRY_MAX_ATTEMPTS,
        base_delay=config.RETRY_BASE_DELAY,
        max_delay=config.RETRY_MAX_DELAY,
        exceptions=(PlaywrightTimeoutError, Exception),
    )
    async def _navigate_with_retry(self, page: Page, url: str) -> None:
        """재시도 로직이 적용된 페이지 네비게이션.

        @async_retry 데코레이터를 통해 타임아웃 등의 오류 발생 시
        지수 백오프로 재시도합니다.

        Args:
            page: Playwright Page 인스턴스.
            url: 이동할 URL.
        """
        await page.goto(url, wait_until="domcontentloaded", timeout=_PAGE_TIMEOUT)

    @staticmethod
    def _extract_source_id(url: str) -> str:
        """URL에서 원티드 공고 ID를 추출한다.

        /wd/{id} 패턴에서 숫자 ID를 추출합니다.

        Args:
            url: 원티드 채용공고 URL.

        Returns:
            공고 ID 문자열. 패턴 매칭 실패 시 URL의 마지막 경로 세그먼트.
        """
        match = re.search(r"/wd/(\d+)", url)
        if match:
            return match.group(1)
        # 폴백: 마지막 경로 세그먼트
        return url.rstrip("/").split("/")[-1]
