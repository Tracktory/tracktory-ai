"""Crawl Wanted metadata/filter tags into a standalone CSV.

This script does not modify the existing crawler, raw job data, or RAG files.

Examples:
    uv run python scripts/crawl_wanted_metadata_csv.py --ids 341074 40159
    uv run python scripts/crawl_wanted_metadata_csv.py --ids-from-rag-jobs
    uv run python scripts/crawl_wanted_metadata_csv.py --filter-tags-default --max-per-filter 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.async_api import Page, async_playwright

DEFAULT_FILTER_TAGS = [
    "\uae00\ub85c\ubc8c TOP \uae30\uc5c5",  # 글로벌 TOP 기업
    "\ub300\uaddc\ubaa8 \ucc44\uc6a9 \uc911",  # 대규모 채용 중
    "\uc801\uadf9 \ucc44\uc6a9 \uc911",  # 적극 채용 중
    "\uc778\uc6d0 \uae09\uc131\uc7a5",  # 인원 급성장
    "\ub204\uc801\ud22c\uc790100\uc5b5\uc774\uc0c1",  # 누적투자100억이상
    "\uc608\ube44 \uc720\ub2c8\ucf58",  # 예비 유니콘
    "1,001~10,000\uba85",  # 1,001~10,000명
    "\ud1f4\uc0ac\uc7285%\uc774\ud558",  # 퇴사율5%이하
]

DETAIL_CSV_FIELDS = [
    "source_id",
    "url",
    "company",
    "title",
    "response_rate",
    "detail_tags",
    "category_tags",
    "category_tag_ids",
    "reward_total",
    "employment_type",
    "is_remote_work",
    "industry_name",
    "filter_tags",
    "crawl_error",
]

JOB_URL_RE = re.compile(r"https://www\.wanted\.co\.kr/wd/(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Wanted metadata tags to CSV.")
    parser.add_argument("--ids", nargs="*", default=[], help="Wanted job IDs to inspect.")
    parser.add_argument("--urls", nargs="*", default=[], help="Wanted detail URLs to inspect.")
    parser.add_argument(
        "--ids-from-rag-jobs",
        action="store_true",
        help="Read Wanted job IDs from data/processed/rag/jobs/*.txt.",
    )
    parser.add_argument(
        "--rag-jobs-dir",
        default="data/processed/rag/jobs",
        help="RAG job document directory for --ids-from-rag-jobs.",
    )
    parser.add_argument(
        "--filter-tags",
        nargs="*",
        default=[],
        help="Wanted list filter tag names to crawl.",
    )
    parser.add_argument(
        "--filter-tags-default",
        action="store_true",
        help="Crawl the default company/filter tags shown on the Wanted list page.",
    )
    parser.add_argument("--max-per-filter", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--wait-ms", type=int, default=2_000)
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path. Default: data/raw/wanted_metadata_YYYYMMDD.csv",
    )
    parser.add_argument("--no-headless", action="store_true")
    return parser.parse_args()


def wanted_url(job_id: str) -> str:
    return f"https://www.wanted.co.kr/wd/{job_id}"


def extract_job_id(url: str) -> str:
    match = re.search(r"/wd/(\d+)", url)
    return match.group(1) if match else url.rstrip("/").split("/")[-1]


def output_path(value: str | None) -> Path:
    if value:
        return Path(value)
    today = datetime.now().strftime("%Y%m%d")
    return Path("data") / "raw" / f"wanted_metadata_{today}.csv"


def load_rag_job_ids(directory: str) -> list[str]:
    job_ids: set[str] = set()
    for path in Path(directory).glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in JOB_URL_RE.finditer(text):
            job_ids.add(match.group(1))
    return sorted(job_ids, key=int)


async def get_initial_data(page: Page) -> dict:
    raw = await page.evaluate("document.getElementById('__NEXT_DATA__')?.textContent || null")
    if not raw:
        return {}
    data = json.loads(raw)
    return data.get("props", {}).get("pageProps", {}).get("initialData", {})


async def extract_detail_metadata(page: Page, url: str, wait_ms: int) -> dict[str, str]:
    await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(wait_ms)

    initial = await get_initial_data(page)
    company = initial.get("company") or {}
    category_tag = initial.get("category_tag") or {}
    child_tags = category_tag.get("child_tags") or []
    reward = initial.get("reward") or {}

    visible = await page.evaluate("document.body.innerText")
    lines = [line.strip() for line in visible.splitlines() if line.strip()]

    response_rate = ""
    response_label = "\uc751\ub2f5\ub960"
    if response_label in lines:
        idx = lines.index(response_label)
        if idx + 1 < len(lines):
            response_rate = lines[idx + 1]

    detail_tags: list[str] = []
    tag_label = "\ud0dc\uadf8"
    stop_labels = {
        "\ub9c8\uac10\uc77c",
        "\uadfc\ubb34\uc9c0\uc5ed",
        "\ubcf8 \ucc44\uc6a9\uc815\ubcf4",
        "\ud574\ub2f9 \ud3ec\uc9c0\uc158\uc740",
        "\ube44\uc2b7\ud55c \ud3ec\uc9c0\uc158",
    }
    if tag_label in lines:
        idx = lines.index(tag_label)
        for line in lines[idx + 1 : idx + 20]:
            if any(line.startswith(stop) for stop in stop_labels):
                break
            if len(line) <= 30 and line not in detail_tags:
                detail_tags.append(line)

    category_tags = [
        str(tag.get("text", "")).strip()
        for tag in child_tags
        if isinstance(tag, dict) and str(tag.get("text", "")).strip()
    ]
    category_tag_ids = [
        str(tag.get("id", "")).strip()
        for tag in child_tags
        if isinstance(tag, dict) and str(tag.get("id", "")).strip()
    ]

    return {
        "source_id": str(initial.get("id") or extract_job_id(url)),
        "url": url,
        "company": str(company.get("company_name") or ""),
        "title": str(initial.get("position") or ""),
        "response_rate": response_rate,
        "detail_tags": "|".join(detail_tags),
        "category_tags": "|".join(category_tags),
        "category_tag_ids": "|".join(category_tag_ids),
        "reward_total": str(reward.get("formatted_total") or ""),
        "employment_type": str(initial.get("employment_type") or ""),
        "is_remote_work": str(initial.get("is_remote_work") or ""),
        "industry_name": str(company.get("industry_name") or ""),
        "filter_tags": "",
        "crawl_error": "",
    }


async def extract_detail_metadata_safe(context, url: str, wait_ms: int) -> dict[str, str]:
    page = await context.new_page()
    try:
        return await extract_detail_metadata(page, url, wait_ms)
    except Exception as exc:
        return {
            **{field: "" for field in DETAIL_CSV_FIELDS},
            "source_id": extract_job_id(url),
            "url": url,
            "crawl_error": repr(exc),
        }
    finally:
        await page.close()


async def crawl_detail_rows(
    context, urls: list[str], concurrency: int, wait_ms: int
) -> list[dict[str, str]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    total = len(urls)
    completed = 0

    async def run_one(url: str) -> dict[str, str]:
        nonlocal completed
        async with semaphore:
            row = await extract_detail_metadata_safe(context, url, wait_ms)
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == total:
                print(f"details: {completed}/{total}")
            return row

    return await asyncio.gather(*(run_one(url) for url in urls))


async def discover_filter_tag_id(page: Page, tag_name: str) -> str:
    base_url = (
        "https://www.wanted.co.kr/wdlist/518"
        "?country=kr&job_sort=job.latest_order&years=-1&locations=all"
    )
    await page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(5_000)
    await page.get_by_text(tag_name, exact=True).click(timeout=10_000)
    await page.wait_for_timeout(2_000)
    query = parse_qs(urlparse(page.url).query)
    return (query.get("tags") or [""])[0]


async def collect_filter_tag_jobs(page: Page, tag_name: str, max_jobs: int) -> set[str]:
    tag_id = await discover_filter_tag_id(page, tag_name)
    if not tag_id:
        return set()

    list_url = (
        "https://www.wanted.co.kr/wdlist/518"
        f"?country=kr&job_sort=job.latest_order&years=-1&tags={tag_id}&locations=all"
    )
    await page.goto(list_url, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(4_000)

    found: set[str] = set()
    stagnant = 0
    while len(found) < max_jobs and stagnant < 3:
        urls = await page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href*="/wd/"]'))
                .map(a => a.href.split("?")[0])
                .filter(href => /\\/wd\\/\\d+/.test(href))"""
        )
        before = len(found)
        for url in urls:
            found.add(extract_job_id(url))
            if len(found) >= max_jobs:
                break
        stagnant = stagnant + 1 if len(found) == before else 0
        if len(found) < max_jobs:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2_000)
    return found


async def main() -> None:
    args = parse_args()
    out_path = output_path(args.output)
    filter_tags = list(args.filter_tags)
    if args.filter_tags_default:
        filter_tags.extend(tag for tag in DEFAULT_FILTER_TAGS if tag not in filter_tags)

    urls = list(args.urls)
    if args.ids_from_rag_jobs:
        args.ids.extend(load_rag_job_ids(args.rag_jobs_dir))
    urls.extend(wanted_url(job_id) for job_id in args.ids)
    urls = list(dict.fromkeys(urls))

    filter_memberships: dict[str, set[str]] = defaultdict(set)
    rows_by_id: dict[str, dict[str, str]] = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.no_headless)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="ko-KR",
            timezone_id="Asia/Seoul",
        )
        for row in await crawl_detail_rows(context, urls, args.concurrency, args.wait_ms):
            rows_by_id[row["source_id"]] = row

        page = await context.new_page()
        for tag_name in filter_tags:
            job_ids = await collect_filter_tag_jobs(page, tag_name, args.max_per_filter)
            for job_id in job_ids:
                filter_memberships[job_id].add(tag_name)
                rows_by_id.setdefault(
                    job_id,
                    {field: "" for field in DETAIL_CSV_FIELDS},
                )
                rows_by_id[job_id]["source_id"] = job_id
                rows_by_id[job_id]["url"] = wanted_url(job_id)

        await context.close()
        await browser.close()

    for job_id, tags in filter_memberships.items():
        rows_by_id[job_id]["filter_tags"] = "|".join(sorted(tags))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows_by_id.values())

    print(f"saved: {out_path}")
    print(f"rows: {len(rows_by_id)}")


if __name__ == "__main__":
    asyncio.run(main())
