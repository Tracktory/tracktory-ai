"""기존 수집 데이터의 URL 목록으로 재크롤링하는 스크립트.

Usage:
    uv run python scripts/recrawl_wanted.py
    uv run python scripts/recrawl_wanted.py --input data/raw/wanted_20260322.json
    uv run python scripts/recrawl_wanted.py --max 10  # 처음 10건만
"""
import argparse
import asyncio
import json
from pathlib import Path

from tracktory.crawler.wanted.crawler import WantedCrawler


def main():
    parser = argparse.ArgumentParser(description="원티드 채용공고 재크롤링")
    parser.add_argument(
        "--input",
        default="data/raw/wanted_20260322.json",
        help="기존 수집 JSON 파일 경로"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="최대 크롤링 수 (0=전체)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"파일을 찾을 수 없습니다: {input_path}")
        return

    with input_path.open("r", encoding="utf-8") as f:
        jobs = json.load(f)

    urls = [job["url"] for job in jobs]
    if args.max > 0:
        urls = urls[:args.max]

    print(f"재크롤링 대상: {len(urls)}건")

    crawler = WantedCrawler(verbose=args.verbose, headless=True)
    result = asyncio.run(crawler.crawl_urls(urls))
    print(result.summary())


if __name__ == "__main__":
    main()
