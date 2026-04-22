"""카테고리별 원티드 채용공고 크롤링 CLI 스크립트."""

import argparse
import asyncio

from tracktory.crawler.config import config
from tracktory.crawler.wanted.crawler import WantedCrawler


def main():
    parser = argparse.ArgumentParser(
        description="원티드 채용공고 카테고리별 크롤링",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="수집할 카테고리 목록 (공백 구분). 미지정 시 전체 카테고리.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=config.WANTED_DEFAULT_PER_CATEGORY,
        help=f"카테고리당 최대 수집 건수 (기본: {config.WANTED_DEFAULT_PER_CATEGORY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="URL 수집만 확인 (상세 크롤링 수행 안 함)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="브라우저 표시 (디버깅용)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="resume 비활성화 (진행 상태 무시, 처음부터 수집)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="사용 가능한 카테고리 목록 출력 후 종료",
    )
    args = parser.parse_args()

    # --list-categories: print available categories and exit
    if args.list_categories:
        print("\n사용 가능한 카테고리:")
        print(f"{'카테고리':<20s} {'tag_ids':>10s}")
        print("-" * 35)
        for cat, tags in config.WANTED_CATEGORY_TAGS.items():
            tag_str = ", ".join(str(t) for t in tags)
            print(f"  {cat:<18s} {tag_str:>10s}")
        print(f"\n총 {len(config.WANTED_CATEGORY_TAGS)}개 카테고리")
        print("\nNOTE: 보안, 게임, DBA, 데이터분석은 직접 태그가 없어")
        print("      전체 목록에서 수집 후 텍스트 기반 분류로 커버됩니다.")
        return

    # Run crawler
    crawler = WantedCrawler(
        verbose=args.verbose,
        headless=not args.no_headless,
    )

    # --no-resume: use a temp progress path so existing progress is ignored
    progress_path = None
    if args.no_resume:
        import tempfile
        from pathlib import Path

        progress_path = Path(tempfile.mkdtemp()) / "no_resume_progress.json"

    results = asyncio.run(
        crawler.crawl_by_category(
            categories=args.categories,
            max_per_category=args.max_per_category,
            dry_run=args.dry_run,
            progress_path=progress_path,
        )
    )

    # Print summary table
    if results:
        print("\n" + "=" * 60)
        print("카테고리별 수집 결과")
        print("=" * 60)
        print(f"{'카테고리':<18s} {'수집':>6s} {'실패':>6s} {'스킵':>6s} {'소요(초)':>8s}")
        print("-" * 50)
        total_c = total_f = total_s = 0
        for cat, result in results.items():
            print(
                f"  {cat:<16s} {result.total_collected:>6d} "
                f"{result.total_failed:>6d} {result.total_skipped:>6d} "
                f"{result.duration_seconds:>8.1f}"
            )
            total_c += result.total_collected
            total_f += result.total_failed
            total_s += result.total_skipped
        print("-" * 50)
        print(f"  {'합계':<16s} {total_c:>6d} {total_f:>6d} {total_s:>6d}")
        print("=" * 60)


if __name__ == "__main__":
    main()
