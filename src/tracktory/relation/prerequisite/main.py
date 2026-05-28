"""
선수과목 관계 파이프라인 진입점.

전체 흐름:
  1. rename_syllabi   — syllabi/*.txt → syllabi_rename/*.txt (과목명 기반 파일명 변경)
  2. normalize_syllabi — syllabi_rename/*.txt → syllabi_clean/*.txt (선수과목 필드 정규화)
  3. build_prerequisites — syllabi_clean/*.txt + courses.csv + prereq_alias.json
                           → prerequisites.json
  4. validate_prerequisites — 자기 자신 참조 및 순환참조 검증

실행:
    uv run python -m tracktory.relation.prerequisite.main
"""

from __future__ import annotations

import logging
import sys

from tracktory.relation.prerequisite.build_prerequisites import (
    build_prerequisites,
    validate_prerequisites,
)
from tracktory.relation.prerequisite.logging_setup import setup_logging
from tracktory.relation.prerequisite.normalize_prereq import normalize_syllabi
from tracktory.relation.prerequisite.rename_syllabi import rename_syllabi

logger = logging.getLogger("prereq.main")


def _banner(title: str) -> None:
    logger.info("=" * 60)
    logger.info("%s", title)
    logger.info("=" * 60)


def run() -> int:
    """파이프라인 전 단계를 순서대로 실행하고 종료코드를 반환한다.

    Returns:
        0 = 검증 통과, 1 = 단계 실패 또는 검증 실패.
    """
    try:
        _banner("1단계: 강의계획서 파일명 변경 (syllabi → syllabi_rename)")
        rename_syllabi()

        _banner("2단계: 선수과목 필드 정규화 (syllabi_rename → syllabi_clean)")
        normalize_syllabi()

        _banner("3단계: 선수과목 JSON 생성 (syllabi_clean → prerequisites.json)")
        result = build_prerequisites()
        has_prereq = sum(1 for v in result.values() if v)
        total_edges = sum(len(v) for v in result.values())
        logger.info("전체 과목 수     : %d개", len(result))
        logger.info("선수과목 있음    : %d개", has_prereq)
        logger.info("선수과목 없음    : %d개", len(result) - has_prereq)
        logger.info("선수과목 관계 수 : %d개", total_edges)

        _banner("4단계: 선수과목 그래프 검증 (자기 자신 참조 / 순환참조)")
        return 0 if validate_prerequisites(result) else 1

    except (FileNotFoundError, ValueError) as e:
        logger.error("파이프라인 실패: %s", e)
        return 1


def main() -> None:
    """CLI 진입점."""
    setup_logging("prereq")
    sys.exit(run())


if __name__ == "__main__":
    main()
