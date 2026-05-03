"""
선수과목 관계 파이프라인 진입점.

전체 흐름:
  1. rename_syllabi   — syllabi/*.txt → syllabi_rename/*.txt (과목명 기반 파일명 변경)
  2. normalize_prereq — syllabi_rename/*.txt → syllabi_clean/*.txt (선수과목 필드 정규화)
  3. build_prerequisites — syllabi_clean/*.txt + courses.csv + prereq_alias.json
                           → prerequisites.json
  4. validate_prerequisites — 자기 자신 참조 및 순환참조 검증

실행:
    uv run python -m tracktory.relation.prerequisite.main
"""

from __future__ import annotations

from tracktory.relation.prerequisite import rename_syllabi, normalize_prereq
from tracktory.relation.prerequisite.build_prerequisites import (
    build_prerequisites,
    validate_prerequisites,
)


def run() -> None:
    """선수과목 파이프라인 전 단계를 순서대로 실행한다."""
    # 1단계: 파일명 변경
    print("=" * 60)
    print("1단계: 강의계획서 파일명 변경 (syllabi → syllabi_rename)")
    print("=" * 60)
    rename_syllabi.main()

    # 2단계: 선수과목 정규화
    print()
    print("=" * 60)
    print("2단계: 선수과목 필드 정규화 (syllabi_rename → syllabi_clean)")
    print("=" * 60)
    normalize_prereq.main()

    # 3단계: prerequisites.json 생성
    print()
    print("=" * 60)
    print("3단계: 선수과목 JSON 생성 (syllabi_clean → prerequisites.json)")
    print("=" * 60)
    result = build_prerequisites()

    has_prereq = sum(1 for v in result.values() if v)
    total_edges = sum(len(v) for v in result.values())
    print(f"전체 과목 수     : {len(result)}개")
    print(f"선수과목 있음    : {has_prereq}개")
    print(f"선수과목 없음    : {len(result) - has_prereq}개")
    print(f"선수과목 관계 수 : {total_edges}개")

    # 4단계: 검증
    print()
    print("=" * 60)
    print("4단계: 선수과목 그래프 검증 (자기 자신 참조 / 순환참조)")
    print("=" * 60)
    validate_prerequisites(result)


if __name__ == "__main__":
    run()
