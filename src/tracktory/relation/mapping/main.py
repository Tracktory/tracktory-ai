"""
직무·트랙 역량 매핑 파이프라인 진입점.

전체 흐름:
  1. normalize_wanted   — wanted_cleaned.json → wanted_by_job.json
  2. job_competency     — wanted_by_job.json + job_tech_stacks.json → job_competency_map.json
  3. track_competency   — courses.csv + syllabi_rename/*.txt → track_competency_map.json

실행:
    uv run python -m tracktory.relation.mapping.main
"""

from __future__ import annotations

from tracktory.relation.mapping.preprocessing.normalize_wanted import normalize_wanted
from tracktory.relation.mapping.job_competency import (
    build_job_competency_map,
    save_job_competency_map,
)
from tracktory.relation.mapping.track_competency import (
    build_track_competency_map,
    save_track_competency_map,
)


def run() -> None:
    """직무·트랙 역량 매핑 파이프라인 전 단계를 순서대로 실행한다."""
    # 1단계: 채용공고 직무별 정규화
    print("=" * 60)
    print("1단계: 채용공고 직무 정규화 (wanted_cleaned → wanted_by_job)")
    print("=" * 60)
    postings = normalize_wanted()
    print(f"처리 완료: {len(postings)}건")

    # 2단계: 직무별 역량 매핑
    print()
    print("=" * 60)
    print("2단계: 직무별 역량 매핑 (wanted_by_job + job_tech_stacks → job_competency_map)")
    print("=" * 60)
    job_result = build_job_competency_map()
    save_job_competency_map(job_result)
    total_competencies = sum(len(v["competencies"]) for v in job_result.values())
    print(f"직무 수          : {len(job_result)}개")
    print(f"전체 역량 항목 수 : {total_competencies}개")

    # 3단계: 트랙별 역량 매핑
    print()
    print("=" * 60)
    print("3단계: 트랙별 역량 매핑 (courses.csv + syllabi_rename → track_competency_map)")
    print("=" * 60)
    track_result = build_track_competency_map()
    save_track_competency_map(track_result)
    total_courses = sum(e["total_courses"] for e in track_result.values())
    total_with_syllabus = sum(e["courses_with_syllabus"] for e in track_result.values())
    print(f"트랙 수          : {len(track_result)}개")
    print(f"총 과목 수       : {total_courses}개")
    print(f"강의계획서 있음  : {total_with_syllabus}개")
    print(f"강의계획서 없음  : {total_courses - total_with_syllabus}개")


if __name__ == "__main__":
    run()
