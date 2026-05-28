"""한성대 학사 데이터 RAG 전처리 파이프라인

data/raw/hansung/ 아래 CSV들을 읽어 data/processed/rag/ 에 RAGFlow 적재용 txt 파일을 생성한다.

실행:
    uv run python -m tracktory.rag.preprocessing.pipeline
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from tracktory.common.config import CommonConfig
from tracktory.rag.preprocessing.courses import build_courses
from tracktory.rag.preprocessing.extract_rag_metadata import extract_all as extract_metadata
from tracktory.rag.preprocessing.find_syllabus_duplicates import (
    find_duplicates,
    move_duplicates,
)
from tracktory.rag.preprocessing.find_syllabus_duplicates import (
    write_outputs as write_duplicate_outputs,
)
from tracktory.rag.preprocessing.jobs import build_jobs_from_csv
from tracktory.rag.preprocessing.syllabus import build_syllabi
from tracktory.rag.preprocessing.tracks.builder import build_all, load_college_map
from tracktory.rag.preprocessing.tracks.cleaner import clean
from tracktory.rag.preprocessing.tracks.parser import parse

logger = logging.getLogger(__name__)

_DIRECTORY_PAGE_SIGNALS: list[str] = ["대학_트랙", "대학전체", "공유하기", "팝업존"]
_COLLEGE_JSON_FILES: tuple[str, ...] = ("한성대_트랙구조.json", "track_structure.json")
_TRACK_CSV_FILES: tuple[str, ...] = ("한성대_트랙정보.csv", "tracks.csv")
_COURSE_CSV_FILES: tuple[str, ...] = ("한성대_강의정보.csv", "courses.csv")
_JOB_CSV_FILES: tuple[str, ...] = ("wanted_cleaned.csv",)
_SYLLABUS_INPUTS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("2026-1", ("한성대_강의계획서.csv", "syllabi.csv"), ("syllabi", "1학기")),
    ("2025-2", ("hansung_syllabuses_raw_20252.csv",), ("syllabi", "2학기")),
]


def _resolve_raw_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path:
    for filename in candidates:
        path = raw_dir / filename
        if path.exists():
            return path
    return raw_dir / candidates[0]


def process_tracks(
    track_csv: str,
    college_json: str,
    output_dir: str,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """크롤링 결과에 대학 전체 목록 페이지가 섞여 스킵 처리 필요. clean → parse → build 3단계 + 유효하지 않은 row 제거.

    Args:
        track_csv: tracks.csv 경로.
        college_json: track_structure.json 경로 (부재 시 college_map 빈 dict로 fallback).
        output_dir: txt 파일 출력 디렉터리.

    Returns:
        (results, skipped) — results는 build_all() 반환값, skipped는 (트랙명, 사유) 튜플 리스트.
    """
    college_map = load_college_map(college_json)
    df = pd.read_csv(track_csv, encoding="utf-8")

    sections_by_track: dict[str, dict[str, str]] = {}
    skipped: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        track_name: str = str(row["Track_Name"])
        raw_text: str = str(row["Raw_Text"])

        if any(sig in raw_text for sig in _DIRECTORY_PAGE_SIGNALS):
            skipped.append((track_name, "대학 전체 목록 페이지 — 개별 트랙 데이터 없음"))
            continue

        clean_lines = clean(raw_text, track_name)
        if len(clean_lines) < 3:
            skipped.append((track_name, f"정제 후 내용 부족 ({len(clean_lines)}줄)"))
            continue

        sections_by_track[track_name] = parse(clean_lines)

    for name, _ in skipped:
        safe_name = name.replace("/", "_").replace("ㆍ", "_").replace("·", "_")
        old_file = os.path.join(output_dir, f"트랙소개_{safe_name}.txt")
        if os.path.exists(old_file):
            os.remove(old_file)

    results = build_all(sections_by_track, college_map, output_dir)
    return results, skipped


def run_all(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """경로 설정과 4개 파이프라인 순서를 하나의 진입점에서 처리하기 위함. CommonConfig 기반 경로로 트랙·강의·강의계획서·채용공고 순 실행.

    Args:
        raw_dir: 원본 CSV 디렉터리. 기본값은 data/raw/hansung/.
        output_dir: txt 출력 루트 디렉터리. 기본값은 data/processed/rag/.
    """
    raw = raw_dir or CommonConfig.DATA_RAW_DIR / "hansung"
    out = output_dir or CommonConfig.DATA_PROCESSED_DIR / "rag"
    college_json = _resolve_raw_file(raw, _COLLEGE_JSON_FILES)

    # ── 1. 트랙 소개 ───────────────────────────────────────────────────────────
    logger.info("[1/6] 트랙 소개 처리 중...")
    results, skipped = process_tracks(
        track_csv=str(_resolve_raw_file(raw, _TRACK_CSV_FILES)),
        college_json=str(college_json),
        output_dir=str(out / "tracks"),
    )
    logger.info("  → tracks/ (%d개 / 스킵: %d개)", len(results), len(skipped))
    for name, reason in skipped:
        logger.info("    [SKIP] %s: %s", name, reason)

    # ── 2. 강의정보 ────────────────────────────────────────────────────────────
    logger.info("[2/6] 강의정보 (교과목 목록) 처리 중...")
    course_results = build_courses(
        track_csv=str(_resolve_raw_file(raw, _COURSE_CSV_FILES)),
        college_json=str(college_json),
        output_dir=str(out / "courses"),
    )
    logger.info("  → courses/ (%d개)", len(course_results))

    # ── 3. 강의계획서 ──────────────────────────────────────────────────────────
    logger.info("[3/6] 강의계획서 처리 중...")
    total_syllabi = 0
    total_skipped = 0
    for label, filenames, output_parts in _SYLLABUS_INPUTS:
        syllabus_results, syllabus_skipped = build_syllabi(
            syllabus_csv=str(_resolve_raw_file(raw, filenames)),
            output_dir=str(out.joinpath(*output_parts)),
        )
        total_syllabi += len(syllabus_results)
        total_skipped += len(syllabus_skipped)
        logger.info(
            "  → %s/ [%s] (%d개 / 스킵: %d개)",
            "/".join(output_parts),
            label,
            len(syllabus_results),
            len(syllabus_skipped),
        )
    logger.info("  → syllabi 합계 (%d개 / 스킵: %d개)", total_syllabi, total_skipped)

    # ── 4. 채용공고 ────────────────────────────────────────────────────────────
    raw_jobs_csv = _resolve_raw_file(raw, _JOB_CSV_FILES)
    processed_jobs_csv = CommonConfig.DATA_PROCESSED_DIR / "wanted_cleaned.csv"
    jobs_csv = str(raw_jobs_csv if raw_jobs_csv.exists() else processed_jobs_csv)
    if os.path.exists(jobs_csv):
        logger.info("[4/6] 채용공고 처리 중...")
        job_results = build_jobs_from_csv(
            csv_path=jobs_csv,
            output_dir=str(out / "jobs"),
        )
        logger.info("  → jobs/ (%d개)", len(job_results))
    else:
        logger.info("[4/6] 채용공고 건너뜀 — 파일 없음: %s", jobs_csv)

    # ── 5. 강의계획서 중복 탐지 + 격리 ────────────────────────────────────────
    logger.info("[5/6] 강의계획서 중복 탐지 중...")
    syllabi_dir = out / "syllabi"
    if syllabi_dir.exists():
        duplicate_groups = find_duplicates(syllabi_dir)
        groups_path, delete_list_path = write_duplicate_outputs(duplicate_groups, out)
        duplicate_count = sum(len(g["duplicates"]) for g in duplicate_groups)
        logger.info(
            "  → 중복 그룹 %d개 / 중복 파일 %d개 (groups: %s, delete-list: %s)",
            len(duplicate_groups),
            duplicate_count,
            groups_path.name,
            delete_list_path.name,
        )
        if duplicate_groups:
            quarantine_dir = out / "_duplicates" / "syllabi"
            moved = move_duplicates(duplicate_groups, syllabi_dir, quarantine_dir)
            logger.info("  → 격리 이동 %d개 → %s", moved, quarantine_dir)
    else:
        logger.info("[5/6] 중복 탐지 건너뜀 — 디렉터리 없음: %s", syllabi_dir)

    # ── 6. 메타데이터 추출 ────────────────────────────────────────────────────
    logger.info("[6/6] 메타데이터 추출 중...")
    metadata = extract_metadata(out)
    metadata_path = out / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("  → metadata.json (%d개 문서)", len(metadata))

    logger.info("[완료] 전체 전처리 파이프라인 완료")


if __name__ == "__main__":
    run_all()
