"""한성대 학사 데이터 RAG 전처리 파이프라인

data/raw/hansung/ 아래 CSV들을 읽어 data/processed/rag/ 에 RAGFlow 적재용 txt 파일을 생성한다.

실행:
    uv run python -m tracktory.rag.preprocessing.pipeline
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd

from tracktory.common.config import CommonConfig
from tracktory.rag.preprocessing.courses import build_courses
from tracktory.rag.preprocessing.jobs import build_jobs_from_csv
from tracktory.rag.preprocessing.syllabus import build_syllabi
from tracktory.rag.preprocessing.tracks.builder import build_all, load_college_map
from tracktory.rag.preprocessing.tracks.cleaner import clean
from tracktory.rag.preprocessing.tracks.parser import parse

_DIRECTORY_PAGE_SIGNALS: list[str] = ["대학_트랙", "대학전체", "공유하기", "팝업존"]


def process_tracks(
    track_csv: str,
    college_json: str,
    output_dir: str,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """트랙 소개 파이프라인: clean → parse → build.

    Args:
        track_csv: 한성대_트랙정보.csv 경로.
        college_json: 한성대_트랙구조.json 경로.
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
    """네 가지 데이터(트랙·강의·강의계획서·채용공고) 전처리를 순서대로 실행한다.

    Args:
        raw_dir: 원본 CSV 디렉터리. 기본값은 data/raw/hansung/.
        output_dir: txt 출력 루트 디렉터리. 기본값은 data/processed/rag/.
    """
    raw = raw_dir or CommonConfig.DATA_RAW_DIR / "hansung"
    out = output_dir or CommonConfig.DATA_PROCESSED_DIR / "rag"
    college_json = str(raw / "한성대_트랙구조.json")

    # ── 1. 트랙 소개 ───────────────────────────────────────────────────────────
    print("[1/4] 트랙 소개 처리 중...")
    results, skipped = process_tracks(
        track_csv=str(raw / "한성대_트랙정보.csv"),
        college_json=college_json,
        output_dir=str(out / "tracks"),
    )
    print(f"  → tracks/ ({len(results)}개 / 스킵: {len(skipped)}개)")
    for name, reason in skipped:
        print(f"    [SKIP] {name}: {reason}")

    # ── 2. 강의정보 ────────────────────────────────────────────────────────────
    print("[2/4] 강의정보 (교과목 목록) 처리 중...")
    course_results = build_courses(
        track_csv=str(raw / "한성대_강의정보.csv"),
        college_json=college_json,
        output_dir=str(out / "courses"),
    )
    print(f"  → courses/ ({len(course_results)}개)")

    # ── 3. 강의계획서 ──────────────────────────────────────────────────────────
    print("[3/4] 강의계획서 처리 중...")
    syl_results, syl_skipped = build_syllabi(
        syllabus_csv=str(raw / "한성대_강의계획서.csv"),
        output_dir=str(out / "syllabi"),
    )
    print(f"  → syllabi/ ({len(syl_results)}개 / 스킵: {len(syl_skipped)}개)")

    # ── 4. 채용공고 ────────────────────────────────────────────────────────────
    jobs_csv = str(CommonConfig.DATA_PROCESSED_DIR / "wanted_cleaned.csv")
    if os.path.exists(jobs_csv):
        print("[4/4] 채용공고 처리 중...")
        job_results = build_jobs_from_csv(
            csv_path=jobs_csv,
            output_dir=str(out / "jobs"),
        )
        print(f"  → jobs/ ({len(job_results)}개)")
    else:
        print(f"[4/4] 채용공고 건너뜀 — 파일 없음: {jobs_csv}")

    print("\n[완료] 전체 전처리 파이프라인 완료")


if __name__ == "__main__":
    run_all()
