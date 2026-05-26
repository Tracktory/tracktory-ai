"""RAG 전처리 결과(.txt)에서 메타데이터 추출."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tracktory.common.config import CommonConfig

DEFAULT_INPUT_DIR = CommonConfig.DATA_PROCESSED_DIR / "rag"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "metadata.json"


def _first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_track_like(filename: str, text: str, doc_type: str) -> dict[str, Any]:
    header = text.splitlines()[0] if text.splitlines() else ""
    metadata: dict[str, Any] = {"doc_type": doc_type}

    track = re.search(r"\[트랙:\s*([^|]+)", header)
    college = re.search(r"대학:\s*([^|]+)", header)
    department = re.search(r"학부:\s*([^\]]+)", header)

    if track:
        metadata["track_name"] = track.group(1).strip()
    else:
        prefix = "트랙소개_" if doc_type == "track_intro" else "교육과정_"
        metadata["track_name"] = filename.removeprefix(prefix).removesuffix(".txt")
    if college:
        metadata["college"] = college.group(1).strip()
    if department:
        metadata["department"] = department.group(1).strip()

    return metadata


def _parse_syllabus(path: Path, filename: str, text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"doc_type": "syllabus"}

    if path.parent.name in {"1학기", "2학기"}:
        metadata["semester"] = path.parent.name

    # 파일명: 강의계획서_{과목명}_{코드}.txt — 코드는 항상 마지막 토큰
    code = filename.removeprefix("강의계획서_").removesuffix(".txt").rsplit("_", 1)[-1]
    if code:
        metadata["course_code"] = code

    fields = {
        "course_name": r"^과목명\s*:?\s*(.+)$",
        "credits_type": r"^학점/구분:\s*(.+)$",
        "target_grade": r"^수강대상\s*:?\s*(.+)$",
        "professor_raw": r"^담당교수:\s*(.+)$",
        "email": r"^이메일\s*:?\s*(.+)$",
    }
    for key, pattern in fields.items():
        value = _first_match(pattern, text)
        if value:
            metadata[key] = value

    credits_type = metadata.pop("credits_type", "")
    if credits_type:
        parts = [part.strip() for part in credits_type.split("_", maxsplit=1)]
        metadata["credits"] = parts[0].replace("학점", "").strip()
        if len(parts) > 1:
            metadata["course_type"] = parts[1]

    professor_raw = metadata.pop("professor_raw", "")
    if professor_raw:
        match = re.match(r"(.+?)\s*\((.+)\)", professor_raw)
        if match:
            metadata["professor"] = match.group(1).strip()
            metadata["department"] = match.group(2).strip()
        else:
            metadata["professor"] = professor_raw

    return metadata


def _parse_job(filename: str, text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"doc_type": "job_posting"}
    # 파일명: 채용공고_{직무명}_{source_id}.txt — id는 항상 마지막 토큰
    metadata["job_id"] = filename.removeprefix("채용공고_").removesuffix(".txt").rsplit("_", 1)[-1]

    fields = {
        "title": r"^직무:\s*(.+)$",
        "category": r"^카테고리:\s*(.+)$",
        "industry": r"^업종:\s*(.+)$",
        "experience": r"^경력:\s*(.+)$",
        "location": r"^위치:\s*(.+)$",
        "salary": r"^연봉:\s*(.+)$",
        "deadline": r"^마감:\s*(.+)$",
        "url": r"^공고 URL:\s*(.+)$",
    }
    for key, pattern in fields.items():
        value = _first_match(pattern, text)
        if value:
            metadata[key] = value

    tech_stack = _first_match(r"^기술스택:\s*(.+)$", text)
    if tech_stack:
        metadata["tech_stack"] = [item.strip() for item in tech_stack.split(",") if item.strip()]

    return metadata


def parse_metadata(path: Path, root: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    filename = path.name

    if filename.startswith("트랙소개_"):
        metadata = _parse_track_like(filename, text, "track_intro")
    elif filename.startswith("교육과정_"):
        metadata = _parse_track_like(filename, text, "curriculum")
    elif filename.startswith("강의계획서_"):
        metadata = _parse_syllabus(path, filename, text)
    elif filename.startswith("채용공고_"):
        metadata = _parse_job(filename, text)
    else:
        metadata = {"doc_type": "unknown"}

    return {
        "path": path.relative_to(root).as_posix(),
        "filename": filename,
        "metadata": metadata,
    }


def extract_all(input_dir: Path) -> list[dict[str, Any]]:
    files = sorted(
        path
        for folder in ("tracks", "courses", "syllabi", "jobs")
        for path in (input_dir / folder).rglob("*.txt")
    )
    return [parse_metadata(path, input_dir) for path in files]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    results = extract_all(args.input_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(item["metadata"].get("doc_type", "unknown") for item in results)
    print(f"metadata written: {args.output}")
    print(f"total files: {len(results)}")
    for doc_type, count in sorted(counts.items()):
        print(f"{doc_type}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
