"""정규화된 강의 내용 해시로 중복 강의계획서 탐지."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from tracktory.common.config import CommonConfig

DEFAULT_SYLLABI_DIR = CommonConfig.DATA_PROCESSED_DIR / "rag" / "syllabi"
DEFAULT_OUTPUT_DIR = CommonConfig.DATA_PROCESSED_DIR / "rag"
CONTENT_MARKER = "■ 교과목 개요"


def extract_comparable_content(text: str) -> str:
    marker_index = text.find(CONTENT_MARKER)
    content = text[marker_index:] if marker_index != -1 else text
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _semester_rank(path: Path) -> tuple[int, str]:
    semester = path.parent.name
    if semester == "1학기":
        return (0, path.name)
    if semester == "2학기":
        return (1, path.name)
    return (2, path.as_posix())


def find_duplicates(syllabi_dir: Path) -> list[dict[str, Any]]:
    hash_map: dict[str, list[Path]] = {}
    for path in sorted(syllabi_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8")
        comparable = extract_comparable_content(text)
        content_hash = hash_content(comparable)
        hash_map.setdefault(content_hash, []).append(path)

    groups: list[dict[str, Any]] = []
    for content_hash, paths in sorted(hash_map.items()):
        if len(paths) < 2:
            continue

        ordered = sorted(paths, key=_semester_rank)
        keep = ordered[0]
        duplicates = ordered[1:]
        groups.append(
            {
                "hash": content_hash,
                "keep": keep.relative_to(syllabi_dir).as_posix(),
                "duplicates": [path.relative_to(syllabi_dir).as_posix() for path in duplicates],
                "count": len(ordered),
            }
        )

    return groups


def write_outputs(groups: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / "syllabi_duplicate_groups.json"
    delete_list_path = output_dir / "syllabi_duplicate_delete_list.txt"

    groups_path.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")

    duplicate_paths = [duplicate for group in groups for duplicate in group["duplicates"]]
    delete_list_path.write_text("\n".join(duplicate_paths), encoding="utf-8")

    return groups_path, delete_list_path


def move_duplicates(groups: list[dict[str, Any]], syllabi_dir: Path, duplicate_dir: Path) -> int:
    moved = 0
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        for rel_path in group["duplicates"]:
            source = syllabi_dir / rel_path
            target = duplicate_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved += 1
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syllabi-dir", type=Path, default=DEFAULT_SYLLABI_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--move-duplicates-to",
        type=Path,
        help="Optional directory to move duplicate files into. Omitted by default.",
    )
    args = parser.parse_args()

    syllabi_dir = args.syllabi_dir.resolve()
    groups = find_duplicates(syllabi_dir)
    groups_path, delete_list_path = write_outputs(groups, args.output_dir.resolve())

    duplicate_count = sum(len(group["duplicates"]) for group in groups)
    print(f"duplicate groups: {len(groups)}")
    print(f"duplicate files: {duplicate_count}")
    print(f"groups written: {groups_path}")
    print(f"delete list written: {delete_list_path}")

    if args.move_duplicates_to:
        moved = move_duplicates(groups, syllabi_dir, args.move_duplicates_to.resolve())
        print(f"duplicates moved: {moved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
