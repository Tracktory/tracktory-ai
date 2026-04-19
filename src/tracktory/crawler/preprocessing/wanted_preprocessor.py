"""
Wanted job posting preprocessor.

Pipeline:
  1. Load raw JSON
  2. Clean bullet-point formatting in extra fields
  3. Convert empty strings to None for preferred / experience / salary
  4. Save processed JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tracktory.common.tech_keywords import classify_job_category


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_raw_json(path: str) -> list[dict]:
    """Load raw JSON array from *path* and return as a list of dicts."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_processed(data: list[dict], path: str) -> None:
    """Write *data* to *path* as pretty-printed UTF-8 JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Bullet cleanup (A2)
# ---------------------------------------------------------------------------

def clean_bullet_text(text: str) -> str:
    """Normalise bullet / numbered list formatting in *text*.

    Rules applied in order:
    1. Replace Korean middle dot ``ㆍ`` with ``• `` so it is treated
       uniformly by the steps below.
    2. Insert a newline before every ``•`` that is NOT already preceded by
       a newline (handles concatenated bullets such as
       ``"• item A• item B"``).
    3. Replace leading ``•`` (possibly with surrounding spaces) on each
       line with ``- ``.
    4. Insert a newline before concatenated numbered items like ``2. ``
       (i.e. a digit + period + space that is NOT at the start of a line).
    5. Collapse runs of 3+ consecutive newlines down to two, then strip
       leading/trailing whitespace.

    Returns the cleaned string, or an empty string when *text* is falsy.
    """
    if not text:
        return text

    # 1. Normalise Korean middle dot to bullet
    result = text.replace("ㆍ", "•")

    # 2. Insert newline before • that is not already preceded by \n
    #    Pattern: any char that is NOT \n, followed by •
    result = re.sub(r"(?<!\n)•", "\n•", result)

    # 3. Replace '• ' or '•' at the start of a line with '- '
    result = re.sub(r"^•\s*", "- ", result, flags=re.MULTILINE)

    # 4. Insert newline before concatenated numbered items
    #    e.g. "...개선2. 시스템..."  →  "...개선\n2. 시스템..."
    #    Only trigger when the digit is NOT already at the start of a line.
    result = re.sub(r"(?<!\n)(\d+\. )", r"\n\1", result)

    # 5. Collapse 3+ consecutive newlines to exactly two
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


# ---------------------------------------------------------------------------
# Category reclassification
# ---------------------------------------------------------------------------

def reclassify_category(record: dict) -> str:
    """Return a (possibly improved) category for *record*.

    If the current category is not "기타" it is left unchanged.
    When the category IS "기타", ``classify_job_category`` is used with the
    job title plus the concatenated responsibilities/requirements text to
    attempt a more specific classification.
    """
    current = record.get("category", "기타")
    if current != "기타":
        return current

    title = record.get("title", "")
    extra = record.get("extra") or {}
    responsibilities = extra.get("responsibilities", "") or ""
    requirements = extra.get("requirements", "") or ""
    text = (responsibilities + " " + requirements).strip()

    return classify_job_category(title, text)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (A4)
# ---------------------------------------------------------------------------

_BULLET_FIELDS = ("responsibilities", "requirements", "preferred", "benefits")
_NULLABLE_FIELDS = ("preferred", "experience", "salary")


def _preprocess_record(record: dict) -> tuple[dict, bool, int, bool]:
    """Return (processed_record, bullet_was_cleaned, null_conversions, category_reclassified)."""
    rec = dict(record)
    extra = dict(rec.get("extra") or {})

    bullet_cleaned = False
    null_count = 0

    # --- bullet cleanup ---
    for field in _BULLET_FIELDS:
        original = extra.get(field, "")
        if not isinstance(original, str):
            continue
        cleaned = clean_bullet_text(original)
        if cleaned != original:
            bullet_cleaned = True
        extra[field] = cleaned

    rec["extra"] = extra

    # --- empty string → None ---
    for field in _NULLABLE_FIELDS:
        # top-level fields
        if field in rec and rec[field] == "":
            rec[field] = None
            null_count += 1
        # extra fields
        if field in extra and extra[field] == "":
            extra[field] = None
            null_count += 1

    # --- category reclassification ---
    # reclassify_category reads from rec (which already has updated extra)
    new_category = reclassify_category(rec)
    category_reclassified = (rec.get("category") == "기타" and new_category != "기타")
    rec["category"] = new_category

    return rec, bullet_cleaned, null_count, category_reclassified


_CSV_COLUMNS = [
    "source", "source_id", "company", "title", "category",
    "tech_stacks", "location", "salary", "url", "experience",
    "collected_at", "responsibilities", "requirements",
    "preferred", "benefits", "deadline", "company_id", "industry_name",
]


def _flatten(record: dict) -> dict:
    """Flatten a record by merging extra fields into top-level."""
    flat = {k: v for k, v in record.items() if k != "extra"}
    extra = record.get("extra") or {}
    flat.update(extra)
    # tech_stacks list → comma-separated string
    ts = flat.get("tech_stacks")
    if isinstance(ts, list):
        flat["tech_stacks"] = ", ".join(ts)
    return flat


def _save_csv(data: list[dict], path: Path) -> None:
    """Write *data* to *path* as UTF-8 CSV with flattened extra fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in data:
            writer.writerow(_flatten(record))


def main() -> None:
    """CLI entry point: load → clean → save → print statistics."""
    parser = argparse.ArgumentParser(description="Preprocess wanted job data")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input JSON file path. Default: auto-detect latest in data/raw/")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file path. Default: data/processed/wanted_cleaned.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]

    if args.input:
        raw_path = Path(args.input)
    else:
        raw_dir = project_root / "data" / "raw"
        candidates = sorted(raw_dir.glob("wanted_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("Error: No wanted_*.json files found in data/raw/")
            sys.exit(1)
        raw_path = candidates[0]

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = project_root / "data" / "processed" / "wanted_cleaned.json"

    print(f"Loading  : {raw_path}")
    raw_data = load_raw_json(str(raw_path))

    processed: list[dict] = []
    total_bullet_cleaned = 0
    total_null_converted = 0
    total_reclassified = 0

    # Track original categories before reclassification for comparison
    original_category_dist: Counter = Counter()

    for record in raw_data:
        original_category_dist[record.get("category", "unknown")] += 1
        rec, was_cleaned, null_count, was_reclassified = _preprocess_record(record)
        processed.append(rec)
        if was_cleaned:
            total_bullet_cleaned += 1
        total_null_converted += null_count
        if was_reclassified:
            total_reclassified += 1

    save_processed(processed, str(out_path))
    print(f"Saved    : {out_path}")

    # --- CSV export ---
    csv_path = out_path.with_suffix(".csv")
    _save_csv(processed, csv_path)
    print(f"Saved    : {csv_path}")

    # --- statistics ---
    final_category_dist = Counter(r.get("category", "unknown") for r in processed)

    print()
    print("=== Preprocessing Statistics ===")
    print(f"  Total records              : {len(processed)}")
    print(f"  Bullet-cleaned records     : {total_bullet_cleaned}")
    print(f"  Empty→null conversions     : {total_null_converted}")
    print(f"  기타→specific reclassified : {total_reclassified}")
    print()
    print("  Category distribution (before → after):")
    all_cats = sorted(
        set(original_category_dist) | set(final_category_dist),
        key=lambda c: -final_category_dist.get(c, 0),
    )
    for cat in all_cats:
        before = original_category_dist.get(cat, 0)
        after = final_category_dist.get(cat, 0)
        change = f"  ({after - before:+d})" if before != after else ""
        print(f"    {cat:<25} {before:>5} → {after:>5}{change}")


if __name__ == "__main__":
    main()
