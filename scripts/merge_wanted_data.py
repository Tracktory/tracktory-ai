import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple wanted crawl JSON files with deduplication."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="JSON file paths or glob patterns (default: data/raw/wanted_*.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/raw/wanted_merged_YYYYMMDD.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def resolve_inputs(patterns, verbose=False):
    paths = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if matched:
            paths.extend(matched)
        else:
            p = Path(pattern)
            if p.exists():
                paths.append(str(p))
            elif verbose:
                print(f"  [warn] no files matched: {pattern}")
    return sorted(set(paths))


def load_file(path, verbose):
    p = Path(path)
    if not p.exists():
        print(f"  [skip] file not found: {path}")
        return []
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"  [skip] expected list in {path}, got {type(data).__name__}")
        return []
    if verbose:
        print(f"  loaded {len(data):>5} records from {path}")
    return data


def merge(records, verbose):
    seen = {}
    for record in records:
        sid = record.get("source_id")
        if sid is None:
            continue
        if sid not in seen:
            seen[sid] = record
        else:
            existing_at = seen[sid].get("collected_at", "")
            incoming_at = record.get("collected_at", "")
            if incoming_at > existing_at:
                if verbose:
                    print(
                        f"  [dedup] source_id={sid}: replacing {existing_at!r} -> {incoming_at!r}"
                    )
                seen[sid] = record
    return seen


def main():
    args = parse_args()
    verbose = args.verbose

    default_patterns = ["data/raw/wanted_*.json"]
    input_patterns = args.inputs if args.inputs else default_patterns

    today = datetime.now().strftime("%Y%m%d")
    output_path = Path(args.output) if args.output else Path(f"data/raw/wanted_merged_{today}.json")

    print("Resolving input files...")
    input_files = resolve_inputs(input_patterns, verbose=verbose)
    if not input_files:
        print("No input files found. Exiting.")
        return

    print(f"Found {len(input_files)} input file(s).")

    all_records = []
    for path in input_files:
        records = load_file(path, verbose)
        all_records.extend(records)

    total_loaded = len(all_records)
    print(f"\nTotal records loaded: {total_loaded}")

    print("Deduplicating by source_id...")
    merged = merge(all_records, verbose)

    duplicates_removed = total_loaded - len(merged)
    final_records = sorted(merged.values(), key=lambda r: r.get("collected_at", ""), reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)

    print("\n--- Merge Statistics ---")
    print(f"  Input files      : {len(input_files)}")
    print(f"  Records loaded   : {total_loaded}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Final count      : {len(final_records)}")
    print(f"  Output           : {output_path}")

    category_counts = defaultdict(int)
    for record in final_records:
        cat = record.get("category") or "unknown"
        category_counts[cat] += 1

    print("\n--- Category Distribution ---")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {count}")


if __name__ == "__main__":
    main()
