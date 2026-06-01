"""채용공고 메타데이터의 기술스택을 직무 카테고리별로 집계해 yaml 에 주입한다.

``data/processed/rag/metadata.json`` 의 채용공고(job_posting) 메타에서 카테고리별
``tech_stack`` 을 모아 빈도 상위 N개를 ``config/category_to_job_type.yaml`` 의
각 카테고리 항목에 ``tech_stacks`` 필드로 기록한다.

집계는 런타임 직무 검색 어댑터(``RagflowJobSearchClient``)와 **동일한 정규화**
(``extract_tech_keywords``)를 거친다. 어휘가 어긋나면 노드의 competency_tags
분리(공고 기술 - 집계)가 부정확해지기 때문이다.

집계 대상 카테고리는 yaml 에 이미 매핑된 카테고리로 한정한다 — 미매핑 카테고리
공고는 노드가 어차피 스킵한다. yaml 의 헤더 주석과 항목 순서는 보존한다.

실행:
    uv run python scripts/aggregate_category_tech_stacks.py --top-n 15
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import yaml

from tracktory.common.tech_keywords import extract_tech_keywords

_ROOT = Path(__file__).resolve().parents[1]
_METADATA_PATH = _ROOT / "data" / "processed" / "rag" / "metadata.json"
_YAML_PATH = _ROOT / "src" / "tracktory" / "config" / "category_to_job_type.yaml"

# 공고 수가 적어(게임 6건·보안 27건) 빈도 집계가 노이즈에 휘둘리는 카테고리는
# 도메인 지식으로 직접 큐레이션한다 — 집계 결과를 이 목록으로 덮어쓴다.
_CURATED_TECH_STACKS: dict[str, list[str]] = {
    "게임": ["Unity", "Unreal Engine", "C#", "C++", "Blender", "Maya", "HLSL", "ARKit"],
    "보안": ["Linux", "Python", "OWASP", "Wireshark", "Nmap", "Burp Suite", "SIEM", "Metasploit"],
}


def _iter_job_postings(metadata_path: Path) -> Iterator[tuple[str, list[str]]]:
    """metadata.json 에서 (카테고리, 원시 tech_stack 리스트) 쌍을 순회한다."""
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    for entry in data:
        meta = entry.get("metadata") or {}
        if meta.get("doc_type") != "job_posting":
            continue
        category = meta.get("category")
        if not category:
            continue
        tech_stack = meta.get("tech_stack") or []
        yield str(category), [str(t) for t in tech_stack]


def aggregate(
    metadata_path: Path, categories: list[str], top_n: int
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """카테고리별 기술스택 빈도 상위 ``top_n`` 개와 공고 수를 집계한다.

    Args:
        metadata_path: metadata.json 경로.
        categories: 집계 대상 카테고리(yaml 에 매핑된 것).
        top_n: 카테고리별 상위 기술 수. 0이면 전체.

    Returns:
        ``({카테고리: [기술, ...]}, {카테고리: 공고수})`` 튜플.
    """
    counters: dict[str, Counter[str]] = {c: Counter() for c in categories}
    totals: dict[str, int] = dict.fromkeys(categories, 0)

    for category, raw_tags in _iter_job_postings(metadata_path):
        if category not in counters:
            continue
        totals[category] += 1
        # 런타임 어댑터와 동일 정규화. 공고 내 중복은 set 으로 1회만 카운트.
        for tech in set(extract_tech_keywords(", ".join(raw_tags))):
            counters[category][tech] += 1

    aggregated = {
        category: [name for name, _ in counter.most_common(top_n if top_n > 0 else None)]
        for category, counter in counters.items()
    }
    return aggregated, totals


def _split_header(text: str) -> str:
    """yaml 선두의 주석/공백 블록(문서화 헤더)을 그대로 떼어낸다."""
    header_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            header_lines.append(line)
        else:
            break
    return "".join(header_lines)


def write_yaml(yaml_path: Path, aggregated: dict[str, list[str]]) -> None:
    """기존 yaml 의 헤더·항목 순서를 보존하며 각 항목에 tech_stacks 를 주입한다.

    ``default_flow_style=None`` 은 스칼라만 담은 컬렉션(=tech_stacks 리스트)만
    flow(``[a, b]``)로, 나머지 매핑은 block 으로 직렬화한다. 항목 순서는
    ``sort_keys=False`` 로 파일 순서를 보존한다.
    """
    text = yaml_path.read_text(encoding="utf-8")
    header = _split_header(text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"category_to_job_type 파일은 매핑이어야 함: {yaml_path}")

    for category, entry in data.items():
        if isinstance(entry, dict):
            entry["tech_stacks"] = aggregated.get(str(category), [])

    body = yaml.safe_dump(
        data, allow_unicode=True, sort_keys=False, default_flow_style=None, width=10_000
    )
    yaml_path.write_text(header + body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=_METADATA_PATH)
    parser.add_argument("--yaml", type=Path, default=_YAML_PATH)
    parser.add_argument(
        "--top-n", type=int, default=8, help="카테고리별 상위 기술 수. 0이면 전체."
    )
    args = parser.parse_args()

    data = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    categories = [str(k) for k in data] if isinstance(data, dict) else []

    aggregated, totals = aggregate(args.metadata, categories, args.top_n)
    # 데이터가 적은 카테고리는 큐레이션 목록으로 덮어쓴다.
    for category, curated in _CURATED_TECH_STACKS.items():
        if category in aggregated:
            aggregated[category] = list(curated)
    write_yaml(args.yaml, aggregated)

    print(f"updated: {args.yaml}")
    for category in categories:
        tag = " (curated)" if category in _CURATED_TECH_STACKS else ""
        print(f"  {category}: postings={totals[category]} tech={len(aggregated[category])}{tag}")


if __name__ == "__main__":
    main()
