"""
wanted_cleaned.json → data/processed/wanted_by_job.json

원티드 공고의 category를 한글 직무명에서 job_id로 변환하고, 기술 태그를 정규화한다.
레코드 구조는 wanted_cleaned.json과 동일하며 두 필드만 변경된다.
  - category : "백엔드" → "BE" (CATEGORY_TO_JOB_ID 변환)
  - tech_stacks : NORMALIZATION_MAP 적용 후 정규화된 태그 목록

카테고리 매핑이 없는 공고는 category를 원본 그대로 유지한다.

실행:
    uv run python -m tracktory.relation.mapping.preprocessing.normalize_wanted
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tracktory.common.tech_keywords import NORMALIZATION_MAP
from tracktory.relation.mapping.config import WANTED_CLEANED_PATH, WANTED_BY_JOB_PATH

# wanted_cleaned.json의 category 값 → job_id
CATEGORY_TO_JOB_ID: dict[str, str] = {
    "프론트엔드": "FE",
    "백엔드": "BE",
    "데이터엔지니어": "DE",
    "데이터엔지니어/사이언스": "DE",
    "데이터분석": "DA",
    "데이터사이언스": "DA",
    "AI/ML": "AI",
    "보안": "SEC",
    "모바일": "MOB",
    "DevOps/인프라": "DEVOPS",
    "게임": "GAME",
}


def _normalize_tag(tag: str) -> str:
    """기술 태그를 NORMALIZATION_MAP 기준 정규 이름으로 변환한다.

    맵에 없으면 원본 대소문자를 유지한다. .title() 폴백을 쓰지 않는 이유:
    "SQL"→"Sql", "AWS"→"Aws" 같은 오변환 방지.

    Args:
        tag: 원본 기술 태그 문자열.

    Returns:
        정규화된 태그 이름.
    """
    stripped = tag.strip()
    if not stripped:
        return stripped
    if stripped in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[stripped]
    lower = stripped.lower()
    for variant, canonical in NORMALIZATION_MAP.items():
        if variant.lower() == lower:
            return canonical
    return stripped


def normalize_wanted(
    wanted_path: Path = WANTED_CLEANED_PATH,
    out_path: Path = WANTED_BY_JOB_PATH,
) -> list[dict]:
    """wanted_cleaned.json의 category·tech_stacks를 정규화하여 저장한다.

    레코드 구조는 그대로 유지하고 category 값과 tech_stacks만 변경한다.
    category 매핑이 없는 공고는 category를 원본 그대로 유지한다.

    Args:
        wanted_path: wanted_cleaned.json 경로.
        out_path: 출력 JSON 경로.

    Returns:
        category가 정규화된 공고 리스트 (전체 레코드 포함).
    """
    with wanted_path.open(encoding="utf-8") as f:
        postings: list[dict] = json.load(f)

    result: list[dict] = []

    for post in postings:
        record = dict(post)
        job_id = CATEGORY_TO_JOB_ID.get(post.get("category", ""))
        if job_id is not None:
            record["category"] = job_id
        raw_tags: list[str] = record.get("tech_stacks") or []
        record["tech_stacks"] = [_normalize_tag(str(t)) for t in raw_tags if str(t).strip()]
        result.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    with WANTED_CLEANED_PATH.open(encoding="utf-8") as _f:
        _total_raw = len(json.load(_f))

    result = normalize_wanted()
    job_counts: Counter[str] = Counter(r["category"] for r in result)
    mapped = sum(v for k, v in job_counts.items() if k in CATEGORY_TO_JOB_ID.values())

    print(f"\n완료: {WANTED_BY_JOB_PATH}")
    print(f"총 공고 수    : {_total_raw}건")
    print(f"카테고리 매핑 : {mapped}건")
    print(f"미매핑 유지   : {_total_raw - mapped}건")
