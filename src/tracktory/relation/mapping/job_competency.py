"""
직무별 필요역량·기술스택 매핑 테이블 생성 스크립트.

두 소스를 병합하여 직무별 기술스택 competency map을 만든다.
  - job_tech_stacks.json : 전문가 큐레이션 (rank·importance·type)
  - wanted_by_job.json  : 전처리된 채용공고 빈도 (job_id·정규화된 tech_stacks)

큐레이션 목록에 없는 기술도 실제 채용공고에 등장하면 "추가 역량(extra)"으로 포함한다.

실행:
    uv run python -m tracktory.relation.mapping.job_competency
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

from tracktory.common.tech_split import split_tech_name
from tracktory.relation.mapping.config import (
    JOB_STACKS_PATH,
    WANTED_BY_JOB_PATH,
    JOB_COMPETENCY_OUT_PATH,
)


class CompetencyItem(TypedDict):
    name: str
    type: str                  # "language" | "framework" | "tool" | "library" | "extra"
    importance: str | None     # "필수" | "권장" | None(큐레이션 없음)
    curated_rank: int | None   # 큐레이션 순위 (없으면 None)
    wanted_count: int          # 해당 직무 공고 중 등장 횟수
    wanted_rate: float         # wanted_count / total_postings


class JobCompetencyEntry(TypedDict):
    job_id: str
    job_name_ko: str
    job_name_en: str
    total_postings: int
    competencies: list[CompetencyItem]


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_wanted_freq_per_job(
    postings: list[dict],
) -> dict[str, tuple[int, Counter[str], list[set[str]]]]:
    """wanted_by_job.json 레코드를 job_id별로 집계한다.

    Args:
        postings: wanted_by_job.json 레코드 리스트.

    Returns:
        {job_id: (total_count, tech_counter, posting_tag_sets)} 딕셔너리.
    """
    total_counts: Counter[str] = Counter()
    tech_counters: dict[str, Counter[str]] = {}
    posting_tag_sets: dict[str, list[set[str]]] = {}

    for post in postings:
        job_id: str = post["category"]
        total_counts[job_id] += 1
        if job_id not in tech_counters:
            tech_counters[job_id] = Counter()
            posting_tag_sets[job_id] = []

        tag_set: set[str] = set()
        for tag in post.get("tech_stacks") or []:
            if tag:
                tech_counters[job_id][tag] += 1
                tag_set.add(tag.lower())
        posting_tag_sets[job_id].append(tag_set)

    return {
        job_id: (total_counts[job_id], tech_counters[job_id], posting_tag_sets[job_id])
        for job_id in total_counts
    }


def _merge_competencies(
    curated_stacks: list[dict],
    total_postings: int,
    tech_counter: Counter[str],
    posting_tag_sets: list[set[str]],
) -> list[CompetencyItem]:
    """큐레이션 목록과 원티드 빈도를 병합하여 CompetencyItem 리스트를 반환한다.

    Args:
        curated_stacks: job_tech_stacks.json의 tech_stacks 리스트.
        total_postings: 해당 직무의 원티드 공고 총 수.
        tech_counter: 해당 직무 공고의 기술스택 Counter.
        posting_tag_sets: 공고별 정규화 태그 집합(소문자) 리스트.

    Returns:
        정렬된 CompetencyItem 리스트.
    """
    items: list[CompetencyItem] = []
    curated_names: set[str] = set()

    for stack in curated_stacks:
        name: str = stack["name"]
        tags = split_tech_name(name)
        if tags is not None:
            tag_lowers = {t.lower() for t in tags}
            count = sum(1 for posting_tags in posting_tag_sets if posting_tags & tag_lowers)
        else:
            lower = name.lower()
            count = sum(cnt for key, cnt in tech_counter.items() if key.lower() == lower)

        rate = round(count / total_postings, 4) if total_postings > 0 else 0.0
        items.append(
            CompetencyItem(
                name=name,
                type=stack.get("type", "tool"),
                importance=stack.get("importance"),
                curated_rank=stack.get("rank"),
                wanted_count=count,
                wanted_rate=rate,
            )
        )
        curated_names.add(name.lower())
        for tag in (split_tech_name(name) or []):
            curated_names.add(tag.lower())

    if total_postings > 0:
        for tech, count in tech_counter.most_common():
            if tech.lower() in curated_names:
                continue
            rate = round(count / total_postings, 4)
            items.append(
                CompetencyItem(
                    name=tech,
                    type="extra",
                    importance=None,
                    curated_rank=None,
                    wanted_count=count,
                    wanted_rate=rate,
                )
            )

    _importance_order = {"필수": 0, "권장": 1, None: 2}

    def _sort_key(item: CompetencyItem) -> tuple[int, int]:
        return (_importance_order.get(item["importance"], 2), -item["wanted_count"])

    items.sort(key=_sort_key)
    return items


def build_job_competency_map(
    job_stacks_path: Path = JOB_STACKS_PATH,
    wanted_by_job_path: Path = WANTED_BY_JOB_PATH,
) -> dict[str, JobCompetencyEntry]:
    """직무별 역량·기술스택 매핑 테이블을 빌드한다.

    Args:
        job_stacks_path: job_tech_stacks.json 경로.
        wanted_by_job_path: normalize_wanted.py 출력 경로.

    Returns:
        {job_id: JobCompetencyEntry} 딕셔너리.
    """
    job_stacks_raw = _load_json(job_stacks_path)
    assert isinstance(job_stacks_raw, dict)
    categories: list[dict] = job_stacks_raw["job_categories"]

    postings = _load_json(wanted_by_job_path)
    assert isinstance(postings, list)

    wanted_freq = _build_wanted_freq_per_job(postings)
    result: dict[str, JobCompetencyEntry] = {}

    for cat in categories:
        job_id: str = cat["category_id"]
        total_postings, tech_counter, posting_tag_sets = wanted_freq.get(
            job_id, (0, Counter(), [])
        )
        competencies = _merge_competencies(
            curated_stacks=cat["tech_stacks"],
            total_postings=total_postings,
            tech_counter=tech_counter,
            posting_tag_sets=posting_tag_sets,
        )
        result[job_id] = JobCompetencyEntry(
            job_id=job_id,
            job_name_ko=cat["category_name_ko"],
            job_name_en=cat["category_name_en"],
            total_postings=total_postings,
            competencies=competencies,
        )

    return result


def save_job_competency_map(
    result: dict[str, JobCompetencyEntry],
    out_path: Path = JOB_COMPETENCY_OUT_PATH,
) -> None:
    """빌드된 결과를 JSON 파일로 저장한다.

    Args:
        result: build_job_competency_map() 반환값.
        out_path: 저장할 JSON 경로.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    result = build_job_competency_map()
    save_job_competency_map(result)
    total_competencies = sum(len(v["competencies"]) for v in result.values())
    print(f"\n완료: {JOB_COMPETENCY_OUT_PATH}")
    print(f"직무 수          : {len(result)}개")
    print(f"전체 역량 항목 수 : {total_competencies}개")
