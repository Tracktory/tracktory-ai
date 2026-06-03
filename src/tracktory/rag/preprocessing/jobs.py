"""채용공고 전처리: 공고별 RAG 문서 생성

입력 방법 두 가지:
  1. build_jobs_from_csv()     — data/processed/wanted_cleaned.csv (배치 처리용)
  2. build_jobs_from_postings() — list[JobPosting] (크롤러 직접 연계용)

출력: data/processed/rag/jobs/채용공고_{source}_{source_id}.txt
"""

import json
import os
from typing import Any

import pandas as pd

from tracktory.common.config import CommonConfig
from tracktory.common.models import JobPosting

# 온보딩 매칭 라벨 사전계산 산출물 (scripts/generate_job_labels.py 산출).
# source_id → {"env": [...], "interests": [...], "category": ...}.
_DEFAULT_JOB_LABELS_PATH = CommonConfig.DATA_PROCESSED_DIR / "job_env_labels.json"


def _val(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _load_job_labels(path: str | os.PathLike[str] | None) -> dict[str, dict[str, list[str]]]:
    """source_id → {"env": [...], "interests": [...]}. 파일 부재 시 빈 dict(주입 생략)."""
    if path is None or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict[str, list[str]]] = {}
    for sid, entry in raw.items():
        if isinstance(entry, dict):
            out[str(sid)] = {
                "env": list(entry.get("env", [])),
                "interests": list(entry.get("interests", [])),
            }
    return out


def _build_job_document(data: dict[str, str]) -> str:
    """CSV·JobPosting 두 입력 경로가 동일 포맷 공유해야 해 분리. 공통 dict를 RAG 문서 텍스트로 변환."""
    title = data.get("title", "")
    company = data.get("company", "")
    category = data.get("category", "")
    tech_stacks = data.get("tech_stacks", "")
    industry = data.get("industry_name", "")
    deadline = data.get("deadline", "")
    responsibilities = data.get("responsibilities", "")
    requirements = data.get("requirements", "")
    preferred = data.get("preferred", "")

    header_parts: list[str] = []
    if title:
        header_parts.append(f"채용공고: {title}")
    if company:
        header_parts.append(f"회사: {company}")
    if category:
        header_parts.append(f"분야: {category}")

    lines: list[str] = [f"[{' | '.join(header_parts)}]"]

    # 온보딩 매칭 라벨을 헤더 바로 아래에 둔다. 원raw 태그/기술이 아니라 온보딩
    # 어휘(표면형)여야 질의 토큰과 매칭된다.
    interests = data.get("interests", "")
    if interests:
        lines.append(f"[관심분야] {interests}")
    work_env = data.get("work_env", "")
    if work_env:
        lines.append(f"[환경] {work_env}")

    lines.append("")

    if title:
        lines.append(f"직무: {title}")
    if company:
        lines.append(f"회사: {company}")
    if category:
        lines.append(f"카테고리: {category}")
    if industry:
        lines.append(f"업종: {industry}")
    if deadline:
        lines.append(f"마감: {deadline}")
    if tech_stacks:
        lines.append(f"기술스택: {tech_stacks}")

    if responsibilities:
        lines += ["", "■ 주요 업무", responsibilities]
    if requirements:
        lines += ["", "■ 자격 요건", requirements]
    if preferred:
        lines += ["", "■ 우대 사항", preferred]

    return "\n".join(lines).strip()


def build_jobs_from_csv(
    csv_path: str,
    output_dir: str,
    job_labels_path: str | os.PathLike[str] | None = _DEFAULT_JOB_LABELS_PATH,
) -> list[dict[str, str]]:
    """배치 전처리용. 크롤링 결과 CSV를 한 번에 RAG 문서로 변환."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8")
    labels_by_sid = _load_job_labels(job_labels_path)

    results: list[dict[str, str]] = []
    seen_core: set[str] = set()
    for idx, row in df.iterrows():
        data: dict[str, str] = {
            "title": _val(row.get("title")),
            "company": _val(row.get("company")),
            "category": _val(row.get("category")),
            "tech_stacks": _val(row.get("tech_stacks")),
            "location": _val(row.get("location")),
            "salary": _val(row.get("salary")),
            "experience": _val(row.get("experience")),
            "industry_name": _val(row.get("industry_name")),
            "deadline": _val(row.get("deadline")),
            "url": _val(row.get("url")),
            "source": _val(row.get("source")),
            "source_id": _val(row.get("source_id")),
            "responsibilities": _val(row.get("responsibilities")),
            "requirements": _val(row.get("requirements")),
            "preferred": _val(row.get("preferred")),
            "benefits": _val(row.get("benefits")),
        }
        source = data["source"] or "job"
        source_id = data["source_id"] or str(idx)
        labels = labels_by_sid.get(source_id, {})
        data["interests"] = ", ".join(labels.get("interests", []))
        data["work_env"] = ", ".join(labels.get("env", []))

        # 동일 공고 중복 제거: 주요 업무+자격 요건 본문이 같으면 한 건만 남긴다
        # (같은 채용을 다른 source_id 로 재게시한 케이스). 본문이 비면 dedup 대상 제외.
        core = (data["responsibilities"] + "\n" + data["requirements"]).strip()
        if core:
            if core in seen_core:
                continue
            seen_core.add(core)

        doc = _build_job_document(data)
        filename = f"채용공고_{source}_{source_id}.txt"
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(doc)

        results.append(
            {
                "source": source,
                "source_id": source_id,
                "title": data["title"],
                "company": data["company"],
            }
        )

    return results


def build_jobs_from_postings(jobs: list[JobPosting], output_dir: str) -> list[dict[str, str]]:
    """크롤러 직접 연계용. 실시간 수집된 JobPosting 객체를 즉시 RAG 문서로 변환."""
    os.makedirs(output_dir, exist_ok=True)

    results: list[dict[str, str]] = []
    for job in jobs:
        data: dict[str, str] = {
            "title": job.title,
            "company": job.company,
            "category": job.category,
            "tech_stacks": ", ".join(job.tech_stacks),
            "location": job.location,
            "salary": job.salary,
            "experience": job.experience,
            "industry_name": str(job.extra.get("industry_name", "")),
            "deadline": str(job.extra.get("deadline", "")),
            "url": job.url,
            "source": job.source,
            "source_id": job.source_id,
            "responsibilities": str(job.extra.get("responsibilities", "")),
            "requirements": str(job.extra.get("requirements", "")),
            "preferred": str(job.extra.get("preferred", "")),
            "benefits": str(job.extra.get("benefits", "")),
        }

        doc = _build_job_document(data)
        filename = f"채용공고_{job.source}_{job.source_id}.txt"
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(doc)

        results.append(
            {
                "source": job.source,
                "source_id": job.source_id,
                "title": job.title,
                "company": job.company,
            }
        )

    return results
