"""채용공고 전처리: 공고별 RAG 문서 생성

입력 방법 두 가지:
  1. build_jobs_from_csv()     — data/processed/wanted_cleaned.csv (배치 처리용)
  2. build_jobs_from_postings() — list[JobPosting] (크롤러 직접 연계용)

출력: data/processed/rag/jobs/채용공고_{source}_{source_id}.txt
"""

import os
from typing import Any

import pandas as pd

from tracktory.common.models import JobPosting


def _val(v: Any) -> str:
    """pandas NaN이면 빈 문자열, 아니면 문자열로 변환한다."""
    if pd.isna(v):
        return ""
    return str(v).strip()


def _build_job_document(data: dict[str, str]) -> str:
    """공고 데이터 dict를 RAG 문서 텍스트로 변환한다."""
    title = data.get("title", "")
    company = data.get("company", "")
    category = data.get("category", "")
    tech_stacks = data.get("tech_stacks", "")
    location = data.get("location", "")
    salary = data.get("salary", "")
    experience = data.get("experience", "")
    industry = data.get("industry_name", "")
    deadline = data.get("deadline", "")
    url = data.get("url", "")
    responsibilities = data.get("responsibilities", "")
    requirements = data.get("requirements", "")
    preferred = data.get("preferred", "")
    benefits = data.get("benefits", "")

    header_parts: list[str] = []
    if title:
        header_parts.append(f"채용공고: {title}")
    if company:
        header_parts.append(f"회사: {company}")
    if category:
        header_parts.append(f"분야: {category}")

    lines: list[str] = [f"[{' | '.join(header_parts)}]", ""]

    if title:
        lines.append(f"직무: {title}")
    if company:
        lines.append(f"회사: {company}")
    if category:
        lines.append(f"카테고리: {category}")
    if industry:
        lines.append(f"업종: {industry}")
    if experience:
        lines.append(f"경력: {experience}")
    if location:
        lines.append(f"위치: {location}")
    lines.append(f"연봉: {salary}" if salary else "연봉: 미공개")
    if deadline:
        lines.append(f"마감: {deadline}")
    if tech_stacks:
        lines.append(f"기술스택: {tech_stacks}")
    if url:
        lines.append(f"공고 URL: {url}")

    if responsibilities:
        lines += ["", "■ 주요 업무", responsibilities]
    if requirements:
        lines += ["", "■ 자격 요건", requirements]
    if preferred:
        lines += ["", "■ 우대 사항", preferred]
    if benefits:
        lines += ["", "■ 복지 혜택", benefits]

    return "\n".join(lines).strip()


def build_jobs_from_csv(csv_path: str, output_dir: str) -> list[dict[str, str]]:
    """CSV 파일에서 공고별 txt 파일을 생성한다."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8")

    results: list[dict[str, str]] = []
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

        doc = _build_job_document(data)
        filename = f"채용공고_{source}_{source_id}.txt"
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(doc)

        results.append({
            "source": source,
            "source_id": source_id,
            "title": data["title"],
            "company": data["company"],
        })

    return results


def build_jobs_from_postings(jobs: list[JobPosting], output_dir: str) -> list[dict[str, str]]:
    """JobPosting 리스트에서 공고별 txt 파일을 생성한다."""
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

        results.append({
            "source": job.source,
            "source_id": job.source_id,
            "title": job.title,
            "company": job.company,
        })

    return results
