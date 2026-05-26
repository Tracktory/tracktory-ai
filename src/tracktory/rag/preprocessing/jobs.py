"""채용공고 전처리: 공고별 RAG 문서 생성

입력 방법 두 가지:
  1. build_jobs_from_csv()     — data/processed/wanted_cleaned.csv (배치 처리용)
  2. build_jobs_from_postings() — list[JobPosting] (크롤러 직접 연계용)

출력: data/processed/rag/jobs/채용공고_{직무명}_{source_id}.txt
"""

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from tracktory.common.models import JobPosting
from tracktory.rag.preprocessing.normalize import (
    normalize_text,
    safe_filename_part,
    sanitize_body_text,
)

_LEADING_TITLE_TAG_RE = re.compile(r"^(?:\[[^\]]+\]\s*)+")
_REMOTE_LOCATION_RE = re.compile(r"(리모트|원격|재택)")
_LOCATION_LABEL_RE = re.compile(
    r"(?P<label>(?:근무지|근무\s*장소|근무장소|주소|위치)\s*[:\uFF1A]?\s*)"
    r"(?P<location>(?:서울특별시|서울시|서울|경기도|경기|인천광역시|인천|부산광역시|부산|"
    r"대구광역시|대구|대전광역시|대전|광주광역시|광주|울산광역시|울산|세종특별자치시|세종|"
    r"강원특별자치도|강원도|강원|충청북도|충북|충청남도|충남|전북특별자치도|전라북도|전북|"
    r"전라남도|전남|경상북도|경북|경상남도|경남|제주특별자치도|제주도|제주|"
    r"[가-힣]+시|[가-힣]+군|[가-힣]+구)[^\n\r■]{0,120})"
)
_PARENTHETICAL_LOCATION_RE = re.compile(
    r"\((?P<location>(?:서울특별시|서울시|서울|경기도|경기|인천광역시|인천|부산광역시|부산|"
    r"대구광역시|대구|대전광역시|대전|광주광역시|광주|울산광역시|울산|세종특별자치시|세종|"
    r"강원특별자치도|강원도|강원|충청북도|충북|충청남도|충남|전북특별자치도|전라북도|전북|"
    r"전라남도|전남|경상북도|경북|경상남도|경남|제주특별자치도|제주도|제주)[^)]{0,120})\)"
)
_LOCATION_TEXT_SPLIT_RE = re.compile(r"[,()/]|(?:\s+-\s+)")
_TOP_LEVEL_LOCATIONS = {
    "서울특별시": "서울",
    "서울시": "서울",
    "서울": "서울",
    "경기도": "경기",
    "경기": "경기",
    "인천광역시": "인천",
    "인천": "인천",
    "부산광역시": "부산",
    "부산시": "부산",
    "부산": "부산",
    "대구광역시": "대구",
    "대구": "대구",
    "대전광역시": "대전",
    "대전": "대전",
    "광주광역시": "광주",
    "광주": "광주",
    "울산광역시": "울산",
    "울산": "울산",
    "세종특별자치시": "세종",
    "세종": "세종",
    "강원특별자치도": "강원",
    "강원도": "강원",
    "강원": "강원",
    "충청북도": "충북",
    "충북": "충북",
    "충청남도": "충남",
    "충남": "충남",
    "전북특별자치도": "전북",
    "전라북도": "전북",
    "전북": "전북",
    "전라남도": "전남",
    "전남": "전남",
    "경상북도": "경북",
    "경북": "경북",
    "경상남도": "경남",
    "경남": "경남",
    "제주특별자치도": "제주",
    "제주도": "제주",
    "제주": "제주",
}
_LOCAL_LOCATION_SUFFIXES = ("시", "군", "구", "읍", "면")


def _val(v: Any) -> str:
    if pd.isna(v):
        return ""
    return normalize_text(str(v)).strip()


def _company_name_variants(company: str) -> list[str]:
    """회사명 전체와 괄호 안팎 별칭을 함께 제거하기 위한 후보 목록을 만든다."""
    company = normalize_text(company).strip()
    if not company:
        return []

    variants: set[str] = {company}
    parenthetical_parts = re.findall(r"\(([^)]+)\)", company)
    variants.update(part.strip() for part in parenthetical_parts if part.strip())

    outside_parentheses = re.sub(r"\s*\([^)]*\)", "", company).strip()
    if outside_parentheses:
        variants.add(outside_parentheses)

    return sorted(variants, key=len, reverse=True)


def _iter_company_names(companies: str | Iterable[str]) -> Iterable[str]:
    if isinstance(companies, str):
        yield companies
    else:
        yield from companies


def _company_name_patterns(companies: str | Iterable[str]) -> tuple[re.Pattern[str], ...]:
    variants: set[str] = set()
    for company in _iter_company_names(companies):
        variants.update(_company_name_variants(company))

    return tuple(
        re.compile(re.escape(variant).replace(r"\ ", r"\s+"), flags=re.IGNORECASE)
        for variant in sorted(variants, key=len, reverse=True)
    )


def _remove_company_patterns(text: str, patterns: Iterable[re.Pattern[str]]) -> str:
    """채용공고 텍스트에 포함된 회사명 표기를 제거한다."""
    text = normalize_text(text)
    for pattern in patterns:
        text = pattern.sub("", text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _remove_company_names(text: str, companies: str | Iterable[str]) -> str:
    return _remove_company_patterns(text, _company_name_patterns(companies))


def _sanitize_location(text: str) -> str:
    text = normalize_text(text).strip()
    if not text:
        return ""
    if _REMOTE_LOCATION_RE.search(text):
        return "원격"

    text = _LOCATION_TEXT_SPLIT_RE.split(text, maxsplit=1)[0].strip()
    tokens = [token.strip(" ,.") for token in text.split() if token.strip(" ,.")]
    if not tokens:
        return ""

    result: list[str] = []
    index = 0
    top_level = _TOP_LEVEL_LOCATIONS.get(tokens[0])
    if top_level:
        result.append(top_level)
        index = 1

    for token in tokens[index:]:
        if token.endswith(_LOCAL_LOCATION_SUFFIXES):
            result.append(token)
            if len(result) >= 3:
                break
            continue
        break

    if result:
        return " ".join(result)

    first = tokens[0]
    if first.endswith(_LOCAL_LOCATION_SUFFIXES):
        return first

    return ""


def _sanitize_location_mentions(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        location = _sanitize_location(match.group("location"))
        if not location:
            return match.group(0)
        return f"{match.group('label')}{location}"

    text = _LOCATION_LABEL_RE.sub(replace, text)

    def replace_parenthetical(match: re.Match[str]) -> str:
        location = _sanitize_location(match.group("location"))
        if not location:
            return match.group(0)
        return f"({location})"

    return _PARENTHETICAL_LOCATION_RE.sub(replace_parenthetical, text)


def _remove_leading_title_tags(text: str) -> str:
    return _LEADING_TITLE_TAG_RE.sub("", normalize_text(text)).strip()


def _sanitize_job_title(text: str, company_patterns: Iterable[re.Pattern[str]]) -> str:
    return _remove_leading_title_tags(_remove_company_patterns(text, company_patterns))


def _sanitize_job_text(text: str, company_patterns: Iterable[re.Pattern[str]]) -> str:
    text = sanitize_body_text(text)
    text = _sanitize_location_mentions(text)
    return _remove_company_patterns(text, company_patterns)


def _clean_output_dir(output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for txt_file in path.glob("*.txt"):
        txt_file.unlink()


def _build_job_document(data: dict[str, str], company_patterns: Iterable[re.Pattern[str]]) -> str:
    """CSV·JobPosting 두 입력 경로가 동일 포맷 공유해야 해 분리. 공통 dict를 RAG 문서 텍스트로 변환."""
    title = _sanitize_job_title(data.get("title", ""), company_patterns)
    category = data.get("category", "")
    tech_stacks = data.get("tech_stacks", "")
    location = _sanitize_location(data.get("location", ""))
    salary = data.get("salary", "")
    experience = data.get("experience", "")
    industry = data.get("industry_name", "")
    deadline = data.get("deadline", "")
    responsibilities = data.get("responsibilities", "")
    requirements = data.get("requirements", "")
    preferred = data.get("preferred", "")
    benefits = data.get("benefits", "")

    header_parts: list[str] = []
    if title:
        header_parts.append(f"채용공고: {title}")
    if category:
        header_parts.append(f"분야: {category}")

    lines: list[str] = [f"[{' | '.join(header_parts)}]", ""]

    if title:
        lines.append(f"직무: {title}")
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

    if responsibilities:
        lines += ["", "■ 주요 업무", _sanitize_job_text(responsibilities, company_patterns)]
    if requirements:
        lines += ["", "■ 자격 요건", _sanitize_job_text(requirements, company_patterns)]
    if preferred:
        lines += ["", "■ 우대 사항", _sanitize_job_text(preferred, company_patterns)]
    if benefits:
        lines += ["", "■ 복지 혜택", _sanitize_job_text(benefits, company_patterns)]

    return _remove_company_patterns("\n".join(lines).strip(), company_patterns)


def build_jobs_from_csv(csv_path: str, output_dir: str) -> list[dict[str, str]]:
    """배치 전처리용. 크롤링 결과 CSV를 한 번에 RAG 문서로 변환."""
    _clean_output_dir(output_dir)
    df = pd.read_csv(csv_path, encoding="utf-8")
    company_names = [_val(company) for company in df.get("company", [])]
    company_patterns = _company_name_patterns(company_names)

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

        doc = _build_job_document(data, company_patterns)
        scrubbed_title = _sanitize_job_title(data["title"], company_patterns)
        safe_title = (
            safe_filename_part(scrubbed_title, max_len=40) if scrubbed_title else "직무미상"
        )
        filename = f"채용공고_{safe_title}_{source_id}.txt"
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(doc)

        results.append(
            {
                "source": source,
                "source_id": source_id,
                "title": scrubbed_title,
            }
        )

    return results


def build_jobs_from_postings(jobs: list[JobPosting], output_dir: str) -> list[dict[str, str]]:
    """크롤러 직접 연계용. 실시간 수집된 JobPosting 객체를 즉시 RAG 문서로 변환."""
    _clean_output_dir(output_dir)
    company_names = [normalize_text(job.company).strip() for job in jobs]
    company_patterns = _company_name_patterns(company_names)

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

        doc = _build_job_document(data, company_patterns)
        scrubbed_title = _sanitize_job_title(data["title"], company_patterns)
        safe_title = (
            safe_filename_part(scrubbed_title, max_len=40) if scrubbed_title else "직무미상"
        )
        filename = f"채용공고_{safe_title}_{job.source_id}.txt"
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(doc)

        results.append(
            {
                "source": job.source,
                "source_id": job.source_id,
                "title": scrubbed_title,
            }
        )

    return results
