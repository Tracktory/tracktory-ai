"""
사람인 API XML 응답 파싱 모듈
==============================

사람인 Open API가 반환하는 XML 응답을 파싱하여
JobPosting 객체 리스트로 변환한다.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from tracktory.common.models import JobPosting
from tracktory.common.tech_keywords import classify_job_category

logger = logging.getLogger("crawling.saramin.parser")


def get_total_count(xml_root: ET.Element) -> int:
    """XML 응답에서 전체 검색 결과 수를 추출한다.

    Args:
        xml_root: 사람인 API XML 응답의 루트 엘리먼트.

    Returns:
        전체 결과 수. 파싱에 실패하면 0을 반환한다.
    """
    total_elem = xml_root.find(".//total")
    if total_elem is not None and total_elem.text:
        try:
            return int(total_elem.text.strip())
        except ValueError:
            logger.warning("전체 결과 수 파싱 실패: '%s'", total_elem.text)
    return 0


def parse_job_list(xml_root: ET.Element, search_keyword: str) -> list[JobPosting]:
    """XML 응답에서 채용공고 목록을 파싱하여 JobPosting 리스트로 변환한다.

    각 <job> 엘리먼트에서 공고 ID, 회사명, 제목, 근무지, 급여, 경력,
    상세 URL 등을 추출한다. 직무 카테고리는 제목 기반 분류를 시도하고,
    분류 결과가 '기타'이면 검색 키워드를 카테고리로 사용한다.
    tech_stacks는 이 단계에서는 빈 리스트로 설정되며, 상세 페이지
    방문 후 별도로 채워진다.

    Args:
        xml_root: 사람인 API XML 응답의 루트 엘리먼트.
        search_keyword: 해당 검색에 사용된 키워드. 카테고리 폴백 및
                        메타데이터 용도로 사용한다.

    Returns:
        파싱된 JobPosting 객체 리스트. XML 형식 오류 시 빈 리스트를 반환한다.
    """
    jobs: list[JobPosting] = []

    try:
        job_elements = xml_root.findall(".//job")
    except Exception as exc:
        logger.warning("XML job 엘리먼트 탐색 실패: %s", exc)
        return jobs

    for job_elem in job_elements:
        try:
            job = _parse_single_job(job_elem, search_keyword)
            if job is not None:
                jobs.append(job)
        except Exception as exc:
            logger.warning("개별 공고 파싱 실패 (건너뜀): %s", exc)
            continue

    return jobs


def _parse_single_job(
    job_elem: ET.Element,
    search_keyword: str,
) -> JobPosting | None:
    """단일 <job> 엘리먼트를 JobPosting 객체로 변환한다.

    Args:
        job_elem: 개별 채용공고 XML 엘리먼트.
        search_keyword: 검색 키워드.

    Returns:
        JobPosting 객체. 필수 데이터(ID)가 없으면 None을 반환한다.
    """

    def _text(path: str, default: str = "") -> str:
        """XML 경로에서 텍스트를 안전하게 추출한다."""
        elem = job_elem.find(path)
        if elem is not None and elem.text:
            return elem.text.strip()
        return default

    # source_id: position/job-code 우선, 없으면 id 태그 사용
    source_id = _text("position/job-code") or _text("id")
    if not source_id:
        logger.debug("source_id를 추출할 수 없는 공고 건너뜀")
        return None

    title = _text("position/title")
    company = _text("company/detail/name")
    location = _text("position/location/name")
    salary = _text("salary/name")
    experience = _text("position/experience-level/name")
    detail_url = _text("url")

    # 카테고리 분류: 제목 기반 분류를 시도하고, '기타'이면 검색 키워드 사용
    category = classify_job_category(title)
    if category == "기타":
        category = search_keyword

    return JobPosting(
        source="saramin",
        source_id=source_id,
        company=company,
        title=title,
        category=category,
        tech_stacks=[],
        location=location,
        salary=salary,
        url=detail_url,
        experience=experience,
        extra={"search_keyword": search_keyword},
    )
