"""
원티드 채용공고 DOM 셀렉터 상수 및 추출 헬퍼 함수
=================================================

원티드 상세 페이지에서 구조화된 데이터를 추출하기 위한
CSS 셀렉터 상수와 비동기 추출 함수를 제공합니다.

기술스택 태그 추출은 3단계 전략을 사용합니다:
    1. CSS 셀렉터 직접 매칭 (알려진 클래스명 패턴)
    2. '기술스택' 헤더 텍스트 기반 인접 태그 탐색
    3. __NEXT_DATA__ JSON 파싱 (재귀 탐색)

NOTE: 2026-03 기준 원티드에서 __NEXT_DATA__가 더 이상 존재하지 않습니다.
      섹션 추출은 DOM 기반(h2/h3/h4 sibling walking)으로 수행합니다.
      "상세 정보 더 보기" 버튼을 먼저 클릭하여 콘텐츠를 확장합니다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from playwright.async_api import Page

logger = logging.getLogger("crawling.wanted.selectors")

# ---------------------------------------------------------------------------
# CSS 셀렉터 상수
# ---------------------------------------------------------------------------

SKILL_TAG_SELECTORS: list[str] = [
    "[class*='SkillTag']",
    "[class*='skill_tag']",
    "[class*='skill-tag']",
    "[class*='JobSkill']",
    "[class*='job_skill']",
    "[class*='SkillItem']",
    "[class*='skillItem']",
    "[class*='skill']",
    "[data-attribute-id='skill']",
    "[data-attribute-id*='skill']",
]

SECTION_HEADERS: dict[str, str] = {
    "responsibilities": "주요업무",
    "requirements": "자격요건",
    "preferred": "우대사항",
    "benefits": "혜택 및 복지",
}


# ---------------------------------------------------------------------------
# 내부 헬퍼 함수
# ---------------------------------------------------------------------------


async def _safe_text(page: Page, selectors: list[str]) -> str:
    """여러 CSS 셀렉터를 순서대로 시도하여 첫 번째 매칭 요소의 텍스트를 반환한다.

    모든 셀렉터가 실패하면 빈 문자열을 반환합니다.

    Args:
        page: Playwright Page 인스턴스.
        selectors: 시도할 CSS 셀렉터 목록 (우선순위 순).

    Returns:
        매칭된 요소의 텍스트 또는 빈 문자열.
    """
    for selector in selectors:
        try:
            el = await page.query_selector(selector)
            if el:
                text = await el.inner_text()
                if text and text.strip():
                    return text.strip()  # type: ignore[no-any-return]
        except Exception:
            continue
    return ""


def _deduplicate(items: list[str]) -> list[str]:
    """순서를 유지하면서 중복을 제거한다.

    Args:
        items: 원본 문자열 리스트.

    Returns:
        중복이 제거된 문자열 리스트.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


async def _get_next_data(page: Page) -> dict[str, Any] | None:
    """페이지에서 __NEXT_DATA__ JSON을 추출하여 파싱한다.

    Args:
        page: Playwright Page 인스턴스.

    Returns:
        파싱된 JSON 딕셔너리 또는 None.
    """
    try:
        raw = await page.evaluate("""
            () => {
                const el = document.getElementById('__NEXT_DATA__');
                return el ? el.textContent : null;
            }
        """)
        if raw:
            return json.loads(raw)  # type: ignore[no-any-return]
    except Exception:
        pass
    return None


def _find_detail_html(data: Any, depth: int = 0) -> str | None:
    """__NEXT_DATA__ JSON에서 채용공고 상세 HTML 콘텐츠를 재귀적으로 찾는다.

    'detail' 또는 'description' 키에서 HTML 형식의 긴 문자열을 탐색합니다.

    Args:
        data: JSON 데이터 (dict, list, 또는 기본 타입).
        depth: 현재 재귀 깊이.

    Returns:
        발견된 HTML 문자열 또는 None.
    """
    if depth > 15:
        return None

    detail_keys = {
        "detail",
        "description",
        "jobDetail",
        "job_detail",
        "content",
        "body",
        "position_detail",
    }

    if isinstance(data, dict):
        # 직접 키 매칭
        for key, value in data.items():
            if (
                key.lower() in detail_keys
                and isinstance(value, str)
                and len(value) > 100
                and ("<" in value or "주요업무" in value or "자격요건" in value)
            ):
                return value
        # 재귀 탐색
        for value in data.values():
            result = _find_detail_html(value, depth + 1)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _find_detail_html(item, depth + 1)
            if result:
                return result

    return None


def _parse_sections_from_html(html: str) -> dict[str, str]:
    """HTML 문자열에서 섹션별 텍스트를 추출한다.

    <h2>주요업무</h2><p>...</p> 등의 구조에서 각 섹션을 분리합니다.

    Args:
        html: 채용공고 상세 HTML 문자열.

    Returns:
        섹션별 텍스트 딕셔너리.
    """
    import re as _re

    result: dict[str, str] = {
        "responsibilities": "",
        "requirements": "",
        "preferred": "",
        "benefits": "",
    }

    section_patterns = [
        ("responsibilities", ["주요업무", "담당업무", "주요 업무", "담당 업무"]),
        ("requirements", ["자격요건", "필수조건", "자격 요건", "필수 조건"]),
        ("preferred", ["우대사항", "우대 사항", "우대조건"]),
        ("benefits", ["혜택 및 복지", "복리후생", "혜택", "Benefits", "Perks"]),
    ]

    # HTML 태그 제거 함수
    def strip_tags(s: str) -> str:
        return _re.sub(r"<[^>]+>", "\n", s)

    # 각 헤더의 위치를 찾아서 다음 헤더까지의 콘텐츠를 추출
    all_keywords = []
    for key, keywords in section_patterns:
        for kw in keywords:
            all_keywords.append((kw, key))

    # 패턴으로 모든 헤더 위치 찾기 (태그 내부 텍스트 포함)
    header_positions: list[tuple[int, str, str]] = []  # (position, key, keyword)
    for kw, key in all_keywords:
        # <h2>주요업무</h2> 또는 단순 텍스트 매칭
        for m in _re.finditer(_re.escape(kw), html):
            header_positions.append((m.start(), key, kw))

    # 위치 순으로 정렬
    header_positions.sort(key=lambda x: x[0])

    # 중복 키 제거 (첫 번째 매칭만 사용)
    seen_keys: set[str] = set()
    unique_positions: list[tuple[int, str, str]] = []
    for pos, key, kw in header_positions:
        if key not in seen_keys:
            seen_keys.add(key)
            unique_positions.append((pos, key, kw))

    # 각 섹션의 콘텐츠 추출
    for i, (pos, key, kw) in enumerate(unique_positions):
        # 키워드 이후부터 다음 섹션 시작 전까지
        start = pos + len(kw)
        end = unique_positions[i + 1][0] if i + 1 < len(unique_positions) else len(html)

        section_html = html[start:end]
        text = strip_tags(section_html).strip()
        # 연속 줄바꿈 정리
        text = _re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if text:
            result[key] = text

    return result


def _find_experience_in_json(data: Any, depth: int = 0) -> str | None:
    """__NEXT_DATA__ JSON에서 경력 요건 정보를 재귀적으로 찾는다.

    Args:
        data: JSON 데이터.
        depth: 현재 재귀 깊이.

    Returns:
        경력 요건 문자열 또는 None.
    """
    if depth > 15:
        return None

    experience_keys = {
        "experience",
        "experience_level",
        "career",
        "experienceLevel",
        "experience_type",
        "career_type",
        "position_experience",
    }

    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in experience_keys:
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, dict):
                    # e.g., {"name": "신입", "id": 1}
                    name = value.get("name") or value.get("title") or value.get("label", "")
                    if name and isinstance(name, str):
                        return name.strip()  # type: ignore[no-any-return]
                elif isinstance(value, int):
                    # e.g., experience_level: 1 -> "신입"
                    level_map = {0: "경력무관", 1: "신입", 2: "경력"}
                    return level_map.get(value, str(value))
        for value in data.values():
            result = _find_experience_in_json(value, depth + 1)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _find_experience_in_json(item, depth + 1)
            if result:
                return result

    return None


def _extract_from_next_data(data: Any, depth: int = 0) -> list[str]:
    """__NEXT_DATA__ JSON에서 재귀적으로 스킬 태그를 탐색한다.

    원티드 Next.js 데이터 구조에서 'skill', 'tag', 'tech' 등의
    키를 찾아 태그 문자열을 수집합니다.

    Args:
        data: JSON 파싱 결과 (dict, list, 또는 기본 타입).
        depth: 현재 재귀 깊이 (무한 재귀 방지용).

    Returns:
        발견된 스킬 태그 문자열 리스트.
    """
    if depth > 10:
        return []

    tags: list[str] = []
    skill_keys = {
        "skill",
        "skills",
        "tag",
        "tags",
        "tech",
        "techs",
        "skill_tags",
        "skillTags",
    }

    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in skill_keys:
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            tags.append(item.strip())
                        elif isinstance(item, dict):
                            name = item.get("name") or item.get("title") or item.get("keyword", "")
                            if name and isinstance(name, str):
                                tags.append(name.strip())
            else:
                tags.extend(_extract_from_next_data(value, depth + 1))
    elif isinstance(data, list):
        for item in data:
            tags.extend(_extract_from_next_data(item, depth + 1))

    return tags


# ---------------------------------------------------------------------------
# 공개 추출 함수
# ---------------------------------------------------------------------------


async def extract_skill_tags(page: Page) -> list[str]:
    """원티드 채용공고 페이지에서 기술스택 태그를 추출한다.

    3단계 추출 전략을 순서대로 시도합니다:
        1. CSS 셀렉터 직접 매칭: 알려진 스킬 태그 클래스명 패턴으로 요소를 찾음
        2. 헤더 텍스트 기반 탐색: '기술스택' 헤더를 찾아 인접/형제 태그를 수집
        3. __NEXT_DATA__ JSON 파싱: Next.js 서버 데이터에서 skill/tag 키를 재귀 탐색

    정규화는 수행하지 않으며, 원본 태그 문자열을 반환합니다.
    정규화는 호출자가 common.tech_keywords.normalize_tech_tags()를 사용합니다.

    Args:
        page: 채용공고 상세 페이지가 로드된 Playwright Page 인스턴스.

    Returns:
        추출된 기술스택 태그 문자열 리스트 (중복 제거됨).
    """
    tags: list[str] = []

    # 1단계: CSS 셀렉터 직접 매칭
    for selector in SKILL_TAG_SELECTORS:
        try:
            elements = await page.query_selector_all(selector)
            if elements:
                for el in elements:
                    text = await el.inner_text()
                    text = text.strip()
                    if text and len(text) < 60:
                        tags.append(text)
                if tags:
                    return _deduplicate(tags)
        except Exception:
            continue

    # 2단계: '기술스택' 헤더 텍스트 기반 인접 태그 수집
    try:
        tags = await page.evaluate("""
            () => {
                const headers = document.querySelectorAll(
                    'h3, h4, dt, [class*="title"], [class*="Title"], strong'
                );
                const skillKeywords = [
                    '기술스택', '기술 스택', 'skill', 'Skill',
                    'tech stack', 'Tech Stack'
                ];

                for (const header of headers) {
                    const headerText = header.textContent.trim().toLowerCase();
                    const isSkillHeader = skillKeywords.some(
                        kw => headerText.includes(kw.toLowerCase())
                    );

                    if (isSkillHeader) {
                        const parent = header.closest(
                            'section, article, div[class*="content"], li'
                        ) || header.parentElement;
                        if (!parent) continue;

                        const tagEls = parent.querySelectorAll('span, button, a');
                        const tags = [];
                        for (const el of tagEls) {
                            const text = el.textContent.trim();
                            if (
                                text
                                && text.length < 50
                                && !skillKeywords.some(
                                    kw => text.toLowerCase().includes(kw.toLowerCase())
                                )
                            ) {
                                tags.push(text);
                            }
                        }
                        if (tags.length > 0) return tags;
                    }
                }
                return [];
            }
        """)
        if tags:
            return _deduplicate(tags)
    except Exception:
        pass

    # 3단계: __NEXT_DATA__ JSON 파싱
    try:
        next_data_raw = await page.evaluate("""
            () => {
                const el = document.getElementById('__NEXT_DATA__');
                return el ? el.textContent : null;
            }
        """)
        if next_data_raw:
            data = json.loads(next_data_raw)
            tags = _extract_from_next_data(data)
            if tags:
                return _deduplicate(tags)
    except Exception:
        pass

    return tags


async def extract_sections(page: Page) -> dict[str, str]:
    """채용공고의 주요업무/자격요건/우대사항/혜택 및 복지 섹션 텍스트를 추출한다.

    "상세 정보 더 보기" 버튼을 클릭하여 콘텐츠를 확장한 뒤,
    h2/h3/h4 태그를 기준으로 sibling walking 방식으로 섹션을 분리합니다.

    Args:
        page: 채용공고 상세 페이지가 로드된 Playwright Page 인스턴스.

    Returns:
        섹션 키('responsibilities', 'requirements', 'preferred', 'benefits')를
        값으로 매핑한 딕셔너리. 추출 실패 시 빈 딕셔너리.
    """
    empty_result: dict[str, str] = {
        "responsibilities": "",
        "requirements": "",
        "preferred": "",
        "benefits": "",
    }

    # "상세 정보 더 보기" 버튼 클릭하여 콘텐츠 확장
    try:
        await page.evaluate("""() => {
            const buttons = document.querySelectorAll('button, a, span');
            for (const btn of buttons) {
                if (btn.textContent.trim().includes('더 보기')) {
                    btn.click();
                    return true;
                }
            }
            return false;
        }""")
        await page.wait_for_timeout(1000)
    except Exception:
        pass

    # DOM 기반 섹션 추출: h2/h3/h4 헤더 + sibling walking
    try:
        sections_data = await page.evaluate("""
            () => {
                const result = {
                    responsibilities: '',
                    requirements: '',
                    preferred: '',
                    benefits: '',
                };

                const sectionMap = [
                    {
                        key: 'responsibilities',
                        keywords: ['주요업무', '담당업무', '주요 업무', '담당 업무']
                    },
                    {
                        key: 'requirements',
                        keywords: ['자격요건', '필수조건', '자격 요건', '필수 조건']
                    },
                    {
                        key: 'preferred',
                        keywords: ['우대사항', '우대 사항', '우대조건']
                    },
                    {
                        key: 'benefits',
                        keywords: ['혜택 및 복지', '복리후생', '혜택', 'Benefits']
                    },
                ];

                const headers = document.querySelectorAll('h2, h3, h4');

                for (const header of headers) {
                    const headerText = header.textContent.trim();

                    for (const { key, keywords } of sectionMap) {
                        if (result[key]) continue;  // already found

                        const matched = keywords.some(kw => headerText.includes(kw));
                        if (!matched) continue;

                        // Walk next siblings until next h2/h3/h4
                        let bodyText = '';
                        let sibling = header.nextElementSibling;
                        while (sibling && !['H2', 'H3', 'H4'].includes(sibling.tagName)) {
                            bodyText += sibling.textContent.trim() + '\\n';
                            sibling = sibling.nextElementSibling;
                        }

                        if (bodyText.trim()) {
                            result[key] = bodyText.trim();
                        }
                        break;
                    }
                }

                return result;
            }
        """)
        return sections_data or empty_result
    except Exception:
        return empty_result


def _find_company_info_in_json(data: Any, depth: int = 0) -> dict[str, str]:
    """__NEXT_DATA__ JSON에서 회사 메타정보를 재귀적으로 찾는다.

    company_id 또는 company_name 키를 가진 딕셔너리를 탐색합니다.

    Args:
        data: JSON 데이터 (dict, list, 또는 기본 타입).
        depth: 현재 재귀 깊이.

    Returns:
        company_id, industry_name 키를 포함한 딕셔너리. 찾지 못하면 빈 딕셔너리.
    """
    if depth > 15:
        return {}

    if isinstance(data, dict):
        if "company_id" in data or "company_name" in data:
            result: dict[str, str] = {}
            company_id = data.get("company_id")
            if company_id is not None:
                result["company_id"] = str(company_id)
            industry_name = data.get("industry_name")
            if industry_name is not None:
                result["industry_name"] = str(industry_name)
            return result
        for value in data.values():
            found = _find_company_info_in_json(value, depth + 1)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_company_info_in_json(item, depth + 1)
            if found:
                return found

    return {}


async def extract_company_info(page: Page) -> dict[str, str]:
    """채용공고 페이지에서 회사 메타정보를 추출한다.

    URL에서 job_id를 추출하고, 페이지 본문에서 업종 정보를 탐색합니다.

    Args:
        page: 채용공고 상세 페이지가 로드된 Playwright Page 인스턴스.

    Returns:
        회사 메타정보 딕셔너리. 키: job_id, industry_name.
    """
    try:
        result = await page.evaluate(r"""() => {
            const info = {};

            // Extract job_id from URL
            const urlMatch = window.location.pathname.match(/\/wd\/(\d+)/);
            info.job_id = urlMatch ? urlMatch[1] : '';

            // Look for industry info near company name at bottom of page
            // Pattern: "회사명\nIT, 컨텐츠" appears after "팔로우" or near footer
            const body = document.body.innerText;
            const lines = body.split('\n').map(l => l.trim()).filter(l => l);
            const industryPatterns = [
                'IT', '금융', '제조', '의료', '건설', '유통', '미디어',
                '전문, 과학기술', '교육 서비스', '도소매', '운수', '농업'
            ];
            // Search from bottom up to avoid nav bar false positives
            for (let i = lines.length - 1; i >= Math.max(0, lines.length - 50); i--) {
                const line = lines[i];
                if (line.length < 30 && industryPatterns.some(p => line.startsWith(p))) {
                    info.industry_name = line;
                    break;
                }
            }

            return info;
        }""")
        return result or {}
    except Exception:
        return {}


async def extract_company_name(page: Page) -> str:
    """채용공고 페이지에서 회사명을 추출한다.

    여러 알려진 셀렉터를 우선순위 순으로 시도합니다.

    Args:
        page: 채용공고 상세 페이지가 로드된 Playwright Page 인스턴스.

    Returns:
        회사명 문자열. 추출 실패 시 빈 문자열.
    """
    return await _safe_text(
        page,
        [
            "a[class*='CompanyName']",
            "a[class*='company_name']",
            "[class*='company-name']",
            "header a",
        ],
    )


async def extract_job_title(page: Page) -> str:
    """채용공고 페이지에서 공고 제목을 추출한다.

    여러 알려진 셀렉터를 우선순위 순으로 시도합니다.

    Args:
        page: 채용공고 상세 페이지가 로드된 Playwright Page 인스턴스.

    Returns:
        공고 제목 문자열. 추출 실패 시 빈 문자열.
    """
    return await _safe_text(
        page,
        [
            "h1[class*='JobHeader']",
            "h2[class*='JobHeader']",
            "[class*='position-name']",
            "[class*='job-title']",
            "h1",
        ],
    )


async def extract_location(page: Page) -> str:
    """채용공고 페이지에서 근무지 정보를 추출한다.

    Args:
        page: 채용공고 상세 페이지가 로드된 Playwright Page 인스턴스.

    Returns:
        근무지 문자열. 추출 실패 시 빈 문자열.
    """
    return await _safe_text(
        page,
        [
            "[class*='location']",
            "[class*='Location']",
            "span[class*='address']",
        ],
    )


async def extract_deadline(page: Page) -> str:
    """채용공고 페이지에서 마감일 정보를 추출한다.

    Args:
        page: 채용공고 상세 페이지가 로드된 Playwright Page 인스턴스.

    Returns:
        마감일 문자열. 추출 실패 시 빈 문자열.
    """
    return await _safe_text(
        page,
        [
            "[class*='deadline']",
            "[class*='Deadline']",
            "[class*='expire']",
        ],
    )


async def extract_experience(page: Page) -> str:
    """채용공고 페이지에서 경력 요건을 추출한다.

    페이지 상단 subtitle 영역에서 "경력 X-Y년", "신입", "경력무관" 등의
    패턴을 탐색합니다. (예: "소울엑스∙서울 마포구∙경력 3-5년")

    Args:
        page: 채용공고 상세 페이지가 로드된 Playwright Page 인스턴스.

    Returns:
        경력 요건 문자열 (예: "신입", "경력 3-5년", "경력무관").
        추출 실패 시 빈 문자열.
    """
    try:
        result = await page.evaluate(r"""() => {
            const body = document.body.innerText;
            const lines = body.split('\n');
            // Look for pattern like "경력 3-5년", "신입", "경력무관" in first 15 lines
            for (let i = 0; i < Math.min(lines.length, 15); i++) {
                const line = lines[i].trim();
                // Match "경력 X-Y년" or "경력 X년 이상"
                const expMatch = line.match(/경력\s*([\d~\-]+년?(?:\s*이상)?)/);
                if (expMatch) return '경력 ' + expMatch[1];
                // Match "신입" or "경력무관"
                if (line.includes('신입') && line.includes('경력')) return '신입/경력';
                if (line === '신입' || line.includes('신입')) return '신입';
                if (line.includes('경력무관')) return '경력무관';
            }
            // Also check the subtitle text (format: "회사∙위치∙경력 3-5년")
            const subtitle = document.body.innerText.substring(0, 500);
            const match = subtitle.match(/경력\s*([\d~\-]+년?(?:\s*이상)?)/);
            if (match) return '경력 ' + match[1];
            if (subtitle.includes('신입')) return '신입';
            return '';
        }""")
        return result or ""
    except Exception:
        return ""
