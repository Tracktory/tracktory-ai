"""Step 1: Raw 텍스트에서 UI 노이즈 제거 (한성대 트랙 페이지)"""

import re

from tracktory.rag.preprocessing.normalize import normalize_text, sanitize_body_text

_NAV_ITEMS: frozenset[str] = frozenset(
    {
        "로그인",
        "종합정보시스템",
        "한성대 바로가기",
        "전체메뉴",
        "홈으로",
        "메인화면",
        "인쇄",
        "공유",
        "즐겨찾기",
        "트랙소개",
        "전공소개",
        "학과소개",
        "교수소개",
        "교육과정소개",
        "교과목로드맵",
        "비교과프로그램",
        "트랙졸업요건",
        "게시판",
        "트랙홍보영상",
        "전공홍보영상",
        "전공 홍보영상",
        "대학소개",
        "학사안내",
        "정보마당",
        "입학안내",
        "입학 안내",
        "사이트맵",
        "Q&A",
        "교수 소개",
        "비교과 프로그램",
        "트랙 홈페이지",
        "홈페이지 바로가기",
        "홈페이지바로가기",
        "홈페이지 바로 가기",
        "글로벌인재대학 상세학과 닫기",
        "미래플러스대학 상세학과 닫기",
        "상상력교양대학 상세학과 닫기",
        "산학 및 졸업생 멘토풀 현황",
        "산학 및 졸업생 멘토 풀 현황",
        "산학 멘토 풀 현황",
        "위치",
        "전화",
        "팩스",
    }
)

_FOOTER_EXACT: frozenset[str] = frozenset(
    {
        "개인정보처리방침",
        "이메일무단수집거부",
        "이메일주소무단수집거부",
        "교내주요사이트",
        "관련기관",
        "학내 주요 서비스",
        "대외 협력·연계",
        "발전기금",
        "TOP",
    }
)

_FOOTER_CONTAINS: tuple[str, ...] = (
    "[02876]",
    "(02876)",
    "COPYRIGHT",
    "copyright",
    "TEL :",
    "FAX :",
    "ARS안내",
    "1544-4113",
    "페이스북",
    "인스타그램",
    "유튜브",
    "카카오채널",
    "HANSUNG UNIVERSITY",
)

_BULLET_RE: re.Pattern[str] = re.compile(r"^[▶·\-⦁●■○‣＊。．•]+\s*")  # noqa: RUF001
_LOCATION_RE: re.Pattern[str] = re.compile(r"^[가-힣\s]+(?:\d+호|관)$")
_PHONE_RE: re.Pattern[str] = re.compile(r"^[\d\s\-]+$")
_CONTENT_NOISE: frozenset[str] = frozenset({"위치", "전화", "팩스"})

# fallback: 즐겨찾기 이후 콘텐츠 시작을 판단하는 알려진 섹션 헤더
_REAL_STARTERS: frozenset[str] = frozenset(
    {
        "트랙 소개",
        "전공 소개",
        "학과소개",
        "학과 교육목표",
        "학과교육목표",
    }
)


def _is_footer(line: str) -> bool:
    if line in _FOOTER_EXACT:
        return True
    return any(p in line for p in _FOOTER_CONTAINS)


def _find_content_start(lines: list[str], track_name: str) -> int:
    """트랙 페이지마다 본문 시작 위치가 달라 탐지 로직 필요. 본문 시작 인덱스 반환."""
    edu_idx: int | None = None
    for i, line in enumerate(lines):
        if line == "교육과정소개":
            edu_idx = i

    if edu_idx is not None:
        for i in range(edu_idx + 1, len(lines)):
            line = lines[i]
            if line in _NAV_ITEMS or line == track_name:
                continue
            if re.match(r"^[\d\s\-]+$", line):
                continue
            if len(line) < 20 and not any(c in line for c in "소개목표인력진로역량교과자격"):
                continue
            return i

    fav_seen = False
    for i, line in enumerate(lines):
        if line == "즐겨찾기":
            fav_seen = True
            continue
        if fav_seen:
            if line in _REAL_STARTERS:
                return i
            if len(line) > 40 and line not in _NAV_ITEMS:
                return i

    return 0


def clean(raw_text: str, track_name: str) -> list[str]:
    """nav/footer 텍스트가 RAG 검색 노이즈로 작동. 본문 범위를 찾아 불필요한 줄 제거 후 반환."""
    lines = [line.strip() for line in normalize_text(raw_text).split("\n") if line.strip()]

    end = len(lines)
    for i, line in enumerate(lines):
        if _is_footer(line):
            end = i
            break
    lines = lines[:end]

    start = _find_content_start(lines, track_name)
    lines = lines[start:]

    result = []
    for line in lines:
        line = _BULLET_RE.sub("", line).strip()
        if not line or line == track_name:
            continue
        if line in _CONTENT_NOISE:
            continue
        if _LOCATION_RE.match(line) or _PHONE_RE.match(line):
            continue
        result.append(sanitize_body_text(line))

    return result
