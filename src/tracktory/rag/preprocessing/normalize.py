"""RAG 전처리 공통 텍스트 정규화

토크나이저/임베딩 모델에서 NaN을 유발하거나 파싱을 오염시키는
유니코드 문자를 첫 파싱 입력 시점에 제거한다.
"""

from __future__ import annotations

import re

_STRIP_CODEPOINTS: list[int] = [
    # C0 제어문자 (\t=0x09, \n=0x0A, \r=0x0D 제외)
    *range(0x0001, 0x0009),  # U+0001~U+0008
    0x000B,  # U+000B Vertical Tab
    0x000C,  # U+000C Form Feed
    *range(0x000E, 0x0020),  # U+000E~U+001F
    # DEL
    0x007F,  # U+007F
    # C1 제어문자
    *range(0x0080, 0x00A0),  # U+0080~U+009F
    # 소프트 하이픈
    0x00AD,  # U+00AD Soft Hyphen
    # 영폭(zero-width) 문자
    0x200B,  # U+200B Zero Width Space
    0x200C,  # U+200C Zero Width Non-Joiner
    0x200D,  # U+200D Zero Width Joiner
    # 양방향 오버라이드
    *range(0x202A, 0x202F),  # U+202A~U+202E
    # 단어 결합자
    0x2060,  # U+2060 Word Joiner
    # 양방향 격리자
    *range(0x2066, 0x206A),  # U+2066~U+2069
    # 한글 필러
    0x3164,  # U+3164 Hangul Filler
    0xFFA0,  # U+FFA0 Halfwidth Hangul Filler
    # BOM
    0xFEFF,  # U+FEFF
    # 인코딩 오류 대체 문자
    0xFFFD,  # U+FFFD Replacement Character
]

_TRANS = str.maketrans(
    "",
    "",
    "".join(chr(cp) for cp in _STRIP_CODEPOINTS),
)

# BGE-M3 등 임베딩 모델에서 NaN 유발이 확인된 문자 → 대체
_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("ㆍ", "·"),  # ㆍ(U+318D) → ·(U+00B7)
)


_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE_INLINE = re.compile(r"\b\d{2,4}[-\s]\d{3,4}[-\s]\d{4}\b")

# Windows 파일명 금지문자 + 구분 기호
_ILLEGAL_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename_part(text: str, max_len: int = 50) -> str:
    """식별자 텍스트를 파일명 한 조각으로 안전하게 정규화한다.

    파일명은 RAGFlow의 ``docnm_kwd`` 가 되어 제목 임베딩(``filename_embd_weight``)과
    키워드 인덱스에 반영된다. 따라서 식별자 의미는 보존하되, Windows 금지문자를
    치환하고 경로 길이 폭주를 막기 위해 길이를 제한한다.
    """
    text = _ILLEGAL_FILENAME_RE.sub("_", text)
    text = text.replace("ㆍ", "_").replace("·", "_")
    text = re.sub(r"\s+", " ", text).strip().strip("._ ")
    if len(text) > max_len:
        text = text[:max_len].strip("._ ")
    return text or "미상"


def normalize_text(text: str) -> str:
    """문제 문자를 제거·치환하여 정제된 텍스트를 반환한다."""
    text = text.translate(_TRANS)
    for src, dst in _REPLACEMENTS:
        text = text.replace(src, dst)
    return text


def sanitize_body_text(text: str) -> str:
    """본문 텍스트에서 URL, 이메일, 전화번호를 제거한다.

    청크 구분자(■)와 함께 쓸 때 URL·이메일·전화번호가 토큰 경계를
    오염시키는 문제를 방지하기 위해 메타데이터가 아닌 본문 필드에만 적용한다.
    """
    text = _URL_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)
    text = _PHONE_RE_INLINE.sub("", text)
    return text
