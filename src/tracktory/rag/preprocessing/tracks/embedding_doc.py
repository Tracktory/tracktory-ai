"""트랙 임베딩 비교용 문서 생성 (트랙 시너지 meta_vector 입력).

기존 트랙소개 .txt 는 전 트랙이 동일 템플릿 + 공통 마케팅 문구("4차 산업혁명 융합
인재 양성" 류) + 스크랩된 네비게이션 메뉴를 공유한다. 이 공통 장르 성분이 임베딩을
지배해 트랙 간 코사인이 모두 높게 뭉치고(동적 범위 압축), 시너지 MMR 다양성 항
(``w_meta``)이 트랙을 변별하지 못한다.

본 모듈은 트랙소개 .txt 를 입력으로 받아 **변별 신호(역량·진로·양성인력·도메인 키워드)
는 보존하고 공통 보일러플레이트는 제거**한 임베딩 전용 문서를 규칙 기반으로 조립한다.
순수 함수만 두어(파일 I/O 는 호출 스크립트가 담당) mock 없이 테스트 가능하다.

설계 근거(``synergy.yaml`` w_meta = "도메인/서사 거리" 별도 축): 과목 ID 중복은 시너지
``course_overlap`` 항이 이미 본다. 따라서 본 문서는 과목 **ID** 리스트가 아니라 의미적
도메인 텍스트만 농축한다(필수 교과목은 과목 *명* 이라 도메인 신호로 가볍게 포함).
"""

from __future__ import annotations

import re

# 트랙소개 .txt 의 ``■ {label}`` 헤더 라벨 → 통일 섹션 키.
# (builder._SECTION_ORDER 의 출력 라벨과 동일. "관련 홈페이지" 는 URL 노이즈라 미수록.)
_LABEL_TO_KEY: dict[str, str] = {
    "소개": "소개",
    "교육목표": "교육목표",
    "목표 양성 인력": "양성인력",
    "졸업 후 진로": "진로",
    "전공역량": "역량",
    "필수 교과목": "필수교과목",
    "관련 자격증": "자격증",
    "산학협력업체": "산학협력",
}

# 임베딩 문서에 실을 섹션과 출력 순서 — 고변별 섹션을 앞에 둔다.
_DOC_SECTIONS: list[tuple[str, str]] = [
    ("양성인력", "양성 인력"),
    ("진로", "진로"),
    ("역량", "전공역량"),
    ("필수교과목", "필수 교과목"),
    ("자격증", "자격증"),
    ("산학협력", "산학협력"),
    ("소개", "소개"),
    ("교육목표", "교육목표"),
]

# 스크랩된 UI 네비게이션·제네릭 라벨 (전 트랙 공통, 변별 가치 0). 줄 통째 제거.
_NAV_NOISE: frozenset[str] = frozenset(
    {
        "교육과정 로드맵",
        "졸업요건",
        "비교과프로그램",
        "교과프로그램",
        "게시판",
        "FAQ",
        "Q&A",
        "학과홍보영상",
        "바로가기",
        "홈페이지",
        "문의처",
        "주요 교과목",
        "주요교과목",
        "관련자격증",
        "주요 직업 :",
        "주요 직업:",
        "융합 직업 :",
        "융합 직업:",
        "취업처:",
        "취업처 :",
        "교과목 개요 및 특징",
        "핵심역량",
        "함양",
    }
)

# 공통 마케팅 문구 — 같은 문장에 도메인 토큰이 섞여 있어 줄을 버리지 않고 *부분* 제거한다.
_BOILERPLATE_PHRASES: tuple[str, ...] = (
    "4차 산업혁명 시대를 선도할",
    "4차 산업혁명을 이끄는",
    "4차 산업혁명 시대를",
    "4차 산업혁명",
    "여러분의 미래를 시작해보세요",
    "여러분의 미래",
    "미래 산업 환경에 즉시 적응 가능한",
    "즉시 적응 가능한",
    "전문가로 성장할 수 있습니다",
    "전문가로 성장",
    "창의적인 융합인재를 양성",
    "창의적인 융합인재",
    "융합형 실무 인재 양성을 목표로 합니다",
    "융합형 실무 인재",
    "융합형 인재",
    "융합 인재",
    "융합인재",
    "인재 양성을 목표로 합니다",
    "인재 양성을 목표로",
    "양성을 목표로 합니다",
    "양성하는 것을 목표로 합니다",
    "미래를 선도하는",
    "미래를 선도",
    "글로벌 리더",
    "핵심 인재",
)

_WS_RE = re.compile(r"[ \t]+")
_HEADER_RE = re.compile(r"^\[트랙:\s*(?P<name>.*?)\s*[|\]]")


def _clean_line(line: str) -> str | None:
    """한 줄을 정제. 네비게이션 노이즈면 ``None``, 아니면 보일러플레이트 제거 후 반환.

    보일러플레이트 부분 제거로 줄이 사실상 비면(2자 미만) ``None`` 으로 떨군다.
    """
    stripped = line.strip()
    if not stripped or stripped in _NAV_NOISE:
        return None
    for phrase in _BOILERPLATE_PHRASES:
        if phrase in stripped:
            stripped = stripped.replace(phrase, " ")
    stripped = _WS_RE.sub(" ", stripped).strip(" ,.·")
    if len(stripped) < 2:
        return None
    return stripped


def _clean_section(text: str) -> str:
    """섹션 본문을 줄 단위로 정제하고 빈 줄을 제거해 재조립."""
    cleaned = [c for line in text.splitlines() if (c := _clean_line(line)) is not None]
    return "\n".join(cleaned)


def parse_track_txt(text: str) -> tuple[str, dict[str, str]]:
    """트랙소개 .txt 한 개를 ``(track_name, 섹션 dict)`` 로 파싱한다.

    첫 줄 ``[트랙: 이름 | 대학: .. | 학부: ..]`` 헤더에서 트랙명을 그대로(가운뎃점 보존)
    추출한다. ``tracks.yaml`` 의 ``track_id`` 도 가운뎃점을 보존하므로 정규화 없이 원문
    이름이 후속 임베딩 단계의 join 키가 된다.

    Args:
        text: 트랙소개 .txt 전체 내용.

    Returns:
        ``(헤더 트랙명, {통일 섹션 키: 본문})``. 헤더 파싱 실패 시 트랙명은 ``""``.
    """
    lines = text.splitlines()
    track_name = ""
    body_start = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        match = _HEADER_RE.match(line.strip())
        if match:
            track_name = match.group("name").strip()
            body_start = i + 1
        break

    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in lines[body_start:]:
        stripped = line.strip()
        if stripped.startswith("■"):
            label = stripped.lstrip("■").strip()
            current_key = _LABEL_TO_KEY.get(label)  # 미수록 라벨(홈페이지 등)은 None → 스킵
            if current_key is not None:
                sections.setdefault(current_key, [])
            continue
        if current_key is not None:
            sections[current_key].append(line)

    return track_name, {k: "\n".join(v).strip() for k, v in sections.items()}


def build_embedding_doc(track_name: str, sections: dict[str, str]) -> str:
    """변별 신호를 농축한 임베딩 전용 문서를 조립한다.

    트랙명을 머리에 두고 고변별 섹션부터 정제 본문을 잇는다. 정제 후 빈 섹션은 생략한다.
    """
    parts = [track_name] if track_name else []
    for key, label in _DOC_SECTIONS:
        cleaned = _clean_section(sections.get(key, ""))
        if cleaned:
            parts.append(f"{label}: {cleaned}")
    return "\n".join(parts)


def build_doc_from_txt(text: str) -> tuple[str, str]:
    """트랙소개 .txt 한 개 → ``(track_name, 임베딩 문서)``. parse + build 조합."""
    track_name, sections = parse_track_txt(text)
    return track_name, build_embedding_doc(track_name, sections)
