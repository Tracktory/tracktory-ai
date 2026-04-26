"""Step 2: 정제된 줄 리스트를 섹션별로 파싱 (한성대 트랙 페이지)"""

_SECTION_HEADERS: dict[str, frozenset[str]] = {
    "소개": frozenset(
        {
            "트랙 소개",
            "트랙소개",
            "전공 소개",
            "전공소개",
            "학과소개",
            "학과 소개",
            "학부소개",
            "학부 소개",
        }
    ),
    "교육목표": frozenset(
        {
            "교육목표",
            "교육 목표",
            "학과 교육목표",
            "학과교육목표",
        }
    ),
    "양성인력": frozenset(
        {
            "목표 양성 인력",
            "목표양성인력",
            "양성 인력",
            "양성인력",
        }
    ),
    "진로": frozenset(
        {
            "졸업 후 진로",
            "졸업후진로",
            "진로 및 전망",
            "주요 직업",
        }
    ),
    "역량": frozenset(
        {
            "트랙전공역량",
            "전공역량",
            "전공 역량",
            "학과전공역량",
            "학과 육성역량",
            "학과육성역량",
            "학과 전공역량",
        }
    ),
    "연계트랙": frozenset(
        {
            "연계트랙",
            "연계 트랙",
        }
    ),
    "필수교과목": frozenset(
        {
            "필수 교과목",
            "필수교과목",
        }
    ),
    "자격증_직무": frozenset(
        {
            "직무역량 관련 자격",
            "직무역량관련자격",
        }
    ),
    "자격증_기본": frozenset(
        {
            "기본역량 관련 자격",
            "기본역량관련자격",
            "심화역량 관련 자격",
            "심화역량관련자격",
        }
    ),
    "자격증": frozenset(
        {
            "자격증",
            "관련 자격증",
        }
    ),
    "산학협력": frozenset(
        {
            "산학협력업체",
        }
    ),
    "관련홈페이지": frozenset(
        {
            "관련 홈페이지",
            "관련홈페이지",
        }
    ),
}

_HEADER_TO_SECTION: dict[str, str] = {}
for _sec, _headers in _SECTION_HEADERS.items():
    for _h in _headers:
        _HEADER_TO_SECTION[_h] = _sec


def _classify(line: str) -> str | None:
    """줄이 섹션 헤더면 섹션 key를 반환하고, 아니면 None을 반환한다."""
    if line in _HEADER_TO_SECTION:
        return _HEADER_TO_SECTION[line]
    if (
        line.endswith(" 소개")
        and len(line) < 20
        and any(w in line for w in ("학부", "학과", "전공", "트랙"))
    ):
        return "소개"
    if "교육목표 및 핵심역량" in line or line.endswith("교육목표"):
        return "교육목표"
    return None


def parse(lines: list[str]) -> dict[str, str]:
    """정제된 줄 리스트를 섹션별 텍스트 dict으로 변환한다."""
    sections: dict[str, list[str]] = {k: [] for k in _SECTION_HEADERS}
    current_sec = "소개"

    for line in lines:
        sec = _classify(line)
        if sec is not None:
            current_sec = sec
        else:
            sections[current_sec].append(line)

    cert_parts: list[str] = []
    if sections["자격증_직무"]:
        cert_parts.append("직무역량 관련 자격: " + ", ".join(sections["자격증_직무"]))
    if sections["자격증_기본"]:
        cert_parts.append("기본역량 관련 자격: " + ", ".join(sections["자격증_기본"]))
    if sections["자격증"]:
        cert_parts.extend(sections["자격증"])

    if cert_parts:
        sections["자격증"] = cert_parts
    del sections["자격증_직무"]
    del sections["자격증_기본"]

    return {k: "\n".join(v).strip() for k, v in sections.items()}
