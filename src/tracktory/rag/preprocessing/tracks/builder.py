"""Step 3: 파싱된 섹션 데이터를 RAG용 자기완결 문서(.txt)로 변환"""

import json
import os
from typing import Any

# 추천·RAG·카탈로그에서 영구 제외할 단과대. 계약학과(미래플러스대학)는 일반 전공
# 추천 대상이 아니므로 전처리 단계에서부터 산출물을 생성하지 않는다. 다른 단과대를
# 추가로 빼야 하면 이 튜플에만 단과대명(접두사)을 추가하면 전 경로에 반영된다.
EXCLUDED_COLLEGES: tuple[str, ...] = ("미래플러스대학",)


def is_excluded_college(college: str | None) -> bool:
    """제외 단과대 여부. ``미래플러스대학(계약학과)`` 처럼 접미사가 붙는 표기를 흡수하려 prefix 매칭한다."""
    if not college:
        return False
    return any(college.startswith(prefix) for prefix in EXCLUDED_COLLEGES)


_SECTION_ORDER: list[tuple[str, str]] = [
    ("소개", "소개"),
    ("교육목표", "교육목표"),
    ("양성인력", "목표 양성 인력"),
    ("진로", "졸업 후 진로"),
    ("역량", "전공역량"),
    ("연계트랙", "연계트랙"),
    ("필수교과목", "필수 교과목"),
    ("자격증", "관련 자격증"),
    ("산학협력", "산학협력업체"),
    ("관련홈페이지", "관련 홈페이지"),
]


def load_college_map(json_path: str) -> dict[str, dict[str, str]]:
    """트랙명만으로 소속 대학·학부 알 수 없어 RAG 문서 헤더 구성에 별도 JSON 필요. 트랙 → 소속 매핑 로드.

    JSON 부재 시 빈 dict 반환. 다운스트림 ``build_document`` 가
    ``college_info.get("college", "한성대학교")`` 로 fallback 처리하므로 한성대학교 헤더로 진행.
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    mapping: dict[str, dict[str, str]] = {}
    for college in data:
        college_name: str = college["college"]
        for dept in college.get("departments", []):
            dept_name: str = dept["name"]
            for track in dept.get("tracks", []):
                mapping[track["name"]] = {
                    "college": college_name,
                    "department": dept_name,
                }
    return mapping


def _normalize_name(name: str) -> str:
    """U+318D(ㆍ)이 포함되면 GraphRAG 엔티티 추출에서 NaN 임베딩이 발생할 수 있어 치환."""
    return name.replace("ㆍ", "·")


def safe_filename_part(name: str) -> str:
    """파일명에 못 쓰는 구분자(`/`, 가운뎃점류)를 `_` 로 치환."""
    return name.replace("/", "_").replace("ㆍ", "_").replace("·", "_")


def doc_filename(prefix: str, track_name: str, department: str) -> str:
    """전처리 산출물 파일명 단일 규칙: ``{prefix}_{트랙}_{학부}.txt`` (학부 없으면 트랙만).

    트랙소개·교육과정·스킵 제거가 모두 이 규칙을 공유해야 재실행 시 stale orphan 이
    남지 않는다. 학부는 college_map 에 트랙이 있을 때만 채워지며, 없으면 트랙명만 쓴다.
    """
    stem = safe_filename_part(track_name)
    dept = safe_filename_part(department.strip()) if department else ""
    return f"{prefix}_{stem}_{dept}.txt" if dept else f"{prefix}_{stem}.txt"


def build_document(
    track_name: str,
    sections: dict[str, str],
    college_info: dict[str, str],
) -> str:
    """RAG 청크가 잘려 나와도 어느 트랙인지 알 수 있어야 해 헤더 필요. 첫 줄에 트랙·대학·학부 컨텍스트 헤더 포함하여 문서 생성."""
    college = college_info.get("college", "한성대학교")
    department = college_info.get("department", "")
    track_name = _normalize_name(track_name)

    if department:
        header = f"[트랙: {track_name} | 대학: {college} | 학부: {department}]"
    else:
        header = f"[트랙: {track_name} | 대학: {college}]"

    parts = [header]
    for sec_key, sec_label in _SECTION_ORDER:
        content = sections.get(sec_key, "").strip()
        if not content:
            continue
        parts.append(f"\n■ {sec_label}")
        parts.append(content)

    return "\n".join(parts)


def build_all(
    sections_by_track: dict[str, dict[str, str]],
    college_map: dict[str, dict[str, str]],
    output_dir: str,
) -> list[dict[str, Any]]:
    """txt 수백 개 생성 후 파싱 결과 일괄 검토 필요. 전 트랙 txt 생성 + tracks_master.json 출력."""
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict[str, Any]] = []

    for track_name, sections in sections_by_track.items():
        college_info = college_map.get(track_name, {"college": "한성대학교", "department": ""})
        doc_text = build_document(track_name, sections, college_info)

        file_path = os.path.join(
            output_dir, doc_filename("트랙소개", track_name, college_info.get("department", ""))
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc_text)

        results.append(
            {
                "track": track_name,
                "college": college_info.get("college"),
                "department": college_info.get("department"),
                "sections": {k: v for k, v in sections.items() if v},
            }
        )

    master_path = os.path.join(output_dir, "..", "tracks_master.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
