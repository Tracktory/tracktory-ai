"""Step 3: 파싱된 섹션 데이터를 RAG용 자기완결 문서(.txt)로 변환"""

import json
import os
from typing import Any

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
    """트랙명만으로 소속 대학·학부 알 수 없어 RAG 문서 헤더 구성에 별도 JSON 필요. 트랙 → 소속 매핑 로드."""
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


def build_document(
    track_name: str,
    sections: dict[str, str],
    college_info: dict[str, str],
) -> str:
    """RAG 청크가 잘려 나와도 어느 트랙인지 알 수 있어야 해 헤더 필요. 첫 줄에 트랙·대학·학부 컨텍스트 헤더 포함하여 문서 생성."""
    college = college_info.get("college", "한성대학교")
    department = college_info.get("department", "")

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

        safe_name = track_name.replace("/", "_").replace("ㆍ", "_").replace("·", "_")
        file_path = os.path.join(output_dir, f"트랙소개_{safe_name}.txt")
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
