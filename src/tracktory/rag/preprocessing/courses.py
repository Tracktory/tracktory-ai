"""강의정보 전처리: 트랙별 교과목 목록 RAG 문서 생성

한성대_강의정보.csv → data/processed/rag/courses/교육과정_{트랙명}.txt
"""

import os

import pandas as pd

from tracktory.rag.preprocessing.tracks.builder import load_college_map

_CATEGORY_LABEL: dict[str, str] = {
    "전기": "전공기초",
    "전선": "전공선택",
    "전필": "전공필수",
    "전선(상호)": "전공선택(상호인정)",
    "교필": "교양필수",
    "선필교": "선택필수교양",
    "일교": "일반교양",
}


def build_courses(track_csv: str, college_json: str, output_dir: str) -> list[dict[str, str | int]]:
    """이수 가능 시점 정보를 RAG에 포함해야 학년별 질문 대응 가능. 학년·학기별로 구조화하여 교과목 txt 생성."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(track_csv, encoding="utf-8")
    college_map = load_college_map(college_json)

    results: list[dict[str, str | int]] = []
    for track_name, group in df.groupby("트랙"):
        info = college_map.get(str(track_name), {"college": "한성대학교", "department": ""})

        header_parts = [f"트랙: {track_name}"]
        if info["college"]:
            header_parts.append(f"대학: {info['college']}")
        if info["department"]:
            header_parts.append(f"학부: {info['department']}")

        lines: list[str] = [f"[{' | '.join(header_parts)}]", "", "■ 교육과정 (교과목 목록)", ""]

        for year in sorted(group["학년"].unique()):
            for sem in sorted(group["학기"].unique()):
                subset = group[(group["학년"] == year) & (group["학기"] == sem)]
                if subset.empty:
                    continue
                lines.append(f"{year}학년 {sem}학기")
                for _, row in subset.iterrows():
                    label = _CATEGORY_LABEL.get(str(row["교과구분"]), str(row["교과구분"]))
                    lines.append(
                        f"  - [{label}] {row['교과목']} ({row['교과목코드']}, {row['학점']}학점)"
                    )
                lines.append("")

        content = "\n".join(lines).strip()
        safe_name = str(track_name).replace("/", "_").replace("ㆍ", "_").replace("·", "_")
        out_path = os.path.join(output_dir, f"교육과정_{safe_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        results.append({"track": str(track_name), "course_count": len(group)})

    return results
