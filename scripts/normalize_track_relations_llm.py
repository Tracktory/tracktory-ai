"""외부 택소노미 없이 LLM이 추출한 트랙 관계를 정규화한다.

이 스크립트는 ``track_relations_llm.json``을 기준 데이터로 삼고
``competency_taxonomy.json``이나 직무 데이터를 읽지 않는다. LLM이 만든
표현은 유지하되, 추천 기능에서 더 안전하게 사용할 수 있도록 다음 처리를 한다.

- 도구/언어/플랫폼 이름을 표준화한다.
- 도구가 아닌 실무 스킬을 ``tech_stacks``에서 분리한다.
- 트랙 간 유사한 역량 표현을 중복 제거한다.
- 기술 출처 과목에서 ``uses_technology`` 관계를 생성한다.

실행:
    uv run python scripts/normalize_track_relations_llm.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_IN_PATH = _ROOT / "data" / "processed" / "track_relations_llm" / "track_relations_llm.json"
_OUT_DIR = _ROOT / "data" / "processed" / "track_relations_llm_normalized"


_TECH_EXPANSIONS: dict[str, list[str]] = {
    "html/css/javascript": ["HTML", "CSS", "JavaScript"],
    "ai 도구 (gpt, gemini)": ["GPT", "Gemini"],
    "ai 툴 (google gemini)": ["Gemini"],
    "포토샵 (photoshop)": ["Photoshop"],
    "캡컷 (capcut)": ["CapCut"],
    "pbl(problem-based learning)": [],
}

_TECH_ALIASES: dict[str, str] = {
    "3d max": "3ds Max",
    "3ds max": "3ds Max",
    "adobe after effects": "After Effects",
    "after effects": "After Effects",
    "adobe illustrator": "Illustrator",
    "illustrator": "Illustrator",
    "일러스트레이터": "Illustrator",
    "adobe photoshop": "Photoshop",
    "photoshop": "Photoshop",
    "포토샵": "Photoshop",
    "adobe premiere pro": "Premiere Pro",
    "adobe xd": "Adobe XD",
    "autocad": "AutoCAD",
    "cad": "CAD",
    "css3": "CSS",
    "css": "CSS",
    "davinci resolve": "DaVinci Resolve",
    "excel": "Excel",
    "microsoft excel": "Excel",
    "figma": "Figma",
    "html5": "HTML",
    "html": "HTML",
    "jupyter notebook": "Jupyter",
    "ms sql server": "SQL Server",
    "sql server": "SQL Server",
    "sql": "SQL",
    "ros2": "ROS 2",
    "sap s/4hana": "SAP S/4HANA",
    "sketchup": "SketchUp",
    "stable diffusion": "Stable Diffusion",
    "substance painter": "Substance Painter",
    "textom": "TEXTOM",
    "zbrush": "ZBrush",
    "이사도라": "Isadora",
    "스마트스토어": "Smart Store",
}

_KEEP_TECH_EXACT = {
    "Android",
    "AntConc",
    "C",
    "C++",
    "CapCut",
    "DaVinci Resolve",
    "EasyEDA",
    "Gemini",
    "GPT",
    "iOS",
    "Isadora",
    "Java",
    "JavaScript",
    "Kling",
    "Kotlin",
    "MATLAB",
    "Midjourney",
    "NVIDIA DLI",
    "Notion",
    "OMEKA",
    "OpenCV",
    "OpenGL",
    "Pandas",
    "Power BI",
    "Power Query",
    "Python",
    "QGIS",
    "R",
    "React",
    "SAP S/4HANA",
    "SAS",
    "SPSS",
    "SQL Server",
    "Tableau",
    "TensorFlow",
    "Unity",
    "Verilog",
}

_NON_TECH_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^ai$",
        r"^ai\s*기술$",
        r"^haccp$",
        r"ai\s*생성.*도구$",
        r"생성\s*이미지\s*도구$",
        r"기술$",
        r"기법$",
        r"이론$",
        r"방법론$",
        r"재료$",
        r"조리법$",
        r"기본 조리",
        r"상품 기획$",
        r"교육 자료 설계$",
        r"영어학$",
        r"코퍼스$",
        r"댄스$",
        r"무용$",
        r"발레$",
        r"카메라$",
        r"제작 소프트웨어$",
        r"분석 도구$",
        r"애니메이션 제작 도구$",
    )
)

_GENERIC_COMP_WORDS = (
    "능력",
    "역량",
    "이해",
    "활용",
    "적용",
    "수행",
    "기술",
)


@dataclass
class CompetencyCluster:
    label: str
    key: str
    members: set[str] = field(default_factory=set)


def main() -> None:
    args = _parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("input JSON must be an object keyed by track name")

    comp_label_by_raw = _cluster_competencies(raw, args.competency_similarity)
    normalized = {
        track_name: _normalize_track(track_name, item, comp_label_by_raw)
        for track_name, item in raw.items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(normalized, args.out_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=_IN_PATH)
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    parser.add_argument("--competency-similarity", type=float, default=0.86)
    return parser.parse_args()


def _normalize_track(
    track_name: str,
    item: dict[str, Any],
    comp_label_by_raw: dict[str, str],
) -> dict[str, Any]:
    competencies = _normalize_competencies(item.get("competencies", []), comp_label_by_raw)
    tech_stacks, practical_skills = _normalize_techs(item.get("tech_stacks", []))
    relations = _normalize_relations(item.get("relations", []), comp_label_by_raw)
    relations.extend(_technology_relations(tech_stacks))
    relations.extend(_practical_skill_relations(practical_skills))

    return {
        "track_id": item.get("track_id", track_name),
        "track_name": item.get("track_name", track_name),
        "college_id": item.get("college_id", ""),
        "department_id": item.get("department_id", ""),
        "course_ids": item.get("course_ids", []),
        "source": "normalized:llm_only",
        "source_model": item.get("model"),
        "track_summary": item.get("track_summary", ""),
        "competencies": competencies,
        "tech_stacks": tech_stacks,
        "practical_skills": practical_skills,
        "relations": _dedupe_relations(relations),
    }


def _normalize_competencies(
    raw_competencies: list[dict[str, Any]],
    comp_label_by_raw: dict[str, str],
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for comp in raw_competencies:
        raw_name = _clean_text(str(comp.get("name", "")))
        if not raw_name:
            continue
        name = comp_label_by_raw.get(raw_name, raw_name)
        current = by_name.setdefault(
            name,
            {
                "name": name,
                "raw_names": [],
                "categories": [],
                "evidence": [],
                "source_courses": [],
                "confidence": 0.0,
            },
        )
        _append_unique(current["raw_names"], raw_name)
        _append_unique(current["categories"], _clean_text(str(comp.get("category", ""))))
        _append_unique(current["evidence"], _clean_text(str(comp.get("evidence", ""))))
        for course in comp.get("source_courses") or []:
            _append_unique(current["source_courses"], str(course))
        current["confidence"] = max(float(comp.get("confidence", 0.0)), current["confidence"])
    return sorted(by_name.values(), key=lambda x: (-x["confidence"], x["name"]))


def _normalize_techs(
    raw_techs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tech_by_name: dict[str, dict[str, Any]] = {}
    skill_by_name: dict[str, dict[str, Any]] = {}

    for tech in raw_techs:
        raw_name = _clean_text(str(tech.get("name", "")))
        if not raw_name:
            continue
        expanded = _expand_tech(raw_name)
        if expanded is not None:
            for name in expanded:
                _merge_tech_entry(tech_by_name, name, raw_name, tech)
            if not expanded:
                _merge_skill_entry(skill_by_name, raw_name, raw_name, tech)
            continue

        canonical = _canonical_tech(raw_name)
        if canonical is None:
            _merge_skill_entry(skill_by_name, raw_name, raw_name, tech)
        else:
            _merge_tech_entry(tech_by_name, canonical, raw_name, tech)

    tech_stacks = sorted(tech_by_name.values(), key=lambda x: x["name"].lower())
    practical_skills = sorted(skill_by_name.values(), key=lambda x: x["name"])
    return tech_stacks, practical_skills


def _merge_tech_entry(
    by_name: dict[str, dict[str, Any]],
    name: str,
    raw_name: str,
    source: dict[str, Any],
) -> None:
    current = by_name.setdefault(
        name,
        {
            "name": name,
            "raw_names": [],
            "evidence": [],
            "source_courses": [],
            "confidence": 0.0,
        },
    )
    _append_unique(current["raw_names"], raw_name)
    _append_unique(current["evidence"], _clean_text(str(source.get("evidence", ""))))
    for course in source.get("source_courses") or []:
        _append_unique(current["source_courses"], str(course))
    current["confidence"] = max(float(source.get("confidence", 0.0)), current["confidence"])


def _merge_skill_entry(
    by_name: dict[str, dict[str, Any]],
    name: str,
    raw_name: str,
    source: dict[str, Any],
) -> None:
    current = by_name.setdefault(
        name,
        {
            "name": name,
            "raw_names": [],
            "evidence": [],
            "source_courses": [],
            "confidence": 0.0,
        },
    )
    _append_unique(current["raw_names"], raw_name)
    _append_unique(current["evidence"], _clean_text(str(source.get("evidence", ""))))
    for course in source.get("source_courses") or []:
        _append_unique(current["source_courses"], str(course))
    current["confidence"] = max(float(source.get("confidence", 0.0)), current["confidence"])


def _expand_tech(name: str) -> list[str] | None:
    key = _tech_key(name)
    if key not in _TECH_EXPANSIONS:
        return None
    return _TECH_EXPANSIONS[key]


def _canonical_tech(name: str) -> str | None:
    key = _tech_key(name)
    if key in _TECH_ALIASES:
        return _TECH_ALIASES[key]
    if name in _KEEP_TECH_EXACT:
        return name
    if any(pattern.search(name) for pattern in _NON_TECH_PATTERNS):
        return None
    if _looks_like_named_tool(name):
        return name
    return None


def _looks_like_named_tool(name: str) -> bool:
    if len(name) <= 1:
        return False
    return bool(re.search(r"[A-Za-z0-9+#./-]", name))


def _normalize_relations(
    raw_relations: list[dict[str, Any]],
    comp_label_by_raw: dict[str, str],
) -> list[dict[str, Any]]:
    relations = []
    for rel in raw_relations:
        obj = _clean_text(str(rel.get("object", "")))
        predicate = rel.get("predicate")
        if predicate == "teaches_competency":
            obj = comp_label_by_raw.get(obj, obj)
        relations.append(
            {
                "subject": _clean_text(str(rel.get("subject", ""))),
                "predicate": predicate,
                "object": obj,
                "evidence": _clean_text(str(rel.get("evidence", ""))),
                "source_course": rel.get("source_course"),
                "confidence": float(rel.get("confidence", 0.0)),
            }
        )
    return relations


def _technology_relations(tech_stacks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations = []
    for tech in tech_stacks:
        for course in tech["source_courses"]:
            relations.append(
                {
                    "subject": course,
                    "predicate": "uses_technology",
                    "object": tech["name"],
                    "evidence": tech["evidence"][0] if tech["evidence"] else "",
                    "source_course": course,
                    "confidence": tech["confidence"],
                }
            )
    return relations


def _practical_skill_relations(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations = []
    for skill in skills:
        for course in skill["source_courses"]:
            relations.append(
                {
                    "subject": course,
                    "predicate": "teaches_practical_skill",
                    "object": skill["name"],
                    "evidence": skill["evidence"][0] if skill["evidence"] else "",
                    "source_course": course,
                    "confidence": skill["confidence"],
                }
            )
    return relations


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for rel in relations:
        key = (
            rel.get("subject"),
            rel.get("predicate"),
            rel.get("object"),
            rel.get("source_course"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(rel)
    return result


def _cluster_competencies(raw: dict[str, Any], threshold: float) -> dict[str, str]:
    names: list[str] = []
    for item in raw.values():
        for comp in item.get("competencies", []):
            name = _clean_text(str(comp.get("name", "")))
            if name:
                names.append(name)

    clusters: list[CompetencyCluster] = []
    for name in sorted(set(names), key=lambda value: (len(value), value)):
        key = _competency_key(name)
        if not key:
            continue
        match = _find_cluster(key, clusters, threshold)
        if match is None:
            clusters.append(CompetencyCluster(label=name, key=key, members={name}))
            continue
        match.members.add(name)
        match.label = _choose_cluster_label(match.members)
        match.key = _competency_key(match.label)

    label_by_raw: dict[str, str] = {}
    for cluster in clusters:
        for member in cluster.members:
            label_by_raw[member] = cluster.label
    return label_by_raw


def _find_cluster(
    key: str,
    clusters: list[CompetencyCluster],
    threshold: float,
) -> CompetencyCluster | None:
    best: tuple[float, CompetencyCluster] | None = None
    for cluster in clusters:
        score = _competency_similarity(key, cluster.key)
        if score >= threshold and (best is None or score > best[0]):
            best = (score, cluster)
    return best[1] if best else None


def _competency_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 5 and len(b) >= 5 and (a in b or b in a):
        return 0.92
    return SequenceMatcher(None, a, b).ratio()


def _competency_key(name: str) -> str:
    text = _clean_text(name).lower()
    text = re.sub(r"[ㆍ·/()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for word in _GENERIC_COMP_WORDS:
        text = text.replace(word, "")
    return re.sub(r"\s+", "", text)


def _choose_cluster_label(members: set[str]) -> str:
    return sorted(members, key=lambda value: (len(_competency_key(value)), len(value), value))[0]


def _write_outputs(normalized: dict[str, dict[str, Any]], out_dir: Path) -> None:
    lite = {
        track_name: {
            "track_id": item["track_id"],
            "track_name": item["track_name"],
            "competencies": [comp["name"] for comp in item["competencies"]],
            "tech_stacks": [tech["name"] for tech in item["tech_stacks"]],
            "practical_skills": [skill["name"] for skill in item["practical_skills"]],
            "relation_count": len(item["relations"]),
        }
        for track_name, item in normalized.items()
    }
    synergy = {
        track_name: {
            "competencies": [comp["name"] for comp in item["competencies"]],
            "tech_stacks": [tech["name"] for tech in item["tech_stacks"]],
        }
        for track_name, item in normalized.items()
    }
    relation_counts: dict[str, int] = defaultdict(int)
    for item in normalized.values():
        for rel in item["relations"]:
            relation_counts[str(rel["predicate"])] += 1

    report = {
        "tracks": len(normalized),
        "tracks_with_tech_stacks": sum(1 for item in normalized.values() if item["tech_stacks"]),
        "tracks_with_practical_skills": sum(
            1 for item in normalized.values() if item["practical_skills"]
        ),
        "total_competencies": sum(len(item["competencies"]) for item in normalized.values()),
        "total_tech_stacks": sum(len(item["tech_stacks"]) for item in normalized.values()),
        "total_practical_skills": sum(
            len(item["practical_skills"]) for item in normalized.values()
        ),
        "relation_counts": dict(sorted(relation_counts.items())),
        "source": "track_relations_llm only; no external taxonomy/job data",
    }

    (out_dir / "track_relations_llm_normalized.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "track_relations_llm_normalized_lite.json").write_text(
        json.dumps(lite, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "track_synergy_features.json").write_text(
        json.dumps(synergy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "track_relations_llm_normalized_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"full: {out_dir / 'track_relations_llm_normalized.json'}")
    print(f"lite: {out_dir / 'track_relations_llm_normalized_lite.json'}")
    print(f"synergy: {out_dir / 'track_synergy_features.json'}")


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _clean_text(value: str) -> str:
    value = value.replace("\u3000", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip(" .;:,")


def _tech_key(name: str) -> str:
    key = _clean_text(name).lower()
    key = re.sub(r"\s+", " ", key)
    return key


if __name__ == "__main__":
    main()
