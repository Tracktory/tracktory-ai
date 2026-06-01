"""트랙 전용 원천 문서에서 LLM으로 트랙 관계를 추출한다.

이 스크립트는 이전에 생성한 프로필 파일, Graph RAG 관계, 채용/커리어 공고
데이터를 사용하지 않는다. 아래 파일만 읽는다.

- ``src/tracktory/config/tracks.yaml``
- ``src/tracktory/config/courses.yaml``
- ``data/processed/rag/tracks`` 트랙 소개
- ``data/processed/rag/courses`` 교육과정 요약
- ``data/processed/rag/syllabi`` 강의계획서

과목 관계가 없거나 매칭된 강의계획서 텍스트가 없는 트랙은 건너뛴다.
기본값으로 ``tracks.yaml`` 순서의 대상 트랙 중 앞 50개를 처리한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Literal, cast

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[1]
_TRACKS_YAML = _ROOT / "src" / "tracktory" / "config" / "tracks.yaml"
_COURSES_YAML = _ROOT / "src" / "tracktory" / "config" / "courses.yaml"
_RAG_DIR = _ROOT / "data" / "processed" / "rag"
_OUT_DIR = _ROOT / "data" / "processed" / "track_relations_llm"

_SYSTEM_PROMPT = """\
너는 대학 트랙/교과 데이터를 정제하는 커리큘럼 분석가다.

반드시 지켜야 할 원칙:
- 제공된 트랙 소개, 교육과정, 강의계획서 내용만 근거로 사용한다.
- 채용공고, 직무 데이터, 이전 생성 결과, 일반적인 시장 지식은 사용하지 않는다.
- 회사명, 취업처, 졸업 후 진로, 자격증 목록은 기술/역량 근거로 사용하지 않는다.
- 기술스택은 수업에서 실제로 배우거나 사용하는 언어, 프레임워크, 도구, 플랫폼만 추출한다.
- 역량은 학생이 수업을 통해 기르게 되는 능력을 구체적으로 쓴다.
- 너무 일반적인 표현(예: 성실성, 관심, 태도)은 제외한다.
- 한 글자 기술명(C, R 등)은 강의계획서에서 프로그래밍 언어/통계 도구로 명시된 경우에만 포함한다.
- 모든 항목은 입력 텍스트에 있는 짧은 evidence를 가져야 한다.
- 불확실하면 제외한다.
"""


class ExtractedCompetency(BaseModel):
    name: str = Field(description="커리큘럼에서 학생이 습득하는 구체 역량")
    category: str = Field(description="역량의 상위 범주. 예: 프로그래밍, 데이터분석, 디자인, 행정")
    evidence: str = Field(description="입력 텍스트에서 가져온 짧은 근거 문장")
    source_courses: list[str] = Field(default_factory=list, description="근거가 된 과목명")
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedTech(BaseModel):
    name: str = Field(description="정규화된 기술/도구/언어 이름")
    evidence: str = Field(description="입력 텍스트에서 가져온 짧은 근거 문장")
    source_courses: list[str] = Field(default_factory=list, description="근거가 된 과목명")
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    subject: str = Field(description="트랙명 또는 과목명")
    predicate: Literal[
        "teaches_competency",
        "uses_technology",
        "requires_prerequisite",
        "builds_project",
        "covers_topic",
    ]
    object: str = Field(description="역량, 기술, 선수과목, 프로젝트, 주제")
    evidence: str = Field(description="입력 텍스트에서 가져온 짧은 근거 문장")
    source_course: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class TrackExtraction(BaseModel):
    track_summary: str = Field(description="트랙의 교육 내용 요약. 1-2문장.")
    competencies: list[ExtractedCompetency] = Field(default_factory=list)
    tech_stacks: list[ExtractedTech] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


def main() -> None:
    args = _parse_args()
    load_dotenv(_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY가 .env에 없습니다.")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    tracks = _load_tracks(args.tracks_yaml)
    courses = _load_courses(args.courses_yaml)
    intro_by_track = _load_track_texts(args.rag_dir / "tracks")
    curriculum_by_track = _load_track_texts(args.rag_dir / "courses")
    syllabi_by_course = _load_syllabi(args.rag_dir / "syllabi", courses)
    eligible = _eligible_tracks(tracks, syllabi_by_course)
    if args.limit > 0:
        eligible = eligible[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    item_dir = args.out_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=True)

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,  # type: ignore[arg-type]
        timeout=args.timeout,
        max_retries=args.max_retries,
    ).with_structured_output(TrackExtraction)

    print(f"eligible tracks: {len(_eligible_tracks(tracks, syllabi_by_course))}")
    print(f"processing: {len(eligible)} tracks | model={model}")
    print(f"out: {args.out_dir}")

    results: dict[str, dict] = {}
    skipped_existing = 0
    failed: list[dict[str, str]] = []

    for index, track in enumerate(eligible, 1):
        track_name = str(track["track_name"])
        out_path = item_dir / f"{_safe_filename(track_name)}.json"
        if args.resume and out_path.exists():
            results[track_name] = json.loads(out_path.read_text(encoding="utf-8"))
            skipped_existing += 1
            continue

        prompt = _build_prompt(
            track=track,
            courses=courses,
            intro_text=intro_by_track.get(track_name, ""),
            curriculum_text=curriculum_by_track.get(track_name, ""),
            syllabi_by_course=syllabi_by_course,
            max_chars=args.max_chars,
        )

        started = time.perf_counter()
        try:
            extraction = cast(
                TrackExtraction,
                llm.invoke(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                ),
            )
        except Exception as exc:
            failed.append({"track_name": track_name, "error": str(exc)[:500]})
            print(f"[fail] {index}/{len(eligible)} {track_name}: {exc}")
            continue

        item = {
            "track_id": track.get("track_id"),
            "track_name": track_name,
            "college_id": track.get("college_id"),
            "department_id": track.get("department_id"),
            "course_ids": track.get("course_ids") or [],
            "source": "llm:track_intro+curriculum+syllabi",
            "model": model,
            **extraction.model_dump(mode="json"),
        }
        out_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results[track_name] = item
        elapsed = time.perf_counter() - started
        print(
            f"[ok] {index}/{len(eligible)} {track_name} "
            f"comp={len(item['competencies'])} tech={len(item['tech_stacks'])} "
            f"rel={len(item['relations'])} {elapsed:.1f}s"
        )

    _write_outputs(args.out_dir, results, failed, skipped_existing, len(eligible), model)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-yaml", type=Path, default=_TRACKS_YAML)
    parser.add_argument("--courses-yaml", type=Path, default=_COURSES_YAML)
    parser.add_argument("--rag-dir", type=Path, default=_RAG_DIR)
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    parser.add_argument("--limit", type=int, default=50, help="처리할 대상 트랙 수. 0이면 전체.")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-chars", type=int, default=45000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_tracks(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tracks = []
    for item in raw.get("tracks") or []:
        if not isinstance(item, dict):
            continue
        track_name = str(item.get("track_name") or item.get("track_id") or "").strip()
        if track_name:
            item = dict(item)
            item["track_name"] = track_name
            tracks.append(item)
    return tracks


def _load_courses(path: Path) -> dict[str, dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    courses = {}
    for item in raw.get("courses") or []:
        if isinstance(item, dict) and item.get("course_id"):
            courses[str(item["course_id"])] = item
    return courses


def _load_track_texts(path: Path) -> dict[str, str]:
    result = {}
    for text_file in path.glob("*.txt"):
        text = text_file.read_text(encoding="utf-8")
        match = re.search(r"\[트랙:\s*([^|\]]+)", text)
        if match:
            result[_normalize_track_name(match.group(1))] = text
    return result


def _load_syllabi(path: Path, courses: dict[str, dict]) -> dict[str, list[str]]:
    course_name_to_id = {
        str(course.get("course_name") or "").strip(): course_id
        for course_id, course in courses.items()
        if str(course.get("course_name") or "").strip()
    }
    by_course_id: dict[str, list[str]] = {}
    for text_file in path.rglob("*.txt"):
        text = text_file.read_text(encoding="utf-8")
        course_name = _parse_course_name(text)
        course_id = ""
        for candidate in courses:
            if candidate in text_file.name:
                course_id = candidate
                break
        if not course_id and course_name:
            course_id = course_name_to_id.get(course_name, "")
        if course_id:
            by_course_id.setdefault(course_id, []).append(text)
    return by_course_id


def _parse_course_name(text: str) -> str:
    match = re.search(r"과목명:\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _eligible_tracks(tracks: list[dict], syllabi_by_course: dict[str, list[str]]) -> list[dict]:
    result = []
    for track in tracks:
        course_ids = [str(cid) for cid in track.get("course_ids") or []]
        if not course_ids:
            continue
        if not any(cid in syllabi_by_course for cid in course_ids):
            continue
        result.append(track)
    return result


def _build_prompt(
    *,
    track: dict,
    courses: dict[str, dict],
    intro_text: str,
    curriculum_text: str,
    syllabi_by_course: dict[str, list[str]],
    max_chars: int,
) -> str:
    track_name = str(track["track_name"])
    course_ids = [str(cid) for cid in track.get("course_ids") or []]
    course_lines = []
    syllabus_blocks = []
    used_chars = 0

    for cid in course_ids:
        course = courses.get(cid, {})
        course_name = str(course.get("course_name") or cid)
        course_lines.append(
            f"- {course_name} ({cid}, {course.get('course_type', '')}, {course.get('stage', '')})"
        )
        for syllabus in syllabi_by_course.get(cid, []):
            focused = _extract_focus(syllabus)
            if not focused:
                continue
            block = f"\n[강의계획서: {course_name}]\n{focused}\n"
            if used_chars + len(block) > max_chars:
                continue
            syllabus_blocks.append(block)
            used_chars += len(block)

    intro = _strip_noisy_sections(intro_text)
    curriculum = curriculum_text[:6000]

    return f"""\
[트랙 메타]
트랙명: {track_name}
대학: {track.get("college_id", "")}
학부/학과: {track.get("department_id", "")}

[트랙 소개]
{intro[:8000]}

[교육과정 요약]
{curriculum}

[트랙 소속 과목]
{chr(10).join(course_lines)}

[강의계획서]
{"".join(syllabus_blocks)}
"""


def _extract_focus(text: str) -> str:
    sections = ("교과목 개요", "수업 목표", "역량 성취기준", "주차별 수업 주제", "선수과목")
    lines = text.splitlines()
    chunks = []
    keep = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("■ "):
            keep = any(section in stripped for section in sections)
        if stripped.startswith("선수과목"):
            keep = True
        if keep:
            chunks.append(stripped)
    return "\n".join(chunks)[:5000]


def _strip_noisy_sections(text: str) -> str:
    if not text:
        return ""
    stop_markers = ("■ 졸업 후 진로", "취업처:", "관련자격증")
    cut = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut]


def _write_outputs(
    out_dir: Path,
    results: dict[str, dict],
    failed: list[dict[str, str]],
    skipped_existing: int,
    requested: int,
    model: str,
) -> None:
    ordered = dict(sorted(results.items(), key=lambda pair: pair[0]))
    lite = {
        name: {
            "track_id": item["track_id"],
            "track_name": item["track_name"],
            "competencies": [comp["name"] for comp in item["competencies"]],
            "tech_stacks": [tech["name"] for tech in item["tech_stacks"]],
            "relation_count": len(item["relations"]),
        }
        for name, item in ordered.items()
    }
    report = {
        "requested_tracks": requested,
        "completed_tracks": len(results),
        "failed_tracks": len(failed),
        "skipped_existing": skipped_existing,
        "model": model,
        "source": "llm:track_intro+curriculum+syllabi_only",
        "failed": failed,
    }
    (out_dir / "track_relations_llm.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "track_relations_llm_lite.json").write_text(
        json.dumps(lite, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "track_relations_llm_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\ncompleted={len(results)} failed={len(failed)} skipped_existing={skipped_existing}")
    print(f"full: {out_dir / 'track_relations_llm.json'}")
    print(f"lite: {out_dir / 'track_relations_llm_lite.json'}")
    print(f"report: {out_dir / 'track_relations_llm_report.json'}")


def _normalize_track_name(value: str) -> str:
    return value.strip().replace("·", "ㆍ")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.ㆍ&+-]+", "_", value).strip("_")[:120]


if __name__ == "__main__":
    main()
