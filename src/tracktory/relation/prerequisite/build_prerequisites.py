"""
선수과목 관계 JSON 생성 스크립트.

syllabi_clean/*.txt + courses.csv + prereq_alias.json → prerequisites.json

소스 파일:
  - data/output/syllabi_clean/*.txt   : normalize_prereq.py 출력
    과목명: COURSE_NAME
    선수과목: null | 원문
  - data/courses.csv                  : 공식 과목명 목록 (정확 매칭용)
  - data/processed/prereq_alias.json  : LLM 매핑 결과 (없으면 정확 매칭만 수행)
    형식: {"객체지향2": "객체지향언어2", ...}

처리 흐름:
  1. syllabi_clean/*.txt → 과목명 + 선수과목 원문 추출
     (같은 과목명 파일이 여러 개면 선수과목이 null이 아닌 파일 우선)
  2. 원문 → 쉼표·슬래시·또는·및으로 분리
  3. 각 조각 → courses.csv 정확 매칭 → 미매칭 시 prereq_alias.json 조회
  4. 전체 과목 (선수과목 없는 과목 포함) → prerequisites.json

출력 파일: data/processed/prerequisites.json
  형식: {"과목명": ["선수과목1", ...], ...}  (선수과목 없으면 [])

실행:
    uv run python -m tracktory.relation.prerequisite.build_prerequisites
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("prereq.build")

_ROOT = Path(__file__).resolve().parents[4]
_SYLLABI_CLEAN_DIR = _ROOT / "data" / "processed" / "rag" / "output" / "syllabi_clean"
_COURSES_CSV = _ROOT / "data" / "raw" / "hansung" / "courses.csv"
_ALIAS_MAP_PATH = _ROOT / "data" / "processed" / "prereq_alias.json"
_OUT_PATH = _ROOT / "data" / "processed" / "prerequisites.json"

_SPLIT_RE = re.compile(r"[,，/]|또는|및")
_PAREN_RE = re.compile(r"[（(（][^）)）]*[）)）]")


def _normalize(name: str) -> str:
    """괄호 제거·공백 제거·소문자 변환으로 과목명을 정규화한다."""
    name = _PAREN_RE.sub("", name)
    return name.replace(" ", "").lower()


def _build_lookup(courses_csv: Path) -> dict[str, str]:
    """courses.csv → {normalize(교과목): 교과목 원본명} lookup.

    Args:
        courses_csv: 강의정보 CSV 경로.

    Returns:
        정규화된 과목명을 키, 원본 과목명을 값으로 하는 딕셔너리.
    """
    lookup: dict[str, str] = {}
    with courses_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["교과목"].strip()
            if name:
                norm = _normalize(name)
                if norm not in lookup:
                    lookup[norm] = name
    return lookup


def _parse_course_name(text: str) -> str:
    """txt 텍스트에서 과목명을 추출한다."""
    for line in text.splitlines():
        if line.strip().startswith("과목명:"):
            return line.strip()[len("과목명:"):].strip()
    return ""


def _parse_prereq_raw(text: str) -> str:
    """txt 텍스트에서 선수과목 원문을 추출한다. 없으면 'null' 반환."""
    for line in text.splitlines():
        if line.strip().startswith("선수과목:"):
            return line.strip()[len("선수과목:"):].strip()
    return "null"


def _split_prereq_raw(raw: str) -> list[str]:
    """선수과목 원문을 과목명 조각 리스트로 분리한다.

    콜론 앞 설명 제거 → 괄호 제거 → 구분자로 분리.
    """
    if ":" in raw:
        raw = raw.split(":")[-1]
    raw = _PAREN_RE.sub("", raw)
    return [p.strip().strip(".") for p in _SPLIT_RE.split(raw) if p.strip().strip(".")]


def _load_syllabi_map(syllabi_clean_dir: Path) -> dict[str, str]:
    """syllabi_clean 디렉터리에서 과목명 → 선수과목 원문 맵을 반환한다.

    같은 과목명 파일이 여러 개면 선수과목이 null이 아닌 파일을 우선 선택한다.

    Args:
        syllabi_clean_dir: normalize_prereq.py 출력 디렉터리.

    Returns:
        {과목명: 선수과목_원문} 딕셔너리 (원문은 'null' 또는 실제 값).
    """
    texts_by_course: dict[str, list[str]] = defaultdict(list)

    for txt_file in sorted(syllabi_clean_dir.glob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = txt_file.read_text(encoding="utf-8-sig")

        course_name = _parse_course_name(text)
        if course_name:
            texts_by_course[course_name].append(text)

    result: dict[str, str] = {}
    for course_name, texts in texts_by_course.items():
        preferred = next((t for t in texts if _parse_prereq_raw(t) != "null"), texts[0])
        result[course_name] = _parse_prereq_raw(preferred)

    return result


def build_prerequisites(
    syllabi_clean_dir: Path = _SYLLABI_CLEAN_DIR,
    courses_csv: Path = _COURSES_CSV,
    alias_map_path: Path = _ALIAS_MAP_PATH,
    out_path: Path = _OUT_PATH,
) -> dict[str, list[str]]:
    """선수과목 관계를 빌드하여 prerequisites.json으로 저장한다.

    선수과목이 없는 과목도 빈 리스트로 포함한다.
    자기 자신이 선수과목으로 매칭된 경우는 제외한다.

    Args:
        syllabi_clean_dir: normalize_prereq.py 출력 디렉터리.
        courses_csv: 강의정보 CSV 경로.
        alias_map_path: prereq_alias.json 경로 (없으면 정확 매칭만 수행).
        out_path: 저장할 JSON 경로.

    Returns:
        {과목명: [선수과목명, ...]} 딕셔너리.
    """
    lookup = _build_lookup(courses_csv)

    alias_map: dict[str, str | None] = {}
    if alias_map_path.exists():
        with alias_map_path.open(encoding="utf-8") as f:
            raw_alias = json.load(f)
        alias_map = {_normalize(k): v for k, v in raw_alias.items()}
    else:
        logger.warning("prereq_alias.json 없음 — 정확 매칭만 수행 (%s)", alias_map_path)

    syllabi_map = _load_syllabi_map(syllabi_clean_dir)
    result: dict[str, list[str]] = {}

    for course_name, prereq_raw in sorted(syllabi_map.items()):
        prereq_names: list[str] = []

        if prereq_raw != "null":
            for fragment in _split_prereq_raw(prereq_raw):
                if not fragment:
                    continue
                # 1. courses.csv 정확 매칭
                official = lookup.get(_normalize(fragment))
                # 2. prereq_alias.json 조회
                if official is None:
                    official = alias_map.get(_normalize(fragment))
                if official and official not in prereq_names and official != course_name:
                    prereq_names.append(official)

        result[course_name] = prereq_names

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def validate_prerequisites(graph: dict[str, list[str]]) -> bool:
    """선수과목 그래프에서 자기 자신 참조·순환참조·dangling 참조를 검증한다.

    dangling 참조(graph 에 노드가 없는 선수과목)는 경고로 분류하여 통과 여부에는
    영향을 주지 않지만 로그로 보고한다. 자기 자신 참조와 순환참조는 에러로 분류한다.

    Args:
        graph: build_prerequisites() 반환값.

    Returns:
        에러가 없으면 True, 하나라도 발견되면 False.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. 자기 자신 참조
    for course, prereqs in graph.items():
        if course in prereqs:
            errors.append(f"자기 자신 참조: {course}")

    # 2. 순환참조 — DFS (흰색/회색/검정 3색 마킹)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {course: WHITE for course in graph}

    def dfs(node: str, path: list[str]) -> None:
        color[node] = GRAY
        for prereq in graph.get(node, []):
            if prereq not in color:
                # graph 에 노드가 없는 선수과목 — alias/courses.csv 매칭은 됐지만
                # syllabi 파일이 없어 자체 그래프 항목이 없는 경우.
                warnings.append(f"존재하지 않는 선수과목 참조: {node} → {prereq}")
                continue
            if color[prereq] == GRAY:
                cycle = path[path.index(prereq):] + [prereq] if prereq in path else path + [prereq]
                errors.append(f"순환참조: {' → '.join(cycle)}")
            elif color[prereq] == WHITE:
                dfs(prereq, path + [prereq])
        color[node] = BLACK

    for course in graph:
        if color[course] == WHITE:
            dfs(course, [course])

    if warnings:
        logger.warning("[검증 경고] dangling 참조 %d건", len(warnings))
        for w in warnings:
            logger.warning("  ⚠ %s", w)

    if errors:
        logger.error("[검증 실패] %d건", len(errors))
        for e in errors:
            logger.error("  ✗ %s", e)
        return False

    logger.info("[검증 통과] 자기 자신 참조 및 순환참조 없음")
    return True


if __name__ == "__main__":
    from tracktory.relation.prerequisite.logging_setup import setup_logging

    setup_logging("prereq")
    result = build_prerequisites()
    validate_prerequisites(result)

    has_prereq = sum(1 for v in result.values() if v)
    total_edges = sum(len(v) for v in result.values())
    logger.info("완료: %s", _OUT_PATH)
    logger.info("전체 과목 수     : %d개", len(result))
    logger.info("선수과목 있음    : %d개", has_prereq)
    logger.info("선수과목 없음    : %d개", len(result) - has_prereq)
    logger.info("선수과목 관계 수 : %d개", total_edges)
