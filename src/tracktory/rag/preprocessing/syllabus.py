"""강의계획서 전처리: 과목별 RAG 문서 생성

한성대_강의계획서.csv → data/processed/rag/syllabi/강의계획서_{Course_Code}.txt

유지: 과목명, 교수 정보, 역량성취기준, 교과목개요, 수업목표, 선수과목, 주교재, 주차별 주제
제거: 인재상·교수학습방법·수업유형·성적평가 (보일러플레이트)
"""

import os
import re

import pandas as pd

_SECTION_HEADERS: frozenset[str] = frozenset(
    {
        "수업정보",
        "교수정보",
        "인재상",
        "교수학습방법",
        "수업유형",
        "역량성취기준",
        "성적평가",
        "수업계획",
        "주차별 수업계획",
    }
)

_SKIP_SECTIONS: frozenset[str] = frozenset({"인재상", "교수학습방법", "수업유형", "성적평가"})

_WEEK_RE: re.Pattern[str] = re.compile(r"^(\d+)주")
_METHOD_MARKER_RE: re.Pattern[str] = re.compile(r"\s*T\s+PT\s+P\s+D\s*")
_LEGEND_RE: re.Pattern[str] = re.compile(r"^- \*\s*T\s*:\s*Theory\s*이론")
_WEEK_NOISE_RE: re.Pattern[str] = re.compile(r"^- .+주차$")

_WEEKLY_TABLE_HEADER = "주차 보강시 예정일 강의주제 및 내용 강의방법"


def _parse_syllabus(text: str) -> dict[str, str | list[str]]:
    """섹션 구분자가 과목마다 불일치해 정규식 파싱 필요. 섹션 분리 후 필드별 dict 추출."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        if line in _SECTION_HEADERS:
            current = line
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    result: dict[str, str | list[str]] = {}

    for line in sections.get("수업정보", []):
        if "과목명" in line and "학점 및 이수구분" in line:
            m = re.match(r"과목명\s+(.+?)\s+학점 및 이수구분\s+(.+?)(?:\s+수업시간|$)", line)
            if m:
                result["과목명"] = m.group(1).strip()
                result["학점구분"] = m.group(2).strip()
        if "수강대상" in line:
            m = re.search(r"수강대상\s+(.+)", line)
            if m:
                result["수강대상"] = m.group(1).strip()

    for line in sections.get("교수정보", []):
        m = re.match(r"성명\s+(.+?)\s+소속\s+(.+)", line)
        if m:
            result["교수"] = m.group(1).strip()
            result["소속"] = m.group(2).strip()
        m = re.match(r"이메일\s+(\S+)", line)
        if m and m.group(1) not in ("-", ""):
            result["이메일"] = m.group(1).strip()

    criteria: list[str] = []
    for line in sections.get("역량성취기준", []):
        if line in ("역량성취기준 평가방법 비고", "역량성취기준 평가방법"):
            continue
        if len(line) > 10:
            criteria.append(line)
    if criteria:
        result["역량성취기준"] = criteria

    skip_prefixes = ("학점구성", "부교재/참고문헌", "기타 안내사항", "장애학생지원")
    for line in sections.get("수업계획", []):
        if line.startswith(skip_prefixes):
            continue
        for key in ("교과목개요", "수업목표", "선수과목", "주교재"):
            if line.startswith(key):
                result[key] = line[len(key) :].strip()
                break

    weeks_data: list[tuple[int, list[str]]] = []
    cur_week: int | None = None
    cur_lines: list[str] = []

    for line in sections.get("주차별 수업계획", []):
        if line == _WEEKLY_TABLE_HEADER or _LEGEND_RE.match(line):
            continue
        m = _WEEK_RE.match(line)
        if m:
            if cur_week is not None:
                weeks_data.append((cur_week, cur_lines))
            cur_week = int(m.group(1))
            cur_lines = []
        elif cur_week is not None:
            cur_lines.append(line)

    if cur_week is not None:
        weeks_data.append((cur_week, cur_lines))

    weekly: list[str] = []
    for week_num, week_lines in weeks_data:
        if any("일반수업활동 :" in line for line in week_lines):
            topic_parts: list[str] = []
            collecting = False
            for line in week_lines:
                if "일반수업활동 :" in line:
                    topic_parts = [line.split("일반수업활동 :", 1)[1].strip()]
                    collecting = True
                elif collecting and not line.startswith("-") and not line.startswith("PBL"):
                    topic_parts.append(line)
                elif collecting:
                    break
            topic = " ".join(topic_parts).strip()
        else:
            topic_lines = [
                line
                for line in week_lines
                if not _WEEK_NOISE_RE.match(line) and not line.startswith("PBL")
            ]
            topic = "\n".join(topic_lines).strip()

        topic = _METHOD_MARKER_RE.sub("", topic).strip()
        if topic:
            weekly.append(f"{week_num}주: {topic}")

    if weekly:
        result["주차별수업"] = weekly

    return result


def _build_document(course_code: str, parsed: dict[str, str | list[str]]) -> str:
    course_name = parsed.get("과목명", course_code)
    professor = parsed.get("교수", "")
    affiliation = parsed.get("소속", "")

    header_parts = [f"강의계획서: {course_name}"]
    if professor:
        header_parts.append(f"교수: {professor}")
    if affiliation:
        header_parts.append(f"소속: {affiliation}")

    lines: list[str] = [f"[{' | '.join(header_parts)}]", ""]

    if parsed.get("과목명"):
        lines.append(f"과목명: {parsed['과목명']}")
    if parsed.get("학점구분"):
        lines.append(f"학점/구분: {parsed['학점구분']}")
    if parsed.get("수강대상"):
        lines.append(f"수강대상: {parsed['수강대상']}")
    if professor:
        prof_line = f"담당교수: {professor}"
        if affiliation:
            prof_line += f" ({affiliation})"
        lines.append(prof_line)
    if parsed.get("이메일"):
        lines.append(f"이메일: {parsed['이메일']}")

    if parsed.get("교과목개요"):
        lines += ["", "■ 교과목 개요", str(parsed["교과목개요"])]
    if parsed.get("수업목표"):
        lines += ["", "■ 수업 목표", str(parsed["수업목표"])]

    extras: list[str] = []
    if parsed.get("선수과목"):
        extras.append(f"선수과목: {parsed['선수과목']}")
    if parsed.get("주교재"):
        extras.append(f"주교재: {parsed['주교재']}")
    if extras:
        lines += ["", *extras]

    criteria = parsed.get("역량성취기준")
    if criteria and isinstance(criteria, list):
        lines += ["", "■ 역량 성취기준"]
        for c in criteria:
            lines.append(f"- {c}")

    weekly = parsed.get("주차별수업")
    if weekly and isinstance(weekly, list):
        lines += ["", "■ 주차별 수업 주제"]
        lines.extend(weekly)

    return "\n".join(lines).strip()


def build_syllabi(
    syllabus_csv: str,
    output_dir: str,
) -> tuple[list[dict[str, str]], list[str]]:
    """일부 텍스트 파싱 실패해도 전체 중단 방지 필요. 과목명 추출 실패 시 skipped 분리 처리."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(syllabus_csv, encoding="utf-8")

    results: list[dict[str, str]] = []
    skipped: list[str] = []

    for _, row in df.iterrows():
        course_code = str(row["Course_Code"])
        text = str(row["Syllabus_Text"])

        parsed = _parse_syllabus(text)
        course_name = parsed.get("과목명", "")

        if not course_name:
            skipped.append(course_code)
            continue

        doc = _build_document(course_code, parsed)
        out_path = os.path.join(output_dir, f"강의계획서_{course_code}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc)

        results.append({"course_code": course_code, "course_name": str(course_name)})

    return results, skipped
