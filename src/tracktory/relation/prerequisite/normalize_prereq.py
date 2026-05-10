"""
data/output/syllabi_rename/ 의 강의계획서 txt 파일에서 선수과목 필드를 정규화하여
data/output/syllabi_clean/ 에 복사한다. 원본 파일은 그대로 유지된다.

정규화 규칙:
  - 선수과목 라인이 없음              → 선수과목: null 라인 추가
  - 선수과목: 없음 / 해당없음 / 빈값  → 선수과목: null
  - 선수과목: 자료구조                → 선수과목: 자료구조 (원문 유지)

실행:
    uv run python -m tracktory.relation.prerequisite.normalize_prereq
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_SRC_DIR = _ROOT / "data" / "processed" / "rag" / "output" / "syllabi_rename"
_OUT_DIR = _ROOT / "data" / "processed" / "rag" / "output" / "syllabi_clean"
_LOG_FILE = _ROOT / "logs" / "prereq_raw.log"

# 괄호 제거·공백 제거·소문자 변환 후 정확 일치로 null 판정하는 값 집합
_NULL_EXACT: set[str] = {
    "", "null", "없음", "ㅇ해당사항없음", "-해당없음", "해당없음", "해당사항없음",
    "선수과목없음", "별도없음", "무", "무관",
    "x", "?음", "-na", "n/a", "na", "none", "-", "--", "`",
}

# 공백 제거 후 이 접두사로 시작하면 null (자유서술 패턴)
_NULL_STARTS: tuple[str, ...] = (
    "없",                    # 없음, 없습니다, 없음(설명), 없음, 필수: ...
    "선수과목없음",           # 선수과목 없음. 단, 기본적인 ... 같은 자유서술
    "특정교과목에제한",       # 특정 교과목에 제한은 없으나 ...
    "필수선수과목은없으나",
)

# 선수과목 조각 구분자 (자유서술 감지용)
_SPLIT_RE = re.compile(r"[,，/]|또는|및")

# 괄호 패턴
_PAREN_RE = re.compile(r"[（(（][^）)）]*[）)）]")


def _normalize_prereq_value(raw: str) -> str:
    """선수과목 원문이 실질적으로 비어있으면 'null'을 반환한다.

    판정 순서:
      1. 괄호 제거 후 _NULL_EXACT 정확 일치
      2. 공백 제거 후 _NULL_STARTS 접두사 일치
      3. 구분자 없이 60자 초과 → 자유서술로 판단

    괄호 제거를 먼저 하므로:
      - '없음 (설명...)' → '없음' → null
      - '신호 및 시스템 (설명...)' → '신호 및 시스템' → 유지

    Args:
        raw: 선수과목 원문 (콜론 이후 값).

    Returns:
        정규화된 값 — 비어있으면 'null', 아니면 괄호 제거된 원문.
    """
    # 괄호 안 부연설명 제거
    without_paren = _PAREN_RE.sub("", raw).strip().rstrip(".")

    # 공백 제거 + 소문자 정규화 (이후 모든 비교에 사용)
    nospace = without_paren.replace(" ", "").lower()

    # 1. 정확 일치 (공백 무시 — '해당 없음' == '해당없음')
    if nospace in _NULL_EXACT:
        return "null"

    # 2. 접두사 일치
    if any(nospace.startswith(p) for p in _NULL_STARTS):
        return "null"

    # 3. 구분자 없이 60자 초과 → 자유서술
    if len(without_paren) > 60 and not _SPLIT_RE.search(without_paren) and ":" not in without_paren:
        return "null"

    return without_paren if without_paren else "null"


def _process_text(text: str) -> str:
    """txt 텍스트에서 선수과목 라인을 정규화한다.

    선수과목 라인이 없으면 파일 끝에 '선수과목: null' 라인을 추가한다.

    Args:
        text: 강의계획서 전체 텍스트.

    Returns:
        선수과목 라인이 정규화된 텍스트.
    """
    lines = text.splitlines(keepends=True)
    result_lines: list[str] = []
    prereq_found = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("선수과목:"):
            prereq_found = True
            value = stripped[len("선수과목:"):].strip()
            normalized = _normalize_prereq_value(value)
            ending = "\n" if line.endswith(("\n", "\r\n")) else ""
            result_lines.append(f"선수과목: {normalized}{ending}")
        else:
            result_lines.append(line)

    if not prereq_found:
        if result_lines and not result_lines[-1].endswith("\n"):
            result_lines.append("\n")
        result_lines.append("선수과목: null\n")

    return "".join(result_lines)


def main() -> None:
    if not _SRC_DIR.exists():
        print(f"오류: {_SRC_DIR} 가 존재하지 않습니다.")
        sys.exit(1)

    txt_files = sorted(_SRC_DIR.glob("*.txt"))
    print(f"탐색 경로: {_SRC_DIR}")
    print(f"발견된 파일: {len(txt_files)}개\n")

    if not txt_files:
        print("파일 없음 — 경로를 확인하세요.")
        sys.exit(1)

    null_count = 0
    has_prereq_count = 0
    prereq_log: list[tuple[str, str]] = []

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = txt_file.read_text(encoding="utf-8-sig")

        processed = _process_text(text)

        # 과목명 추출
        course_name = ""
        for line in processed.splitlines():
            if line.strip().startswith("과목명:"):
                course_name = line.strip()[len("과목명:"):].strip()
                break

        # 선수과목 값 추출 및 통계 집계
        for line in processed.splitlines():
            if line.strip().startswith("선수과목:"):
                value = line.strip()[len("선수과목:"):].strip()
                if value == "null":
                    null_count += 1
                else:
                    has_prereq_count += 1
                    prereq_log.append((course_name or txt_file.stem, value))
                break

        (_OUT_DIR / txt_file.name).write_text(processed, encoding="utf-8")

    # 로그 파일 저장 (선수과목 있는 과목만)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("w", encoding="utf-8") as f:
        f.write(f"선수과목 원문 목록 ({len(prereq_log)}건)\n")
        f.write("=" * 60 + "\n\n")
        for course, raw in prereq_log:
            f.write(f"[{course}]\n")
            f.write(f"  {raw}\n\n")

    print("처리 결과:")
    print(f"  선수과목 있음  : {has_prereq_count}개")
    print(f"  선수과목 null  : {null_count}개")
    print(f"\n완료: {len(txt_files)}개 파일 → {_OUT_DIR}")
    print(f"로그  : {_LOG_FILE}")


if __name__ == "__main__":
    main()
