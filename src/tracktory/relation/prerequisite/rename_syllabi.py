"""
data/output/syllabi/ 의 강의계획서 txt 파일을 과목명 기반으로 이름 변경하여
data/output/syllabi_rename/ 에 복사한다. 원본 파일은 그대로 유지된다.

원본 파일명 끝 문자(A, B, 7, N 등)를 suffix로 활용한다.

  강의계획서_20261110065CTA0017B.txt  → 강의계획서_기초시각디자인.txt   (단독)
  강의계획서_20261110092T0120037.txt  → 강의계획서_의복구성_7.txt      (중복)
  강의계획서_20261110092T012003A.txt  → 강의계획서_의복구성_A.txt
  강의계획서_20261110092T012003N.txt  → 강의계획서_의복구성_N.txt

실행:
    uv run python -m tracktory.relation.prerequisite.rename_syllabi
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_SYLLABI_DIR = _ROOT / "data" / "processed" / "rag" / "output" / "syllabi"
_OUT_DIR = _ROOT / "data" / "processed" / "rag" / "output" / "syllabi_rename"


def _extract_course_name(text: str) -> str:
    """txt 텍스트에서 과목명을 추출한다."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("과목명:"):
            return stripped[len("과목명:"):].strip()
    return ""


def _safe_filename(name: str) -> str:
    """파일명으로 쓸 수 없는 문자를 제거한다."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def _original_suffix(stem: str) -> str:
    """강의계획서_{code} 에서 code 맨 끝 문자를 반환한다."""
    code = stem.removeprefix("강의계획서_")
    return code[-1] if code else ""


def main() -> None:
    print(f"탐색 경로: {_SYLLABI_DIR}")
    if not _SYLLABI_DIR.exists():
        print("오류: 디렉터리가 존재하지 않습니다.")
        sys.exit(1)

    txt_files = sorted(
        p for p in _SYLLABI_DIR.glob("*.txt")
        if p.name.startswith("강의계획서_")
    )
    print(f"발견된 파일: {len(txt_files)}개\n")
    if not txt_files:
        print("파일 없음 — 경로를 확인하세요.")
        sys.exit(1)

    course_files: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    unnamed: list[Path] = []

    for path in txt_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")

        name = _extract_course_name(text)
        if name:
            suffix = _original_suffix(path.stem)
            course_files[name].append((path, suffix))
        else:
            unnamed.append(path)

    rename_plan: list[tuple[Path, Path]] = []

    for course_name, entries in course_files.items():
        safe = _safe_filename(course_name)
        if len(entries) == 1:
            rename_plan.append((entries[0][0], _OUT_DIR / f"강의계획서_{safe}.txt"))
        else:
            suffixes = [s for _, s in entries]
            suffix_unique = len(suffixes) == len(set(suffixes))
            suffix_counter: dict[str, int] = defaultdict(int)
            for path, suffix in entries:
                if suffix_unique:
                    new_name = f"강의계획서_{safe}_{suffix}.txt"
                else:
                    suffix_counter[suffix] += 1
                    count = suffix_counter[suffix]
                    new_name = (
                        f"강의계획서_{safe}_{suffix}.txt"
                        if count == 1
                        else f"강의계획서_{safe}_{suffix}{count}.txt"
                    )
                rename_plan.append((path, _OUT_DIR / new_name))

    new_names = [dst for _, dst in rename_plan]
    if len(new_names) != len(set(new_names)):
        duplicates = [n for n in new_names if new_names.count(n) > 1]
        print("오류: 새 파일명 충돌 발생:")
        for d in set(duplicates):
            print(f"  {d.name}")
        sys.exit(1)

    multi = sum(1 for _, entries in course_files.items() if len(entries) > 1)
    print(f"총 {len(rename_plan)}개 파일 / 중복 과목 {multi}개\n")

    if unnamed:
        print(f"과목명 추출 실패 {len(unnamed)}개 (변경 안 함):")
        for p in unnamed:
            print(f"  {p.name}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in rename_plan:
        dst.write_bytes(src.read_bytes())

    print(f"완료: {len(rename_plan)}개 파일 → {_OUT_DIR}")


if __name__ == "__main__":
    main()
