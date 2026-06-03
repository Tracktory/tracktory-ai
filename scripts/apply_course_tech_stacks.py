"""강의계획서 산문에서 추출한 기술 토큰을 정적 과목 카탈로그에 병합한다.

이수 과목 부스팅은 사용자가 들은 과목과 직무 채용공고 기술스택의 교집합으로
적합도를 가산한다. 그런데 과목은 카탈로그에 이름 (예: "데이터베이스") 만 갖고
있어 직무 토큰 (예: "MySQL") 과 절대 겹치지 않는다. 이 스크립트는 강의계획서
산문에 직무 기술 어휘 (``extract_tech_keywords``) 를 돌려 추출한 기술 토큰을
``Course.tech_stacks`` 로 채워, 교집합이 실제로 잡히도록 만든다.

토큰은 ``extract_tech_keywords`` 가 돌려준 표시 표기 그대로 (예: ``"Python"``,
``"Spring Boot"``) 저장한다 — casefold·변형 없이 둔다. 런타임 부스팅 단계가
``canonical_tech_keys`` 로 비교 시점에 정합 키로 변환하기 때문이다.

실행:
    uv run python scripts/apply_course_tech_stacks.py --write

카탈로그를 수정하지 않고 건수만 확인하려면 ``--dry-run`` 을 사용한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_COURSES_YAML = _ROOT / "src" / "tracktory" / "config" / "courses.yaml"
_SYLLABI_DIR = _ROOT / "data" / "processed" / "rag" / "syllabi"
_TECH_KEYWORDS_PY = _ROOT / "src" / "tracktory" / "common" / "tech_keywords.py"

# 강의계획서 파일명 = "강의계획서_<11자리 학정 등록번호><7자리 과목코드><1자리 분반>.txt".
# 11자리 등록번호 뒤 7자가 카탈로그 ``course_id`` 와 1:1 대응하는 과목코드이며,
# 마지막 1자는 분반 (같은 과목의 여러 강의) 이라 union 대상이다. 카탈로그 id 는
# 모두 7자 (예: "CTA0017", "W080023", "M03A002") 라 7자 anchor 가 데이터 정합.
_FILENAME_CODE_PATTERN = re.compile(r"강의계획서_\d{11}([A-Z0-9]{7})")

# 강의계획서 헤더의 교수 이메일·참고 URL 은 기술 토큰이 아니다. 그런데 도메인
# 꼬리 ``.net`` (hanmail.net / wikidocs.net 등) 이 ``.NET`` 으로, 경로 단어가
# 다른 기술로 오인식돼 직무와 무관한 과목에 토큰이 새어든다. 추출 전에 이메일·
# URL·소문자 호스트 도메인을 지운다. ``ASP.NET`` 같은 본문의 정상 표기는
# 소문자 호스트가 앞서지 않아 보존된다.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+")
_URL_RE = re.compile(r"https?://\S+")
_BARE_DOMAIN_RE = re.compile(r"(?<![A-Za-z0-9])[a-z0-9][a-z0-9-]*\.(?:net|com|org|io|kr)\b")


def main() -> None:
    args = _parse_args()
    extract = _load_extract_tech_keywords(args.tech_keywords_py)
    catalog = _load_yaml(args.courses_yaml)
    courses = catalog.get("courses")
    if not isinstance(courses, list):
        raise SystemExit(f"{args.courses_yaml} must contain a top-level 'courses' list")

    course_tokens, scanned, no_course = _aggregate_syllabus_tokens(
        args.syllabi_dir, _catalog_ids(courses), extract
    )
    stats = _apply_tokens(courses, course_tokens)
    print(
        "\n".join(
            [
                f"courses: {stats['courses']}",
                f"courses with tokens: {stats['courses_with_tokens']}",
                f"syllabi scanned: {scanned}",
                f"syllabi with no matching course: {no_course}",
            ]
        )
    )

    if args.dry_run and not args.write:
        print("dry-run: courses.yaml not modified")
        return
    if not args.write:
        raise SystemExit("pass --write to modify courses.yaml, or --dry-run to inspect only")

    # 카탈로그 생성기가 CRLF 로 산출하므로 같은 줄바꿈으로 되써야 한다. LF 로
    # 쓰면 토큰 몇 줄만 바뀌어도 전 줄이 diff 로 잡혀 (CRLF↔LF) 리뷰 불가 +
    # 다음 카탈로그 재생성 때 도로 뒤집히는 churn 이 난다.
    args.courses_yaml.write_text(
        yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\r\n",
    )
    print(f"updated: {args.courses_yaml}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--courses-yaml", type=Path, default=_COURSES_YAML)
    parser.add_argument("--syllabi-dir", type=Path, default=_SYLLABI_DIR)
    parser.add_argument("--tech-keywords-py", type=Path, default=_TECH_KEYWORDS_PY)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_extract_tech_keywords(path: Path) -> Callable[[str], list[str]]:
    """``tech_keywords.py`` 를 파일 경로로 직접 로드해 추출 함수만 꺼낸다.

    패키지 경로 (``tracktory.common.tech_keywords``) 로 import 하면 패키지
    ``__init__`` 이 환경변수 검증 (RAGFlow 설정) 을 강제해 오프라인 배치에서
    실패한다. 모듈 파일을 직접 로드하면 그 부작용을 우회한다.

    Raises:
        SystemExit: 모듈 로드 실패 또는 추출 함수 부재 시.
    """
    spec = importlib.util.spec_from_file_location("_tech_keywords_isolated", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extract: Callable[[str], list[str]] | None = getattr(module, "extract_tech_keywords", None)
    if extract is None or not callable(extract):
        raise SystemExit(f"{path} has no callable extract_tech_keywords")
    return extract


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must be a YAML mapping")
    return raw


def _catalog_ids(courses: list[Any]) -> set[str]:
    ids: set[str] = set()
    for course in courses:
        if isinstance(course, dict):
            course_id = str(course.get("course_id", "")).strip()
            if course_id:
                ids.add(course_id)
    return ids


def _aggregate_syllabus_tokens(
    syllabi_dir: Path,
    catalog_ids: set[str],
    extract: Callable[[str], list[str]],
) -> tuple[dict[str, set[str]], int, int]:
    """강의계획서 파일을 훑어 ``course_id`` 별 기술 토큰 집합을 합친다.

    한 과목의 여러 분반 파일은 같은 ``course_id`` 로 모여 토큰이 union 된다.
    카탈로그에 없는 과목 (교양·타 트랙 등) 의 파일은 건너뛴다 — 부스팅 대상이
    추천 카탈로그 안의 과목뿐이라, 카탈로그 밖 토큰은 쓰일 곳이 없다.

    Returns:
        (course_id -> 토큰 집합, 스캔한 파일 수, 매칭 과목이 없던 파일 수).
    """
    tokens_by_course: dict[str, set[str]] = {}
    scanned = 0
    no_course = 0
    for path in sorted(syllabi_dir.glob("*.txt")):
        scanned += 1
        course_id = _extract_course_id(path.name)
        if course_id is None or course_id not in catalog_ids:
            no_course += 1
            continue
        text = _strip_contact_noise(path.read_text(encoding="utf-8"))
        bucket = tokens_by_course.setdefault(course_id, set())
        bucket.update(extract(text))
    return tokens_by_course, scanned, no_course


def _strip_contact_noise(text: str) -> str:
    """추출 전에 이메일·URL·소문자 호스트 도메인을 공백으로 지운다.

    강의계획서 헤더의 교수 이메일과 참고 URL 은 기술 토큰의 출처가 아니지만,
    도메인 꼬리 (``.net`` 등) 가 기술 어휘로 오인식돼 무관한 과목을 오염시킨다.
    본문의 정상 표기 (``ASP.NET`` 등) 는 소문자 호스트가 선행하지 않아 남는다.
    """
    cleaned = _EMAIL_RE.sub(" ", text)
    cleaned = _URL_RE.sub(" ", cleaned)
    return _BARE_DOMAIN_RE.sub(" ", cleaned)


def _extract_course_id(filename: str) -> str | None:
    """강의계획서 파일명에서 7자리 과목코드 (= ``course_id``) 를 꺼낸다."""
    match = _FILENAME_CODE_PATTERN.search(filename)
    return match.group(1) if match else None


def _apply_tokens(courses: list[Any], tokens_by_course: dict[str, set[str]]) -> dict[str, int]:
    """각 과목 엔트리에 정렬·중복 제거된 토큰 리스트를 ``tech_stacks`` 로 채운다.

    강의계획서가 없거나 토큰이 안 잡힌 과목은 빈 리스트를 부여한다 (필드 부재로
    런타임 KeyError 가 나지 않도록 명시적으로 채운다).
    """
    courses_with_tokens = 0
    for course in courses:
        if not isinstance(course, dict):
            continue
        course_id = str(course.get("course_id", "")).strip()
        tokens = sorted(tokens_by_course.get(course_id, set()))
        course["tech_stacks"] = tokens
        if tokens:
            courses_with_tokens += 1
    return {
        "courses": len(courses),
        "courses_with_tokens": courses_with_tokens,
    }


if __name__ == "__main__":
    main()
