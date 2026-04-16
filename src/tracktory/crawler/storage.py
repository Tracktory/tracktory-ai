"""
통합 저장 모듈 - CSV/JSON 출력 및 재개(resume) 추적
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracktory.common.models import JobPosting

from tracktory.crawler.config import config

# CSV 컬럼 순서 (models.JobPosting.to_csv_row 와 일치)
_CSV_FIELDNAMES: list[str] = [
    "source",
    "source_id",
    "company",
    "title",
    "category",
    "tech_stacks",
    "location",
    "salary",
    "url",
    "experience",
    "collected_at",
    "responsibilities",
    "requirements",
    "preferred",
    "benefits",
]


def save_jobs_csv(
    jobs: "list[JobPosting]",
    path: Path,
    append: bool = True,
) -> int:
    """채용 공고 목록을 CSV 파일로 저장합니다.

    파일이 없거나 append=False 인 경우 헤더 행을 먼저 기록합니다.
    tech_stacks 는 '|' 구분자로 직렬화됩니다.
    Excel 호환을 위해 utf-8-sig 인코딩을 사용합니다.

    Args:
        jobs: 저장할 채용 공고 목록.
        path: 저장 대상 파일 경로.
        append: True 이면 기존 파일에 이어 쓰고, False 이면 덮어씁니다.

    Returns:
        int: 실제로 저장된 공고 수.
    """
    if not jobs:
        return 0

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not append or not path.exists() or path.stat().st_size == 0
    file_mode = "a" if append else "w"

    with path.open(file_mode, encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for job in jobs:
            writer.writerow(job.to_csv_row())

    return len(jobs)


def save_jobs_json(jobs: "list[JobPosting]", path: Path) -> int:
    """채용 공고 목록을 JSON 파일로 저장합니다.

    기존 파일이 있으면 덮어씁니다.
    한글이 그대로 유지되도록 ensure_ascii=False 를 사용하며,
    indent=2 로 읽기 좋은 형식으로 출력합니다.

    Args:
        jobs: 저장할 채용 공고 목록.
        path: 저장 대상 파일 경로.

    Returns:
        int: 실제로 저장된 공고 수.
    """
    if not jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        return 0

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [job.to_dict() for job in jobs]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    return len(records)


def load_collected_ids(source: str, path: Path) -> set[str]:
    """기존 CSV 파일에서 이미 수집된 source_id 집합을 로드합니다.

    크롤링 재개(resume) 시 중복 수집을 방지하기 위해 사용합니다.
    파일이 존재하지 않거나 읽기에 실패한 경우 빈 집합을 반환합니다.

    Args:
        source: 필터링할 데이터 출처 ('saramin' 또는 'wanted').
        path: 읽어올 CSV 파일 경로.

    Returns:
        set[str]: 이미 수집된 source_id 문자열 집합.
    """
    path = path.resolve()
    if not path.exists():
        return set()

    collected: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("source") == source and row.get("source_id"):
                    collected.add(row["source_id"])
    except Exception:
        return set()

    return collected


def get_output_path(source: str, fmt: str = "csv") -> Path:
    """날짜가 포함된 출력 파일 경로를 생성합니다.

    형식: data/raw/{source}_{YYYYMMDD}.{fmt}
    부모 디렉터리가 없으면 자동으로 생성합니다.

    Args:
        source: 데이터 출처 식별자 (예: 'saramin', 'wanted').
        fmt: 파일 확장자. 기본값은 'csv'.

    Returns:
        Path: 날짜가 포함된 절대 경로 객체.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = config.DATA_RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{source}_{date_str}.{fmt}"


def load_category_progress(path: Path) -> dict:
    """카테고리별 크롤링 진행 상황을 JSON 파일에서 로드합니다.

    각 카테고리의 수집 상태(status), 수집된 공고 수(collected),
    목표 수(target), 발견된 URL 수(urls_found) 등을 추적합니다.
    파일이 존재하지 않거나 읽기에 실패한 경우 빈 딕셔너리를 반환합니다.

    Args:
        path: 진행 상황 JSON 파일 경로.

    Returns:
        dict: 카테고리명을 키로 하는 진행 상황 딕셔너리.
              예: {"백엔드": {"status": "done", "collected": 200,
                             "target": 200, "urls_found": 250}, ...}
    """
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_category_progress(progress: dict, path: Path) -> None:
    """카테고리별 크롤링 진행 상황을 JSON 파일로 저장합니다.

    부모 디렉터리가 없으면 자동으로 생성합니다.
    한글이 그대로 유지되도록 ensure_ascii=False 를 사용하며,
    indent=2 로 읽기 좋은 형식으로 출력합니다.

    Args:
        progress: 저장할 카테고리 진행 상황 딕셔너리.
        path: 저장 대상 JSON 파일 경로.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(progress, fh, ensure_ascii=False, indent=2)


def get_category_output_path(category: str, fmt: str = "csv") -> Path:
    """카테고리명과 날짜가 포함된 출력 파일 경로를 생성합니다.

    형식: data/raw/wanted_cat_{category}_{YYYYMMDD}.{fmt}
    카테고리명에 포함된 '/' 문자는 '_' 로 치환하여 파일명 안전성을 확보합니다.
    부모 디렉터리가 없으면 자동으로 생성합니다.

    Args:
        category: 카테고리명 (예: '백엔드', 'DevOps/인프라').
        fmt: 파일 확장자. 기본값은 'csv'.

    Returns:
        Path: 카테고리와 날짜가 포함된 절대 경로 객체.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    safe_category = category.replace("/", "_")
    output_dir = config.DATA_RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"wanted_cat_{safe_category}_{date_str}.{fmt}"
