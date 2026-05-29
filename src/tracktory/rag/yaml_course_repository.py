"""오프라인 YAML 카탈로그 기반 ``CourseRepository`` 구현체.

과목 카탈로그도 정적 학사 데이터이므로, RAGFlow 를 추천 요청마다 부르지 않고
``src/tracktory/config/courses.yaml`` 에서 한 번 읽어 메모리에 올린다. YAML 은
``scripts/generate_course_catalog.py`` 가 ``RagflowCourseRepository`` 로 생성한다.

``list_for_tracks(track_ids)`` 는 메모리에 올린 과목 중 ``track_ids`` 와 한 트랙
이상 겹치는 과목을 반환한다(이미 ``course_id`` dedup 된 카탈로그).

호출 측은 ``YamlCourseRepository`` 를 직접 import 하지 않고 ``CourseRepository``
Protocol 타입으로만 주입받는다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tracktory.graph.models import Course

__all__ = ["CourseCatalogError", "YamlCourseRepository"]

logger = logging.getLogger(__name__)

# 패키지 config 디렉터리의 courses.yaml (src/tracktory/rag/ 기준 parents[1]).
_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "courses.yaml"


class CourseCatalogError(Exception):
    """과목 카탈로그 YAML 로딩/검증 실패.

    학습 로드맵 노드의 안전 종료 분기가 raw 예외가 아닌 본 예외를 보도록
    boundary 에서 좁힌다.
    """


class YamlCourseRepository:
    """``CourseRepository`` Protocol 의 정적 YAML 카탈로그 구현체.

    생성 시 YAML 을 1회 로드해 ``Course`` 리스트를 메모리에 보관하고, 이후
    ``list_for_tracks`` 는 메모리 조회만 한다.
    """

    def __init__(self, catalog_path: Path | None = None) -> None:
        self._path = catalog_path or _DEFAULT_CATALOG_PATH
        # fail-fast: 카탈로그 부재/손상은 생성 시점에 드러낸다.
        self._courses: list[Course] = self._load()

    def list_for_tracks(self, track_ids: list[str]) -> list[Course]:
        """``track_ids`` 와 한 트랙 이상 겹치는 과목을 반환한다.

        카탈로그가 이미 ``course_id`` dedup 되어 있어 추가 dedup 은 불필요하다.
        입력에 없는 트랙은 자연히 매칭 0건이 된다.
        """
        wanted = set(track_ids)
        if not wanted:
            return []
        return [c for c in self._courses if wanted.intersection(c.track_ids)]

    def _load(self) -> list[Course]:
        try:
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CourseCatalogError(f"과목 카탈로그 파일 없음: {self._path}") from exc
        except OSError as exc:
            raise CourseCatalogError(f"과목 카탈로그 읽기 실패: {self._path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise CourseCatalogError(f"과목 카탈로그 YAML 파싱 실패: {self._path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise CourseCatalogError(f"과목 카탈로그 최상위는 매핑이어야 함: {self._path}")
        entries = raw.get("courses")
        if not isinstance(entries, list):
            raise CourseCatalogError(f"과목 카탈로그에 'courses' 리스트가 없음: {self._path}")

        courses: list[Course] = []
        for entry in entries:
            course = self._to_course(entry)
            if course is not None:
                courses.append(course)
        return courses

    @staticmethod
    def _to_course(entry: Any) -> Course | None:
        """YAML 항목 1건을 ``Course`` 로 검증한다. 손상 항목은 스킵."""
        if not isinstance(entry, dict):
            logger.warning("과목 카탈로그 항목이 매핑이 아님 스킵: %r", entry)
            return None
        try:
            return Course.model_validate(entry)
        except ValidationError as exc:
            logger.warning("과목 카탈로그 항목 검증 실패 스킵: %s", exc)
            return None
