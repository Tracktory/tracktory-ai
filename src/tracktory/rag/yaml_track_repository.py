"""오프라인 YAML 카탈로그 기반 ``TrackRepository`` 구현체.

트랙 카탈로그는 거의 변하지 않는 정적 학사 데이터인데, RAGFlow 기반 어댑터
(``RagflowTrackRepository``)는 트랙마다 교육과정 본문을 fetch 해 추천 요청 1건당
수십 번의 외부 호출이 발생한다. 본 구현체는 그 카탈로그를
``src/tracktory/config/tracks.yaml`` 에서 **한 번 읽어 메모리에 올려** 런타임
외부 I/O 를 제거한다.

YAML 은 ``scripts/generate_track_catalog.py`` 가 ``RagflowTrackRepository`` 로
생성한다(데이터 갱신 시 재실행). 즉 RAGFlow 호출은 카탈로그 생성 시점으로
이동하고, 추천 경로(트랙 시너지 노드)는 파일 1회 로드 외 네트워크가 없다.

호출 측은 ``YamlTrackRepository`` 를 직접 import 하지 않고 ``TrackRepository``
Protocol 타입으로만 주입받는다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tracktory.graph.models import Track

__all__ = ["TrackCatalogError", "YamlTrackRepository"]

logger = logging.getLogger(__name__)

# 패키지 config 디렉터리의 tracks.yaml (src/tracktory/rag/ 기준 parents[1]).
_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "tracks.yaml"


class TrackCatalogError(Exception):
    """트랙 카탈로그 YAML 로딩/검증 실패.

    트랙 시너지 노드의 안전 종료 분기가 raw 예외가 아닌 본 예외를 보도록
    boundary 에서 좁힌다.
    """


class YamlTrackRepository:
    """``TrackRepository`` Protocol 의 정적 YAML 카탈로그 구현체.

    생성 시 YAML 을 1회 로드해 ``Track`` 리스트를 메모리에 보관하고, 이후
    ``list_all`` / ``find_by_track_ids`` 는 메모리 조회만 한다.
    """

    def __init__(self, catalog_path: Path | None = None) -> None:
        self._path = catalog_path or _DEFAULT_CATALOG_PATH
        # fail-fast: 카탈로그 부재/손상은 생성 시점에 드러낸다.
        self._tracks: list[Track] = self._load()

    def list_all(self) -> list[Track]:
        """전체 트랙 목록을 반환한다."""
        return list(self._tracks)

    def find_by_track_ids(self, track_ids: list[str]) -> list[Track]:
        """주어진 ``track_ids`` 에 해당하는 트랙만 반환한다.

        입력에 없는 식별자는 조용히 무시한다(Protocol contract).
        """
        wanted = set(track_ids)
        return [track for track in self._tracks if track.track_id in wanted]

    def _load(self) -> list[Track]:
        try:
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise TrackCatalogError(f"트랙 카탈로그 파일 없음: {self._path}") from exc
        except OSError as exc:
            raise TrackCatalogError(f"트랙 카탈로그 읽기 실패: {self._path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise TrackCatalogError(f"트랙 카탈로그 YAML 파싱 실패: {self._path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise TrackCatalogError(f"트랙 카탈로그 최상위는 매핑이어야 함: {self._path}")
        entries = raw.get("tracks")
        if not isinstance(entries, list):
            raise TrackCatalogError(f"트랙 카탈로그에 'tracks' 리스트가 없음: {self._path}")

        tracks: list[Track] = []
        for entry in entries:
            track = self._to_track(entry)
            if track is not None:
                tracks.append(track)
        return tracks

    @staticmethod
    def _to_track(entry: Any) -> Track | None:
        """YAML 항목 1건을 ``Track`` 으로 검증한다. 손상 항목은 스킵."""
        if not isinstance(entry, dict):
            logger.warning("트랙 카탈로그 항목이 매핑이 아님 스킵: %r", entry)
            return None
        try:
            return Track.model_validate(entry)
        except ValidationError as exc:
            logger.warning("트랙 카탈로그 항목 검증 실패 스킵: %s", exc)
            return None
