"""직무 브리핑 조회 서비스 — 큐레이션 카탈로그에서 직무별 브리핑을 읽는다.

데모는 사전 큐레이션된 고정 출력을 쓰므로, 본 서비스는 정적 YAML 카탈로그를
생성자에서 1회 로드하고 호출 경로(``get_briefings``)는 순수 조회만 수행한다
(외부 검색·LLM 호출 없음). 추천 파이프라인과 독립적이라 추천 그래프를 거치지
않고 직무 식별자만으로 호출된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tracktory.briefing.models import BriefingSource, JobBriefing

DEFAULT_BRIEFING_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "job_briefings.yaml"
)


class BriefingCatalogError(Exception):
    """브리핑 카탈로그 YAML 부재/손상 시 raise — 호출자가 5xx 로 매핑."""


def load_briefing_catalog(path: Path) -> dict[str, list[JobBriefing]]:
    """큐레이션 카탈로그에서 ``{job_id: [JobBriefing, ...]}`` 색인을 로드한다.

    한 직무가 여러 트렌드 카드를 가질 수 있어 직무 식별자 → 카드 리스트로
    색인한다. ``job_name`` 은 직무 단위로 한 번 선언하고 각 카드에 전개해,
    카드마다 직무명을 반복 기입하지 않으면서도 ``JobBriefing`` 이 자기완결적인
    (job_id + job_name 동반) 상태를 유지한다.

    Args:
        path: ``job_briefings.yaml`` 의 경로.

    Returns:
        ``{job_id: [JobBriefing, ...]}`` 색인. 카탈로그가 비면 ``{}``.

    Raises:
        BriefingCatalogError: 파일 부재, YAML 파싱 실패, 최상위 구조가
            ``briefings`` mapping 이 아닐 때, 또는 카드가 모델 검증
            (출처 1개 이상 등) 을 위배할 때.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise BriefingCatalogError(f"briefing catalog not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise BriefingCatalogError(f"briefing catalog YAML parse failed: {path}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict) or not isinstance(raw.get("briefings"), dict):
        raise BriefingCatalogError(
            f"briefing catalog {path} must contain a top-level 'briefings' mapping"
        )

    index: dict[str, list[JobBriefing]] = {}
    for job_id, entry in raw["briefings"].items():
        index[str(job_id)] = _parse_job_entry(str(job_id), entry, path)
    return index


def _parse_job_entry(job_id: str, entry: Any, path: Path) -> list[JobBriefing]:
    """직무 한 건의 카탈로그 entry 를 ``JobBriefing`` 리스트로 전개한다."""
    if not isinstance(entry, dict):
        raise BriefingCatalogError(f"briefing entry for '{job_id}' in {path} must be a mapping")
    job_name = entry.get("job_name")
    cards = entry.get("cards")
    if not isinstance(job_name, str) or not job_name.strip():
        raise BriefingCatalogError(f"briefing entry '{job_id}' in {path} requires a 'job_name'")
    if not isinstance(cards, list):
        raise BriefingCatalogError(f"briefing entry '{job_id}' in {path} requires a 'cards' list")

    briefings: list[JobBriefing] = []
    for card in cards:
        if not isinstance(card, dict):
            raise BriefingCatalogError(f"a card of '{job_id}' in {path} must be a mapping")
        try:
            briefings.append(
                JobBriefing(
                    job_id=job_id,
                    job_name=job_name,
                    headline=card.get("headline", ""),
                    summary=card.get("summary", ""),
                    skills=card.get("skills") or [],
                    sources=[BriefingSource.model_validate(s) for s in card.get("sources") or []],
                )
            )
        except ValidationError as exc:
            raise BriefingCatalogError(f"invalid briefing card for '{job_id}' in {path}") from exc
    return briefings


class BriefingService:
    """추천 직무 식별자로 큐레이션 브리핑을 조회하는 서비스.

    카탈로그는 생성자에서 1회 로드하고 조회 경로는 파일 I/O 없는 순수 함수다.
    단위 테스트는 ``catalog_path`` 로 합성 카탈로그를 주입해 외부 I/O 없이 검증한다.
    """

    def __init__(self, catalog_path: Path | None = None) -> None:
        self._index = load_briefing_catalog(catalog_path or DEFAULT_BRIEFING_CATALOG_PATH)

    def get_briefings(self, job_ids: list[str]) -> list[JobBriefing]:
        """요청한 직무들의 브리핑 카드를 요청 순서대로 모아 반환한다.

        각 카드는 자신의 ``job_id`` 를 실어 추천 직무와의 연결을 명시한다.
        카탈로그에 없는 직무는 조용히 건너뛰며(부분 충족 허용), 중복 요청은
        한 번만 전개해 같은 카드가 중복 노출되지 않게 한다.

        Args:
            job_ids: 추천 직무 식별자 목록 (메인 백엔드가 전달).

        Returns:
            ``JobBriefing`` 리스트. 매칭 직무가 하나도 없으면 빈 리스트.
        """
        briefings: list[JobBriefing] = []
        seen: set[str] = set()
        for job_id in job_ids:
            if job_id in seen:
                continue
            seen.add(job_id)
            briefings.extend(self._index.get(job_id, []))
        return briefings


__all__ = [
    "DEFAULT_BRIEFING_CATALOG_PATH",
    "BriefingCatalogError",
    "BriefingService",
    "load_briefing_catalog",
]
