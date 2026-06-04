"""직무 브리핑 서비스·로더 단위 테스트.

로더는 합성 카탈로그(tmp_path)로 구조 계약을, 조회는 색인 주입으로 순서·스킵·
dedup·연결 명시를 검증한다. 끝으로 실제 출하 카탈로그가 로드 가능한지(출처
강제 포함) 1건으로 가드한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tracktory.briefing.models import BriefingSource, JobBriefing
from tracktory.briefing.service import (
    DEFAULT_BRIEFING_CATALOG_PATH,
    BriefingCatalogError,
    BriefingService,
    load_briefing_catalog,
)

_VALID_CATALOG = """\
briefings:
  BE:
    job_name: 백엔드 개발자
    cards:
      - headline: 백엔드 트렌드 1
        summary: 요약 1
        skills: [Spring Boot, AWS]
        sources:
          - { title: 출처 A, url: https://example.com/a, published_at: "2024" }
      - headline: 백엔드 트렌드 2
        summary: 요약 2
        sources:
          - { title: 출처 B, url: https://example.com/b }
  FE:
    job_name: 프론트엔드 개발자
    cards:
      - headline: 프론트 트렌드
        summary: 요약
        sources:
          - { title: 출처 C, url: https://example.com/c }
"""


def _write_catalog(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "job_briefings.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _service(tmp_path: Path, content: str = _VALID_CATALOG) -> BriefingService:
    return BriefingService(catalog_path=_write_catalog(tmp_path, content))


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------


def test_load_expands_job_name_to_every_card(tmp_path: Path) -> None:
    """직무 단위로 선언한 job_name 이 각 카드에 전개되어 카드가 자기완결적이다."""
    index = load_briefing_catalog(_write_catalog(tmp_path, _VALID_CATALOG))

    assert set(index) == {"BE", "FE"}
    assert len(index["BE"]) == 2
    assert all(card.job_id == "BE" for card in index["BE"])
    assert all(card.job_name == "백엔드 개발자" for card in index["BE"])


def test_load_empty_file_returns_empty_index(tmp_path: Path) -> None:
    """빈 YAML 은 손상이 아니라 빈 카탈로그로 취급한다."""
    assert load_briefing_catalog(_write_catalog(tmp_path, "")) == {}


def test_load_missing_file_raises(tmp_path: Path) -> None:
    assert not (tmp_path / "absent.yaml").exists()
    with pytest.raises(BriefingCatalogError):
        load_briefing_catalog(tmp_path / "absent.yaml")


def test_load_without_briefings_mapping_raises(tmp_path: Path) -> None:
    with pytest.raises(BriefingCatalogError):
        load_briefing_catalog(_write_catalog(tmp_path, "jobs: []\n"))


def test_load_card_without_sources_raises(tmp_path: Path) -> None:
    """출처 없는 카드는 검증 불가하므로 로드 단계에서 차단한다."""
    content = """\
briefings:
  BE:
    job_name: 백엔드 개발자
    cards:
      - headline: 출처 없는 트렌드
        summary: 요약
        sources: []
"""
    with pytest.raises(BriefingCatalogError):
        load_briefing_catalog(_write_catalog(tmp_path, content))


def test_load_card_with_malformed_url_raises(tmp_path: Path) -> None:
    """형식이 깨진 출처 URL 은 사전 검증 불가하므로 적재 단계에서 차단한다."""
    content = """\
briefings:
  BE:
    job_name: 백엔드 개발자
    cards:
      - headline: 트렌드
        summary: 요약
        sources:
          - { title: 깨진 출처, url: not-a-url }
"""
    with pytest.raises(BriefingCatalogError):
        load_briefing_catalog(_write_catalog(tmp_path, content))


def test_load_entry_without_job_name_raises(tmp_path: Path) -> None:
    content = """\
briefings:
  BE:
    cards:
      - headline: h
        summary: s
        sources:
          - { title: t, url: https://example.com }
"""
    with pytest.raises(BriefingCatalogError):
        load_briefing_catalog(_write_catalog(tmp_path, content))


# ---------------------------------------------------------------------------
# 조회 (get_briefings)
# ---------------------------------------------------------------------------


def test_get_briefings_preserves_request_order(tmp_path: Path) -> None:
    """요청 직무 순서대로(직무 내부는 카탈로그 순서) 카드를 모은다."""
    service = _service(tmp_path)

    result = service.get_briefings(["FE", "BE"])

    assert [card.job_id for card in result] == ["FE", "BE", "BE"]


def test_get_briefings_skips_unknown_job_ids(tmp_path: Path) -> None:
    """큐레이션 없는 직무는 조용히 건너뛴다(부분 충족 허용)."""
    service = _service(tmp_path)

    result = service.get_briefings(["UNKNOWN", "FE"])

    assert [card.job_id for card in result] == ["FE"]


def test_get_briefings_dedups_repeated_ids(tmp_path: Path) -> None:
    """같은 직무를 두 번 요청해도 카드가 중복 전개되지 않는다."""
    service = _service(tmp_path)

    result = service.get_briefings(["BE", "BE"])

    assert [card.job_id for card in result] == ["BE", "BE"]  # BE 카드 2장, 1회만 전개


def test_get_briefings_empty_when_no_match(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.get_briefings(["NONE"]) == []


def test_every_returned_card_links_its_job_and_carries_sources(tmp_path: Path) -> None:
    """각 카드는 자신의 job_id 로 추천 직무에 연결되고 출처를 1개 이상 동반한다."""
    service = _service(tmp_path)

    for card in service.get_briefings(["BE", "FE"]):
        assert isinstance(card, JobBriefing)
        assert card.job_id in {"BE", "FE"}
        assert len(card.sources) >= 1
        assert all(isinstance(src, BriefingSource) for src in card.sources)


# ---------------------------------------------------------------------------
# 출하 카탈로그 가드
# ---------------------------------------------------------------------------


def test_default_catalog_loads_with_sources() -> None:
    """출하되는 큐레이션 카탈로그가 로드 가능하고 모든 카드가 출처를 갖는다."""
    index = load_briefing_catalog(DEFAULT_BRIEFING_CATALOG_PATH)

    assert index, "출하 카탈로그가 비어 있으면 안 된다"
    for cards in index.values():
        for card in cards:
            assert card.sources, f"{card.job_id} 카드에 출처가 없다"


def test_default_catalog_job_ids_are_standard_codes() -> None:
    """브리핑 직무 코드가 직무 카탈로그 표준 코드 밖으로 새지 않는다.

    추천 직무 식별자와 같은 표준 코드 어휘를 써야 메인 백엔드가 별칭 매핑 없이
    추천 직무 ↔ 브리핑을 조인할 수 있다. 표준 코드는 직무 검색 어댑터가 쓰는
    카테고리→코드 매핑(``category_to_job_type.yaml``)의 ``job_id`` 값을 권위로 삼는다.
    """
    config_dir = DEFAULT_BRIEFING_CATALOG_PATH.parent
    raw = yaml.safe_load((config_dir / "category_to_job_type.yaml").read_text(encoding="utf-8"))
    standard_codes = {entry["job_id"] for entry in raw.values()}

    briefing_codes = set(load_briefing_catalog(DEFAULT_BRIEFING_CATALOG_PATH))

    assert briefing_codes <= standard_codes, (
        f"표준 코드 밖 직무: {sorted(briefing_codes - standard_codes)}"
    )
