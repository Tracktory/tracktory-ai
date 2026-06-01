"""[offline 생성기] 전처리 txt → ``src/tracktory/config/tracks.yaml`` 트랙 카탈로그 덤프.

학사 트랙 카탈로그는 정적이므로 매 추천 요청마다 RAGFlow 를 부르지 않고, 본
스크립트로 한 번 생성해 둔 YAML 을 런타임(``YamlTrackRepository``)이 읽는다.
트랙소개/교육과정 데이터가 갱신되면 재실행한다.

입력은 ``data/processed/rag/tracks/*.txt`` 와 ``data/processed/rag/courses/*.txt`` 이다.
RAGFlow 는 검색 인덱스 소비자로만 두고 카탈로그 생성 경로에는 외부 호출이 없다.

실행:  uv run python scripts/generate_track_catalog.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tracktory.rag.preprocessed_catalog_repository import PreprocessedTrackRepository

_OUT_PATH = Path(__file__).resolve().parents[1] / "src" / "tracktory" / "config" / "tracks.yaml"

# 추천 파이프라인에서 영구 제외할 트랙 ID. 교양 과정은 전공 추천 대상이 아님.
_BLACKLIST: frozenset[str] = frozenset(
    [
        "교양영어과정",
        "기초교양학부",
        "소양핵심교양학부",
        "자율공학학부",
    ]
)


def main() -> None:
    repo = PreprocessedTrackRepository()
    tracks = repo.list_all()

    all_tracks = sorted(tracks, key=lambda t: t.track_id)
    skipped_no_courses = [t.track_name for t in all_tracks if not t.course_ids]
    skipped_blacklist = [t.track_name for t in all_tracks if t.track_id in _BLACKLIST]
    entries: list[dict[str, Any]] = [
        {
            "track_id": t.track_id,
            "track_name": t.track_name,
            "college_id": t.college_id,
            "department_id": t.department_id,
            "major_id": t.major_id,
            "course_ids": t.course_ids,
        }
        for t in all_tracks
        if t.course_ids and t.track_id not in _BLACKLIST
    ]

    if skipped_no_courses:
        print(f"강의 정보 없어 제외: {len(skipped_no_courses)}건 {skipped_no_courses}")
    if skipped_blacklist:
        print(f"블랙리스트 제외: {len(skipped_blacklist)}건 {skipped_blacklist}")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(
        yaml.safe_dump({"tracks": entries}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"트랙 카탈로그 {len(entries)}건 → {_OUT_PATH}")


if __name__ == "__main__":
    main()
