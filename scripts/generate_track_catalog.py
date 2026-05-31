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


def main() -> None:
    repo = PreprocessedTrackRepository()
    tracks = repo.list_all()

    # 비어있는 필드(meta_text/meta_vector/competencies/tech_stacks)는 모델 기본값에
    # 맡기고 YAML 에는 의미 있는 식별·소속·과목만 덤프한다.
    entries: list[dict[str, Any]] = [
        {
            "track_id": t.track_id,
            "track_name": t.track_name,
            "college_id": t.college_id,
            "department_id": t.department_id,
            "major_id": t.major_id,
            "course_ids": t.course_ids,
        }
        for t in sorted(tracks, key=lambda t: t.track_id)
    ]

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(
        yaml.safe_dump({"tracks": entries}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with_courses = sum(1 for e in entries if e["course_ids"])
    print(f"트랙 카탈로그 {len(entries)}건 (과목목록 있음 {with_courses}건) → {_OUT_PATH}")


if __name__ == "__main__":
    main()
