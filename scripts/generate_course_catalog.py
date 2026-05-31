"""[offline 생성기] 전처리 txt → ``src/tracktory/config/courses.yaml`` 과목 카탈로그 덤프.

과목 카탈로그도 정적이므로 추천 요청마다 RAGFlow 를 부르지 않고, 본 스크립트로
한 번 생성해 둔 YAML 을 런타임(``YamlCourseRepository``)이 읽는다. 교육과정
데이터가 갱신되면 (트랙 카탈로그와 함께) 재실행한다.

트랙 식별자는 ``tracks.yaml``(``YamlTrackRepository``)에서 가져와 과목의
``track_ids`` 가 트랙 카탈로그의 ``track_id`` 와 정합하도록 보장한다.

실행:  uv run python scripts/generate_course_catalog.py
       (선행: scripts/generate_track_catalog.py 로 tracks.yaml 생성)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from tracktory.rag.preprocessed_catalog_repository import PreprocessedCourseRepository
from tracktory.rag.prerequisite_resolver import resolve_prereq_ids
from tracktory.rag.yaml_track_repository import YamlTrackRepository

_ROOT = Path(__file__).resolve().parents[1]
_OUT_PATH = _ROOT / "src" / "tracktory" / "config" / "courses.yaml"
# 선수과목 입력은 build_prerequisites.py 산출물(미추적 data/) 을 그대로 읽는다.
_PREREQ_PATH = _ROOT / "data" / "processed" / "prerequisites.json"


def main() -> None:
    # 트랙 카탈로그의 track_id 를 그대로 써서 course.track_ids 정합 보장.
    track_ids = [t.track_id for t in YamlTrackRepository().list_all()]
    repo = PreprocessedCourseRepository()
    courses = repo.list_for_tracks(track_ids)

    # prerequisites.json(과목명 키) → 카탈로그 course_id 로 해석. 카탈로그 밖
    # 선수(교양/타과)·자기참조는 드롭되므로 dangling prereq 가 생기지 않는다.
    prerequisites = json.loads(_PREREQ_PATH.read_text(encoding="utf-8"))
    resolution = resolve_prereq_ids(prerequisites, [(c.course_name, c.course_id) for c in courses])
    prereq_by_id = resolution.prereq_ids_by_course

    # priority(1) 는 모델 기본값에 맡기고 의미 있는 필드만 덤프.
    entries: list[dict[str, Any]] = [
        {
            "course_id": c.course_id,
            "course_name": c.course_name,
            "credits": c.credits,
            "stage": c.stage,
            "course_type": c.course_type,
            "track_ids": c.track_ids,
            "prereq_ids": prereq_by_id.get(c.course_id, []),
            "available_grades": c.available_grades,
            "available_semesters": c.available_semesters,
        }
        for c in sorted(courses, key=lambda c: c.course_id)
    ]

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(
        yaml.safe_dump({"courses": entries}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    courses_with_prereq = sum(1 for e in entries if e["prereq_ids"])
    print(f"과목 카탈로그 {len(entries)}건 → {_OUT_PATH}")
    print(
        f"  선수과목 입력: {courses_with_prereq}과목 / "
        f"{resolution.total_resolved_edges}엣지 (카탈로그 밖 드롭 {len(resolution.dropped_edges)})"
    )


if __name__ == "__main__":
    main()
