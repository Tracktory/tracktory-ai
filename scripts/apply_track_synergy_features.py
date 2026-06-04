"""트랙 시너지 입력(역량·기술스택)을 정적 트랙 카탈로그에 병합한다.

``TrackSynergyNode``는 ``YamlTrackRepository``를 통해
``src/tracktory/config/tracks.yaml``에서 ``Track.competencies``와
``Track.tech_stacks``를 읽는다. 두 필드는 서로 다른 출처에서 합성한다.

- ``competencies``: 정규화된 LLM 트랙 특징 (커리큘럼 산문에서 추출).
- ``tech_stacks``: LLM 추출 기술과 트랙 소속 과목의 기술 토큰 집계의 합집합.

기술스택을 두 출처의 합집합으로 채우는 이유:
    LLM 추출만으로는 트랙당 기술 토큰이 적어(보수적 추출) 조합 간 직무 커버율이
    소수의 값으로 뭉쳐 시너지 점수가 동률로 평탄해진다. 동률 조합은 점수가 아니라
    정렬 tie-break(조합 키)로 갈려 추천 순위가 임의로 정해진다. 강의계획서에서 이미
    추출해 둔 과목별 기술 토큰(``Course.tech_stacks``)을 트랙 단위로 합치면 토큰이
    풍부해져 점수가 조합별로 구분되고, 순위가 점수 근거로 정해진다.

``tech_stacks``는 두 출처 모두 직무 기술 어휘 기준 정규 표기(``canonical_tech_token``)
로 통합한다. 시너지의 직무 커버율·상보성은 트랙·직무 토큰의 교집합으로 계산되는데,
트랙이 직무와 같은 기술을 다른 표기로 적어두면(예: "spring boot" vs "Spring Boot")
교집합이 비어 점수가 평탄해진다. 저장 시점에 직무 어휘로 맞춰 두면 표시 어휘와 비교
어휘가 모두 직무-역량 매핑과 정합한다.

실행:
    uv run python scripts/apply_track_synergy_features.py --write

카탈로그를 수정하지 않고 건수만 확인하려면 ``--dry-run``을 사용한다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from tracktory.common.tech_keywords import canonical_tech_token

_ROOT = Path(__file__).resolve().parents[1]
_TRACKS_YAML = _ROOT / "src" / "tracktory" / "config" / "tracks.yaml"
_COURSES_YAML = _ROOT / "src" / "tracktory" / "config" / "courses.yaml"
_FEATURES_JSON = (
    _ROOT / "data" / "processed" / "track_relations_llm_normalized" / "track_synergy_features.json"
)


def main() -> None:
    args = _parse_args()
    catalog = _load_yaml(args.tracks_yaml)
    features = _load_features(args.features_json)
    course_tech = _load_course_tech(args.courses_yaml)
    tracks = catalog.get("tracks")
    if not isinstance(tracks, list):
        raise SystemExit(f"{args.tracks_yaml} must contain a top-level 'tracks' list")

    stats = _merge_features(tracks, features, course_tech)
    print(
        "\n".join(
            [
                f"tracks: {stats['tracks']}",
                f"feature tracks: {stats['feature_tracks']}",
                f"matched: {stats['matched']}",
                f"unmatched features: {stats['unmatched_features']}",
                f"tracks with competencies: {stats['tracks_with_competencies']}",
                f"tracks with tech_stacks: {stats['tracks_with_tech_stacks']}",
            ]
        )
    )

    if args.dry_run and not args.write:
        print("dry-run: tracks.yaml not modified")
        return
    if not args.write:
        raise SystemExit("pass --write to modify tracks.yaml, or --dry-run to inspect only")

    args.tracks_yaml.write_text(
        yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"updated: {args.tracks_yaml}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-yaml", type=Path, default=_TRACKS_YAML)
    parser.add_argument("--courses-yaml", type=Path, default=_COURSES_YAML)
    parser.add_argument("--features-json", type=Path, default=_FEATURES_JSON)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must be a YAML mapping")
    return raw


def _load_features(path: Path) -> dict[str, dict[str, list[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must be a JSON object")
    return raw


def _load_course_tech(path: Path) -> dict[str, list[str]]:
    """``course_id`` → 기술 토큰 매핑을 과목 카탈로그에서 로드한다.

    트랙별 기술스택 집계의 입력이다. 과목 토큰은 강의계획서 산문에서 추출돼
    ``Course.tech_stacks`` 에 이미 적혀 있으며, 본 함수는 정합 표기 변환 없이
    원본 토큰을 넘긴다(병합 단계가 직무 어휘로 통일한다).
    """
    raw = _load_yaml(path)
    courses = raw.get("courses")
    if not isinstance(courses, list):
        raise SystemExit(f"{path} must contain a top-level 'courses' list")
    result: dict[str, list[str]] = {}
    for course in courses:
        if not isinstance(course, dict):
            continue
        course_id = str(course.get("course_id", ""))
        techs = course.get("tech_stacks") or []
        if course_id and isinstance(techs, list):
            result[course_id] = [str(t) for t in techs]
    return result


def _merge_features(
    tracks: list[Any],
    features: dict[str, dict[str, list[str]]],
    course_tech: dict[str, list[str]],
) -> dict[str, int]:
    feature_keys = set(features)
    matched = 0
    tracks_with_comp = 0
    tracks_with_tech = 0

    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("track_id", ""))
        track_name = str(track.get("track_name", ""))
        feature = features.get(track_id) or features.get(track_name)

        if feature is not None:
            matched += 1
            feature_keys.discard(track_id)
            feature_keys.discard(track_name)
            track["competencies"] = _dedupe(feature.get("competencies") or [])
            llm_tech = feature.get("tech_stacks") or []
        else:
            track.setdefault("competencies", [])
            llm_tech = []

        course_ids = track.get("course_ids") or []
        aggregated_tech = [tech for cid in course_ids for tech in course_tech.get(str(cid), [])]
        track["tech_stacks"] = _dedupe_canonical([*llm_tech, *aggregated_tech])

        if track.get("competencies"):
            tracks_with_comp += 1
        if track["tech_stacks"]:
            tracks_with_tech += 1

    return {
        "tracks": len(tracks),
        "feature_tracks": len(features),
        "matched": matched,
        "unmatched_features": len(feature_keys),
        "tracks_with_competencies": tracks_with_comp,
        "tracks_with_tech_stacks": tracks_with_tech,
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _dedupe_canonical(values: Iterable[str]) -> list[str]:
    """정합 표기로 통합한 뒤 중복을 제거한다 (정렬 — 출처 순서 비의존 결정성).

    직무 커버율·상보성 비교는 정합 키로 이뤄지므로 같은 기술의 다른 표기는 한
    토큰으로 묶여야 한다. LLM·과목 두 출처를 합치면 표기가 섞이는데, 저장 시점에
    통일해 두면 표시·비교 어휘가 모두 직무-역량 매핑과 정합한다.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = canonical_tech_token(str(value))
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return sorted(result)


if __name__ == "__main__":
    main()
