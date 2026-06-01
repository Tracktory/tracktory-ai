"""정규화된 LLM 트랙 특징을 정적 트랙 카탈로그에 병합한다.

``TrackSynergyNode``는 ``YamlTrackRepository``를 통해
``src/tracktory/config/tracks.yaml``에서 ``Track.competencies``와
``Track.tech_stacks``를 읽는다. 정규화된 LLM 출력은 런타임 카탈로그 밖에
보관하므로, 이 스크립트는 시너지 점수 계산에 필요한 두 필드만 카탈로그로
복사한다.

실행:
    uv run python scripts/apply_track_synergy_features.py --write

카탈로그를 수정하지 않고 건수만 확인하려면 ``--dry-run``을 사용한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_TRACKS_YAML = _ROOT / "src" / "tracktory" / "config" / "tracks.yaml"
_FEATURES_JSON = (
    _ROOT / "data" / "processed" / "track_relations_llm_normalized" / "track_synergy_features.json"
)


def main() -> None:
    args = _parse_args()
    catalog = _load_yaml(args.tracks_yaml)
    features = _load_features(args.features_json)
    tracks = catalog.get("tracks")
    if not isinstance(tracks, list):
        raise SystemExit(f"{args.tracks_yaml} must contain a top-level 'tracks' list")

    stats = _merge_features(tracks, features)
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


def _merge_features(tracks: list[Any], features: dict[str, dict[str, list[str]]]) -> dict[str, int]:
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
        if feature is None:
            track.setdefault("competencies", [])
            track.setdefault("tech_stacks", [])
            continue

        matched += 1
        feature_keys.discard(track_id)
        feature_keys.discard(track_name)
        track["competencies"] = _dedupe(feature.get("competencies") or [])
        track["tech_stacks"] = _dedupe(feature.get("tech_stacks") or [])
        if track["competencies"]:
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


if __name__ == "__main__":
    main()
