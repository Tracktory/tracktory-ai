"""[offline 생성기] RAGFlow KB 의 트랙 청크 벡터 → ``config/track_meta_vectors.yaml``.

트랙소개 .txt 가 bge-m3 로 적재된 KB 에서 청크 벡터를 받아 트랙(doc_name) 단위로
mean-pool + L2 정규화한 뒤, 전체 트랙 평균 벡터를 빼고 다시 L2 정규화하여
``track_id`` 키로 사이드카 YAML 에 덤프한다. 런타임
(``YamlTrackRepository``)이 이 사이드카를 tracks.yaml 옆에서 읽어 ``Track.meta_vector``
로 주입하고, 트랙 시너지의 메타 코사인 항이 이를 쓴다.

벡터를 tracks.yaml 에 인라인하지 않고 사이드카로 분리하는 이유는 차원이 커 사람이 읽을 수
없게 되기 때문이다.

RAGFlow 호출이 발생하므로 추천 런타임이 아니라 본 생성 시점에만 외부 I/O 가 있다.

실행:  uv run python scripts/generate_track_meta_vectors.py [dataset_id]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from tracktory.rag.ragflow_client import RagflowClient, RagflowConfig

# 사용자 제공: 트랙소개 .txt 가 bge-m3 로 적재된 KB.
_DEFAULT_DATASET_ID = "fdda2e3e5c5711f18721935010848eaf"
_ROOT = Path(__file__).resolve().parents[1]
_TRACKS_YAML = _ROOT / "src" / "tracktory" / "config" / "tracks.yaml"
_OUT_PATH = _ROOT / "src" / "tracktory" / "config" / "track_meta_vectors.yaml"
_ROUND = 6  # 코사인에 영향 없는 선에서 파일 크기 절감


def _safe_name(track_id: str) -> str:
    """KB doc_name 이 쓰는 파일안전 이름 규칙(builder 와 동일): ``/ ㆍ ·`` → ``_``."""
    return track_id.replace("/", "_").replace("ㆍ", "_").replace("·", "_")


def _track_id_by_safe_name() -> dict[str, str]:
    """tracks.yaml 의 ``track_id`` 를 safe_name 키로 역인덱싱. doc_name → track_id 매핑용."""
    ids = re.findall(r"^- track_id:\s*(.+)$", _TRACKS_YAML.read_text(encoding="utf-8"), re.M)
    mapping: dict[str, str] = {}
    for raw in ids:
        track_id = raw.strip()
        safe = _safe_name(track_id)
        if safe in mapping:
            print(f"⚠ safe_name 충돌: {mapping[safe]} vs {track_id} → {safe}")
        mapping[safe] = track_id
    return mapping


def main() -> None:
    load_dotenv()
    dataset_id = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DATASET_ID

    # 벡터 페이로드가 커 기본 5초 타임아웃은 빠듯 → 넉넉히.
    cfg = replace(RagflowConfig.from_env(), timeout=30.0)
    client = RagflowClient(cfg)
    chunks = client.fetch_chunk_vectors(dataset_id=dataset_id)

    by_doc: dict[str, list[list[float]]] = defaultdict(list)
    for chunk in chunks:
        name = chunk.get("doc_name")
        vector = chunk.get("vector")
        if isinstance(name, str) and isinstance(vector, list) and vector:
            by_doc[name].append(vector)

    safe_to_id = _track_id_by_safe_name()
    vectors: dict[str, list[float]] = {}
    unmapped: list[str] = []
    for doc_name, vecs in by_doc.items():
        stem = re.sub(r"\.txt$", "", doc_name).strip()
        track_id = safe_to_id.get(_safe_name(stem))
        if track_id is None:
            unmapped.append(doc_name)
            continue
        mean = np.mean(np.asarray(vecs, dtype=float), axis=0)
        unit = mean / (np.linalg.norm(mean) + 1e-12)
        vectors[track_id] = [float(x) for x in unit]

    if vectors:
        track_ids = list(vectors)
        mat = np.asarray([vectors[track_id] for track_id in track_ids], dtype=float)
        centered = mat - mat.mean(axis=0, keepdims=True)
        centered /= np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
        vectors = {
            track_id: [round(float(x), _ROUND) for x in row]
            for track_id, row in zip(track_ids, centered, strict=True)
        }

    # 트랙당 한 줄로 덤프 — 차원이 커 indent 전체 전개는 비대해지므로, 키는 줄바꿈하되
    # 벡터 배열은 한 줄에 둬 사람이 트랙 목록을 훑을 수 있게 한다.
    lines = [
        "# Mean-centered + L2-normalized track meta vectors.",
        "# Generated from RAGFlow chunk vectors.",
    ]
    lines.extend(
        f"{json.dumps(tid, ensure_ascii=False)}: {json.dumps(vec)}" for tid, vec in vectors.items()
    )
    _OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dim = len(next(iter(vectors.values()))) if vectors else 0
    print(f"트랙 meta_vector {len(vectors)}건 (dim={dim}) → {_OUT_PATH}")
    if unmapped:
        print(f"track_id 미매핑 {len(unmapped)}건(스킵): {', '.join(unmapped)}")


if __name__ == "__main__":
    main()
