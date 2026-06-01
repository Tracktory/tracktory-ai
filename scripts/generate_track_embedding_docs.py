"""[offline 생성기] 트랙소개 .txt → 트랙 임베딩 비교용 문서 + 매니페스트.

트랙 시너지 MMR 다양성 항(``w_meta``)은 ``Track.meta_vector`` 간 코사인으로 트랙을
변별한다. 원본 트랙소개 .txt 는 공통 템플릿·마케팅·네비게이션 노이즈를 공유해 임베딩이
뭉치므로, 본 스크립트가 변별 신호만 농축한 임베딩 전용 문서를 만들어 둔다.

후속 PR 이 이 매니페스트(``track_id`` → 문서)를 BGE-M3 로 임베딩해 ``meta_vector`` 로
적재한다. 본 스크립트는 RAGFlow·임베딩 키 없이 오프라인으로 동작한다.

실행:  uv run python scripts/generate_track_embedding_docs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tracktory.rag.preprocessing.tracks.embedding_doc import build_doc_from_txt

_ROOT = Path(__file__).resolve().parents[1]
_IN_DIR = _ROOT / "data" / "processed" / "rag" / "tracks"
_OUT_DIR = _ROOT / "data" / "processed" / "rag" / "track_embed_docs"
_MANIFEST_PATH = _ROOT / "data" / "processed" / "rag" / "track_embed_docs.manifest.json"


def _safe_filename(track_id: str) -> str:
    """파일시스템 안전 이름. 매니페스트 키는 원문 ``track_id`` 를 쓰고 파일명만 치환한다."""
    return track_id.replace("/", "_").replace("ㆍ", "_").replace("·", "_")


def main() -> None:
    txt_paths = sorted(_IN_DIR.glob("트랙소개_*.txt"))
    if not txt_paths:
        raise FileNotFoundError(f"트랙소개 .txt 가 없습니다: {_IN_DIR}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    skipped: list[str] = []

    for path in txt_paths:
        track_name, doc = build_doc_from_txt(path.read_text(encoding="utf-8"))
        if not track_name:
            skipped.append(path.name)
            continue
        (_OUT_DIR / f"{_safe_filename(track_name)}.txt").write_text(doc, encoding="utf-8")
        manifest[track_name] = doc

    _MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"임베딩 문서 {len(manifest)}건 → {_OUT_DIR}")
    print(f"매니페스트 → {_MANIFEST_PATH}")
    if skipped:
        print(f"헤더 파싱 실패로 스킵 {len(skipped)}건: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
