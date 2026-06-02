"""직무 커버율 — 기술 표기 정합 전/후 분포 비교.

직무 채용공고 기술과 트랙 커리큘럼 기술은 같은 기술을 서로 다른 표기로 적어둘
수 있다 (대소문자·별칭). 정합 토큰으로 통합하지 않으면 교집합이 0 에 수렴해
트랙 시너지의 직무 커버율 항이 평탄해지고, 트랙 간 점수 변별이 사라진다.

본 스크립트는 실제 트랙 카탈로그 (``tracks.yaml``) 의 모든 트랙 조합에 대해
대표 직무 기술 집합의 커버율 분포를, 정합 전 (raw — 대소문자 구분 교집합) 과
정합 후 (canonical — ``canonical_tech_keys`` 통합) 로 비교한다.

직무 기술 표기는 실제 채용공고 태그에서 흔한 소문자·별칭 형태를 반영한다.
트랙 카탈로그는 정규 표기 ("Java", "Spring Boot") 로 적혀 있어, 정합이 없으면
대소문자만 달라도 교집합이 비어버린다.

실행:
    uv run python scripts/compare_job_coverage_normalization.py
"""

from __future__ import annotations

import itertools
import os
import statistics
from pathlib import Path

import yaml

# tracktory.common 패키지는 import 시점에 Settings() 를 즉시 인스턴스화하며 RAGFlow
# 자격증명 4 종을 요구한다. 본 스크립트는 순수 계산만 하므로 자격증명을 쓰지 않는다 —
# .env 가 없는 환경에서도 돌도록 빈 값을 채운다 (tests/conftest.py 와 동일한 패턴).
for _var in ("RAGFLOW_API_KEY", "RAGFLOW_BASE_URL", "RAGFLOW_DATASET_ID", "RAGFLOW_RERANKER_ID"):
    os.environ.setdefault(_var, "")

from tracktory.common.tech_keywords import canonical_tech_keys  # noqa: E402

_TRACKS_YAML = Path(__file__).resolve().parents[1] / "src" / "tracktory" / "config" / "tracks.yaml"

REPRESENTATIVE_JOBS: dict[str, list[str]] = {
    "백엔드": ["java", "spring boot", "sql", "mysql", "docker", "aws", "kubernetes"],
    "프론트엔드": ["javascript", "typescript", "reactjs", "css", "html", "next.js"],
    "AI/ML": ["python", "pytorch", "tensorflow", "pandas", "numpy", "scikit-learn"],
    "데이터분석": ["python", "sql", "tableau", "power bi", "pandas"],
}


def _raw_coverage(job_tech: list[str], track_union: set[str]) -> float:
    """대소문자·별칭 구분 그대로의 교집합 커버율 (정합 전)."""
    job = {t for t in job_tech if t}
    if not job:
        return 0.0
    return len(job & track_union) / len(job)


def _canonical_coverage(job_tech: list[str], track_tech: list[str]) -> float:
    """정합 토큰 키로 통합한 교집합 커버율 (정합 후, 추천 파이프라인과 동일)."""
    job_keys = canonical_tech_keys(job_tech)
    if not job_keys:
        return 0.0
    return len(job_keys & canonical_tech_keys(track_tech)) / len(job_keys)


def _load_tracks() -> list[tuple[str, list[str]]]:
    data = yaml.safe_load(_TRACKS_YAML.read_text(encoding="utf-8"))
    return [(t["track_id"], t.get("tech_stacks") or []) for t in data["tracks"]]


def _summary(values: list[float]) -> tuple[int, float, float, float]:
    """(커버율 > 0 개수, 평균, 최대, 표준편차)."""
    nonzero = sum(1 for v in values if v > 0)
    mean = statistics.fmean(values) if values else 0.0
    peak = max(values) if values else 0.0
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return nonzero, mean, peak, stdev


def main() -> None:
    tracks = _load_tracks()
    combos = list(itertools.combinations(tracks, 2))
    print(f"트랙 수: {len(tracks)}, 조합 수: {len(combos)}\n")

    for job_name, job_tech in REPRESENTATIVE_JOBS.items():
        raw_vals: list[float] = []
        canonical_vals: list[float] = []
        for (_, a_tech), (_, b_tech) in combos:
            raw_vals.append(_raw_coverage(job_tech, set(a_tech) | set(b_tech)))
            canonical_vals.append(_canonical_coverage(job_tech, a_tech + b_tech))

        raw_nz, raw_mean, raw_max, raw_std = _summary(raw_vals)
        can_nz, can_mean, can_max, can_std = _summary(canonical_vals)
        total = len(combos)

        print(f"[{job_name}] 직무 기술 {len(job_tech)}개")
        print(
            f"  정합 전 (raw)      : 커버>0 {raw_nz:5d}/{total}  "
            f"평균 {raw_mean:.3f}  최대 {raw_max:.3f}  표준편차 {raw_std:.3f}"
        )
        print(
            f"  정합 후 (canonical): 커버>0 {can_nz:5d}/{total}  "
            f"평균 {can_mean:.3f}  최대 {can_max:.3f}  표준편차 {can_std:.3f}"
        )
        print()


if __name__ == "__main__":
    main()
