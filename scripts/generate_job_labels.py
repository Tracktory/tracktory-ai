"""공고별 온보딩 매칭 라벨(환경·관심분야)을 산출한다.

온보딩 입력 중 공고 본문에 대응물이 없던 두 축을 doc-side 라벨로 만든다.

  - 환경(company scale): 원티드 detail_tags → 대기업 / 스타트업
      · 대기업    = 301명 이상 규모 태그
      · 스타트업  = 설립 3년 이하 / 유니콘 / 투자사 / 50명 이하
    (원티드 코퍼스가 스타트업 편중이라 대기업은 중견 규모까지 흡수해 폭을 넓힌다.)

  - 관심분야(14 세부 기술 토픽): 공고 기술스택·본문 텍스트 → 토픽 키워드 매칭

산출물: data/processed/job_env_labels.json
    { source_id: {"env": [...], "interests": [...], "category": ...} }

jobs.py(build_jobs_from_csv)가 이 산출물을 읽어 공고 .txt 하단에
``[관심분야] ...`` / ``[환경] ...`` 줄로 주입한다. 라벨은 원raw 태그가 아니라
온보딩 표면형이어야 질의 토큰과 매칭된다.

실행:
    uv run python scripts/generate_job_labels.py
"""

from __future__ import annotations

import json
import re

import pandas as pd

from tracktory.common.config import CommonConfig

_META_CSV = CommonConfig.DATA_RAW_DIR / "wanted_metadata_all.csv"
_CLEAN_CSV = CommonConfig.DATA_RAW_DIR / "hansung" / "wanted_cleaned.csv"
_OUTPUT = CommonConfig.DATA_PROCESSED_DIR / "job_env_labels.json"

# ── 환경 검출 규칙 (detail_tags 집합 매치) ──────────────────────────────────
_BIG_TAGS = {"301~1,000명", "1,001~10,000명", "10,001명이상", "대기업"}
_STARTUP_TAGS = {
    "설립3년이하",
    "예비 유니콘",
    "아기 유니콘",
    "누적투자100억이상",
    "50명이하",
    "매쉬업벤처스 투자기업",
    "스파크랩 투자기업",
    "더벤처스 투자기업",
    "퓨처플레이 투자기업",
    "IBK창공 투자기업",
}

# ── AI 연구원 검출 (환경 라벨 "AI 연구원") ─────────────────────────────────
# 회사규모(대기업/스타트업)와 다른 축 — 연구직 vs 개발직. Wanted 회사태그로는
# 안 잡혀 공고 제목·요건 텍스트로 검출한다. 단 전체 공고에 "연구" 키워드를 걸면
# 반도체·임베디드 연구 노이즈가 섞이므로, AI/데이터 카테고리로 스코프를 좁힌다.
_AI_DATA_CATEGORIES = {"AI/ML", "데이터사이언스", "데이터분석", "데이터엔지니어"}
_RESEARCH_RE = re.compile(r"연구원|Research|Scientist|논문|Ph\.?D", re.IGNORECASE)
_RESEARCH_TEXT_FIELDS = ["title", "requirements", "preferred"]


def _is_ai_research(category: str, text: str) -> bool:
    """AI/데이터 카테고리 공고 중 연구직 신호가 있으면 True."""
    return category in _AI_DATA_CATEGORIES and bool(_RESEARCH_RE.search(text))


# ── 관심분야 검출 규칙 (텍스트 부분일치, 온보딩 14 토픽) ─────────────────────
_INTEREST_KEYWORDS: dict[str, list[str]] = {
    "데이터분석": [
        "tableau",
        "power bi",
        "looker",
        "superset",
        "metabase",
        "bigquery",
        "pandas",
        "데이터 분석",
        "데이터분석",
        "analytics",
    ],
    "머신러닝/딥러닝": [
        "pytorch",
        "tensorflow",
        "scikit",
        "sklearn",
        "keras",
        "머신러닝",
        "딥러닝",
        "machine learning",
        "deep learning",
        "xgboost",
        "lightgbm",
    ],
    "웹개발": [
        "react",
        "vue",
        "angular",
        "svelte",
        "next.js",
        "nuxt",
        "html",
        "css",
        "javascript",
        "typescript",
        "프론트",
        "frontend",
    ],
    "앱개발": [
        "android",
        "ios",
        "kotlin",
        "swift",
        "flutter",
        "react native",
        "jetpack",
        "swiftui",
        "모바일",
        "mobile",
    ],
    "서버/백엔드": [
        "spring",
        "django",
        "fastapi",
        "express",
        "nestjs",
        "golang",
        "백엔드",
        "backend",
        "grpc",
        "msa",
        "node",
    ],
    "클라우드인프라": [
        "aws",
        "gcp",
        "azure",
        "kubernetes",
        "docker",
        "terraform",
        "helm",
        "인프라",
        "클라우드",
        "ci/cd",
        "jenkins",
    ],
    "보안": ["보안", "security", "취약점", "모의해킹", "owasp", "siem", "penetration"],
    "UI/UX": ["figma", "sketch", "zeplin", "ux", "ui/ux", "prototyp"],
    "블록체인": [
        "블록체인",
        "blockchain",
        "solidity",
        "web3",
        "smart contract",
        "스마트 컨트랙트",
        "ethereum",
    ],
    "게임개발": ["unity", "unreal", "게임", "cocos", "godot"],
    "IoT/임베디드": [
        "임베디드",
        "embedded",
        "펌웨어",
        "firmware",
        "rtos",
        "iot",
        "arduino",
        "raspberry",
    ],
    "자연어처리": ["nlp", "llm", "자연어", "langchain", " rag", "transformer", "bert", "prompt"],
    "컴퓨터비전": [
        "opencv",
        "컴퓨터 비전",
        "컴퓨터비전",
        "yolo",
        "detection",
        "segmentation",
        "vision",
    ],
    "알고리즘": ["알고리즘", "algorithm", "자료구조"],
}

_INTEREST_TEXT_FIELDS = ["title", "tech_stacks", "responsibilities", "requirements", "category"]


def _env_labels(tag_str: object) -> list[str]:
    """detail_tags 문자열에서 환경 라벨(대기업/스타트업)을 판정한다."""
    if not isinstance(tag_str, str):
        return []
    tags = {t.strip() for t in tag_str.split("|") if t.strip()}
    out: list[str] = []
    if tags & _BIG_TAGS:
        out.append("대기업")
    if tags & _STARTUP_TAGS:
        out.append("스타트업")
    return out


def _interest_labels(text: str) -> list[str]:
    """공고 텍스트에서 매칭되는 관심분야 토픽을 선언 순서대로 반환한다."""
    lowered = text.lower()
    return [topic for topic, kws in _INTEREST_KEYWORDS.items() if any(k in lowered for k in kws)]


def main() -> int:
    meta = pd.read_csv(_META_CSV, dtype={"source_id": str})
    clean = pd.read_csv(_CLEAN_CSV, dtype={"source_id": str})

    env_by_sid = {r["source_id"]: _env_labels(r.get("detail_tags")) for _, r in meta.iterrows()}

    labels: dict[str, dict[str, object]] = {}
    for _, row in clean.iterrows():
        sid = row["source_id"]
        category = str(row.get("category", ""))
        text = " ".join(str(row.get(c, "")) for c in _INTEREST_TEXT_FIELDS)
        env = list(env_by_sid.get(sid, []))
        research_text = " ".join(str(row.get(c, "")) for c in _RESEARCH_TEXT_FIELDS)
        if _is_ai_research(category, research_text):
            env.append("AI 연구원")
        labels[sid] = {
            "env": env,
            "interests": _interest_labels(text),
            "category": category,
        }

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")

    env_n = sum(1 for v in labels.values() if v["env"])
    int_n = sum(1 for v in labels.values() if v["interests"])
    research_n = sum(1 for v in labels.values() if "AI 연구원" in v["env"])
    print(f"labels written: {_OUTPUT} ({len(labels)} postings)")
    print(f"  with env label      : {env_n}")
    print(f"    of which AI 연구원 : {research_n}")
    print(f"  with interest label : {int_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
