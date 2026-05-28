# tracktory-ai

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/Package-uv-7C3AED?logo=astral&logoColor=white)
![Ruff](https://img.shields.io/badge/Linter-Ruff-D7FF64?logo=ruff&logoColor=black)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C?logo=langgraph&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)

> 전체 프로젝트 개요: [**Tracktory/tracktory**](https://github.com/Tracktory/tracktory)

**전공과 직무를 선택하고 싶지만, 아직 무엇을 좋아하고 어떤 길을 가야 할지 모르는 학생들을 위한 AI 학습경로 가이드.**

학생의 관심사·흥미·가치관을 입력받아 적합한 **직무를 발견**하고, 그에 맞는 **트랙 조합을 추천**하며, **학기별 수강 로드맵**까지 자연어로 설명합니다.

이 레포는 추천 파이프라인의 AI 서비스 레이어(LangGraph + FastAPI)를 담당합니다.

---

## 팀 구성

<div align="center">

| <img src="https://github.com/jwon0523.png" width="120"> | <img src="https://github.com/ThreeeJ.png" width="120"> | <img src="https://github.com/parkseonghun598.png" width="120"> | <img src="https://github.com/J2H3233.png" width="120"> |
|:---:|:---:|:---:|:---:|
| **이재원** (팀장) | **정종진** | **박성훈** | **전종현** |
| 추천 시스템 + 트랙 시너지 알고리즘 | 챗봇 에이전트 | 데이터 처리 + 프론트엔드 | 논문 + 임베딩/RAG 최적화 |
| LangGraph 추천 파이프라인 | LangGraph 챗봇 노드 + 직무-역량 매핑 | 크롤링·전처리 | RAGFlow 파이프라인·임베딩 |
| Spring Boot ↔ FastAPI 통합 | Spring Boot ↔ FastAPI 통합 | React Native | Spring Boot |
| [@jwon0523](https://github.com/jwon0523) | [@ThreeeJ](https://github.com/ThreeeJ) | [@parkseonghun598](https://github.com/parkseonghun598) | [@J2H3233](https://github.com/J2H3233) |

</div>

---

## 기술 스택

| 기술 | 역할 | 설명 |
|---|---|---|
| **Python 3.13** + **uv** | 런타임 + 패키지 관리 | AI/ML 생태계 표준. uv는 pip 대비 10~100배 빠른 의존성 해결 |
| **LangGraph** | 추천 워크플로 오케스트레이션 | 상태 기반 그래프로 복잡한 다단계 추천 파이프라인을 노드 단위로 분리·테스트 가능 |
| **FastAPI** | Spring Boot ↔ LangGraph 중계 | 비동기 성능 + 자동 OpenAPI 문서 생성. Spring Boot가 트랜잭션을, FastAPI가 AI 워크로드를 분담 |
| **LightRAG** (RAGFlow 내 구현) | 임베딩 검색 + 관계 기반 탐색 | 단순 벡터 유사도가 아닌 과목-트랙-직무 간 **관계 그래프** 기반 검색으로 추천 정확도 향상 |
| **Pydantic** | 상태·출력 스키마 검증 | LLM 구조화 출력을 `with_structured_output`으로 자동 검증. 런타임 타입 안전성 확보 |
| **Ruff** + **mypy** | 린터·포매터 + 타입 체커 | Ruff는 black/isort/flake8/pylint 통합 대체. mypy로 공개 함수 타입 힌트 강제 |

---

## 디렉토리 구조

```
tracktory-ai/
├── src/tracktory/
│   ├── common/             # 도메인 모델, 카테고리 분류, 기술 키워드
│   ├── crawler/            # 채용공고 크롤러 (원티드, 사람인)
│   │   ├── wanted/         # 원티드 크롤러 (주력)
│   │   ├── saramin/        # 사람인 크롤러 (초기 스캐폴드)
│   │   └── preprocessing/  # 채용공고 원시 데이터 → 정제 데이터
│   ├── rag/                # RAGFlow 전처리·적재·검색 (한성대 학사 데이터)
│   │   ├── preprocessing/  # data/raw/hansung/* → RAGFlow 적재용 청크 텍스트
│   │   └── client.py       # RAGFlow SDK wrapper — graph/ 노드가 주입받아 사용
│   ├── graph/              # (예정) LangGraph State, 노드, 엣지
│   ├── prompts/            # (예정) 프롬프트 템플릿
│   └── api/                # (예정) FastAPI 라우터
├── scripts/                # 유틸 스크립트 (크롤링 실행, 데이터 병합)
├── tests/                  # 단위/통합 테스트
├── data/                   # 데이터 (gitignore — 팀 채널로 동기화)
│   ├── raw/                # 원본 수집 데이터 (한성대 학사, 원티드 채용공고)
│   │   └── hansung/        # 한성대 학사 (트랙·강의·강의계획서)
│   ├── processed/          # 전처리·정제 결과
│   ├── eval/               # 평가용 골드셋 (쿼리셋 등)
│   │   └── queryset/
│   └── _legacy/            # 현 파이프라인 미사용, 참조 보존용
├── pyproject.toml          # 의존성 + ruff/mypy/pytest 설정
├── CONTRIBUTING.md         # 개발 규칙 (커밋·코드·테스트 컨벤션)
└── CLAUDE.md               # Claude Code 세션용 프로젝트 컨텍스트
```

---

## 시작하기

### 사전 요구사항

- Python 3.13 이상
- [`uv`](https://github.com/astral-sh/uv) 설치

### 설치 및 실행

```bash
# 1. 클론
git clone https://github.com/Tracktory/tracktory-ai.git
cd tracktory-ai

# 2. 가상환경 + 의존성
uv venv && uv sync

# 3. 환경변수
cp .env.example .env
# .env 파일을 열어 필요한 키 입력 (LLM API 키 등)
```

```bash
# (예정) FastAPI 서버 실행
uv run uvicorn tracktory.api.main:app --reload
```

---

## 개발 루틴

커밋 전에 다음 명령을 실행하세요:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run mypy src
uv run pytest
```

자세한 코드 규약·커밋 규칙은 [`CONTRIBUTING.md`](./CONTRIBUTING.md)를 참조하세요.

---

## 기여하기

이 레포에 코드를 추가하기 전에 [`CONTRIBUTING.md`](./CONTRIBUTING.md)를 읽어주세요.

- 커밋 메시지 규칙 (`feat:`, `fix:`, `chore:` ...)
- 브랜치 네이밍 (`{type}/{issue-number}`)
- 파이썬 / LangGraph 코드 컨벤션
- 테스트 규약
