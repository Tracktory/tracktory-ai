# tracktory-ai

한성대 학생 대상 **AI 직무·트랙·학습 로드맵 추천 시스템**의 AI 서비스 레이어.
**LangGraph + FastAPI** 기반으로 메인 백엔드(Spring Boot)와 연동됩니다.

## 프로젝트 개요

- 메인 백엔드(Spring Boot)로부터 추천 요청을 받아 LangGraph 파이프라인 실행
- 사용자의 관심사·가치관 기반으로 **직무 → 트랙 → 과목** 추천
- 추천 결과에 대한 자연어 설명 생성 ("왜 이 트랙/과목인가")

## 기술 스택

- **Python 3.13** (`pyproject.toml` + [`uv`](https://github.com/astral-sh/uv))
- **LangGraph** — 추천 워크플로 오케스트레이션 (상태 기반 그래프)
- **FastAPI** — Spring Boot ↔ LangGraph 중계 레이어
- **GraphRAG** (RAGFlow 기반) + **LLM** — 임베딩/검색/설명 생성
- **Pydantic** — 상태·출력 스키마 검증

### 역할 분리 (서브프로젝트 관점)

| 시스템 | 담당 |
|---|---|
| Spring Boot (별도 레포) | 사용자/인증/과목 DB 등 트랜잭션성 데이터 |
| **tracktory-ai (이 레포)** | 추천 연산, LLM 호출, RAG 질의 등 AI 워크로드 |

## 디렉토리 구조

```
tracktory-ai/
├── src/tracktory/          # 파이썬 소스
│   ├── graph/              # LangGraph State, 노드, 엣지, 빌더
│   ├── prompts/            # 프롬프트 템플릿
│   ├── schemas/            # Pydantic 입출력 스키마
│   ├── services/           # LLM/RAG/DB 클라이언트 (I/O 격리)
│   └── api/                # FastAPI 라우터
├── scripts/                # 유틸 스크립트 (데이터 처리, 실험용 진입점)
├── tests/                  # 단위/통합 테스트
├── data/                   # 로컬 데이터 (gitignored)
├── logs/                   # 런타임 로그 (gitignored)
├── pyproject.toml          # 의존성 + ruff/mypy/pytest 설정
├── README.md               # 이 파일
├── CONTRIBUTING.md         # 개발 규칙 (커밋·코드·테스트 컨벤션)
└── CLAUDE.md               # Claude Code 세션용 프로젝트 컨텍스트
```

> 구체 아키텍처(노드 구성, State 스키마, 엣지 등)는 별도의 "아키텍처 결정" 단계에서 확정되며, 본 문서는 그 이후 업데이트됩니다.

## 개발 환경 세팅

### 사전 요구사항

- Python 3.13 이상
- [`uv`](https://github.com/astral-sh/uv) 설치

### 초기 세팅

```bash
# 가상환경 생성
uv venv

# 의존성 설치 (dev 의존성 포함)
uv sync

# 환경변수 파일 작성
cp .env.example .env
# .env 파일을 열어 필요한 키 입력 (LLM API 키 등)
```

## 실행

> LangGraph 구축 후 추가됩니다.

```bash
# (예정) FastAPI 서버 실행
uv run uvicorn tracktory.api.main:app --reload
```

## 개발 루틴

커밋 전에 다음 명령을 실행하세요:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run mypy src
uv run pytest
```

자세한 코드 규약·커밋 규칙은 [`CONTRIBUTING.md`](./CONTRIBUTING.md)를 참조하세요.

## 기여하기

이 레포에 코드를 추가하기 전에 [`CONTRIBUTING.md`](./CONTRIBUTING.md)를 읽어주세요. 다음 내용을 포함합니다:

- 커밋 메시지 규칙
- 파이썬/LangGraph 코드 컨벤션
- 테스트 규약
