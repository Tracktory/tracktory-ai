# Contributing to tracktory-ai

이 문서는 tracktory-ai 레포에 코드를 추가하는 팀원이 지켜야 할 규칙을 담고 있습니다.
**사람이 판단해야 하는 규칙**만 정리했습니다. 포매팅·import 순서·네이밍 등 기계가 강제할 수 있는 것은 `pyproject.toml`의 `[tool.ruff]` / `[tool.mypy]` 설정이 자동으로 잡아줍니다.

## 목차

1. [개발 루틴](#1-개발-루틴)
2. [커밋 메시지 규칙](#2-커밋-메시지-규칙)
3. [파이썬 기본 규약](#3-파이썬-기본-규약)
4. [LangGraph 규약](#4-langgraph-규약)
5. [테스트 규약](#5-테스트-규약)

---

## 1. 개발 루틴

### 도구 체인

| 목적 | 도구 | 명령어 |
|---|---|---|
| 패키지 관리 | `uv` | `uv sync`, `uv add <pkg>`, `uv add --dev <pkg>` |
| 포매터 | `ruff format` | `uv run ruff format .` |
| 린터 | `ruff check` | `uv run ruff check .` (자동 수정: `--fix`) |
| 타입 체커 | `mypy` | `uv run mypy src` |
| 테스트 | `pytest` | `uv run pytest` |
| Lock 검증 | `uv lock --check` | `uv lock --check` |

### 커밋 전 루틴 (pre-commit)

최초 1회 설치:

```bash
uv run pre-commit install
```

설치 후에는 `git commit` 시 ruff + mypy가 **자동 실행**됩니다. 수동 실행이 필요하면:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run mypy src
uv run pytest
uv lock --check
```

### 핵심 수치 (변경 시 `pyproject.toml`도 같이 수정)

- 라인 길이: **100**
- Python 버전: **3.13**
- 문자열: **double quote** (`"`)

---

## 2. 커밋 메시지 규칙

### 포맷

```
[TYPE] 작업 내용
```

### Types

| Type | 용도 |
|---|---|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `refactor` | 리팩토링 |
| `docs` | 문서 |
| `chore` | 기타 (빌드, 설정, 의존성 등) |
| `test` | 테스트 추가/수정 |
| `design` | UI/디자인 시스템 |

### 예시

```
feat: 온보딩 관심사 선택 노드 추가
fix: RAG 쿼리 임베딩 정규화 누락 수정
refactor: LangGraph state 스키마 분리
docs: README에 실행 방법 추가
chore: uv.lock 업데이트
test: 추천 파이프라인 통합 테스트 추가
```

### 원칙

- 제목은 한글 또는 영문 모두 허용, **50자 이내** 권장
- 본문이 필요하면 제목 다음 빈 줄 후에 작성
- 하나의 커밋은 하나의 논리적 변경만
- **스쿼시 머지를 전제**로 한다. 브랜치 안의 개별 커밋은 title-only로 충분하며, `Closes #N`은 브랜치 커밋이 아닌 **PR 본문**에 적는다. (squash 시 PR 본문이 main 커밋의 body가 되므로 중복 제거)

### 브랜치 네이밍

브랜치명은 `{type}/{issue-number}` 형식으로 통일한다.

- `{type}`: 커밋 타입과 동일 (`feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `design`)
- `{issue-number}`: 해당 브랜치의 작업 단위가 되는 GitHub 이슈 번호

예시:
- `feat/42` — 이슈 #42의 기능 구현 브랜치
- `fix/87` — 이슈 #87의 버그 수정 브랜치
- `chore/103` — 이슈 #103의 설정/유지보수 브랜치

### 이슈 / PR 제목

**커밋 메시지는 소문자 prefix**(`feat:`, `fix:`, ...), **이슈·PR 제목은 대문자 prefix**로 작성한다.

| 대상 | 형식 | 예시 |
|---|---|---|
| **커밋 메시지** | `{type}: 작업 내용` (소문자) | `chore: 파이썬 프로젝트 초기 세팅` |
| **이슈 제목** | `{이모지} {Type}: 작업 내용` | `🛠️ Chore: 이슈 템플릿 분리` |
| **PR 제목** | `{이모지} [{Type}] 작업 내용` | `🛠️ [Chore] 파이썬 프로젝트 초기 세팅` |

왜 분리하는가:
- 커밋 메시지는 자동 도구(changelog 생성, semantic-release 등)가 파싱하므로 소문자 convention이 표준
- 이슈·PR 제목은 사람이 시각적으로 스캔하므로 대문자가 가독성이 좋음
- 이슈는 `Type:` (콜론), PR은 `[Type]` (브래킷)으로 구분하여 GitHub 목록에서 이슈/PR을 시각적으로 즉시 구별

이슈 제목은 `.github/ISSUE_TEMPLATE/*.yml`에 이모지 + `Type:` 형태로 하드코딩되어 있으므로 템플릿을 그대로 쓰면 규약에 맞는다. PR 제목은 수동 작성이므로 `이모지 [Type] 작업 내용` 형식을 지킨다.

---

## 3. 파이썬 기본 규약

도구가 못 잡지만 팀이 지켜야 하는 것만 나열합니다.

### 3.1 타입 힌트

- **공개 함수/메서드에는 타입 힌트 필수** (mypy가 강제)
- Python 3.10+ 스타일 사용: `list[str]`, `dict[str, int]`, `X | None` (`Optional[X]` 쓰지 말 것)
- 내부 헬퍼 함수는 생략 가능하지만 가급적 붙일 것

```python
# ✅ Do
def classify_intent(query: str, threshold: float = 0.7) -> str | None:
    ...

# ❌ Don't
from typing import Optional, List
def classify_intent(query, threshold=0.7) -> Optional[str]:  # 구식 문법 + 타입 누락
    ...
```

### 3.2 Docstring

- **공개 함수/클래스**: Google 스타일 docstring 권장
- 내부 헬퍼: 한 줄 요약 허용, 자명하면 생략 가능
- **"무엇을 하는가"가 아니라 "왜 존재하는가"를 쓸 것**. 코드만 봐도 알 수 있는 내용은 쓰지 않는다.

```python
def retrieve_tracks(state: GraphState) -> dict:
    """관심사 임베딩으로 후보 트랙을 top-k 검색한다.

    RAGFlow에서 트랙 메타데이터를 받아오므로 네트워크 호출이 발생한다.
    실패 시 빈 리스트를 반환하며, 재시도는 상위 라우팅 노드에서 처리한다.

    Args:
        state: 최소 `user_interests` 필드가 채워져 있어야 한다.

    Returns:
        {"retrieved_tracks": [...]} 형태의 부분 상태.
    """
```

### 3.3 네이밍과 가시성

- `_leading_underscore` → 모듈 내부 전용. 외부에서 import 금지.
- 상수는 **모듈 최상단**에 `UPPER_SNAKE_CASE`로.
- 모호한 약어 피하기: `cfg`보다 `config`, `res`보다 `result`. (단 `id`, `db`, `api`처럼 관용화된 건 OK)

### 3.4 예외 처리

- **맨 `except:` 금지**, `except Exception` 남발 금지. 잡는 예외 타입을 명시한다.
- 예외를 삼키지 말 것. 재발생 또는 로깅 후 상위로 전파.
- 비즈니스 예외는 전용 클래스 정의 (`RecommendationError` 등).

```python
# ✅ Do
try:
    result = call_llm(prompt)
except llm.RateLimitError as e:
    logger.warning("LLM rate limit hit: %s", e)
    raise

# ❌ Don't
try:
    result = call_llm(prompt)
except:  # 어떤 예외든 삼킴
    result = None
```

---

## 4. LangGraph 규약

이 프로젝트의 핵심. 상위 아키텍처 철학은 `CLAUDE.md`의 "아키텍처 철학" 섹션 참조.

### 4.1 State 스키마

- **`TypedDict` + `Annotated`를 기본**으로 사용. 복잡한 검증이 필요하면 Pydantic `BaseModel`.
- **불변 입력 필드와 누적/변경 필드를 반드시 구분하고 주석으로 표시**한다.
- reducer가 필요한 필드(append/merge 등)는 `Annotated[..., reducer]`로 명시.

```python
# ✅ Do
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    # --- 입력 (불변) ---
    user_id: str
    user_interests: list[str]

    # --- 누적 (노드가 append) ---
    messages: Annotated[list, add_messages]
    retrieved_docs: Annotated[list[dict], lambda a, b: a + b]

    # --- 갱신 (노드가 overwrite) ---
    recommended_jobs: list[dict]
    explanation: str | None
```

### 4.2 노드 작성

**규칙 1 — 시그니처 통일**: 모든 노드는 `def node_name(state: GraphState) -> dict` 형태.

**규칙 2 — 부분 상태 반환**: 전체 `state`를 리턴하지 말고, **변경된 필드만** 담은 dict를 리턴한다.

**규칙 3 — 단일 책임**: 하나의 노드는 하나의 논리 연산. "LLM 호출 + DB 저장"처럼 섞지 않는다.

**규칙 4 — 동사형 함수명**: `classify_intent`, `retrieve_tracks`, `generate_explanation`.

```python
# ✅ Do — 변경된 필드만 리턴, 순수한 단일 책임
def retrieve_tracks(state: GraphState) -> dict:
    tracks = rag_client.search(state["user_interests"], top_k=10)
    return {"retrieved_tracks": tracks}

# ❌ Don't — 전체 state 리턴 + 책임 섞임
def retrieve_and_explain(state: GraphState) -> GraphState:
    state["retrieved_tracks"] = rag_client.search(...)
    state["explanation"] = llm.invoke(...)  # 책임 2개
    return state
```

### 4.3 엣지 / 라우팅

- 조건부 엣지 함수는 **노드와 분리된 파일/섹션**에 둔다 (`edges.py` 권장).
- **반환 타입은 `Literal[...]`로 고정**한다. 오타 방지 + mypy 검증.
- 라우팅 함수는 side effect 없이 state만 읽고 문자열만 반환한다.

```python
from typing import Literal

# ✅ Do
def route_after_retrieval(state: GraphState) -> Literal["generate", "retry", "fallback"]:
    if not state["retrieved_tracks"]:
        return "retry"
    if len(state["retrieved_tracks"]) < 3:
        return "fallback"
    return "generate"

# ❌ Don't — 문자열 자유형, 오타 발생 시 런타임에야 발견
def route_after_retrieval(state):
    if not state["retrieved_tracks"]:
        return "retyr"  # 오타 → 런타임 에러
    return "generate"
```

### 4.4 부작용 격리

**규칙**: LLM 호출, RAG 질의, DB/네트워크 I/O는 **전용 노드에만** 둔다.

- 순수 계산 노드는 I/O를 하지 않는다 (테스트에서 mock 없이 검증 가능해야 함).
- 노드 안에서 LLM/RAG 클라이언트를 전역 import로 쓰지 말고, **config 또는 생성자로 주입**한다 → 테스트에서 교체 가능.

```python
# ✅ Do — 의존성 주입
class RetrieveTracksNode:
    def __init__(self, rag_client: RagClient) -> None:
        self._rag = rag_client

    def __call__(self, state: GraphState) -> dict:
        return {"retrieved_tracks": self._rag.search(state["user_interests"])}

# ❌ Don't — 전역 import, mock 불가
from src.services.rag import rag_client  # 전역 싱글톤

def retrieve_tracks(state: GraphState) -> dict:
    return {"retrieved_tracks": rag_client.search(state["user_interests"])}
```

### 4.5 프롬프트 관리

- **프롬프트를 코드 안에 인라인으로 박지 않는다**. `src/tracktory/prompts/` 아래에 별도 모듈로 분리.
- LangChain `ChatPromptTemplate` / `PromptTemplate` 사용. f-string 직접 조립 금지 (escape 문제).
- 프롬프트 변경은 커밋 단위로 독립시킨다 → diff 리뷰 용이.

```python
# src/tracktory/prompts/explanation.py
from langchain_core.prompts import ChatPromptTemplate

EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "너는 한성대 학생의 진로를 추천하는 상담사야..."),
    ("user", "관심사: {interests}\n추천 트랙: {tracks}\n이유를 설명해줘."),
])
```

### 4.6 LLM 출력 검증

- LLM의 구조화 출력은 **Pydantic 스키마로 받고 `with_structured_output` 사용**.
- 자유 텍스트를 정규식/`json.loads`로 파싱하지 말 것.
- 파싱 실패 시 재시도 또는 폴백 노드 경로를 명시.

```python
from pydantic import BaseModel, Field

class JobRecommendation(BaseModel):
    job_name: str = Field(..., description="추천 직무명")
    reason: str = Field(..., description="추천 근거")
    confidence: float = Field(..., ge=0.0, le=1.0)

structured_llm = llm.with_structured_output(JobRecommendation)
result: JobRecommendation = structured_llm.invoke(prompt)
```

---

## 5. 테스트 규약

### 5.1 노드 단위 테스트

- 노드를 **순수 함수처럼** 테스트: state in → dict out 비교.
- LLM/RAG가 들어간 노드는 **반드시 mock**. 실제 호출은 integration 테스트로 분리.
- 파일 위치: `tests/unit/test_<node_name>.py`

```python
def test_retrieve_tracks_returns_top_k(mocker):
    mock_rag = mocker.Mock()
    mock_rag.search.return_value = [{"track": "빅데이터"}]
    node = RetrieveTracksNode(rag_client=mock_rag)

    result = node({"user_interests": ["데이터 분석"]})

    assert result == {"retrieved_tracks": [{"track": "빅데이터"}]}
    mock_rag.search.assert_called_once_with(["데이터 분석"])
```

### 5.2 그래프 통합 테스트

- `graph.invoke(initial_state)` 호출 후 최종 state 검증.
- LLM 호출은 mock 또는 fixture로 응답 고정 (CI 비용/flakiness 방지).
- 파일 위치: `tests/integration/test_graph_<scenario>.py`

### 5.3 원칙

- **`print` 금지**, `logging` 사용.
- 테스트가 실제 네트워크를 타면 `@pytest.mark.integration`으로 표시.
- CI에서는 `pytest -m "not integration"`로 기본 실행.

---

## 변경 이력

| 버전 | 날짜 | 변경자 | 변경 내용 |
|---|---|---|---|
| 0.4 | 2026-05-05 | 이재원 | §1 도구 체인·수동 실행 명령에 `uv lock --check` 추가. pre-commit 자동 검증 + `.gitattributes` `merge=binary` 와 함께 `uv.lock` 손상 차단. |
| 0.3 | 2026-04-19 | 이재원 | §1 pre-commit 설치 단계 추가. |
| 0.2 | 2026-04-13 | 이재원 | §2 브랜치 네이밍·이슈/PR 제목 규칙 추가. PR 제목 `[Type]` 브래킷 형식 도입. squash merge 원칙 명시. |
| 0.1 | 2026-04-10 | 이재원 | 초안. ruff + mypy + LangGraph 규칙 확정. |
