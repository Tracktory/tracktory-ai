# N4 Track Synergy Node — Architecture Design

| 항목 | 내용 |
|---|---|
| **Status** | Design (구현 전 설계 명세) |
| **Owner** | 이재원 |
| **Last Updated** | 2026-04-29 |
| **Audience** | tracktory-ai 팀 (구현·리뷰), 향후 ADR 승격 시 reviewer |

> 본 문서는 N4 트랙 시너지 노드의 **구현 명세** 입니다. 정책 결정의 근거(왜 이 알고리즘인가)는 ADR 디렉토리(`../adr/`)와 본 문서 본문에 inline 으로 보존됩니다. 코드 작성 착수 전, 이 문서를 1차 신뢰 소스로 사용합니다.

---

## 1. Overview

N4 는 Online 추천 파이프라인에서 **트랙 조합 추천**을 담당하는 LangGraph 노드입니다. 직무 매칭 노드(N3)에서 넘어온 직무 후보 3~5개를 입력으로 받아, 학생에게 추천할 **주 추천 2개(primary) + 보조 추천 5개(secondary)** 를 출력합니다.

### 1.1 N4 가 풀어야 할 문제

한성대 전면 트랙제(47개 트랙)에서 두 트랙 조합의 가능한 수는 1,081개입니다. 학생이 자신의 관심사·직무에 맞춰 이를 모두 비교 평가하는 것은 불가능하므로, 다음 3가지 알고리즘 요소가 결합된 추천이 필요합니다.

1. **Synergy Score** — 두 트랙 조합이 학생의 직무 후보군에 얼마나 적합한지 정량화
2. **Multi-tier Similarity** — 두 조합이 한성대 학사 구조(단과대→학부→트랙→과목)상 얼마나 유사한지 측정
3. **Slot-based Diversity Guarantee** — 학과 경계를 넘는 "이색 조합"이 추천 결과에 항상 노출되도록 보장 (단순 다양성 알고리즘만으로는 확률적 보장이라 누락 가능)

### 1.2 핵심 설계 원칙

본 노드는 `CLAUDE.md` 의 아키텍처 철학 5개 원칙을 따릅니다. 자세한 매핑은 §8 참조.

- State-first 설계 (LangGraph State 스키마 선확정)
- 단일 책임 노드 (계산·선택·I/O 단계 분리)
- 부작용 격리 (LLM·RAG 호출은 진입점 1곳에 집중)
- 명시적 엣지 조건 (`Literal[...]` 타입)
- 검증 가능한 출력 (Pydantic 스키마)

### 1.3 단일 임베딩 공간 의존성

본 노드의 트랙 메타 임베딩은 ADR-0001(`docs/adr/0001-single-embedding-boundary.md`)에서 정의한 **단일 임베딩 설정**을 사용합니다. 다른 노드(N2 Profile Embedding, N3 Job Matching)와 동일한 모델·차원·정규화 설정을 공유해야 코사인 유사도 비교가 의미를 가집니다.

---

## 2. Input / Output Interface

N4 는 LangGraph 의 `GraphState` 를 입력으로 받아 부분 상태를 반환합니다 (`def n4_track_synergy(state: GraphState) -> dict`). State 필드는 다음과 같이 분리됩니다.

### 2.1 Input — N3 / N2 / N1 로부터 받는 State 필드

| 필드 | 타입 | 출처 | 설명 |
|---|---|---|---|
| `job_candidates` | `list[JobCandidate]` | N3 (Job Matching) | 직무 후보 3~5개 (직무명, 채용공고 기술스택, 역량 태그) |
| `profile_vector` | `list[float]` | N2 (Profile Embedding) | 사용자 프로필 임베딩. 본 노드에서는 직접 사용하지 않으나 fallback 추천 안전장치로 참조 가능 |
| `current_tracks` | `list[str]` | N1 (Input Normalization) | 2학년+ 사용자의 이수 트랙 ID. 1학년은 `[]` |
| `college_id` | `str` | N1 | 사용자 소속 단과대 ID. 기능명세 HM-001 의 "1트랙은 주전공 소속 트랙 필수" 제약 적용 |

### 2.2 Output — N4 가 State 에 추가하는 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `primary_combos` | `list[RankedCombo]` | 시너지 점수 상위 2개 트랙 조합 |
| `secondary_combos` | `list[RankedCombo]` | 보조 추천 5개 (cross-college 1개 예약 + MMR 4개) |
| `slot3_fallback_triggered` | `bool` | cross-college 슬롯 fallback 발생 여부 |
| `slot3_fallback_level` | `Literal["T2", "MMR"] \| None` | fallback 단계 (`None` = fallback 없음, `T2` = 학부 cross-dept 로 relax, `MMR` = cross-dept 도 0 → MMR 흘림) |

LangGraph 의 "부분 상태 반환" 원칙에 따라 N4 는 N3 이전 필드를 수정하지 않고 위 4개 필드만 새로 추가합니다.

---

## 3. Data Classes

순수 계산 노드와 I/O 노드 분리를 위해 도메인 모델을 먼저 정의합니다. 모든 모델은 Pydantic `BaseModel` 로 LLM·RAG 출력 검증과 동일한 패턴을 사용합니다.

### 3.1 Track — 4-tier hierarchy 표현

```python
class Track(BaseModel):
    """한성대 단일 트랙 메타데이터 + 4-tier hierarchy 식별자."""

    track_id: str
    track_name: str

    # 4-tier hierarchy
    college_id: str       # T1: 단과대 (예: "IT공과대학")
    department_id: str    # T2: 학부 (예: "컴퓨터공학부")
    major_id: str         # T3: 전공/트랙 그룹 (single-track 학과는 department_id 와 동일 — degenerate)
    course_ids: list[str] # T4: 과목 ID 목록 (overlap_ratio 계산 입력)

    # 트랙 메타 임베딩 (ADR-0001 단일 임베딩 공간 사용)
    meta_embedding: list[float]

    # 트랙 소개 원문 (N6 LLM 자연어 설명 컨텍스트로 전달)
    meta_text: str
```

**4-tier 구조의 이유**: 한성대는 단과대 → 학부 → 트랙 의 위계 구조이며, 일부 학과(AI응용학과, 융합보안학과 등)는 단일 트랙만 운영합니다. `major_id == department_id` 로 처리하면 single-track 학과도 동일한 데이터 모델로 표현 가능합니다.

**과목 overlap (T4)**: 두 트랙이 학습 내용을 얼마나 공유하는지 계산하기 위한 입력. `course_ids` 의 교집합/합집합 비율로 산출.

### 3.2 JobCandidate — N3 직무 매칭 결과

```python
class JobCandidate(BaseModel):
    """N3 직무 매칭 결과 단건."""

    job_name: str
    tech_stacks: list[str]      # 채용공고 기술스택 (시너지 점수의 coverage 항 입력)
    competency_tags: list[str]  # 역량 태그 (시너지 점수의 complementarity 항 입력)
    match_score: float          # N3 의 코사인 유사도 (참조용)
```

**NCS 데이터 미활용**: 본 프로젝트는 NCS 직무역량 데이터 수집을 미활용 결정했습니다(2026-04-04). 따라서 직무 역량 표현은 `tech_stacks`(채용공고 기술스택) + `competency_tags`(채용공고에서 추출한 역량 태그) 만 사용합니다. 향후 NCS 수집 시 합집합으로 확장 가능.

### 3.3 TrackCombo — 트랙 조합 단위

```python
class TrackCombo(BaseModel):
    """트랙 조합 단위 (track_a + track_b)."""

    track_a: Track
    track_b: Track
```

**1학년 vs 2학년+ 분기**: `track_a` 는 항상 사용자의 단과대 소속 트랙(`track_a.college_id == user.college_id`) 으로 제한합니다. 이는 기능명세상의 "1트랙은 주전공 소속 트랙 필수" 제약과 정합합니다. 1학년(`current_tracks == []`)은 `track_a` 후보가 단과대 전체 트랙, 2학년+ 는 `current_tracks[0]` 으로 고정됩니다. 이 제약은 후보 생성 단계(§4 Stage 0)에서 filtering 으로 구현하며, `TrackCombo` 모델 자체는 제약을 인코딩하지 않습니다(정책-메커니즘 분리).

### 3.4 RankedCombo — 시너지 점수 + 슬롯 분류 부착 단위

```python
class RankedCombo(BaseModel):
    """시너지 점수 + 슬롯 분류 + 순위가 붙은 트랙 조합."""

    combo: TrackCombo
    synergy_score: float = Field(ge=0.0, le=1.0)  # clip(0,1) 강제 검증
    complementarity: float
    job_coverage: float
    redundancy: float
    slot_type: Literal["primary", "cross_college", "mmr"]
    rank: int  # 1-indexed
```

`slot_type` 의 `Literal` 타입은 라우팅 함수(예: UI 라벨 분기)가 안전하게 사용할 수 있도록 컴파일 시점에 값 집합을 고정합니다.

`synergy_score: float = Field(ge=0.0, le=1.0)` 의 필드 제약은 §4 Stage 1 의 `clip(0, 1)` 결과를 모델 레벨에서 한 번 더 검증합니다 (방어적 프로그래밍).

---

## 4. Pipeline Stages — Slot Reservation Pattern 4

N4 의 알고리즘은 다음 6단계로 구성됩니다. 각 단계는 단일 책임을 가지며, I/O 는 마지막 단계(진입점)에만 집중됩니다.

### Stage 0 — 후보 생성 (순수 계산)

```
function: generate_track_combos(
    all_tracks: list[Track],
    user_college_id: str,
    current_tracks: list[str]
) -> list[TrackCombo]
```

- 전체 트랙 목록(O4 그래프에서 로드)에서 유효한 2-조합 생성
- 1트랙 주전공 제약 적용: `track_a.college_id == user_college_id`
- 1학년(`current_tracks == []`): `track_a` 후보는 사용자 단과대 전체 트랙
- 2학년+: `track_a = current_tracks[0]` 고정
- **I/O 없음 — 순수 계산** (테스트에서 mock 불필요)

### Stage 1 — 시너지 점수 계산 (순수 계산)

```
function: compute_synergy(
    combo: TrackCombo,
    job_candidates: list[JobCandidate],
    config: SynergyConfig
) -> tuple[float, float, float, float]
    # returns (synergy_score, complementarity, job_coverage, redundancy)
```

**시너지 산출 식**:

```
synergy(track_a, track_b | jobs)
  = clip(w_comp · complementarity
       + w_cov  · job_coverage
       − w_red  · redundancy,
        0, 1)
```

| 항 | 정의 | 의미 |
|---|---|---|
| `complementarity` | 두 트랙의 비중복 역량 합집합 / 전체 역량 풀 | 두 트랙이 서로 보완하는 정도 |
| `job_coverage` | 직무 후보 채용공고 기술스택 중 두 트랙 합집합으로 커버되는 비율 | 추천 직무 도달도 |
| `redundancy` | 두 트랙의 과목 overlap 비율 (T4) | 학습 내용 중복 정도 (페널티) |

**가중치 초기값** (직관 할당, S2 ablation 대상):

- `w_comp = 0.3`, `w_cov = 0.5`, `w_red = 0.2`
- 직관: 직무 도달도(cov) 가장 중요 → 보완성(comp) 다음 → 중복(red) 페널티

**`clip(0, 1)` 강제 이유**: 가중치 합산식이 정의역 [0,1] 을 벗어나면 후속 MMR 의 균형 식이 깨지므로 보호. 수식 자체를 재설계하는 대안도 검토했으나, 단순 3항 합산에서 clip 빈도가 낮을 것으로 예상되어 단순화를 선택. clip 빈도가 운영 중 높게 관측되면 가중치 ablation 으로 재조정.

- **I/O 없음 — 순수 계산**

### Stage 2 — 주 추천 선택 (Slot 1~2, 순수 계산)

```
function: select_primary(
    ranked_combos: list[tuple[TrackCombo, float, ...]],
    n: int = 2
) -> list[RankedCombo]
```

- synergy 점수 내림차순으로 상위 `n` 개 선택
- 1트랙 주전공 제약은 Stage 0 에서 이미 적용됨 → 추가 필터 없음
- 동점 처리는 첫 번째 순서(stable sort) — 결정론성 확보
- **I/O 없음**

### Stage 3 — Cross-College 슬롯 선택 (Slot 3, `logger.info` 만 허용)

```
function: select_cross_college_slot(
    all_ranked: list[tuple[TrackCombo, float, ...]],
    primary: list[RankedCombo],
    config: SynergyConfig
) -> tuple[RankedCombo | None, Literal["T2", "MMR"] | None]
    # returns (selected_combo, fallback_level)
```

**5단계 슬롯 채우기 흐름**:

1. **slot 1~2 (Stage 2 결과)**: synergy 상위 2개 = 주 추천. 1트랙 주전공 제약으로 자연스럽게 사용자 단과대 내부 조합.
2. **slot 3 (cross-college 예약)**: `is_cross_college(combo, primary) == True` AND `synergy(combo) ≥ min_cross_synergy` 만족 후보 중 synergy 최대값 선택.
3. **slot 3 fallback (cross-college 후보 0개)**: T2(학부) cross-dept 로 relax — 사용자 학부 외부의 트랙 조합. `min_cross_synergy` 임계값은 동일 유지 (변경 시 별도 ablation).
4. **그래도 0개**: slot 3 = None, fallback_level = "MMR" 반환. Stage 4 의 MMR 후보군에 흘려보냄 (UI 에서 slot 부족 명시).
5. **graduated cross-ness**: cross-college 만 hard gate, cross-dept 차등은 Stage 4 의 MMR 단계에서 sim 가중치(w_dept) 로 계속 표현.

**`is_cross_college(combo, primary)` 정의**:

```python
def is_cross_college(combo: TrackCombo, primary: list[RankedCombo]) -> bool:
    """combo 의 두 트랙 중 최소 한 개의 college_id 가
       primary 의 모든 트랙의 college_id 집합과 다르면 True."""
    primary_colleges = {
        track.college_id
        for ranked in primary
        for track in (ranked.combo.track_a, ranked.combo.track_b)
    }
    combo_colleges = {combo.track_a.college_id, combo.track_b.college_id}
    return not combo_colleges.issubset(primary_colleges)
```

**`is_cross_dept` (T2 fallback 용)**: 위 함수에서 `college_id` 를 `department_id` 로 치환한 형태.

**`min_cross_synergy: 0.3` 외부화**: `src/tracktory/config/synergy.yaml` 에서 주입. 초기값 0.3 은 "synergy 정의역 [0,1] 의 하위 1/3 컷" 직관 할당. S2 ablation 대상.

**Fallback 로깅 (이 단계에서만 허용되는 side effect)**:

```python
logger.info(
    "slot3 fallback applied",
    extra={
        "trigger": "t1_cross_college_empty",  # or "t2_cross_dept_empty"
        "fallback_level": "T2",  # or "MMR"
        "primary_combo": [(t.track_id) for ranked in primary
                          for t in (ranked.combo.track_a, ranked.combo.track_b)],
    },
)
```

- 로그 레벨 `INFO` 잠정. fallback 은 예상 케이스 (단일 단과대 학생의 cross-college 후보 부족 정상). WARNING 으로 알람 띄우면 alarm fatigue 발생.
- 운영 데이터 누적 후 빈도가 비정상적으로 높으면 WARNING 승격 검토.

**부작용 격리**: 본 함수 외 Stage 0/1/2/4 는 `logger` 호출 없음. 모든 fallback 감사 정보는 이 한 곳에 집중.

### Stage 4 — MMR (Slot 4~7, 순수 계산)

```
function: apply_mmr(
    candidates: list[tuple[TrackCombo, float, ...]],
    selected: list[RankedCombo],
    n: int,
    config: SynergyConfig
) -> list[RankedCombo]
```

**MMR 식**:

```
next = argmax_{i ∉ selected} [
    λ · synergy(i)
  − (1 − λ) · max_{j ∈ selected} sim_4tier(i, j)
]
```

`selected` 입력 = `primary + [slot3_combo]` (slot3 = None 이면 primary 만).

**`sim_4tier` 정의** (4-tier hierarchy + 트랙 메타 다양성 통합):

```
sim_4tier(combo_a, combo_b)
  = w_college    · same_T1(a, b)
  + w_department · same_T2(a, b)
  + w_major      · same_T3(a, b)
  + w_overlap    · overlap_ratio(a, b)
  + w_meta       · cos(meta_a, meta_b)
```

| 항 | 정의 | 의미 |
|---|---|---|
| `same_T1(a, b)` | 두 조합이 같은 단과대 트랙을 포함하면 1, 아니면 0 | 가장 큰 위계 신호 |
| `same_T2(a, b)` | 같은 학부 | T1 의 하위 |
| `same_T3(a, b)` | 같은 트랙 (조합 간 트랙 중복) | T2 의 하위 |
| `overlap_ratio(a, b)` | T4 과목 overlap 비율 | 가장 세밀한 학습 내용 신호 |
| `cos(meta_a, meta_b)` | 트랙 메타 텍스트 임베딩의 코사인 유사도 | 도메인/서사 거리 |

**가중치 단조 제약**: `w_college ≥ w_department ≥ w_major` (상위 tier 가 더 강한 신호). `w_meta` 는 별도 축이며 tier 제약 외.

**single-track 학과 처리**: T2=T3 degenerate 케이스에서 `same_T2 + same_T3` 의 효과적 가중치는 `w_department + w_major` 합산. 데이터 모델상 자연스럽게 처리됨.

**`w_meta · cos(meta_a, meta_b)` 의 부호 일관성**: cos 가 높으면 sim 이 높음 → MMR 의 다양성 항이 cos 가 낮은(즉, 도메인 거리가 먼) 조합을 선호 → cross-domain 조합이 자연스럽게 선택됨. 트랙 메타 다양성을 시너지 식 외부 보너스로 가산하는 대안도 검토했으나, 시너지 식이 단순한 3항 형태로 유지되고 부호 반전 없이 sim 방향성이 일관되는 본 통합 방식을 선택.

**가중치 초기값** (S2 ablation 대상):

- `w_college = 0.4`, `w_department = 0.3`, `w_major = 0.2`, `w_overlap = 0.1`, `w_meta = 0.05`
- `λ = 0.6` (synergy vs diversity 균형, 약간 synergy 쪽으로)

**MMR 알고리즘 출처**: Carbonell & Goldstein (1998). 정보 검색 분야 표준 다양화 알고리즘.

- **I/O 없음 — 순수 계산** (트랙 메타 임베딩은 `Track` 모델에 이미 포함)

### Stage 5 — N4 노드 진입점 (I/O 격리)

```python
class N4TrackSynergyNode:
    def __init__(
        self,
        rag_client: RagFlowClient,
        embedding_loader: EmbeddingLoader,
        synergy_config: SynergyConfig,
    ) -> None:
        self._rag = rag_client
        self._embed = embedding_loader
        self._config = synergy_config

    def __call__(self, state: GraphState) -> dict:
        # 1. RAG 에서 트랙 메타데이터 로드 (I/O)
        all_tracks = self._rag.fetch_all_tracks_with_meta(self._embed)

        # 2. Stage 0~4 순수 함수 체이닝
        combos = generate_track_combos(
            all_tracks, state["college_id"], state["current_tracks"]
        )
        ranked = [
            (combo, *compute_synergy(combo, state["job_candidates"], self._config))
            for combo in combos
        ]
        primary = select_primary(ranked, n=self._config.slots.primary_count)

        slot3, fallback_level = select_cross_college_slot(
            ranked, primary, self._config
        )

        excluded_ids = {ranked.combo.track_a.track_id for ranked in primary} | \
                       {ranked.combo.track_b.track_id for ranked in primary}
        if slot3:
            excluded_ids.add(slot3.combo.track_a.track_id)
            excluded_ids.add(slot3.combo.track_b.track_id)
        mmr_candidates = [r for r in ranked if r[0].track_a.track_id not in excluded_ids]

        secondary_count = self._config.slots.secondary_count
        n_mmr = secondary_count - (1 if slot3 else 0)
        mmr_picks = apply_mmr(
            mmr_candidates,
            selected=primary + ([slot3] if slot3 else []),
            n=n_mmr,
            config=self._config,
        )

        secondary = ([slot3] if slot3 else []) + mmr_picks

        return {
            "primary_combos": primary,
            "secondary_combos": secondary,
            "slot3_fallback_triggered": slot3 is None or fallback_level is not None,
            "slot3_fallback_level": fallback_level,
        }
```

**I/O 격리 요약**:

| 함수 | I/O | 단일 책임 |
|---|---|---|
| `generate_track_combos` | 없음 | 후보 생성 + 1트랙 주전공 제약 |
| `compute_synergy` | 없음 | 시너지 3항 + clip |
| `select_primary` | 없음 | 슬롯 1~2 |
| `select_cross_college_slot` | `logger.info` only | 슬롯 3 + T2 fallback |
| `apply_mmr` | 없음 | 슬롯 4~7 |
| `n4_track_synergy` (`__call__`) | RAG + embedding loader | I/O 진입점 격리 |

---

## 5. Dependencies

### 5.1 의존성 주입 패턴

LangGraph 노드는 클래스로 구현하되 LLM/RAG 클라이언트를 **생성자 주입** 으로 받습니다. 전역 import 금지.

```python
n4_node = N4TrackSynergyNode(
    rag_client=ragflow_client,
    embedding_loader=embedding_loader,
    synergy_config=load_synergy_config("src/tracktory/config/synergy.yaml"),
)

graph.add_node("n4_track_synergy", n4_node)
```

테스트 시 mock 객체 주입으로 RAG/embedding 호출을 차단합니다.

### 5.2 신규 패키지 의존성

본 노드 구현에 필요한 패키지 (`pyproject.toml` 추가 필요):

| 패키지 | 용도 |
|---|---|
| `langgraph` | 그래프 오케스트레이션 |
| `langchain-core` | State, BaseMessage 타입 |
| `pyyaml` | `synergy.yaml` 로드 |
| `numpy` | cos 유사도 등 벡터 연산 |

### 5.3 설정 파일

```
src/tracktory/config/
├── synergy.yaml      # 본 노드의 가중치·임계값 (본 PR에서 스켈레톤 생성)
└── embedding.yaml    # ADR-0001 단일 임베딩 설정 (별도 후속 작업)
```

---

## 6. synergy.yaml Schema

가중치·임계값을 코드 상수가 아닌 외부 yaml 로 분리합니다. 가중치 변경(ablation 실험)이 코드 수정 없이 가능하도록 합니다.

```yaml
# src/tracktory/config/synergy.yaml

# ── 시너지 점수 가중치 (Stage 1) ─────────────────────────────────────
weights:
  complementarity: 0.3
  coverage: 0.5
  redundancy: 0.2

# ── sim_4tier 가중치 (Stage 4 MMR 다양성 측정) ───────────────────────
# 단조 제약: w_college >= w_department >= w_major
similarity:
  w_college: 0.4
  w_department: 0.3
  w_major: 0.2
  w_course_overlap: 0.1
  w_meta: 0.05

# ── MMR 파라미터 (Stage 4) ───────────────────────────────────────────
mmr:
  lambda: 0.6

# ── Slot Reservation (Stage 2 + 3 + 4) ──────────────────────────────
slots:
  primary_count: 2
  secondary_count: 5
  cross_college_reserved: 1
  min_cross_synergy: 0.3
```

**`SynergyConfig` Pydantic 모델** (Phase C 코드 작성 시 생성):

```python
class WeightsConfig(BaseModel):
    complementarity: float
    coverage: float
    redundancy: float

class SimilarityConfig(BaseModel):
    w_college: float
    w_department: float
    w_major: float
    w_course_overlap: float
    w_meta: float

class MMRConfig(BaseModel):
    lambda_: float = Field(alias="lambda", ge=0.0, le=1.0)

class SlotsConfig(BaseModel):
    primary_count: int
    secondary_count: int
    cross_college_reserved: int
    min_cross_synergy: float = Field(ge=0.0, le=1.0)

class SynergyConfig(BaseModel):
    weights: WeightsConfig
    similarity: SimilarityConfig
    mmr: MMRConfig
    slots: SlotsConfig
```

---

## 7. Test Strategy

`CONTRIBUTING.md §5` 의 테스트 규약을 따릅니다. 단위 → 통합 → edge case 순서.

### 7.1 단위 테스트 (`tests/unit/graph/test_n4_*.py`)

| 테스트 | 대상 | 검증 |
|---|---|---|
| `test_compute_synergy_clip_upper` | `compute_synergy` | comp=1, cov=1, red=0 → clip 결과 1.0 |
| `test_compute_synergy_clip_lower` | `compute_synergy` | comp=0, cov=0, red=1 → clip 결과 0.0 |
| `test_sim_4tier_same_college` | sim_4tier | 같은 단과대 → `w_college` 반영 |
| `test_sim_4tier_cross_domain` | sim_4tier | 메타가 도메인 차이 큰 두 트랙 → cos 낮음 → sim 낮음 |
| `test_is_cross_college_true` | `is_cross_college` | T1 다른 단과대 트랙 포함 → True |
| `test_is_cross_college_false` | `is_cross_college` | 같은 단과대만 포함 → False |
| `test_select_primary_top2_stable` | `select_primary` | 동점 시 첫 번째 순서 유지 |
| `test_slot3_cross_college_found` | `select_cross_college_slot` | 정상 cross-college 후보 → 정상 선택, fallback_level=None |
| `test_slot3_t1_fallback_to_t2` | `select_cross_college_slot` | T1 후보 0개 → T2 fallback 발동, fallback_level="T2" |
| `test_slot3_both_empty_returns_none` | `select_cross_college_slot` | T1+T2 모두 0개 → (None, "MMR") |
| `test_apply_mmr_excludes_primary` | `apply_mmr` | primary 가 MMR 후보군에서 제외 |
| `test_apply_mmr_lambda_zero` | `apply_mmr` | λ=0 → 순수 다양성, sim 최대 후보부터 회피 |
| `test_apply_mmr_lambda_one` | `apply_mmr` | λ=1 → 순수 시너지, 점수 순위와 일치 |
| `test_generate_combos_hm001_constraint` | `generate_track_combos` | track_a.college_id == user_college_id 항상 True |
| `test_generate_combos_freshman` | `generate_track_combos` | current_tracks=[] → track_a 후보 복수 |
| `test_generate_combos_2nd_year` | `generate_track_combos` | current_tracks=[fixed_id] → track_a 가 fixed_id 로 고정 |

순수 함수는 mock 불필요. `select_cross_college_slot` 만 logger mock.

### 7.2 통합 테스트 (`tests/integration/test_n4_flow.py`)

```python
@pytest.mark.integration
def test_n4_full_flow_mock_rag(mocker):
    mock_rag = mocker.Mock()
    mock_rag.fetch_all_tracks_with_meta.return_value = build_test_tracks()  # 20개
    node = N4TrackSynergyNode(
        rag_client=mock_rag,
        embedding_loader=MockEmbed(),
        synergy_config=load_test_config(),
    )

    state = build_test_state(job_candidates=5, college_id="컴퓨터공학부")
    result = node(state)

    assert len(result["primary_combos"]) == 2
    assert len(result["secondary_combos"]) == 5
    assert result["primary_combos"][0].slot_type == "primary"
    assert any(c.slot_type == "cross_college" for c in result["secondary_combos"])
    for combo in result["primary_combos"] + result["secondary_combos"]:
        assert 0.0 <= combo.synergy_score <= 1.0
```

### 7.3 Edge Cases

| 테스트 | 시나리오 | 기대 |
|---|---|---|
| `test_cross_college_zero_candidates` | 타학과 트랙 0개 데이터셋 | `slot3_fallback_triggered=True`, `slot3_fallback_level="MMR"`, secondary 5개 모두 mmr |
| `test_synergy_all_zero` | 모든 combo synergy=0 | primary 2개 동점 stable, 오류 없음 |
| `test_weighted_config_change_no_regression` | synergy.yaml 가중치 변경 후 재실행 | 슬롯 순서는 변경 가능하지만 pytest 0 fail (가중치 변경 = 기능 변경, 회귀 아님 — 수동 검토 대상) |

### 7.4 테스트 격리 원칙

- 실제 RAGFlow 호출은 `@pytest.mark.integration` 마킹. CI 에서는 `pytest -m "not integration"` 로 제외.
- embedding 벡터는 `numpy.random.seed(42)` 고정.
- `synergy.yaml` 은 test fixture dict literal 로 override (파일 I/O 없음).

---

## 8. CLAUDE.md 5 Principles Mapping

| 원칙 | 본 노드 설계에서의 구현 |
|---|---|
| **1. State-first 설계** | `GraphState` 에 `job_candidates`, `primary_combos`, `secondary_combos`, `slot3_fallback_triggered`, `slot3_fallback_level` 필드를 코드 작성 전에 확정. 노드가 State 필드를 임의 추가하지 않음. |
| **2. 단일 책임 노드** | Stage 0~5 각 함수가 단일 연산 담당. 시너지 계산(Stage 1)과 슬롯 선택(Stage 2~4) 분리. I/O(Stage 5) 와 계산(Stage 0~4) 분리. |
| **3. 부작용 격리** | RAG/embedding I/O 는 `__call__` 진입점에만. Stage 0/1/2/4 순수 함수에는 I/O 없음. `logger.info` 는 Stage 3 에만 (fallback 감사 단일 위치). |
| **4. 명시적 엣지 조건** | `slot_type: Literal["primary", "cross_college", "mmr"]` 와 `slot3_fallback_level: Literal["T2", "MMR"] \| None` 으로 분기 값 고정. mypy 검증 가능. |
| **5. 검증 가능 출력** | `RankedCombo(BaseModel)` 의 `synergy_score: float = Field(ge=0.0, le=1.0)` 로 clip 결과를 모델 레벨에서 한 번 더 검증. `SynergyConfig` 도 Pydantic 으로 yaml 파싱 검증. |

---

## 9. Verification Plan

### 9.1 본 PR 산출물 검증

- [ ] `docs/design/n4-track-synergy.md` 본 문서 신규 생성 확인
- [ ] `src/tracktory/config/synergy.yaml` 스켈레톤 신규 생성 확인
- [ ] 본 문서의 시너지 식 / sim_4tier 식 / 슬롯 수와 `synergy.yaml` 의 가중치·임계값 키 1:1 정합 확인 (수동)
- [ ] pre-commit 훅 통과 (ruff format / ruff check / mypy / pytest — 본 PR 은 마크다운+yaml 만이라 mypy/pytest 영향 없음)

### 9.2 Phase C (코드 작성) 착수 전 완료 기준

- [ ] 본 PR merge
- [ ] 후속 이슈에서 Phase C 작업 진행:
  - [ ] `pyproject.toml` 의존성 추가 (`uv add langgraph langchain-core pyyaml numpy`)
  - [ ] `src/tracktory/graph/state.py` (GraphState TypedDict)
  - [ ] `src/tracktory/graph/models.py` (Track / JobCandidate / TrackCombo / RankedCombo / SynergyConfig)
  - [ ] `src/tracktory/graph/nodes/n4_synergy.py` (Stage 0~5 구현)
  - [ ] `src/tracktory/graph/edges.py` (조건부 엣지)
  - [ ] `src/tracktory/config/embedding.yaml` (ADR-0001 구현)
  - [ ] `tests/unit/graph/test_n4_*.py` (§7.1 항목)
  - [ ] `tests/integration/test_n4_flow.py` (§7.2)
  - [ ] 커밋 전: `ruff format` → `ruff check --fix` → `mypy src` → `pytest`

### 9.3 ADR 승격 후보

본 노드의 핵심 설계가 변경 없이 1+ 스프린트를 통과하면 ADR 승격을 검토합니다 (ADR README 의 승격 트리거 정합).

- N4 트랙 시너지 알고리즘 (Slot Reservation Pattern 4 + sim_4tier 통합 + MMR)
- 4-tier hierarchy 정의 (T1 단과대 / T2 학부 / T3 전공 / T4 트랙)
- MMR 다양성 알고리즘 (`lambda` 초기값 + ablation 결과)

---

## 변경 이력

| 버전 | 일자 | 작성자 | 변경 내용 |
|---|---|---|---|
| 0.1 | 2026-04-29 | 이재원 | 최초 작성. N4 노드 코드 작성 직전 단계의 설계 명세. Slot Reservation Pattern 4 + sim_4tier 5항(트랙 메타 cos 통합) + MMR 통합 구조. `synergy.yaml` 외부화 스키마 명시. CLAUDE.md 5 원칙 매핑 + Pydantic 검증 패턴 명시. 단위·통합·edge case 테스트 전략 17건. |
