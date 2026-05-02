# Track Synergy Node — Architecture Design

| 항목 | 내용 |
|---|---|
| **Status** | Design (구현 전 합의 자료) |
| **Owner** | 이재원 |
| **Last Updated** | 2026-04-29 |
| **Audience** | tracktory-ai 팀 (인터페이스·의사결정 검토), 향후 ADR 승격 시 reviewer |

> 본 문서는 트랙 시너지 노드의 **인터페이스와 알고리즘 결정** 을 기록합니다. 구체적 구현 코드 / 단위 테스트 / 함수 시그니처 디테일은 본 문서의 범위 밖이며, 코드 작성 시 직접 구현하면서 결정합니다. 본 문서의 목적은 "왜 이 알고리즘인가" 와 "노드 외부 인터페이스가 어떤 형태인가" 만.

---

## 1. Overview

트랙 시너지 노드는 Online 추천 파이프라인에서 **트랙 조합 추천**을 담당하는 LangGraph 노드입니다. 직무 매칭 노드에서 넘어온 직무 후보 3~5개를 입력으로 받아, 학생에게 추천할 **주 추천 2개 + 보조 추천 5개** 의 트랙 조합을 출력합니다.

### 1.1 풀어야 할 문제

한성대 전면 트랙제(47개 트랙)에서 두 트랙 조합의 가능한 수는 1,081개입니다. 학생이 자신의 관심사·직무에 맞춰 이를 모두 비교 평가하는 것은 불가능하므로, 다음 3가지 요소가 결합된 추천이 필요합니다.

1. **Synergy Score** — 두 트랙 조합이 학생의 직무 후보군에 얼마나 적합한지 정량화
2. **Multi-tier Similarity** — 두 조합이 한성대 학사 구조(단과대 → 학부 → 트랙 → 과목) 상 얼마나 유사한지 측정
3. **Slot-based Diversity Guarantee** — 학과 경계를 넘는 "이색 조합" 이 추천 결과에 항상 노출되도록 보장. 단순 다양성 알고리즘만으로는 확률적 보장이라 누락 가능 → 슬롯 기반 hard constraint 도입.

### 1.2 핵심 설계 원칙

본 노드는 `CLAUDE.md` 의 아키텍처 철학 5개 원칙을 준수합니다 (state-first / 단일 책임 / 부작용 격리 / Literal 엣지 / Pydantic 검증). 자세한 매핑은 §6.

### 1.3 단일 임베딩 공간 의존성

본 노드의 트랙 메타 임베딩은 ADR-0001(`docs/adr/0001-single-embedding-boundary.md`) 에서 정의한 단일 임베딩 설정을 사용합니다. 추천 파이프라인의 다른 임베딩 위치(프로필 임베딩 노드, 직무 매칭 노드, Offline 임베딩 생성)와 동일한 모델·차원·정규화 설정을 공유해야 코사인 유사도 비교가 의미를 가집니다.

---

## 2. Input / Output Interface

### 2.1 Input — 선행 노드로부터 받는 State 필드

| 필드 | 출처 | 의미 |
|---|---|---|
| `job_candidates` | 직무 매칭 노드 | 직무 후보 3~5개 (직무명, 채용공고 기술스택, 역량 태그) |
| `profile_vector` | 프로필 임베딩 노드 | 사용자 프로필 임베딩. 본 노드에서는 직접 사용하지 않으나 fallback 안전장치로 참조 가능 |
| `current_tracks` | 입력 정규화 노드 | 2학년 이상 사용자의 이수 트랙 ID. 1학년은 빈 리스트 |
| `college_id` | 입력 정규화 노드 | 사용자 소속 단과대 ID. 기능명세상의 1트랙 주전공 제약(주 추천 두 조합 중 하나는 반드시 사용자 단과대 소속 트랙 포함) 적용 입력 |

### 2.2 Output — 본 노드가 State 에 추가하는 필드

| 필드 | 의미 |
|---|---|
| `primary_combos` | 시너지 점수 상위 2개 트랙 조합 |
| `secondary_combos` | 보조 추천 5개 (cross-college 1개 예약 + MMR 4개) |
| `slot3_fallback_triggered` | cross-college 슬롯 fallback 발생 여부 (boolean) |
| `slot3_fallback_level` | fallback 단계 — `None` (없음) / `T2` (학부 cross-dept relax) / `MMR` (cross-dept 도 0 → MMR 흘림) |

LangGraph 의 "부분 상태 반환" 원칙에 따라 본 노드는 선행 노드의 필드를 수정하지 않고 위 4개만 새로 추가합니다.

---

## 3. Domain Models (Interface Sketch)

본 노드가 다루는 도메인 개념. 정확한 필드 / 타입은 코드 작성 시 결정.

- **Track** — 한성대 단일 트랙 메타데이터. 4-tier hierarchy 식별자(`college_id` / `department_id` / `major_id` / `course_ids`) + 트랙 메타 텍스트 + 임베딩 벡터.
- **JobCandidate** — 직무 매칭 노드의 결과 단건. 직무명 + 채용공고 기술스택 + 역량 태그 + 매칭 점수.
- **TrackCombo** — 두 `Track` 의 조합 단위.
- **RankedCombo** — `TrackCombo` 에 시너지 점수 + 슬롯 분류(`primary` / `cross_college` / `mmr`) + 순위가 부착된 단위. 노드 출력 단위.

**`major_id == department_id` degenerate**: 한성대 일부 학과(AI응용학과, 융합보안학과 등)는 단일 트랙만 운영. 이 경우 `major_id` 를 `department_id` 와 동일하게 두면 동일 모델로 표현 가능.

---

## 4. Algorithm

알고리즘은 6단계로 구성. 각 단계의 단일 책임은 명확히 분리되며, I/O 는 마지막 단계(노드 진입점) 에만 집중됩니다.

### 4.1 Slot Reservation Pattern

총 7개 슬롯(주 2 + 보조 5) 을 다음 순서로 채웁니다.

1. **slot 1~2 (주 추천)** — synergy 점수 상위 2개. 1트랙 주전공 제약은 후보 생성 단계에서 filtering 으로 적용되어, 자연스럽게 사용자 단과대 내부 조합이 됨.
2. **slot 3 (cross-college 예약)** — `is_cross_college(combo, primary)` AND `synergy ≥ min_cross_synergy` 만족 후보 중 synergy 최대값 선택.
3. **slot 3 fallback (T1 후보 0개)** — 학부(T2) cross-dept 로 relax. 임계값(`min_cross_synergy`) 은 동일 유지 (변경 시 별도 ablation 후보).
4. **그래도 0개** — slot 3 = `None`, fallback_level = `MMR`. Stage 4 의 MMR 후보군에 흘려보냄.
5. **slot 4~7 (MMR)** — Carbonell-Goldstein (1998) MMR. primary + slot 3 (있으면) 을 selected 로, 나머지 후보군에서 선정. 각 단계에서 synergy 와 다양성(sim 회피) 균형.
6. **graduated cross-ness** — cross-college 만 hard gate, cross-dept 차등은 MMR 단계의 `sim_4tier` 가중치(`w_department`) 가 계속 표현.

### 4.2 Synergy Score

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

**가중치 초기값** (직관 할당, ablation 대상): `w_comp = 0.3`, `w_cov = 0.5`, `w_red = 0.2`.
직관: 직무 도달도(cov) 가장 중요 → 보완성(comp) 다음 → 중복(red) 페널티.

**`clip(0, 1)` 강제 이유**: 가중치 합산식이 정의역 [0, 1] 을 벗어나면 후속 MMR 의 균형 식이 깨짐. 단순 3항 합산에서 clip 빈도는 낮을 것으로 예상. clip 빈도가 운영 중 높게 관측되면 가중치 ablation 으로 재조정.

**NCS 데이터 미활용 결정 반영**: `job_coverage` 는 채용공고 기술스택 100% 사용. NCS 직무역량 데이터는 미수집 결정에 따라 입력에서 제외. 향후 NCS 수집 시 합집합으로 확장 여지 유지.

### 4.3 sim_4tier — Multi-tier Similarity (트랙 메타 다양성 통합)

```
sim_4tier(combo_a, combo_b)
  = w_college    · same_T1(a, b)
  + w_department · same_T2(a, b)
  + w_major      · same_T3(a, b)
  + w_overlap    · overlap_ratio(a, b)
  + w_meta       · cos(meta_a, meta_b)
```

| 항 | 의미 |
|---|---|
| `same_T1` / `same_T2` / `same_T3` | 단과대 / 학부 / 트랙 일치 여부 (0/1) |
| `overlap_ratio` | T4 과목 overlap 비율 |
| `cos(meta_a, meta_b)` | 트랙 메타 텍스트 임베딩의 코사인 유사도 — **도메인/서사 거리** |

**가중치 단조 제약**: `w_college ≥ w_department ≥ w_major` (상위 tier 가 더 강한 신호). `w_meta` 는 별도 축이며 tier 제약 외.

**가중치 초기값** (ablation 대상): `w_college = 0.4`, `w_department = 0.3`, `w_major = 0.2`, `w_overlap = 0.1`, `w_meta = 0.05`.

**`w_meta · cos` 의 부호 일관성**: cos 가 높으면 sim 이 높음 → MMR 의 다양성 항이 cos 가 낮은(도메인 거리가 먼) 조합을 선호 → cross-domain 조합이 자연스럽게 선택됨. 시너지 식 외부 보너스로 가산하는 대안도 검토했으나, 시너지 식이 단순 3항으로 유지되고 부호 반전 없이 sim 방향성이 일관되는 본 통합 방식을 선택.

### 4.4 MMR

```
next = argmax_{i ∉ selected} [
    λ · synergy(i)
  − (1 − λ) · max_{j ∈ selected} sim_4tier(i, j)
]
```

`λ` 초기값 = 0.6 (synergy 약간 우선, ablation 대상). λ=0 → 순수 다양성, λ=1 → 순수 시너지.

**선행 출처**: Carbonell & Goldstein (1998), 정보 검색 분야 표준 다양화 알고리즘.

### 4.5 단일 책임 / 부작용 격리

- 후보 생성 / 시너지 계산 / primary 선택 / cross-college 슬롯 / MMR — 각각 단일 책임 함수로 분리
- 모든 계산 함수는 순수 (I/O 없음) — 테스트에서 mock 불필요
- RAG · embedding loader 호출은 노드 진입점 (LangGraph callable) 단 1곳에 집중
- `logger.info` 는 cross-college fallback 발생 위치(슬롯 3 단계) 에만 허용 — fallback 감사 정보를 단일 위치에 모음. 로그 레벨은 `INFO` 잠정 (fallback 은 예상 케이스), 운영 데이터 누적 후 빈도가 비정상적으로 높으면 `WARNING` 승격 검토.

---

## 5. Configuration

### 5.1 외부화 정책

가중치·임계값은 코드 상수가 아닌 외부 yaml(`src/tracktory/config/synergy.yaml`) 로 분리. 이유:

- ablation 실험 시 코드 수정 없이 가중치 변경 가능
- 운영 중 `min_cross_synergy` 같은 threshold 튜닝이 deploy 와 분리

### 5.2 외부화 대상

| 블록 | 항목 |
|---|---|
| `weights` | 시너지 식의 `complementarity` / `coverage` / `redundancy` |
| `similarity` | sim_4tier 의 `w_college` / `w_department` / `w_major` / `w_course_overlap` / `w_meta` |
| `mmr` | `lambda` |
| `slots` | `primary_count` / `secondary_count` / `cross_college_reserved` / `min_cross_synergy` |

실제 yaml 스키마와 초기값은 동봉된 `src/tracktory/config/synergy.yaml` 파일이 1차 신뢰 소스. 본 문서의 §4 가중치 초기값은 yaml 의 미러.

---

## 6. CLAUDE.md 5 Principles Alignment

| 원칙 | 본 노드 설계의 정렬 |
|---|---|
| **State-first 설계** | `GraphState` 입출력 필드(§2)를 코드 작성 전에 합의. |
| **단일 책임 노드** | 후보 생성 / 시너지 계산 / 슬롯 선택 / MMR 을 분리. 시너지 계산과 슬롯 선택을 한 함수에 섞지 않음 (§4.5). |
| **부작용 격리** | 모든 계산 함수는 순수. RAG · embedding 호출은 노드 진입점에만. `logger.info` 는 fallback 단일 위치에만 (§4.5). |
| **명시적 엣지 조건** | `slot_type` 와 `slot3_fallback_level` 을 `Literal[...]` 로 고정 (§2.2 출력 표). |
| **검증 가능 출력** | `synergy_score` 는 `[0, 1]` 강제 — Pydantic 필드 제약(`ge=0.0, le=1.0`) 으로 모델 레벨 검증. |

---

## 7. ADR 승격 후보

본 노드의 핵심 설계가 변경 없이 1+ 스프린트를 통과하면 ADR 디렉토리(`docs/adr/`) 로 승격을 검토합니다. 승격 1순위 후보:

- 트랙 시너지 알고리즘 — Slot Reservation Pattern (Hard Constraint + MMR) 구조 자체
- 4-tier hierarchy 정의 (T1 단과대 / T2 학부 / T3 트랙 / T4 과목 overlap)
- MMR 다양성 알고리즘 (`λ` 초기값 + ablation 결과 누적 후)
- sim_4tier 의 트랙 메타 cos 통합 (`w_meta` 가중치 ablation 정착 후)

---

## 변경 이력

| 버전 | 일자 | 작성자 | 변경 내용 |
|---|---|---|---|
| 0.2 | 2026-04-29 | 이재원 | 문서 목적 재정립 — 인터페이스·의사결정 중심으로 재구성. 구체적 구현 코드 / 단위 테스트 17건 표 / 통합 테스트 코드 / Edge Case 표 / Pydantic 모델 정의 / 함수 시그니처 디테일 / "코드 작성 착수 전 체크리스트" 모두 제거. 코드와 문서가 어긋날 위험 + 유지보수 포인트 증가 회피. 6 섹션 (Overview / IO Interface / Domain Sketch / Algorithm / Configuration / CLAUDE.md Alignment) + ADR 승격 후보. |
| 0.1 | 2026-04-29 | 이재원 | 최초 작성. 트랙 시너지 노드의 구현 명세. Slot Reservation Pattern + sim_4tier 5항(트랙 메타 cos 통합) + MMR 6단계 구조. 외부화 스키마 명시. |
