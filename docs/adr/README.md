# Architecture Decision Records (ADR)

이 디렉토리는 tracktory-ai 의 안정화된 아키텍처 결정을 ADR (Architecture Decision Record) 형식으로 보존합니다. 본 디렉토리는 **팀 공유 자산** — 외부에 의존하지 않는 self-contained 형식을 사용합니다.

## 명명 규칙

- 파일명: `NNNN-kebab-case.md` (4자리 zero-padded 번호 + 영문 kebab-case 제목)
- 번호는 글로벌 순차 (0001, 0002, ...). 한 번 부여하면 재사용 X.
- 제목 한글 사용 X (디렉토리 검색 친화성). 본문은 한글/영문 혼용 OK.

## ADR 형식 (MADR-inspired)

각 ADR 은 다음 섹션을 포함합니다:

1. **Status** — Proposed / Accepted / Deprecated / Superseded by [#XXXX]
2. **Context** — 결정이 필요한 이유, 배경
3. **Decision Drivers** (선택) — 평가 기준
4. **Decision** — 채택한 결정 한 줄 요약 + 본문
5. **Consequences** — 채택의 결과 (positive / negative / neutral)
6. **Alternatives Considered** — 기각 또는 보류된 대안 (각 대안에 대해 Description / Why rejected)
7. **References** — 외부 공개 자료 only (논문, 표준 문서, 프레임워크 docs URL)

## 운영 원칙

1. **새 ADR 추가 시 본 README 인덱스 갱신 필수** — 인덱스가 없으면 ADR 발견 불가.
2. **자체 완결성 (self-contained)** — ADR 본문은 외부 internal docs 를 참조하지 않습니다. 팀 공유 레포 밖의 개인 문서 자산은 ADR 형식으로 흡수되었거나 외부 공개 자료로 표현됩니다. 외부 의존이 필요하면 References 섹션에 공개 URL 만 사용.
3. **승격 트리거** — 아키텍처 결정이 안정화되면 (= 변경 없이 1+ 스프린트 통과 + 코드 구현 단계 진입) ADR 로 승격.
4. **Status 변경** — 결정 폐기 시 Status 를 `Deprecated` 또는 `Superseded by [#XXXX]` 로 변경. 본문 삭제 X (역사 보존).

## ADR 인덱스

| # | Title | Status | Date | Summary |
|---|-------|--------|------|---------|
| [0001](./0001-single-embedding-boundary.md) | Single Embedding Boundary (단일 임베딩 설정) | Accepted | 2026-04-26 | O3·O4·N2·N3 가 공유하는 단일 `embedding.yaml` 보장. 코사인 유사도가 항상 동일한 임베딩 공간에서 비교됨. |

## 후속 ADR 후보

- N4 트랙 시너지 알고리즘 — Slot Reservation Pattern (Hard Constraint + MMR)
- 4-tier hierarchy 정의 (T1 단과대 / T2 학부 / T3 전공 / T4 트랙)
- MMR 다양성 알고리즘 (lambda 초기값 + ablation 결과)
- 한국어 임베딩 모델 채택 (BGE-M3 등 — 현재 ADR-0001 scope creep 방지로 분리)
