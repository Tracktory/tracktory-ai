# 0001 — Single Embedding Boundary

## Status

Accepted

(변경 이력은 본 문서 최하단 "Revision History" 섹션 참조.)

## Context

tracktory-ai 의 추천 파이프라인은 다섯 위치에서 임베딩 벡터를 사용합니다:

1. **Offline 임베딩 생성** — RAGFlow 가 강의계획서·채용공고·트랙 메타 텍스트를 임베딩하여 벡터 인덱스에 적재.
2. **Offline 그래프 구축** — LightRAG 가 위 벡터를 노드 속성으로 활용하여 관계 그래프 생성.
3. **Online 사용자 프로필 임베딩** — 온보딩 입력을 자연어 문장으로 직렬화한 후 RAGFlow `retrieve()` 의 question 으로 전달. **임베딩 호출 자체는 RAGFlow 내부에서 수행** (자체 코드가 별도 임베딩 API 를 직접 호출 X). RAGFlow 가 question 텍스트를 본 yaml 설정과 동일한 모델로 임베딩하여 KB 내 직무 청크와 유사도 계산.
4. **Online 직무 매칭** — 위 RAGFlow `retrieve()` 가 반환하는 직무 청크의 `similarity` 점수를 그대로 사용. 자체 코드의 cosine 산식 호출 X — RAGFlow 가 hybrid search + reranker 까지 내부 처리.
5. **Online 트랙 메타 다양성 측정** — 두 트랙 메타 텍스트 임베딩 간 코사인 거리 계산. (구체 호출 grain 은 트랙 시너지 노드 구현 단계에서 확정.)

이 다섯 위치가 서로 다른 임베딩 모델·차원·정규화 설정을 사용하면 코사인 유사도가 **다른 임베딩 공간에서의 비교**가 되어 의미를 잃습니다. 예를 들어 5번 (트랙 메타 거리) 의 모델이 384차원이고 4번 (직무 매칭) 모델이 768차원이면 두 신호를 같은 산식 안에서 결합할 수 없습니다. 같은 모델이라도 정규화 정책 (L2 normalize 여부) 이 다르면 코사인 분포가 어긋나 가중치 튜닝이 무의미해집니다.

이를 "임베딩 공간 드리프트 (embedding space drift)" 라 부릅니다. 클린 아키텍처의 "외부 의존성은 단일 Boundary 를 통해 주입" 원칙과 동일한 문제 — 같은 추상화 (벡터 공간) 를 여러 입구에서 다른 형태로 도입하면 내부 도메인이 외부 의존성 변화에 취약해집니다.

파이프라인이 ablation 실험 (가중치·임계값 튜닝) 을 수행할 때, 임베딩 설정이 노드마다 다르면 실험 변수를 통제할 수 없습니다. 임베딩 모델 자체가 결과 품질의 숨은 변수가 되어버립니다.

## Decision Drivers

- **결정의 일관성**: 코사인 유사도 비교가 항상 동일한 공간에서 이루어져야 함.
- **튜닝 비용**: 가중치 / 임계값을 한 곳에서만 조정해도 전체 파이프라인이 일관되게 변하도록.
- **재현성**: ablation 실험에서 임베딩 모델을 통제 변수로 고정하기 위해.
- **운영 단순성**: MVP 단계에서 멀티 모델 라우팅의 운영 부담 회피.

## Decision

**모든 임베딩 호출은 단일 `embedding.yaml` 설정을 공유한다.**

`embedding.yaml` 은 다음을 정의합니다:

- 모델 식별자 (예: RAGFlow 내장 default — 향후 다른 모델로 교체 시 본 yaml 수정 후 전체 인덱스 재생성)
- 벡터 차원 (`dim`)
- 정규화 정책 (L2 / no-normalize / 기타)
- 토크나이저 / 청크 정책 참조 (RAGFlow 청크 정책과 정합)

위 다섯 위치 (Offline 임베딩 생성 / LightRAG 구축 / 프로필 임베딩 / 직무 매칭 / 트랙 메타 다양성) 모두 이 yaml 을 단일 source 로 로드합니다. 노드별 별도 config 허용 X.

**호출 grain 보충 (2026-05-20)**: Position 3·4 는 자체 코드가 임베딩 API 를 직접 호출하지 않고, RAGFlow `retrieve(question=profile_text)` 한 번에 (a) question 임베딩 + (b) KB 인덱스 검색 + (c) reranker + (d) similarity 점수 반환 까지 일괄 처리됩니다. 단일 yaml 정합은 RAGFlow KB 적재 시점의 임베딩 모델과 본 yaml 이 동일 모델·차원·정규화임을 보장하는 것으로 달성됩니다 (자체 코드의 별도 임베딩 클라이언트 미사용). Position 5 (트랙 메타 다양성) 만 자체 임베딩 호출이 필요할 수 있으며, 그 시점에 `EmbeddingClient` Protocol 을 도입하여 본 yaml 을 로드하는 단일 구현체로 단일성 보장합니다.

**Scope creep 방지**: 본 ADR 은 "임베딩 설정 boundary 의 단일성" 만 다룹니다. 현재 RAGFlow 가 OpenAI `text-embedding-3-small` (dim=1536, L2 normalize) 을 사용 중이며 본 yaml 과 정합됩니다 (2026-05-20 확인). **한국어 특화 모델 (BGE-M3 등) 검토는 후속 ADR 로 분리**합니다. 본 ADR 의 결정은 어떤 모델을 쓰든 단일 yaml 로 통일한다는 것 자체에 한정됩니다.

## Consequences

### Positive

- 코사인 유사도 / 거리 계산이 항상 같은 임베딩 공간에서 일관됩니다.
- 가중치 튜닝 / ablation 시 임베딩 모델 변수를 통제할 수 있습니다.
- 모델 교체 시 yaml 한 곳만 수정하면 전체 파이프라인이 일관되게 갱신됩니다.
- 임베딩 설정 관련 버그의 진단 지점이 단일화됩니다.

### Negative

- 노드별 최적 임베딩 모델이 다른 경우에도 단일 모델로 강제됩니다. 예를 들어 직무 매칭에는 sentence-level 모델, 트랙 메타 다양성 측정에는 paragraph-level 모델이 더 적합할 수 있으나 본 ADR 은 이를 허용하지 않습니다.
- 모델 교체 = 전체 벡터 인덱스 재생성. RAGFlow 적재 비용이 발생합니다.

### Neutral

- 운영 단순성 (단일 yaml 모니터링) 과 유연성 손실의 trade-off. MVP 단계의 단순성 우선 결정이며, 성능 요구가 높아지면 후속 ADR 에서 재검토 가능합니다.

## Alternatives Considered

| ID | Description | Why rejected |
|----|-------------|--------------|
| ALT-1 | 노드별 별도 embedding config — 각 위치마다 최적 모델 선택 (sentence-level / paragraph-level / domain-specific 등) | 임베딩 공간 드리프트 발생. 코사인 유사도가 다른 공간에서의 비교가 되어 산식 내 신호 결합 불가. 클린 아키텍처의 "외부 의존성은 단일 Boundary 통해 주입" 원칙 위배. 가중치 / 임계값 튜닝이 노드별로 분기되어 복잡도가 폭발. ablation 실험 시 임베딩 모델이 숨은 변수가 되어 결과 해석 불가. |

## References

- Robert C. Martin, "Clean Architecture: A Craftsman's Guide to Software Structure and Design" (2017) — Boundary 인터페이스 / 외부 의존성 단일 진입점 원칙.
- RAGFlow Embedding Configuration — https://ragflow.io/docs (공식 docs)
- Hugging Face, BGE-M3 model card — https://huggingface.co/BAAI/bge-m3 (후속 ADR 에서 한국어 모델 채택 검토 시 참조)

## Revision History

| 버전 | 일자 | 단계 | 변경 내용 |
|---|---|---|---|
| 1.0 | 2026-04-23 | Decided | 초안 결정 — 다섯 위치 (Offline 임베딩 생성 / LightRAG 구축 / 프로필 임베딩 / 직무 매칭 / 트랙 메타 다양성) 가 단일 `embedding.yaml` 공유. 노드별 별도 config 금지. |
| 1.1 | 2026-04-26 | Promoted to ADR | 안정화 확인 (변경 없이 스프린트 통과) → ADR 로 승격. 본 문서 생성. Status: Accepted. |
| 1.2 | 2026-05-20 | Clarified | Position 3·4 호출 grain 명시화. 자체 코드가 임베딩 API 직접 호출 X — RAGFlow `retrieve(question=profile_text)` 한 번에 question 임베딩 + KB 검색 + reranker + similarity 점수 일괄 처리. 단일 yaml 정합은 RAGFlow KB 적재 모델 ↔ 본 yaml 동일성 보장으로 달성. OpenAI `text-embedding-3-small` (dim=1536, L2 normalize) 사용 중 확인. 결정 본문 (단일 yaml) 은 불변, 호출 grain 만 clarification. (배경: Path A canonical 결정과 정합.) |
