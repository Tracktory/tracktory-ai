"""N2 — Profile Embedding 노드 (Template 직렬화 + 단일 임베딩 boundary).

D-03 (Template 직렬화) · D-04 (단일 ``embedding.yaml``) · D-06
(``completed_courses`` N2 bypass) 결정의 코드 표현.

LLM 직렬화는 결정성 부재로 S1 파이프라인 비교의 통제 변수를 망가뜨리므로
사용하지 않는다. 임베딩 호출은 ``EmbeddingClient`` Protocol 로 추상화하여
구체 구현(RAGFlow / OpenAI 등)을 인프라 레이어로 분리한다 — CONTRIBUTING.md
§ 4.4 의존성 주입 규약.

처리 흐름 (pipeline-design.md § 5.1):
    1. ``n2_template.yaml`` 의 결정론적 패턴으로 자연어 문장 합성.
    2. ``embedding.yaml`` 모델로 단일 벡터 생성 (O3 · O4 · N3 와 동일 공간 보장).
    3. ``completed_courses`` 는 의미 임베딩 미반영 → N5 입력으로 위임.
"""

from pathlib import Path
from typing import Any, Protocol

import yaml

from tracktory.graph.state import GraphState

_DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "config" / "n2_template.yaml"


class EmbeddingClient(Protocol):
    """O3 · O4 · N2 · N3 가 공유하는 임베딩 호출 인터페이스 (D-04 boundary).

    구체 구현은 ``tracktory.rag`` 등 인프라 레이어에서 제공하며, 노드는
    Protocol 만 의존한다 (CONTRIBUTING.md § 4.4 — 전역 import 금지, 주입 사용).
    """

    def embed(self, text: str) -> list[float]:
        """텍스트를 단일 벡터로 변환한다."""
        ...


class ProfileEmbedNode:
    """프로필 정규화 결과를 자연어 문장 + 임베딩 벡터로 변환하는 callable 노드.

    의존성을 ``__init__`` 으로 주입받는 callable 패턴 (CONTRIBUTING.md
    § 4.4 예시). 단위 테스트에서 ``EmbeddingClient`` 를 mock 으로 교체하면
    네트워크 호출 없이 검증 가능하다.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        template_path: Path | None = None,
    ) -> None:
        self._client = embedding_client
        self._template = _load_template(template_path or _DEFAULT_TEMPLATE_PATH)

    def __call__(self, state: GraphState) -> dict[str, Any]:
        profile = state.get("normalized_profile")
        if not profile:
            return {
                "errors": ["N2 skipped: normalized_profile is missing"],
                "trace": ["N2:skip"],
            }

        text = _serialize_profile(profile, self._template)
        vector = self._client.embed(text)
        return {
            "profile_text": text,
            "profile_vector": vector,
            "trace": ["N2:ok"],
        }


def _load_template(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Template file {path} must define a top-level mapping")
    return loaded


def _serialize_profile(profile: dict[str, Any], template: dict[str, Any]) -> str:
    """카테고리형 프로필을 결정론적 자연어 문장으로 변환한다.

    ``completed_courses`` 는 의도적으로 사용하지 않는다 (D-06).
    LLM 미사용이므로 동일 입력은 항상 동일 문장을 생성한다.
    """
    pattern: str = template.get("pattern", "")
    return pattern.format(
        interests=", ".join(profile.get("interests", [])),
        dev_interests=", ".join(profile.get("dev_interests", [])),
        work_values=", ".join(profile.get("work_values", [])),
    )
