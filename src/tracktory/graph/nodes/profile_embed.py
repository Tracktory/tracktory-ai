"""정규화된 사용자 프로필을 자연어 문장으로 변환한 뒤 임베딩 벡터로 인코딩한다.

LLM 직렬화는 동일 입력에서도 출력이 달라질 수 있어 파이프라인 비교 실험의
통제 조건을 깨뜨린다. 그 대신 ``profile_embed_template.yaml`` 의 결정론적 패턴을 사용한다.

임베딩 호출은 ``EmbeddingClient`` Protocol 로 추상화하여 구체 구현을 인프라 레이어로 분리한다.
노드는 Protocol 만 의존하고 구체 구현은 외부에서 주입받는다.

처리 흐름:
    1. ``profile_embed_template.yaml`` 의 결정론적 패턴으로 자연어 문장 합성.
    2. ``embedding.yaml`` 모델로 단일 벡터 생성 (오프라인·온라인 임베딩 단계가
       동일한 임베딩 공간을 공유 — ADR-0001).
    3. ``completed_courses`` 는 의미 임베딩에 포함하지 않는다. 이수 과목은 이후
       선수과목 필터 단계에서만 집합 연산으로 사용한다.
"""

from pathlib import Path
from typing import Any, Protocol

import yaml

from tracktory.graph.state import GraphState

_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "profile_embed_template.yaml"
)


class EmbeddingClient(Protocol):
    """임베딩 호출 인터페이스 (ADR-0001 단일 임베딩 boundary).

    오프라인·온라인 임베딩 단계가 동일한 Protocol 을 통해 같은 임베딩 공간을 보장한다.
    구체 구현은 인프라 레이어에서 제공하며, 노드는 Protocol 만 의존한다.
    """

    def embed(self, text: str) -> list[float]:
        """텍스트를 단일 벡터로 변환한다."""
        ...


class ProfileEmbedNode:
    """정규화 결과를 자연어 문장 + 임베딩 벡터로 변환하는 callable 노드.

    의존성을 ``__init__`` 으로 주입받으므로 단위 테스트에서 ``EmbeddingClient`` 를
    mock 으로 교체하면 네트워크 호출 없이 검증 가능하다.
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
                "errors": ["profile_embed skipped: normalized_profile is missing"],
                "trace": ["profile_embed:skip"],
            }

        text = _serialize_profile(profile, self._template)
        vector = self._client.embed(text)
        return {
            "profile_text": text,
            "profile_vector": vector,
            "trace": ["profile_embed:ok"],
        }


def _load_template(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Template file {path} must define a top-level mapping")
    return loaded


def _serialize_profile(profile: dict[str, Any], template: dict[str, Any]) -> str:
    """카테고리형 프로필을 결정론적 자연어 문장으로 변환한다.

    ``completed_courses`` 는 의도적으로 사용하지 않는다. 이수 과목은 관심사·흥미와
    다른 성격의 정보(집합 필터 대상)이므로 임베딩 공간에 포함하면 의미를 희석한다.
    LLM 미사용이므로 동일 입력은 항상 동일 문장을 생성한다.
    """
    pattern: str = template.get("pattern", "")
    return pattern.format(
        interests=", ".join(profile.get("interests", [])),
        dev_interests=", ".join(profile.get("dev_interests", [])),
        work_values=", ".join(profile.get("work_values", [])),
    )
