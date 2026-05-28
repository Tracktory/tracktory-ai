"""정규화된 사용자 프로필을 자연어 문장으로 직렬화한다.

LLM 직렬화는 동일 입력에서도 출력이 달라질 수 있어 파이프라인 비교 실험의
통제 조건을 깨뜨린다. 그 대신 ``profile_embed_template.yaml`` 의 결정론적
패턴으로 자연어 문장을 합성한다.

본 노드는 더 이상 자체 임베딩 호출을 수행하지 않는다. 직무 검색 boundary
(``JobSearchClient.rag_search_jobs``) 가 자연어 질의를 받아 임베딩·검색·
재정렬을 일괄 처리하므로, 자체 코드는 자연어 문장만 다음 단계로 넘긴다.
``completed_courses`` 는 이수 과목 집합 필터로만 사용되며 의미 직렬화에는
포함하지 않는다.
"""

from pathlib import Path
from typing import Any

import yaml

from tracktory.graph.state import GraphState

_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "profile_embed_template.yaml"
)


class ProfileEmbedNode:
    """정규화 결과를 자연어 문장으로 변환하는 callable 노드.

    템플릿 yaml 은 생성자에서 한 번만 로드한다. 단위 테스트는 별도 의존성
    주입 없이 입력 dict 에 대한 직렬화 출력 문자열만 비교한다.
    """

    def __init__(self, template_path: Path | None = None) -> None:
        self._template = _load_template(template_path or _DEFAULT_TEMPLATE_PATH)

    def __call__(self, state: GraphState) -> dict[str, Any]:
        """정규화된 프로필을 ``profile_text`` 한 필드만 채워 부분 state 로 반환한다.

        반환값 계약:
            정상: ``{"profile_text": str, "trace": [...]}`` — profile_text 는
                비어 있지 않은 자연어 문장임을 보장한다 (template pattern 이 비어 있지
                않은 한 빈 문자열은 발생하지 않음).
            skip: ``{"errors": [...], "trace": [...]}`` — profile_text 키 자체를
                담지 않아 다음 단계 노드가 ``state.get("profile_text")`` 로 skip
                여부를 판단할 수 있게 한다.

        부작용:
            없음. 임베딩 호출 / 외부 I/O 없음 (Path A canonical).
        """
        profile = state.get("normalized_profile")
        if not profile:
            return {
                "errors": ["profile_embed skipped: normalized_profile is missing"],
                "trace": ["profile_embed:skip"],
            }

        text = _serialize_profile(profile, self._template)
        return {
            "profile_text": text,
            "trace": ["profile_embed:ok"],
        }


def _load_template(path: Path) -> dict[str, Any]:
    """template yaml 을 dict 으로 로드한다. top-level 이 mapping 이 아니면 ``ValueError``."""
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Template file {path} must define a top-level mapping")
    return loaded


def _serialize_profile(profile: dict[str, Any], template: dict[str, Any]) -> str:
    """카테고리형 프로필을 결정론적 자연어 문장으로 변환한다.

    ``completed_courses`` 는 의도적으로 사용하지 않는다. 이수 과목은 관심사·흥미와
    다른 성격의 정보(집합 필터 대상)이므로 의미 직렬화에 포함하면 의미를 희석한다.
    LLM 미사용이므로 동일 입력은 항상 동일 문장을 생성한다.
    """
    pattern: str = template.get("pattern", "")
    return pattern.format(
        interests=", ".join(profile.get("interests", [])),
        dev_interests=", ".join(profile.get("dev_interests", [])),
        work_values=", ".join(profile.get("work_values", [])),
    )
