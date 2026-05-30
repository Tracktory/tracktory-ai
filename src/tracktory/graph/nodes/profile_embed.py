"""정규화된 사용자 프로필을 자연어 문장으로 직렬화한다.

LLM 직렬화는 동일 입력에서도 출력이 달라질 수 있어 파이프라인 비교 실험의
통제 조건을 깨뜨린다. 그 대신 ``profile_embed_template.yaml`` 의 결정론적
패턴으로 자연어 문장을 합성한다.

본 노드는 더 이상 자체 임베딩 호출을 수행하지 않는다. 직무 검색 boundary
(``JobSearchClient.rag_search_jobs``) 가 자연어 질의를 받아 임베딩·검색·
재정렬을 일괄 처리하므로, 자체 코드는 자연어 문장만 다음 단계로 넘긴다.
``completed_courses`` 는 이수 과목 집합 필터로만 사용되며 의미 직렬화에는
포함하지 않는다.

흥미 개발 분야 값은 추상적이라 직무 검색 인덱스의 구체 어휘와 격차가 커
유사도가 낮게 잡힌다. 직렬화 직전 정적 확장 사전으로 직무 어휘에 가까운
키워드로 바꿔 질의 신호를 키운다. 사전 확장은 검색 질의에만 영향을 주며
직렬화의 결정론(동일 입력 = 동일 출력)을 깨지 않는다.
"""

from pathlib import Path
from typing import Any

import yaml

from tracktory.graph.state import GraphState

_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "profile_embed_template.yaml"
)
_DEFAULT_DEV_KEYWORDS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "dev_interest_to_keywords.yaml"
)


class ProfileEmbedNode:
    """정규화 결과를 자연어 문장으로 변환하는 callable 노드.

    템플릿 yaml 은 생성자에서 한 번만 로드한다. 단위 테스트는 별도 의존성
    주입 없이 입력 dict 에 대한 직렬화 출력 문자열만 비교한다.
    """

    def __init__(
        self,
        template_path: Path | None = None,
        dev_keywords_path: Path | None = None,
    ) -> None:
        self._template = _load_template(template_path or _DEFAULT_TEMPLATE_PATH)
        self._dev_keywords = _load_dev_keywords(dev_keywords_path or _DEFAULT_DEV_KEYWORDS_PATH)

    def __call__(self, state: GraphState) -> dict[str, Any]:
        """정규화된 프로필을 ``profile_text`` 한 필드만 채워 부분 state 로 반환한다.

        반환값 계약:
            정상: ``{"profile_text": str, "trace": [...]}`` — profile_text 는
                비어 있지 않은 자연어 문장임을 보장한다 (관심사·흥미 개발 분야는
                필수 입력이라 항상 한 개 이상의 절이 렌더링됨).
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

        expanded = _expand_dev_interests(profile, self._dev_keywords)
        text = _serialize_profile(expanded, self._template)
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


def _load_dev_keywords(path: Path) -> dict[str, str]:
    """흥미 개발 분야 확장 사전을 ``{원본값: 확장키워드}`` 매핑으로 로드한다.

    파일이 없거나 ``keywords`` 매핑이 비어 있으면 빈 dict 을 반환해 확장을
    건너뛴다 (확장 없이도 직렬화는 동작해야 하므로 부재를 오류로 보지 않음).
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        return {}
    keywords = loaded.get("keywords")
    if not isinstance(keywords, dict):
        return {}
    return {str(k): str(v) for k, v in keywords.items()}


def _expand_dev_interests(profile: dict[str, Any], dev_keywords: dict[str, str]) -> dict[str, Any]:
    """``dev_interests`` 값을 확장 사전으로 치환한 프로필 사본을 반환한다.

    원본 profile 은 변경하지 않는다 (다른 노드가 같은 normalized_profile 을
    원본 값으로 읽어야 함). 사전에 없는 값은 그대로 두어 새 흥미 분야가
    추가돼도 직렬화가 깨지지 않게 한다.
    """
    values = profile.get("dev_interests")
    if not values or not dev_keywords:
        return profile
    expanded = [dev_keywords.get(v, v) for v in values]
    return {**profile, "dev_interests": expanded}


def _serialize_profile(profile: dict[str, Any], template: dict[str, Any]) -> str:
    """카테고리형 프로필을 결정론적 자연어 문장으로 변환한다.

    템플릿의 절(clause) 정의를 선언 순서대로 순회하며, 값이 비어 있지 않은
    필드만 자연어 관형구로 만들어 이어 붙인다. 선택 입력(취업 가치·선호 회사
    유형·공부해본 분야)이 누락돼도 빈 문구 없이 문장이 완결되도록 빈 필드의
    절은 건너뛴다.

    ``completed_courses`` 및 학적 구조 필드(입학년도·소속·트랙)는 절 정의에
    없으므로 직렬화 대상에서 자연히 제외된다. 이수 과목은 집합 필터 대상이라
    의미 직렬화에 넣으면 관심사·흥미 의미를 희석한다.

    LLM 미사용이므로 동일 입력은 항상 동일 문장을 생성한다 (절 순서·값 순서·
    구분자가 모두 결정론적).
    """
    clauses: list[dict[str, str]] = template.get("clauses", [])
    value_separator: str = template.get("value_separator", ", ")
    clause_separator: str = template.get("clause_separator", ", ")
    suffix: str = template.get("suffix", "")

    rendered: list[str] = []
    for clause in clauses:
        values = profile.get(clause["field"]) or []
        if not values:
            continue
        rendered.append(clause["template"].format(value=value_separator.join(values)))

    return clause_separator.join(rendered) + suffix
