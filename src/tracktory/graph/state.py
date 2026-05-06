"""tracktory 추천 파이프라인의 LangGraph state schema.

필드를 세 부류로 분리한다.

- 입력 (불변): ``raw_input`` 등 이후 노드가 갱신하지 않는 값.
- 누적 (Annotated reducer): 여러 노드가 append 하는 값 (``errors``, ``trace``).
- 갱신 (overwrite): 단일 노드가 결과를 채우는 값.

노드는 GraphState 를 통째로 리턴하지 않고 변경된 필드만 담은 dict 를 리턴한다.

reducer 는 langgraph 의존성을 피하기 위해 ``operator.add`` 를 사용한다.
list 끼리 ``+`` 는 새 list 를 반환하므로 reducer 의미와 정합하다.
"""

from operator import add
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict, total=False):
    """tracktory 파이프라인 state.

    ``total=False`` 로 모든 필드를 optional 로 선언한다. 노드가 부분 state
    만 채워도 LangGraph 가 reducer 또는 overwrite 로 누적할 수 있다.

    필드 그룹:
        입력:
            user_id, raw_input
        정규화 출력:
            normalized_profile (dict 형태로 NormalizedProfile.model_dump 결과)
        임베딩 출력:
            profile_text, profile_vector
        직무 추천 출력:
            recommended_jobs (직무 매칭 노드의 결과 dict 리스트)
        누적:
            errors, trace
    """

    # --- 입력 (불변) ---
    user_id: str
    raw_input: dict[str, Any]

    # --- 정규화 출력 (이후 노드 read-only) ---
    normalized_profile: dict[str, Any] | None

    # --- 임베딩 출력 ---
    profile_text: str | None
    profile_vector: list[float] | None

    # --- 직무 추천 출력 ---
    recommended_jobs: list[dict[str, Any]] | None

    # --- 누적 (reducer = list concatenation) ---
    errors: Annotated[list[str], add]
    trace: Annotated[list[str], add]
