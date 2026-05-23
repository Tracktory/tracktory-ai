"""추천 파이프라인 그래프의 조건부 라우팅 함수.

조건부 엣지는 노드와 분리된 모듈에 두어 단일 책임을 유지한다. 라우팅
함수는 부작용이 없는 순수 함수로 작성되어, ``GraphState`` 만 읽고 다음
노드 라벨 문자열만 반환한다.

반환 라벨은 ``Literal`` 로 고정되어 타입 체커가 오타 / 누락을 잡는다.
그래프 빌더가 본 함수의 반환 라벨을 LangGraph 의 노드명 (또는 ``END``
sentinel) 으로 매핑한다.
"""

from typing import Literal

from tracktory.graph.state import GraphState


def route_after_input_normalize(
    state: GraphState,
) -> Literal["continue", "end"]:
    """입력 정규화 결과를 보고 다음 노드로 진행할지 안전 종료할지 결정한다.

    ``normalize_input`` 은 검증 실패 시 ``normalized_profile`` 키를 흘리지
    않고 ``errors`` 와 ``trace`` 만 채운다. 본 함수는 그 키 부재를 안전
    종료 신호로 해석한다.

    Returns:
        "continue" — 정규화 성공. 다음 노드로 진행.
        "end" — 정규화 실패. 그래프 ``END`` 로 안전 종료.
    """
    return "continue" if state.get("normalized_profile") else "end"
