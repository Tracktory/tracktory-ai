"""validate_prerequisites 의 그래프 무결성 검증 동작을 확인한다.

데이터 무결성을 보장하는 검증 함수라 회귀가 발생하면 잘못된 prerequisites.json
이 무음으로 통과될 수 있다. self-loop·n-사이클·dangling 케이스를 고정한다.
"""

from tracktory.relation.prerequisite.build_prerequisites import validate_prerequisites


def test_validate_passes_for_normal_dag() -> None:
    """정상 DAG: A → B → C 사슬과 별개 D 노드."""
    graph = {"A": ["B"], "B": ["C"], "C": [], "D": []}
    assert validate_prerequisites(graph) is True


def test_validate_passes_for_empty_graph() -> None:
    assert validate_prerequisites({}) is True


def test_validate_detects_self_loop() -> None:
    """A → A 자기 자신 참조."""
    assert validate_prerequisites({"A": ["A"]}) is False


def test_validate_detects_two_node_cycle() -> None:
    """A → B → A 의 2-사이클."""
    assert validate_prerequisites({"A": ["B"], "B": ["A"]}) is False


def test_validate_detects_three_node_cycle() -> None:
    """A → B → C → A 의 3-사이클."""
    assert validate_prerequisites({"A": ["B"], "B": ["C"], "C": ["A"]}) is False


def test_validate_passes_with_dangling_reference() -> None:
    """graph 에 노드가 없는 선수과목 참조는 경고로 분류되어 통과한다."""
    graph = {"A": ["X"]}  # X 는 graph 의 키가 아님
    assert validate_prerequisites(graph) is True
