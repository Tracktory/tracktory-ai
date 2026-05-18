"""강의계획서의 원시 선수과목 표기를 정규 교과목명으로 환원한다.

학사 자유 텍스트는 표기 변종·잡음이 섞여 있어 같은 교과목이 여러 형태로
등장한다. 학습 로드맵 단계에서 교과목 식별이 깨지지 않도록 본 모듈이
원시 텍스트 → 정규 교과목명 매핑을 단일 진입점으로 제공한다.

데이터 단위는 ``{원시_텍스트: 정규_교과목명 | None}`` 형태의 flat 사전이며,
값이 ``None`` 이면 정규화 불가(지나치게 일반적이거나 잡음) 임을 명시한다.
alias 매핑이 체인을 이루는 경우 ``resolve_alias`` 가 차례로 따라간다.
악성 입력 대비 안전망으로 깊이 상한과 path 기반 cycle 감지를 둔다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MAX_RESOLVE_DEPTH = 8
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class AliasMap(BaseModel):
    """원시 텍스트 → 정규 교과목명 매핑.

    Attributes:
        entries: alias 키와 정규 교과목명 값을 보관하는 사전. 값이 ``None`` 인 entry
            는 정규화 불가 표시이며, ``resolve_alias`` 가 ``None`` 을 반환하는 신호로
            사용된다.
    """

    model_config = ConfigDict(frozen=True)

    entries: dict[str, str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_empty_keys(self) -> AliasMap:
        for key in self.entries:
            if not key:
                raise ValueError("alias map 의 키는 빈 문자열일 수 없습니다.")
        return self


def default_alias_map_path() -> Path:
    """기본 alias 사전 파일 경로.

    조회 순서:
        1. ``TRACKTORY_DATA_DIR`` 환경변수가 설정되어 있으면 그 디렉토리의
           ``prereq_alias.json`` 을 가리킨다.
        2. 아니면 패키지 위치 기준 ``../../../../data/processed/prereq_alias.json``
           을 반환한다 (로컬 개발 편의용 — editable install 전제).
    """
    env_dir = os.environ.get("TRACKTORY_DATA_DIR")
    if env_dir:
        return Path(env_dir) / "prereq_alias.json"
    return _PACKAGE_ROOT / "data" / "processed" / "prereq_alias.json"


def load_alias_map(path: Path) -> AliasMap:
    """JSON 파일에서 alias 매핑을 로드한다.

    파일은 flat ``{str: str | None}`` 형태를 가정한다. 그 외 구조 (list, 중첩 dict,
    숫자형 value 등) 는 ``ValueError`` 로 거부한다.

    Args:
        path: JSON 파일 경로.

    Returns:
        검증을 통과한 ``AliasMap``.

    Raises:
        FileNotFoundError: 경로의 파일이 존재하지 않는 경우.
        ValueError: JSON 파싱 실패 또는 최상위가 dict 가 아니거나 value 가
            ``str | None`` 외 타입인 경우.
    """
    text = path.read_text(encoding="utf-8")
    try:
        raw: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"alias map JSON 디코드 실패. path={path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"alias map JSON 최상위는 객체여야 합니다. got={type(raw).__name__}")

    entries: dict[str, str | None] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"alias 키는 문자열이어야 합니다. got={type(key).__name__}")
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"alias 값은 문자열 또는 null 이어야 합니다. key={key!r} got={type(value).__name__}"
            )
        entries[key] = value
    return AliasMap(entries=entries)


def resolve_alias(
    text: str,
    alias_map: AliasMap,
    *,
    max_depth: int | None = None,
) -> str | None:
    """원시 텍스트를 정규 교과목명으로 환원한다.

    동작 규약:
        * 정확 매칭만 사용한다. 키 자체가 freeform 잡음 텍스트라서 substring
          매칭은 false-positive 위험이 크다.
        * 매핑이 없으면 입력 텍스트 그대로 반환한다 (pass-through). 호출자는
          ``None`` (정규화 불가 명시) 과 pass-through (사전 미등재) 를
          의미적으로 구분해야 한다.
        * 매핑 값이 ``None`` 이면 ``None`` 을 반환한다.
        * 결과가 다시 alias 키에 존재하면 다음 단계로 따라간다.
        * 방문한 노드 집합으로 cycle 을 감지하고, 깊이 상한을 초과하면
          ``RuntimeError`` 를 발생시킨다. 사전 검증을 위해
          ``detect_alias_cycles`` 를 사용하는 것을 권장한다.

    Args:
        text: 원시 텍스트.
        alias_map: alias 매핑.
        max_depth: alias 체인 최대 깊이. ``None`` 이면 모듈 기본값 사용.
            테스트용 작은 값 주입을 허용한다.

    Returns:
        정규 교과목명, 또는 정규화 불가 시 ``None``.

    Raises:
        RuntimeError: alias 체인이 cycle 또는 깊이 상한을 초과하는 경우.
    """
    entries = alias_map.entries
    if text not in entries:
        return text

    depth_limit = max_depth if max_depth is not None else _MAX_RESOLVE_DEPTH
    visited: set[str] = set()
    current: str = text
    for _ in range(depth_limit):
        if current in visited:
            raise RuntimeError(f"alias 체인에 cycle 이 감지되었습니다. start={text!r}")
        visited.add(current)

        next_value = entries.get(current)
        if next_value is None:
            return None
        if next_value not in entries:
            return next_value
        current = next_value

    raise RuntimeError(f"alias 체인 깊이가 {depth_limit} 를 초과했습니다. start={text!r}")


def detect_alias_cycles(alias_map: AliasMap) -> list[list[str]]:
    """alias 해석 체인에 cycle 이 있는지 검사한다.

    각 alias 키에서 출발해 값이 또 다른 alias 키인 동안 따라가며 path 를 누적한다.
    같은 노드를 두 번 만나면 cycle 로 보고, 시작점을 정렬 최소 노드로 회전시켜
    동일 cycle 의 중복 보고를 방지한다.

    Args:
        alias_map: 검사할 alias 매핑.

    Returns:
        cycle 경로 목록. 각 경로는 cycle 의 노드 순서를 담고, acyclic 이면 빈 리스트.
    """
    entries = alias_map.entries
    seen: set[tuple[str, ...]] = set()
    cycles: list[list[str]] = []

    for start in entries:
        path: list[str] = []
        position: dict[str, int] = {}
        current: str | None = start
        while current is not None:
            if current in position:
                cycle_nodes = path[position[current] :]
                canonical = _canonicalize_cycle(cycle_nodes)
                if canonical not in seen:
                    seen.add(canonical)
                    cycles.append(list(canonical))
                break
            if current not in entries:
                break
            position[current] = len(path)
            path.append(current)
            current = entries[current]

    return cycles


def _canonicalize_cycle(nodes: list[str]) -> tuple[str, ...]:
    """cycle 의 시작점을 정렬 최소 노드로 회전해 중복 감지를 단순화한다."""
    pivot = min(range(len(nodes)), key=lambda i: nodes[i])
    rotated = nodes[pivot:] + nodes[:pivot]
    return tuple(rotated)
