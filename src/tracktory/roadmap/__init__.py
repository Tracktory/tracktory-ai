"""학습 로드맵 도메인 패키지.

선수과목 alias 정규화·로드맵 생성에 필요한 도메인 모델과 헬퍼를 모은다.
"""

from tracktory.roadmap.prereq import (
    AliasMap,
    default_alias_map_path,
    detect_alias_cycles,
    load_alias_map,
    resolve_alias,
)

__all__ = [
    "AliasMap",
    "default_alias_map_path",
    "detect_alias_cycles",
    "load_alias_map",
    "resolve_alias",
]
