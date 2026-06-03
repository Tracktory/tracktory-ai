"""과목 이름 → 직무 기술 토큰 색인 — 추천 노드 공용 다리.

사용자는 과목을 *이름* (예: "데이터베이스") 으로 다루지만, 직무 적합도·역량
충족도는 기술 토큰 (예: "MySQL") 어휘로 매겨진다. 과목 카탈로그는 각 과목에
커리큘럼 기반 기술 토큰을 미리 부착해 두므로, 본 모듈이 이름 → 토큰 변환의
단일 진입점이 된다. 직무 매칭의 이수 과목 부스팅과 역량 커버리지 분석이 같은
다리를 공유해, 두 노드가 같은 어휘·같은 정규화 규칙으로 토큰을 본다.

본 모듈은 외부 I/O (yaml 로드) 외에는 순수 함수이며, 노드는 생성자에서 색인을
1 회 로드해 호출 경로에서는 파일 I/O 가 발생하지 않게 한다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# 노드들이 공유하는 기본 과목 카탈로그 경로. 단일 상수로 모아 두 노드가 같은
# 카탈로그를 가리키도록 보장한다 (직무 부스팅 ↔ 커버리지 분석 어휘 정합).
DEFAULT_COURSE_CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "courses.yaml"


def load_course_tech_index(path: Path) -> dict[str, list[str]]:
    """과목 카탈로그에서 ``{정규화 과목명: 기술 토큰 목록}`` 색인을 로드한다.

    키는 ``course_name.strip().casefold()`` 로 정규화해 입력 표기 차이를 흡수한다.
    같은 정규화 이름을 가진 과목이 둘 이상이면 기술 토큰을 합집합으로 묶되 중복을
    제거하고 최초 등장 순서를 보존한다. ``tech_stacks`` 가 비었거나 누락된 과목은
    토큰을 기여하지 않는다 (키 자체를 만들지 않음) — 카탈로그가 토큰을 채우기
    전에는 색인이 비어 부스팅·커버리지가 0 이 되도록 의도한 동작이다.

    Args:
        path: ``courses.yaml`` 의 경로.

    Returns:
        ``{정규화 과목명: [기술 토큰, ...]}`` 색인. 토큰이 전혀 없으면 ``{}``.

    Raises:
        ValueError: yaml 최상위가 ``courses`` 리스트를 담은 mapping 이 아닐 때.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not isinstance(raw.get("courses"), list):
        raise ValueError(f"Course catalog file {path} must contain a 'courses' list")

    index: dict[str, list[str]] = {}
    for entry in raw["courses"]:
        if not isinstance(entry, dict):
            continue
        name = entry.get("course_name")
        if not isinstance(name, str) or not name.strip():
            continue
        tech_stacks = entry.get("tech_stacks") or []
        if not isinstance(tech_stacks, list):
            continue
        key = name.strip().casefold()
        bucket = index.setdefault(key, [])
        for token in tech_stacks:
            if isinstance(token, str) and token.strip() and token not in bucket:
                bucket.append(token)
        if not bucket:
            # 토큰을 하나도 더하지 못한 빈 버킷은 색인에 남기지 않아,
            # 토큰 미보유 과목이 lookup 의 hit 로 위장하지 않게 한다.
            del index[key]
    return index


def resolve_course_tokens(
    course_names: list[str],
    course_tech_index: dict[str, list[str]],
) -> list[str]:
    """과목 이름 목록을 카탈로그 색인으로 직무 기술 토큰 목록으로 환산한다.

    교집합·커버리지 계산은 직무 기술 어휘로 이뤄지므로, 사용자·로드맵이 다루는
    과목 *이름* 을 같은 어휘의 토큰으로 먼저 옮겨야 한다. 본 함수가 그 이름 →
    토큰 다리이며, 색인에 없는 과목 이름은 아무 토큰도 기여하지 않는다 (무관
    과목이 결과에 영향을 주지 않도록).

    Args:
        course_names: 과목 이름 목록 (이수 과목 또는 로드맵 추천 과목).
        course_tech_index: ``load_course_tech_index`` 가 만든 이름 → 토큰 색인.

    Returns:
        모든 과목의 기술 토큰 합집합 (중복 제거, 최초 등장 순서 보존).
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for course in course_names:
        for token in course_tech_index.get(course.strip().casefold(), []):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


__all__ = [
    "DEFAULT_COURSE_CATALOG_PATH",
    "load_course_tech_index",
    "resolve_course_tokens",
]
