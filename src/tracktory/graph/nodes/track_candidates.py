"""트랙 시너지 노드 진입부의 후보 페어 빌드 단계.

시너지 점수·다양성·슬롯 예약 산식이 다룰 입력 집합을 만드는 단일 책임을
가진다. 산식 자체는 트랙 시너지 노드에 있다.

분리 이유:
    후보 페어를 만드는 단계를 후처리 노드로 따로 두면 시너지 노드가 이미
    계산한 다양성 순서가 다시 한 번 재정렬되어 산식이 무력화된다. 후보
    페어 빌드를 시너지 노드와 같은 책임 단위의 진입부로 흡수하기 위해
    본 모듈로 분리한다.

순수 함수 원칙:
    외부 I/O 없음. 트랙 목록은 호출자가 외부에서 로드해 인자로 주입하므로
    단위 테스트는 in-memory 트랙 리스트만으로 검증할 수 있다.

1트랙 주전공 제약:
    학생이 트랙을 아직 선택하지 않은 학년 (``current_tracks`` 비어있음) 은
    학생의 주전공 학부에 속한 트랙을 1트랙 풀로 두고 모든 짝을 enumerate
    한다. 트랙을 이미 선택한 학년은 학생이 고른 트랙을 1트랙으로 고정하고
    나머지 모든 트랙과의 짝을 enumerate 한다.

    "주전공 소속" 의 단위는 학부 (예: 컴퓨터공학부) 이지 단과대 (예:
    IT공과대학) 가 아니다. 단과대 단위로 풀면 다른 학부의 트랙까지 1트랙
    후보로 등장하므로 학부 식별자를 인자로 받는다.

self-pair 제외 / 순서 무관 dedup:
    두 트랙 식별자를 정렬한 키로 dedup 하여 같은 두 트랙으로 구성된 페어가
    순서 차이로 두 번 포함되지 않도록 한다.
"""

from __future__ import annotations

from tracktory.graph.models import Track, TrackCombo


def build_candidate_pairs(
    tracks: list[Track],
    user_department_id: str,
    current_tracks: list[str],
) -> list[TrackCombo]:
    """학년 분기 + 1트랙 주전공 제약을 적용한 후보 페어 리스트.

    Args:
        tracks: 후보 풀이 될 전체 트랙 목록. 호출자가 외부에서 로드하여 주입한다.
        user_department_id: 학생 주전공 학부 식별자. 1트랙 풀을 학생 학부 안의
            트랙으로 제한하는 데 사용한다 (트랙 미선택 학년에 한정).
        current_tracks: 학생이 이미 선택한 트랙 식별자 목록. 비어있으면 트랙
            미선택 학년, 차 있으면 트랙 선택 학년 분기를 의미한다.

    Returns:
        후보 ``TrackCombo`` 리스트. self-pair 제외 + 두 트랙 식별자 정렬 키로
        dedup 한 결과. 후보가 없으면 빈 리스트.
    """
    by_id = {track.track_id: track for track in tracks}

    if current_tracks:
        primary_pool: list[Track] = [by_id[tid] for tid in current_tracks if tid in by_id]
    else:
        primary_pool = [track for track in tracks if track.department_id == user_department_id]

    seen: set[str] = set()
    combos: list[TrackCombo] = []
    for primary_track in primary_pool:
        for partner in tracks:
            if primary_track.track_id == partner.track_id:
                continue
            ids_sorted = sorted([primary_track.track_id, partner.track_id])
            key = "::".join(ids_sorted)
            if key in seen:
                continue
            seen.add(key)
            combos.append(TrackCombo(track_a=primary_track, track_b=partner, combo_key=key))
    return combos
