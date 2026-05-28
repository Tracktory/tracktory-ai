"""트랙 데이터 접근 boundary 추상화.

트랙 시너지 노드는 본 모듈의 ``TrackRepository`` Protocol 만 의존하며 구체
구현은 외부 어댑터로 분리된다. 자체 코드는 트랙 식별자만 boundary 너머로
넘기고, 트랙 카탈로그 로딩·메타 임베딩 벡터 준비는 boundary 안에서 일괄
처리된다. ADR-0001 의 단일 임베딩 boundary 원칙을 적재 책임 위치까지
일관되게 흡수시키는 형태이다.

단위 테스트는 ``list_all`` / ``find_by_track_ids`` 를 in-memory 리스트로 대체
하는 fake 구현체로 검증한다. 운영용 RAGFlow / DB / 파일 wrap 구현체는 별도
모듈로 분리한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tracktory.graph.models import Track


@runtime_checkable
class TrackRepository(Protocol):
    """트랙 카탈로그 접근 인터페이스.

    트랙 시너지 노드는 본 Protocol 만 의존하고 구체 구현은 인프라 레이어에서
    주입된다. 호출 단위는 트랙 전체 로드 또는 식별자 목록 조회이며, 적재 시점의
    임베딩 정합·정렬·dedup 책임은 모두 boundary 안에서 끝난다.

    구현체 contract:
        1. ``list_all`` / ``find_by_track_ids`` 반환의 각 ``Track.meta_vector`` 는
           ADR-0001 단일 임베딩 boundary 가 정의한 동일 공간의 L2 정규화 벡터다.
           차원·정규화 불일치는 트랙 메타 코사인을 의미 없는 값으로 무너뜨리므로
           적재 단계에서 차단한다.
        2. ``find_by_track_ids`` 는 입력 ``track_ids`` 에 없는 식별자가 섞여도
           조용히 무시하고 매칭되는 것만 반환한다. 호출 측이 부분 결과를 받아
           다운스트림에서 자체 검증한다.
        3. 외부 호출 실패 (네트워크 / DB / 파일 I/O / 응답 파싱) 는 도메인 의미가
           있는 예외로 변환하여 raise. raw 예외 전파 금지 — 트랙 시너지 노드의
           안전 종료 분기가 작동하지 않는다.
        4. 동기 호출 contract — 트랙 시너지 노드 진입점이 단일 동기 호출로 1 회만
           부른다. async 가 필요해지면 별도 Protocol 로 분리하고 본 Protocol 은
           변경하지 않는다.
    """

    def list_all(self) -> list[Track]:
        """전체 트랙 목록을 반환한다."""
        ...

    def find_by_track_ids(self, track_ids: list[str]) -> list[Track]:
        """주어진 ``track_ids`` 에 해당하는 트랙만 반환한다."""
        ...
