"""과목 메타 데이터 접근 boundary 추상화.

학습 로드맵 노드는 본 모듈의 ``CourseRepository`` Protocol 만 의존하며 구체
구현은 외부 어댑터로 분리된다. 자체 코드는 트랙 식별자만 boundary 너머로
넘기고, 강의계획서 메타 추출·단계 분류·선수 그래프 정규화는 boundary 안에서
일괄 처리된다.

단위 테스트는 ``list_for_tracks`` 를 in-memory 리스트로 대체하는 fake 구현체로
검증한다. 운영용 RAGFlow / DB / 파일 wrap 구현체는 별도 모듈로 분리한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tracktory.graph.models import Course


@runtime_checkable
class CourseRepository(Protocol):
    """과목 카탈로그 접근 인터페이스.

    학습 로드맵 노드는 본 Protocol 만 의존하고 구체 구현은 인프라 레이어에서
    주입된다. 호출 단위는 트랙 식별자 목록이며, 강의계획서 메타로부터 도출되는
    파생 필드는 모두 boundary 안에서 채워져 반환된다.

    구현체 contract:
        1. ``list_for_tracks`` 반환은 두 트랙 모두에 권장되는 과목이 섞여 있더라도
           ``course_id`` 기준으로 dedup 된 리스트여야 한다. 호출 측은 중복 없는
           카탈로그를 가정하고 후속 stream 알고리즘을 작성한다.
        2. ``Course`` 의 5 필드 — ``stage`` (4 단계 학습 깊이), ``prereq_ids``
           (정규화된 선수과목 식별자), ``priority`` (낮은 숫자 우선),
           ``available_grades`` (학년 제약), ``course_type`` (전공필수 / 전공선택
           / 교양) — 는 모두 구현체가 채워야 한다. ``Course`` 모델이 안전한
           기본값을 정의하고 있으나 추천 품질을 위해 강의계획서 메타로부터
           정확히 도출하는 것이 구현체 책임이다.
        3. 외부 호출 실패는 도메인 의미가 있는 예외로 변환하여 raise. raw 예외
           전파 금지.
        4. 동기 호출 contract — 학습 로드맵 노드 진입점이 단일 동기 호출로 1 회만
           부른다.
    """

    def list_for_tracks(self, track_ids: list[str]) -> list[Course]:
        """주어진 트랙 식별자들에 권장되는 과목 목록을 반환한다.

        두 트랙 모두에 권장되는 과목이 있더라도 ``course_id`` 기준으로 dedup 된
        리스트를 반환할 책임은 구현체에 있다.
        """
        ...
