"""RAGFlow 기반 ``TrackRepository`` 구현체.

``track_repository.TrackRepository`` Protocol 의 운영 어댑터. retrieval(의미
검색)이 아니라 문서목록 API(``RagflowClient.list_documents``)로 트랙 문서를
**전수 나열**하므로 카탈로그 누락이 없다. retrieval 은 질문에 따라 결과가
흔들려 전수 수집이 불가능함이 실측으로 확인되었다.

조립 출처:
    - 트랙소개(track_intro) 문서 — ``meta_fields`` 에서 트랙명·소속(대학/학부).
    - 교육과정(curriculum) 문서 — 본문 "[과목구분] 과목명 (코드, N학점)" 줄에서
      추천 대상(전공) 과목코드를 추출해 ``course_ids`` 로. 전공 판정은 과목
      저장소와 ``curriculum_lines.parse_course_line`` 을 공유해 교양류가 섞이지
      않는다. 트랙명으로 트랙소개와 짝짓는다.

``meta_text`` 는 현재 어떤 노드도 읽지 않아(모델상 "디버깅용 보존") 빈 문자열로
둔다 — 채우려면 트랙마다 소개 본문 fetch 가 필요한데 소비처가 없다.

``meta_vector`` 는 현재 빈 리스트로 둔다(트랙 메타 임베딩 보류). 트랙 시너지의
메타 코사인 항은 ADR-0001 임베딩이 적재되기 전까지 0.0 으로 graceful degrade
한다. ``competencies``/``tech_stacks`` 도 RAGFlow 미보유라 빈 리스트.

호출 측은 ``RagflowTrackRepository`` 를 직접 import 하지 않고 ``TrackRepository``
Protocol 타입으로만 주입받는다.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from tracktory.graph.models import Track
from tracktory.rag.curriculum_lines import parse_course_line
from tracktory.rag.hansung_catalog import match_track_name
from tracktory.rag.ragflow_client import RagflowClient, RagflowDocument, RagflowError

__all__ = ["RagflowTrackRepository", "TrackRepositoryError"]

logger = logging.getLogger(__name__)

# 문서 파일명 부분일치 필터 키워드 (예: 트랙소개_*.txt / 교육과정_*.txt). 호출량을
# 줄이는 1차 narrowing 일 뿐이며, 정밀 필터는 doc_type 메타로 한다 — 파일명에
# "교육과정" 이 든 강의계획서(예: "한국어 교육과정의 이해와 적용")가 키워드에
# 딸려오는 false positive 를 doc_type 으로 차단한다.
_TRACK_INTRO_KEYWORD = "트랙소개"
_TRACK_INTRO_DOC_TYPE = "track_intro"
_CURRICULUM_KEYWORD = "교육과정"
_CURRICULUM_DOC_TYPE = "curriculum"


class TrackRepositoryError(Exception):
    """트랙 카탈로그 로딩 실패.

    RAGFlow 호출/응답 파싱 실패를 도메인 예외로 변환한다. 트랙 시너지 노드의
    안전 종료 분기가 raw 예외가 아닌 본 예외를 보도록 boundary 에서 좁힌다.
    """


class RagflowTrackRepository:
    """``TrackRepository`` Protocol 의 RAGFlow 문서목록 어댑터.

    HTTP 호출·재시도·에러 변환·페이지네이션은 주입된 ``RagflowClient`` 가
    담당하고, 본 클래스는 트랙소개·교육과정 문서 → ``Track`` 변환만 책임진다.
    """

    def __init__(self, client: RagflowClient) -> None:
        # 부작용 격리·DI 원칙: 테스트에서 fake client 주입 가능.
        self._client = client

    def list_all(self) -> list[Track]:
        """전체 트랙 목록을 반환한다."""
        return self._build_tracks(wanted=None)

    def find_by_track_ids(self, track_ids: list[str]) -> list[Track]:
        """주어진 ``track_ids`` 에 해당하는 트랙만 반환한다.

        ``track_id`` 는 트랙명과 동일하다. 입력에 없는 식별자는 조용히 무시하고
        매칭되는 것만 반환한다(Protocol contract). 본문 조회는 매칭된 트랙에
        대해서만 수행한다.
        """
        wanted = set(track_ids)
        if not wanted:
            return []
        return self._build_tracks(wanted=wanted)

    def _build_tracks(self, *, wanted: set[str] | None) -> list[Track]:
        """트랙소개를 나열하고 교육과정과 짝지어 ``Track`` 리스트를 만든다.

        ``wanted`` 가 ``None`` 이면 전체, set 이면 해당 트랙명만 대상으로 한다.
        문서목록 나열 실패는 ``TrackRepositoryError`` 로 변환한다.
        """
        try:
            intro_docs = self._list_by_type(_TRACK_INTRO_KEYWORD, _TRACK_INTRO_DOC_TYPE)
            curriculum_by_name = self._index_curriculum_by_track()
        except RagflowError as exc:
            raise TrackRepositoryError(f"트랙 카탈로그 나열 실패: {exc}") from exc

        tracks: list[Track] = []
        for doc in intro_docs:
            track_name = self._meta_str(doc, "track_name")
            if track_name is None:
                logger.warning("track_name 없는 트랙소개 문서 스킵: %s", doc.name)
                continue
            if wanted is not None and track_name not in wanted:
                continue
            track = self._build_one(doc, track_name, curriculum_by_name)
            if track is not None:
                tracks.append(track)
        return tracks

    def _list_by_type(self, keyword: str, doc_type: str) -> list[RagflowDocument]:
        """파일명 키워드로 1차 narrowing 후 doc_type 메타로 정밀 필터한다.

        파일명에 키워드가 우연히 포함된 다른 doc_type 문서(예: 과목명에
        "교육과정" 이 든 강의계획서)를 doc_type 으로 걸러낸다.
        """
        return [
            doc
            for doc in self._client.list_documents(name_keyword=keyword)
            if doc.meta_fields.get("doc_type") == doc_type
        ]

    def _index_curriculum_by_track(self) -> dict[str, RagflowDocument]:
        """교육과정 문서를 정규화 트랙명 → 문서로 색인한다(메타만, 본문 미조회)."""
        index: dict[str, RagflowDocument] = {}
        for doc in self._list_by_type(_CURRICULUM_KEYWORD, _CURRICULUM_DOC_TYPE):
            name = self._meta_str(doc, "track_name")
            if name is not None:
                index[match_track_name(name)] = doc
        return index

    def _build_one(
        self,
        intro_doc: RagflowDocument,
        track_name: str,
        curriculum_by_name: dict[str, RagflowDocument],
    ) -> Track | None:
        """트랙소개 문서 1건 + 매칭 교육과정 → ``Track``. 필수 메타 누락 시 None."""
        college = self._meta_str(intro_doc, "college")
        department = self._meta_str(intro_doc, "department")
        if college is None or department is None:
            logger.warning("소속 메타 누락 트랙 스킵: %s", track_name)
            return None

        try:
            course_ids: list[str] = []
            curriculum = curriculum_by_name.get(match_track_name(track_name))
            if curriculum is not None:
                body = self._client.fetch_document_text(curriculum.document_id)
                course_ids = self._extract_course_ids(body)
        except RagflowError as exc:
            raise TrackRepositoryError(f"트랙 본문 로딩 실패({track_name}): {exc}") from exc

        try:
            return Track(
                college_id=college,
                department_id=department,
                # 데이터에 전공(T3) 개념이 없어 single-track degenerate
                # (모델 docstring: major_id = department_id 허용).
                major_id=department,
                track_id=track_name,
                track_name=track_name,
                course_ids=course_ids,
                # meta_text 는 현재 어떤 노드도 읽지 않는다(모델상 "디버깅용 보존").
                # 채우려면 트랙마다 소개 본문 fetch 가 필요한데 소비처가 없어 생략.
                meta_text="",
                meta_vector=[],  # 트랙 메타 임베딩 보류 → 시너지 메타항 0.0 degrade.
                competencies=[],  # RAGFlow 미보유.
                tech_stacks=[],  # RAGFlow 미보유.
            )
        except ValidationError as exc:
            logger.warning("Track 검증 실패 스킵(%s): %s", track_name, exc)
            return None

    @staticmethod
    def _extract_course_ids(curriculum_text: str) -> list[str]:
        """교육과정 본문에서 추천 대상(전공) 과목코드를 등장 순서대로 dedup 추출한다.

        전공 판정은 과목 저장소와 동일한 ``parse_course_line`` 을 쓴다 — 과목구분
        태그가 전공 계열이 아니면(교양류 등) ``None`` 이 되어 자연히 제외된다.
        과목 카탈로그에 대응 ``Course`` 가 없는 코드가 ``course_ids`` 에 새어
        들어가지 않도록 두 저장소가 같은 기준을 공유한다.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for line in curriculum_text.splitlines():
            parsed = parse_course_line(line)
            if parsed is None:
                continue
            if parsed.course_id not in seen:
                seen.add(parsed.course_id)
                ordered.append(parsed.course_id)
        return ordered

    @staticmethod
    def _meta_str(doc: RagflowDocument, key: str) -> str | None:
        """``meta_fields`` 에서 비어있지 않은 문자열만 꺼낸다."""
        value = doc.meta_fields.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
