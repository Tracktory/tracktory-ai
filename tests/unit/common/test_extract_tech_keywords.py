"""``extract_tech_keywords`` 추출 정밀도 회귀 가드.

강의계획서 산문에서 토큰을 뽑을 때 흔한 한국어 단어가 기술로 오인식되면,
직무와 무관한 과목이 직무 점수를 부풀린다. 가장 잦았던 두 오탐 부류를
고정한다: (1) 한국어 "뷰"(view) → Vue.js, (2) 이메일·URL 의 도메인 꼬리
``.net`` → .NET. 둘 다 추출되어선 안 되고, 영문 정식 표기는 보존돼야 한다.
"""

from __future__ import annotations

import pytest

from tracktory.common.tech_keywords import extract_tech_keywords


@pytest.mark.parametrize(
    "prose",
    [
        "중간고사 리뷰 및 질의응답",  # 리뷰
        "뷰티산업 디자인 기초",  # 뷰티
        "면접 인터뷰 준비",  # 인터뷰
        "다양한 위젯(뷰)를 사용하여 화면을 구성한다",  # UI view (괄호)
        "뷰, 저장 프로시저, 트리거를 정의한다",  # DB view (쉼표)
        "뷰의 계층적 구조를 학습한다",  # view (한글 인접)
    ],
)
def test_korean_view_word_does_not_extract_vue(prose: str) -> None:
    """한국어 "뷰"(view) 는 Vue.js 로 추출되지 않는다 (가장 잦았던 오탐)."""
    assert "Vue.js" not in extract_tech_keywords(prose)


def test_english_vue_is_still_extracted() -> None:
    """영문 Vue 표기는 그대로 추출된다 (정상 신호 보존)."""
    assert "Vue.js" in extract_tech_keywords("Vue.js 와 React 로 SPA 를 구현한다")
    assert "Vue.js" in extract_tech_keywords("프론트엔드는 VueJS 를 사용한다")


def test_asp_dotnet_in_prose_is_still_extracted() -> None:
    """본문의 정식 .NET 표기는 보존된다 (도메인 오탐 제거와 무관)."""
    assert ".NET" in extract_tech_keywords("ASP.NET 으로 서버를 구현한다")
    assert ".NET" in extract_tech_keywords(".NET Core 기반 마이크로서비스")
