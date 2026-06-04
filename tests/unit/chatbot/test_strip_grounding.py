"""strip_grounding 단위 테스트 — 본문 끝 `[근거: ...]` 줄 제거 (LLM 미사용)"""

from __future__ import annotations

import pytest

from tracktory.prompts.chatbot.rag_response import strip_grounding


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 줄바꿈 뒤 근거 줄
        (
            "웹공학트랙은 클라이언트-서버 기술을 배워요.\n[근거: #1, #3]",
            "웹공학트랙은 클라이언트-서버 기술을 배워요.",
        ),
        # 같은 줄 끝에 인라인
        ("두 트랙 모두 좋은 선택이에요. [근거: #1, #3]", "두 트랙 모두 좋은 선택이에요."),
        # 근거 없음 마커
        ("현재 자료로는 답하기 어려워요.\n[근거: 없음]", "현재 자료로는 답하기 어려워요."),
        # 단일 인덱스
        ("데이터베이스부터 들어 보면 좋아요.\n[근거: #2]", "데이터베이스부터 들어 보면 좋아요."),
        # 여러 줄 본문 — 내부 줄바꿈은 보존, 마지막 근거 줄만 제거
        (
            "여러 줄\n답변이에요.\n다음 단계도 좋아요.\n[근거: #2]",
            "여러 줄\n답변이에요.\n다음 단계도 좋아요.",
        ),
        # 근거 줄 뒤 잔여 공백·개행
        ("포트폴리오를 챙겨 보세요.\n[근거: #1]\n\n  ", "포트폴리오를 챙겨 보세요."),
        # 근거 줄이 아예 없으면 원본 유지
        ("포트폴리오 관리가 중요해요.", "포트폴리오 관리가 중요해요."),
    ],
)
def test_strip_grounding(text: str, expected: str) -> None:
    result = strip_grounding(text)
    assert result == expected
    assert "[근거:" not in result
