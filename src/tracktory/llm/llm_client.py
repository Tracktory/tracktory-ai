"""LLM 호출 boundary 추상화.

자연어 설명 생성 노드는 본 모듈의 ``LLMClient`` Protocol 만 의존하며 구체
구현은 외부 LLM 공급자 어댑터로 분리된다. 자체 코드는 메시지 리스트만 boundary
너머로 넘기고, API 호출·구조화 출력 강제·재시도 정책은 boundary 안에서 일괄
처리된다.

단위 테스트는 ``MagicMock(spec=LLMClient)`` 로 외부 호출 없이 검증한다. 운영용
어댑터 (예: ``ChatOpenAI(...).with_structured_output(Explanation)``) 는 별도
모듈로 분리한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.messages import BaseMessage

from tracktory.graph.models import Explanation


@runtime_checkable
class LLMClient(Protocol):
    """LLM 호출 인터페이스 — 단일 책임은 메시지 → 검증된 ``Explanation`` 변환.

    자연어 설명 생성 노드는 본 Protocol 만 의존하고 구체 구현은 인프라 레이어
    에서 주입된다. 호출 단위는 LangChain 메시지 리스트이며, 구조화 출력 검증·
    재시도·rate limit 대응은 모두 boundary 안에서 끝난다.

    구현체 contract:
        1. ``invoke`` 의 반환은 schema 검증을 통과한 ``Explanation`` 인스턴스다.
           구체 구현은 ``ChatOpenAI(...).with_structured_output(Explanation)``
           형태로 LangChain 의 구조화 출력 강제를 사용한다. 노드는 반환값을
           그대로 state 에 흘리므로 schema 위배는 구현체가 사전 차단한다.
        2. 메시지 리스트의 마지막 항목이 ``HumanMessage`` 라는 LangChain
           convention 을 가정한다. 노드는
           ``EXPLANATION_PROMPT.format_messages(...)`` 결과를 그대로 전달하여
           본 가정을 보장한다.
        3. 외부 호출 실패 (rate limit / 네트워크 / 파싱 실패) 는 raw 예외로
           전파해도 무방하다. 노드 본체는 fallback 분기 없이 LangGraph 전역
           에러 핸들러에 의존한다. retry 정책이 필요해지면 구현체에서 흡수한다.
        4. 동기 호출 contract — 자연어 설명 생성 노드 진입점이 단일 동기 호출로
           1 회만 부른다. async 가 필요해지면 별도 Protocol 로 분리한다.
    """

    def invoke(self, messages: list[BaseMessage]) -> Explanation:
        """메시지 리스트를 LLM 으로 보내고 구조화된 ``Explanation`` 을 받는다."""
        ...
