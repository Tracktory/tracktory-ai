"""운영용 ``LLMClient`` 구현체 — OpenAI(LangChain) 구조화 출력 어댑터.

``llm_client.LLMClient`` Protocol 의 운영 구현으로, ``ChatOpenAI`` 에
``with_structured_output(Explanation)`` 를 걸어 LLM 출력을 Pydantic 스키마로
강제한다. 자연어 설명 노드는 본 클래스를 직접 import 하지 않고 ``LLMClient``
Protocol 타입으로만 주입받는다 (``RagflowClient`` / ``Yaml*Repository`` 와 대칭
되는 마지막 외부 호출 어댑터).

자격증명은 코드에 하드코딩하지 않고 ``OpenAILLMConfig.from_env`` 로 환경변수
(``OPENAI_API_KEY`` / ``OPENAI_MODEL``)에서 읽는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tracktory.graph.models import Explanation

__all__ = ["OpenAILLMClient", "OpenAILLMConfig"]

_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class OpenAILLMConfig:
    """OpenAI 호출 파라미터.

    자격증명은 코드/노트북에 하드코딩하지 않고 ``from_env`` 로 환경변수에서
    읽는다. 설명 단락은 결정성이 중요하므로 ``temperature`` 기본값은 0 이다.
    """

    api_key: str
    model: str = _DEFAULT_MODEL
    temperature: float = 0.0
    timeout: float = 30.0  # 외부 호출 타임아웃
    max_retries: int = 2  # rate limit / 일시 네트워크 오류 재시도

    @classmethod
    def from_env(cls) -> OpenAILLMConfig:
        """``OPENAI_*`` 환경변수에서 설정을 읽는다. API 키 누락 시 즉시 실패한다.

        ``OPENAI_MODEL`` 은 선택 — 미설정 시 ``gpt-4o-mini`` 로 fallback 한다.
        """
        import os

        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError as exc:
            raise RuntimeError(f"OpenAI 환경변수 누락: {exc}") from exc
        return cls(api_key=api_key, model=os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL))


class OpenAILLMClient:
    """``LLMClient`` Protocol 의 OpenAI 구조화 출력 어댑터.

    생성 시 ``with_structured_output(Explanation)`` 를 1회 바인딩하고, 이후
    ``invoke`` 는 메시지 리스트를 그대로 흘려 검증된 ``Explanation`` 을 받는다.
    Protocol contract 3 (외부 호출 실패 raw 전파) 에 따라 rate limit / 네트워크 /
    파싱 실패는 catch 하지 않고 전파한다 — 재시도는 ``max_retries`` 로 흡수한다.
    """

    def __init__(self, config: OpenAILLMConfig) -> None:
        # 부작용 격리·DI 원칙: 호출 측은 본 클래스가 아닌 Protocol 타입에 의존한다.
        self._structured = ChatOpenAI(
            model=config.model,
            api_key=SecretStr(config.api_key),
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
        ).with_structured_output(Explanation)

    def invoke(self, messages: list[BaseMessage]) -> Explanation:
        """메시지 리스트를 LLM 으로 보내고 구조화된 ``Explanation`` 을 받는다."""
        result = self._structured.invoke(messages)
        if not isinstance(result, Explanation):
            # with_structured_output 가 Explanation 을 보장하지만, 타입 좁힘 +
            # 계약 위반 조기 노출을 위해 방어한다 (예외를 삼키지 않고 surface).
            raise TypeError(f"구조화 출력이 Explanation 이 아님: {type(result).__name__}")
        return result
