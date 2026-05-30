"""LLM 호출 boundary 모듈.

추천 파이프라인이 LLM 어댑터에 의존할 때 사용하는 Protocol 들의 단일 위치.
RAG 검색 boundary (``tracktory.rag``) 와 대칭되는 외부 호출 경계로, 향후 LLM
공급자별 어댑터 (OpenAI / Anthropic / fake) 는 모두 본 패키지에 위치한다.
"""

from tracktory.llm.llm_client import LLMClient

__all__ = ["LLMClient"]
