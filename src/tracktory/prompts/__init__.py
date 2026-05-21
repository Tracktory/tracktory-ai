"""프롬프트 템플릿 모듈.

LLM 호출 노드는 본 모듈의 ``ChatPromptTemplate`` 만 import 해 사용하고,
인라인 f-string 으로 프롬프트를 조립하지 않는다. 이로써 프롬프트 변경이
독립적인 diff 단위가 되어 리뷰가 용이해진다.
"""
