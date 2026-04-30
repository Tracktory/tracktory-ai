"""tracktory 설정 yaml 자원 패키지.

본 패키지는 코드 모듈이 아닌 yaml 자원(``embedding.yaml``, ``profile_embed_template.yaml``)
을 담는다. 다른 모듈은 명시적 경로 또는 ``importlib.resources`` 로 yaml 을
로드한다. 모든 노드는 동일한 ``embedding.yaml`` 을 공유해야 한다 (ADR-0001).
"""
