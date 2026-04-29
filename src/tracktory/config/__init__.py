"""tracktory 설정 yaml 자원 패키지.

본 패키지는 코드 모듈이 아닌 yaml 자원(``embedding.yaml``, ``n2_template.yaml``)
을 담는다. 다른 모듈은 명시적 경로 또는 ``importlib.resources`` 로 yaml 을
로드한다 (D-04 단일 boundary 강제).
"""
