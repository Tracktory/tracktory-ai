"""테스트 수트 전역 설정 — 외부 자격증명 없이 hermetic 하게 실행되도록 환경 격리.

``tracktory.common.config`` 는 모듈 임포트 시점에 ``Settings()`` 를 즉시
인스턴스화하며, 이때 RAGFlow 자격증명 4 종이 필수다. 실제 ``.env`` 가 없는
로컬·CI 환경에서 API·RAG 관련 테스트 모듈을 import 만 해도 collection 이
깨지므로, pytest 가 어떤 테스트 모듈보다 먼저 로드하는 본 conftest 에서
값을 주입한다. 외부 호출은 모두 모의 객체로 격리되므로 주입된 값이 실제로
사용되는 경로는 없다.

빈 문자열을 쓰는 이유: 필수 ``str`` 필드를 채워 import 는 통과시키되, 실제
RAGFlow 가 설정된 환경에서만 도는 integration 테스트 (자격증명 truthy 여부로
skip 을 판단) 를 hermetic 환경에서 깨우지 않기 위함이다.

``.env`` 를 먼저 로드한 뒤 ``setdefault`` 로 빈 값을 채운다: 실제 ``.env`` /
CI secret 이 있으면 그 값이 그대로 살아 integration 테스트가 정상 실행되고,
자격증명이 없을 때만 빈 문자열이 채워진다. (config 모듈의 ``load_dotenv`` 는
``override=False`` 라, conftest 가 먼저 빈 값을 박으면 실제 ``.env`` 가
가려지므로 로드 순서를 여기서 맞춘다.)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

os.environ.setdefault("RAGFLOW_API_KEY", "")
os.environ.setdefault("RAGFLOW_BASE_URL", "")
os.environ.setdefault("RAGFLOW_DATASET_ID", "")
os.environ.setdefault("RAGFLOW_RERANKER_ID", "")
