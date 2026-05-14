"""챗봇 패키지 공유 설정·상수.

여러 모듈이 동일한 라벨·설정을 재정의하지 않도록 single source of truth 유지.
"""

from typing import Literal

ChatbotIntent = Literal[
    "track_question",  # 트랙
    "job_question",  # 직무
    "course_question",  # 과목
    "general_advice",  # 일반
]
