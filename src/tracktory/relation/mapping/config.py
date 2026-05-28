"""
직무·트랙 역량 매핑 파이프라인 경로 설정
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]

# 입력 파일
WANTED_CLEANED_PATH = _ROOT / "data" / "processed" / "wanted_cleaned.json"
JOB_STACKS_PATH = _ROOT / "data" / "processed" / "job_tech_stacks.json"
COURSES_CSV = _ROOT / "data" / "raw" / "hansung" / "courses.csv"
SYLLABUS_DIR = _ROOT / "data" / "processed" / "rag" / "output" / "syllabi_rename"

# 중간 파일
WANTED_BY_JOB_PATH = _ROOT / "data" / "processed" / "wanted_by_job.json"

# 출력 파일
JOB_COMPETENCY_OUT_PATH = _ROOT / "data" / "processed" / "job_competency_map.json"
TRACK_COMPETENCY_OUT_PATH = _ROOT / "data" / "processed" / "track_competency_map.json"

# 로그
TRACK_LOG_FILE = _ROOT / "logs" / "track_no_syllabus.log"
