"""
IT 직무 카테고리 정의 및 원티드 태그 매핑

크롤러, 에이전트, API 전체에서 공유하는 카테고리 정보.
"""

# IT 직무 카테고리별 텍스트 매칭 키워드
IT_JOB_CATEGORIES: dict[str, list[str]] = {
    "프론트엔드": ["프론트엔드", "프론트", "frontend", "front-end", "퍼블리셔"],
    "백엔드": ["백엔드", "백앤드", "backend", "back-end", "서버개발", "서버 개발"],
    "풀스택": ["풀스택", "fullstack", "full-stack", "풀 스택"],
    "데이터엔지니어": ["데이터 엔지니어", "데이터엔지니어", "data engineer"],
    "데이터분석": ["데이터 분석", "데이터분석", "data analyst", "데이터 사이언"],
    "데이터사이언스": ["데이터 사이언", "데이터사이언", "data scien"],
    "AI/ML": ["머신러닝", "machine learning", "딥러닝", "deep learning", "ai ", "인공지능", "ml engineer", "nlp", "llm", "생성ai", "생성 ai"],
    "모바일": ["안드로이드", "android", "ios", "모바일", "mobile", "flutter", "react native", "swift", "kotlin"],
    "DevOps/인프라": ["devops", "데브옵스", "sre", "인프라", "클라우드", "platform engineer"],
    "보안": ["보안", "security", "시큐리티", "취약점", "모의해킹", "pentest"],
    "게임": ["게임", "game", "unity", "unreal", "언리얼"],
    "QA": ["qa", "테스트", "품질"],
    "임베디드": ["임베디드", "embedded", "펌웨어", "firmware", "IoT"],
    "블록체인": ["블록체인", "blockchain", "스마트 컨트랙트", "web3"],
    "DBA": ["dba", "데이터베이스 관리"],
}

# 원티드 카테고리별 tag_id 매핑
# 각 tag_id는 원티드 서브카테고리 URL에 대응: wdlist/518/{tag_id}
WANTED_CATEGORY_TAGS: dict[str, list[int]] = {
    "백엔드": [872],
    "프론트엔드": [669],
    "풀스택": [873],
    "AI/ML": [1634],
    "데이터엔지니어": [655, 1025],
    "데이터사이언스": [1024],
    "DevOps/인프라": [674],
    "모바일": [677, 678],
    "QA": [676],
    "임베디드": [658],
    "블록체인": [1027],
}
# NOTE: 보안, 게임, DBA, 데이터분석 have no direct wanted tags
# They are collected from the main list and classified by text
