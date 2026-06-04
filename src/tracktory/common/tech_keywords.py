"""
기술 키워드 추출 및 정규화 통합 모듈
=====================================

채용공고 텍스트에서 기술 스택 키워드를 추출하고, 다양한 표기를
정규(canonical) 이름으로 통합하는 공용 모듈이다.

주요 기능:
    - extract_tech_keywords: 자유 텍스트에서 기술 키워드 추출 (사람인 채용공고용)
    - normalize_tech_tag: 단일 태그 정규화 (원티드 구조화 태그용)
    - normalize_tech_tags: 태그 리스트 일괄 정규화
    - classify_job_category: 채용공고 제목으로 IT 직군 분류
    - canonical_tech_token: 표기 차이를 흡수한 정합 토큰 (사전 미스 시 원본 표기 보존)
    - canonical_tech_key: 대소문자 무시 비교 키 (집합 교집합용)
    - canonical_tech_keys: 태그 묶음을 비교 키 집합으로 변환

사용법:
    from common.tech_keywords import extract_tech_keywords, normalize_tech_tag

    techs = extract_tech_keywords("Java 및 JavaScript 경험자")
    # -> ["Java", "JavaScript"]

    tag = normalize_tech_tag("ReactJS")
    # -> "React"
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# 1. 정규화 맵 (변형 -> 정규 이름)
# ---------------------------------------------------------------------------
NORMALIZATION_MAP: dict[str, str] = {
    # -- JavaScript / TypeScript 계열 --
    "ReactJS": "React",
    "React.js": "React",
    "react.js": "React",
    "reactjs": "React",
    "리액트": "React",
    "VueJS": "Vue.js",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "뷰": "Vue.js",
    "AngularJS": "Angular",
    "앵귤러": "Angular",
    "NextJS": "Next.js",
    "next.js": "Next.js",
    "NuxtJS": "Nuxt.js",
    "nuxt.js": "Nuxt.js",
    "NodeJS": "Node.js",
    "node.js": "Node.js",
    "노드": "Node.js",
    "NestJS": "NestJS",
    "타입스크립트": "TypeScript",
    "자바스크립트": "JavaScript",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "ES6": "JavaScript",
    "ES2024": "JavaScript",
    # -- 프론트엔드 도구 --
    "Tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "Material UI": "MUI",
    "MobX": "MobX",
    # -- Java / JVM 계열 --
    "자바": "Java",
    "코틀린": "Kotlin",
    "스프링": "Spring",
    "Spring Boot": "Spring Boot",
    "SpringBoot": "Spring Boot",
    "스프링부트": "Spring Boot",
    "Golang": "Go",
    "golang": "Go",
    # -- Python 계열 --
    "파이썬": "Python",
    "python3": "Python",
    "Scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "사이킷런": "scikit-learn",
    "판다스": "Pandas",
    "넘파이": "NumPy",
    "텐서플로": "TensorFlow",
    "텐서플로우": "TensorFlow",
    "파이토치": "PyTorch",
    "케라스": "Keras",
    "장고": "Django",
    # -- C 계열 --
    "씨샵": "C#",
    "씨쁠쁠": "C++",
    "Objective C": "Objective-C",
    # -- .NET 계열 --
    "dotnet": ".NET",
    "ASP.NET": ".NET",
    "asp.net": ".NET",
    # -- 데이터베이스 --
    "Postgres": "PostgreSQL",
    "postgres": "PostgreSQL",
    "MySQL": "MySQL",
    "몽고디비": "MongoDB",
    "몽고DB": "MongoDB",
    "마리아디비": "MariaDB",
    "MariaDB": "MariaDB",
    "레디스": "Redis",
    "ElasticSearch": "Elasticsearch",
    "MSSQL": "MS-SQL",
    "MS SQL": "MS-SQL",
    # -- 클라우드 / 인프라 --
    "도커": "Docker",
    "docker": "Docker",
    "쿠버네티스": "Kubernetes",
    "K8s": "Kubernetes",
    "k8s": "Kubernetes",
    "테라폼": "Terraform",
    "앤서블": "Ansible",
    "젠킨스": "Jenkins",
    "GitHub Actions": "GitHub Actions",
    "GitLab CI": "GitLab CI",
    # -- 데이터/ML --
    "머신러닝": "Machine Learning",
    "딥러닝": "Deep Learning",
    "자연어처리": "NLP",
    "컴퓨터비전": "Computer Vision",
    "빅데이터": "Big Data",
    "데이터파이프라인": "Data Pipeline",
    "데이터웨어하우스": "Data Warehouse",
    "랭체인": "LangChain",
    # -- 모바일 --
    "안드로이드": "Android",
    "Jetpack Compose": "Jetpack Compose",
    "React Native": "React Native",
    "리액트네이티브": "React Native",
    "플러터": "Flutter",
    # -- 기타 --
    "마이크로서비스": "MSA",
    "Microservice": "MSA",
    "Microservices": "MSA",
    "RESTful": "REST",
    "restful": "REST",
}


# ---------------------------------------------------------------------------
# 2. 기술 키워드 패턴 목록 (regex, canonical_name)
# ---------------------------------------------------------------------------
# 패턴 구성 원칙:
#   - 긴 패턴을 먼저 매칭하여 짧은 패턴이 잘못 매칭되는 것을 방지한다.
#   - Java vs JavaScript, C vs C++ vs C# 등 혼동 방지를 위해 lookahead/lookbehind 사용.
#   - 모든 패턴은 re.IGNORECASE로 컴파일된다.
# ---------------------------------------------------------------------------

_RAW_TECH_KEYWORDS: list[tuple[str, str]] = [
    # ── 프레임워크 / 복합 이름 (긴 패턴 우선) ────────────────────────────────
    (r"React\s*Native", "React Native"),
    (r"Spring\s*Boot", "Spring Boot"),
    (r"Spring\s*Batch", "Spring Batch"),
    (r"Spring\s*MVC", "Spring MVC"),
    (r"Ruby\s+on\s+Rails", "Ruby on Rails"),
    (r"Tailwind\s*CSS", "Tailwind CSS"),
    (r"Material\s*UI", "MUI"),
    (r"Ant\s*Design", "Ant Design"),
    (r"Jetpack\s*Compose", "Jetpack Compose"),
    (r"GitHub\s*Actions", "GitHub Actions"),
    (r"GitLab\s*CI", "GitLab CI"),
    (r"Power\s*BI", "Power BI"),
    (r"Kotlin\s*Multiplatform|KMP", "Kotlin Multiplatform"),
    (r"Unreal\s*Engine", "Unreal Engine"),
    (r"React\.?js|ReactJS", "React"),
    (r"Vue\.?js|VueJS", "Vue.js"),
    (r"Next\.js|NextJS", "Next.js"),
    (r"Nuxt\.js|NuxtJS", "Nuxt.js"),
    (r"Node\.js|NodeJS", "Node.js"),
    (r"Socket\.io", "Socket.io"),
    (r"Scikit[\-\s]?learn|sklearn", "scikit-learn"),
    (r"Objective[\-\s]?C", "Objective-C"),
    # ── 언어 (혼동 방지 패턴) ──────────────────────────────────────────────
    (r"TypeScript", "TypeScript"),
    (r"JavaScript", "JavaScript"),
    (r"Java(?!Script)", "Java"),
    (r"C\+\+", "C++"),
    (r"C\#", "C#"),
    (r"(?<![A-Za-z])C(?![A-Za-z\+\#])", "C"),
    (r"Python", "Python"),
    (r"Kotlin", "Kotlin"),
    (r"(?<![A-Za-z])Go(?:lang)?(?![A-Za-z])", "Go"),
    (r"Rust", "Rust"),
    (r"Scala", "Scala"),
    (r"Swift(?!UI)", "Swift"),
    (r"SwiftUI", "SwiftUI"),
    (r"Dart", "Dart"),
    (r"PHP", "PHP"),
    (r"Ruby(?!\s+on)", "Ruby"),
    (r"(?<![A-Za-z])R(?![A-Za-z])", "R"),
    (r"SQL(?!ite)", "SQL"),
    (r"Bash", "Bash"),
    (r"Lua", "Lua"),
    (r"Perl", "Perl"),
    (r"Elixir", "Elixir"),
    (r"Clojure", "Clojure"),
    (r"Haskell", "Haskell"),
    (r"HLSL", "HLSL"),
    (r"GLSL", "GLSL"),
    # ── 프론트엔드 ─────────────────────────────────────────────────────────
    (r"Angular(?!JS)", "Angular"),
    (r"AngularJS", "AngularJS"),
    (r"Svelte", "Svelte"),
    (r"(?<![A-Za-z])React(?!\s*Native|\.?js|JS)", "React"),
    (r"HTML5?", "HTML"),
    (r"CSS3?", "CSS"),
    (r"SCSS", "SCSS"),
    (r"Sass", "Sass"),
    (r"Less(?![A-Za-z])", "Less"),
    (r"Webpack", "Webpack"),
    (r"Vite(?![A-Za-z])", "Vite"),
    (r"Rollup", "Rollup"),
    (r"Parcel", "Parcel"),
    (r"Redux", "Redux"),
    (r"Recoil", "Recoil"),
    (r"Zustand", "Zustand"),
    (r"MobX", "MobX"),
    (r"Pinia", "Pinia"),
    (r"GraphQL", "GraphQL"),
    (r"Apollo", "Apollo"),
    (r"Storybook", "Storybook"),
    (r"Bootstrap", "Bootstrap"),
    # ── 백엔드 프레임워크 ──────────────────────────────────────────────────
    (r"Express(?:\.js)?", "Express"),
    (r"NestJS", "NestJS"),
    (r"Spring(?!\s*Boot|\s*Batch|\s*MVC)", "Spring"),
    (r"Django", "Django"),
    (r"FastAPI", "FastAPI"),
    (r"Flask", "Flask"),
    (r"Laravel", "Laravel"),
    (r"\.NET(?!ext)", ".NET"),
    (r"gRPC", "gRPC"),
    (r"REST(?:ful)?(?!\s*API)", "REST"),
    (r"JPA", "JPA"),
    (r"Hibernate", "Hibernate"),
    (r"MyBatis", "MyBatis"),
    (r"Maven", "Maven"),
    (r"Gradle", "Gradle"),
    # ── 데이터베이스 ───────────────────────────────────────────────────────
    (r"PostgreSQL|Postgres", "PostgreSQL"),
    (r"MySQL", "MySQL"),
    (r"MariaDB", "MariaDB"),
    (r"Oracle(?!\s*Cloud)", "Oracle"),
    (r"MongoDB", "MongoDB"),
    (r"Cassandra", "Cassandra"),
    (r"CouchDB", "CouchDB"),
    (r"Redis", "Redis"),
    (r"Memcached", "Memcached"),
    (r"Elasticsearch|ElasticSearch", "Elasticsearch"),
    (r"OpenSearch", "OpenSearch"),
    (r"DynamoDB", "DynamoDB"),
    (r"Firestore", "Firestore"),
    (r"Firebase", "Firebase"),
    (r"SQLite", "SQLite"),
    (r"MS[\-\s]?SQL|MSSQL", "MS-SQL"),
    (r"Snowflake", "Snowflake"),
    (r"BigQuery", "BigQuery"),
    # ── 메시지 큐 / 스트리밍 ────────────────────────────────────────────────
    (r"Kafka", "Kafka"),
    (r"RabbitMQ", "RabbitMQ"),
    (r"ActiveMQ", "ActiveMQ"),
    # ── 클라우드 ───────────────────────────────────────────────────────────
    (r"AWS", "AWS"),
    (r"GCP", "GCP"),
    (r"Azure", "Azure"),
    (r"EC2", "EC2"),
    (r"(?<![A-Za-z])S3(?![A-Za-z])", "S3"),
    (r"Lambda(?![A-Za-z])", "Lambda"),
    (r"EKS", "EKS"),
    (r"ECS", "ECS"),
    (r"RDS", "RDS"),
    # ── 컨테이너 / IaC / CI/CD ─────────────────────────────────────────────
    (r"Docker", "Docker"),
    (r"Kubernetes|K8s", "Kubernetes"),
    (r"Terraform", "Terraform"),
    (r"Ansible", "Ansible"),
    (r"Puppet", "Puppet"),
    (r"Chef(?![A-Za-z])", "Chef"),
    (r"CI/CD|CI\s*/\s*CD", "CI/CD"),
    (r"Jenkins", "Jenkins"),
    (r"ArgoCD|Argo\s*CD", "ArgoCD"),
    (r"Helm(?![A-Za-z])", "Helm"),
    (r"Nginx", "Nginx"),
    (r"Apache(?!\s*(?:Spark|Kafka|Airflow|Flink|Hive))", "Apache"),
    (r"Linux", "Linux"),
    (r"Ubuntu", "Ubuntu"),
    (r"CentOS", "CentOS"),
    (r"Prometheus", "Prometheus"),
    (r"Grafana", "Grafana"),
    # ── 데이터 엔지니어링 / ML ─────────────────────────────────────────────
    (r"Hadoop", "Hadoop"),
    (r"(?:Apache\s*)?Spark|PySpark", "Spark"),
    (r"(?:Apache\s*)?Flink", "Flink"),
    (r"(?:Apache\s*)?Hive", "Hive"),
    (r"Presto", "Presto"),
    (r"Trino", "Trino"),
    (r"(?:Apache\s*)?Airflow", "Airflow"),
    (r"Luigi", "Luigi"),
    (r"Prefect", "Prefect"),
    (r"dbt(?![A-Za-z])", "dbt"),
    (r"Fivetran", "Fivetran"),
    (r"Airbyte", "Airbyte"),
    (r"TensorFlow", "TensorFlow"),
    (r"PyTorch", "PyTorch"),
    (r"Keras", "Keras"),
    (r"XGBoost", "XGBoost"),
    (r"LightGBM", "LightGBM"),
    (r"Pandas", "Pandas"),
    (r"NumPy", "NumPy"),
    (r"SciPy", "SciPy"),
    (r"Jupyter", "Jupyter"),
    (r"Databricks", "Databricks"),
    (r"MLflow", "MLflow"),
    (r"Kubeflow", "Kubeflow"),
    (r"Tableau", "Tableau"),
    (r"Looker", "Looker"),
    (r"Superset", "Superset"),
    (r"Metabase", "Metabase"),
    (r"LLM", "LLM"),
    (r"GPT(?![A-Za-z])", "GPT"),
    (r"LangChain", "LangChain"),
    (r"LlamaIndex", "LlamaIndex"),
    (r"RAG(?![A-Za-z])", "RAG"),
    (r"Hugging\s*Face", "Hugging Face"),
    (r"CUDA", "CUDA"),
    (r"Delta\s*Lake", "Delta Lake"),
    (r"SageMaker", "SageMaker"),
    (r"Vertex\s*AI", "Vertex AI"),
    # ── 모바일 ─────────────────────────────────────────────────────────────
    (r"Android", "Android"),
    (r"iOS", "iOS"),
    (r"Flutter", "Flutter"),
    (r"Xamarin", "Xamarin"),
    (r"CocoaPods", "CocoaPods"),
    # ── 보안 ───────────────────────────────────────────────────────────────
    (r"SIEM", "SIEM"),
    (r"SOAR", "SOAR"),
    (r"(?<![A-Za-z])IDS(?![A-Za-z])", "IDS"),
    (r"(?<![A-Za-z])IPS(?![A-Za-z])", "IPS"),
    (r"OWASP", "OWASP"),
    (r"SSL", "SSL"),
    (r"TLS", "TLS"),
    (r"OAuth", "OAuth"),
    (r"JWT", "JWT"),
    (r"Wireshark", "Wireshark"),
    (r"Nessus", "Nessus"),
    (r"Metasploit", "Metasploit"),
    (r"Burp\s*Suite", "Burp Suite"),
    (r"Nmap", "Nmap"),
    (r"SonarQube", "SonarQube"),
    (r"Snyk", "Snyk"),
    # ── 테스팅 ─────────────────────────────────────────────────────────────
    (r"Jest", "Jest"),
    (r"Cypress", "Cypress"),
    (r"Playwright", "Playwright"),
    (r"Vitest", "Vitest"),
    (r"Selenium", "Selenium"),
    (r"JUnit", "JUnit"),
    (r"pytest", "pytest"),
    # ── 협업 / 도구 ────────────────────────────────────────────────────────
    (r"Git(?!Hub|Lab|(?:[A-Za-z]))", "Git"),
    (r"GitHub(?!\s*Actions)", "GitHub"),
    (r"GitLab(?!\s*CI)", "GitLab"),
    (r"Bitbucket", "Bitbucket"),
    (r"Jira", "Jira"),
    (r"Confluence", "Confluence"),
    (r"Notion", "Notion"),
    (r"Perforce", "Perforce"),
    # ── 방법론 / 아키텍처 ──────────────────────────────────────────────────
    (r"Agile", "Agile"),
    (r"Scrum", "Scrum"),
    (r"Kanban", "Kanban"),
    (r"MSA", "MSA"),
    (r"WebSocket", "WebSocket"),
    (r"OpenAPI|Swagger", "OpenAPI"),
    # ── 게임 ───────────────────────────────────────────────────────────────
    (r"Unity", "Unity"),
    (r"Blender", "Blender"),
    (r"Maya(?![A-Za-z])", "Maya"),
    (r"ARCore", "ARCore"),
    (r"ARKit", "ARKit"),
    (r"OpenXR", "OpenXR"),
    # ── 한국어 기술 명칭 ───────────────────────────────────────────────────
    (r"자바(?!스크립트)", "Java"),
    (r"자바스크립트", "JavaScript"),
    (r"파이썬", "Python"),
    (r"코틀린", "Kotlin"),
    (r"타입스크립트", "TypeScript"),
    (r"리액트(?!네이티브|\s*Native)", "React"),
    (r"리액트\s*네이티브", "React Native"),
    # 한글 단독 "뷰"는 Vue.js 로 추출하지 않는다. "뷰"는 UI·DB 의 "view"(예:
    # "웨젯(뷰)", "뷰, 저장 프로시저", "뷰 계층 구조")를 가리키는 일반어라
    # Vue.js 와 구분이 불가능해 오탐이 매우 잦다 ("리뷰"·"뷰티" 포함). 실제
    # Vue 사용은 영문 "Vue"/"Vue.js"/"VueJS" 로 적히며 그 패턴이 따로 잡는다.
    (r"앵귤러", "Angular"),
    (r"노드", "Node.js"),
    (r"스프링(?!부트|\s*Boot)", "Spring"),
    (r"스프링\s*부트", "Spring Boot"),
    (r"도커", "Docker"),
    (r"쿠버네티스", "Kubernetes"),
    (r"젠킨스", "Jenkins"),
    (r"플러터", "Flutter"),
    (r"머신러닝", "Machine Learning"),
    (r"딥러닝", "Deep Learning"),
    (r"자연어처리", "NLP"),
    (r"컴퓨터비전", "Computer Vision"),
    (r"빅데이터", "Big Data"),
    (r"데이터파이프라인", "Data Pipeline"),
    (r"데이터웨어하우스", "Data Warehouse"),
    (r"안드로이드", "Android"),
    (r"텐서플로우?", "TensorFlow"),
    (r"파이토치", "PyTorch"),
    (r"장고", "Django"),
    (r"랭체인", "LangChain"),
]

# 패턴 길이 내림차순 정렬 후 컴파일 (긴 패턴 우선 매칭)
TECH_KEYWORDS: list[tuple[re.Pattern[str], str]] = sorted(
    [(re.compile(pat, re.IGNORECASE), name) for pat, name in _RAW_TECH_KEYWORDS],
    key=lambda t: -len(t[0].pattern),
)


# ---------------------------------------------------------------------------
# 3. 직군 분류용 패턴
# ---------------------------------------------------------------------------

_JOB_CATEGORY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # More specific patterns (higher priority)
    (re.compile(r"프롬프트\s*엔지니어|prompt\s*engineer", re.IGNORECASE), "AI/ML"),
    (re.compile(r"LLM|GPT|생성\s*AI|generative\s*AI", re.IGNORECASE), "AI/ML"),
    (
        re.compile(r"애널리틱스\s*엔지니어|analytics\s*engineer|BI\s*엔지니어", re.IGNORECASE),
        "데이터분석",
    ),
    (re.compile(r"데이터\s*플랫폼|data\s*platform", re.IGNORECASE), "데이터엔지니어"),
    (
        re.compile(r"IT\s*엔지니어|시스템\s*엔지니어|Server\s*Admin|시스템\s*관리", re.IGNORECASE),
        "DevOps/인프라",
    ),
    (re.compile(r"FW\s*개발|펌웨어|firmware|임베디드|embedded", re.IGNORECASE), "임베디드"),
    (re.compile(r"풀\s*스택|full[\-\s]?stack", re.IGNORECASE), "풀스택"),
    (re.compile(r"VueJS|Vue\.?js|React\s+개발|프론트\s*개발", re.IGNORECASE), "프론트엔드"),
    (re.compile(r"SW\s*개발|소프트웨어\s*개발|software\s*develop", re.IGNORECASE), "백엔드"),
    # Existing patterns
    (re.compile(r"프론트엔드|frontend|front[\-\s]?end|UI\s*개발", re.IGNORECASE), "프론트엔드"),
    (re.compile(r"백엔드|backend|back[\-\s]?end|서버\s*개발", re.IGNORECASE), "백엔드"),
    (re.compile(r"풀스택|fullstack|full[\-\s]?stack", re.IGNORECASE), "풀스택"),
    (
        re.compile(r"AI|인공지능|머신러닝|딥러닝|ML\s*엔지니어|NLP|컴퓨터\s*비전", re.IGNORECASE),
        "AI/ML",
    ),
    (
        re.compile(r"데이터\s*엔지니어|data\s*engineer|ETL|데이터\s*파이프라인", re.IGNORECASE),
        "데이터엔지니어",
    ),
    (re.compile(r"데이터\s*분석|data\s*analy|BI\s*분석", re.IGNORECASE), "데이터분석"),
    (re.compile(r"데이터\s*사이언|data\s*scien", re.IGNORECASE), "데이터사이언스"),
    (
        re.compile(
            r"DevOps|데브옵스|SRE|인프라|클라우드\s*엔지니어|platform\s*engineer", re.IGNORECASE
        ),
        "DevOps/인프라",
    ),
    (re.compile(r"보안|security|시큐리티|침투|모의해킹|SOC", re.IGNORECASE), "보안"),
    (re.compile(r"안드로이드|android|iOS|모바일|mobile|앱\s*개발", re.IGNORECASE), "모바일"),
    (re.compile(r"게임|game|unity|unreal", re.IGNORECASE), "게임"),
    (re.compile(r"QA|테스트\s*엔지니어|품질\s*보증|test\s*engineer", re.IGNORECASE), "QA"),
    (re.compile(r"임베디드|embedded|펌웨어|firmware|IoT", re.IGNORECASE), "임베디드"),
    (re.compile(r"블록체인|blockchain|스마트\s*컨트랙트|web3", re.IGNORECASE), "블록체인"),
]


# ---------------------------------------------------------------------------
# 4. 공개 함수
# ---------------------------------------------------------------------------


def extract_tech_keywords(text: str) -> list[str]:
    """
    채용공고 자유 텍스트에서 기술 키워드를 추출한다.

    Args:
        text: 채용공고 본문 텍스트 (HTML 제거된 순수 텍스트)

    Returns:
        정규화된 기술 키워드 리스트 (중복 제거, 오름차순 정렬)

    Examples:
        >>> extract_tech_keywords("Java 및 JavaScript 경험자")
        ['Java', 'JavaScript']
        >>> extract_tech_keywords("C# .NET 개발")
        ['.NET', 'C#']
        >>> extract_tech_keywords("Next.js와 Node.js")
        ['Next.js', 'Node.js']
    """
    if not text:
        return []

    found: set[str] = set()
    for pattern, canonical in TECH_KEYWORDS:
        if pattern.search(text):
            found.add(canonical)

    return sorted(found)


def normalize_tech_tag(tag: str) -> str:
    """
    단일 기술 태그 이름을 정규 이름으로 변환한다.

    NORMALIZATION_MAP에 등록된 변형이면 정규 이름을 반환하고,
    등록되지 않은 태그는 Title Case로 변환하여 반환한다.

    Args:
        tag: 원본 태그 문자열

    Returns:
        정규화된 태그 문자열

    Examples:
        >>> normalize_tech_tag("ReactJS")
        'React'
        >>> normalize_tech_tag("자바")
        'Java'
        >>> normalize_tech_tag("some unknown tag")
        'Some Unknown Tag'
    """
    stripped = tag.strip()
    if not stripped:
        return stripped

    # 정확히 일치하는 경우
    if stripped in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[stripped]

    # 대소문자 무시 검색
    lower = stripped.lower()
    for variant, canonical in NORMALIZATION_MAP.items():
        if variant.lower() == lower:
            return canonical

    return stripped.title()


def normalize_tech_tags(tags: list[str]) -> list[str]:
    """
    기술 태그 리스트를 일괄 정규화한다.

    각 태그를 normalize_tech_tag로 변환한 뒤 중복을 제거하고 정렬한다.

    Args:
        tags: 원본 태그 문자열 리스트

    Returns:
        정규화된 태그 리스트 (중복 제거, 오름차순 정렬)

    Examples:
        >>> normalize_tech_tags(["ReactJS", "React.js", "자바"])
        ['Java', 'React']
    """
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        normalized = normalize_tech_tag(tag)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return sorted(result)


# ---------------------------------------------------------------------------
# 4b. 정합 토큰 사전 — 직무·트랙 기술 어휘 통합 (교집합 비교용)
# ---------------------------------------------------------------------------
# 직무 채용공고와 트랙 커리큘럼은 같은 기술을 서로 다른 표기로 적어둘 수 있다
# (예: "ReactJS" vs "React", "spring boot" vs "Spring Boot"). NORMALIZATION_MAP
# 을 직무 기술 어휘 기준 정합 사전으로 삼아 양쪽 표기를 한 어휘로 통합한다.
# 직무 매칭·트랙 시너지 등 어느 단계든 같은 사전을 재사용해 교집합을 계산한다.
_NORMALIZATION_INDEX: dict[str, str] = {
    variant.casefold(): canonical for variant, canonical in NORMALIZATION_MAP.items()
}


def canonical_tech_token(tag: str) -> str:
    """기술 태그를 직무 기술 어휘 기준 정규 표기로 통합한다.

    ``normalize_tech_tag`` 와 달리 사전에 없는 태그는 ``.title()`` 로 바꾸지
    않고 원본 표기를 유지한다. ``"SQL"`` → ``"Sql"``, ``"iOS"`` → ``"Ios"``
    같은 약어·고유 대소문자 오변환을 막기 위함이다 — 트랙 카탈로그에는 이미
    정규 표기로 적힌 기술이 많아, 오변환은 오히려 직무 어휘와의 정합을 깬다.

    Args:
        tag: 원본 기술 태그.

    Returns:
        ``NORMALIZATION_MAP`` 에 등록된 변형이면 정규 이름, 아니면 공백을 제거한
        원본 표기.
    """
    stripped = tag.strip()
    if not stripped:
        return stripped
    return _NORMALIZATION_INDEX.get(stripped.casefold(), stripped)


def canonical_tech_key(tag: str) -> str:
    """집합 비교용 정규 키를 반환한다 (``canonical_tech_token`` 후 casefold).

    표시는 ``canonical_tech_token`` 으로, 교집합 비교는 본 키로 한다. 별칭
    차이뿐 아니라 대소문자 차이까지 흡수해 ``"SQL"`` 과 ``"sql"`` 을 같은
    기술로 본다.
    """
    return canonical_tech_token(tag).casefold()


_PAREN_CLAUSE = re.compile(r"\s*\([^)]*\)")
_COMPOUND_SEPARATOR = re.compile(r"\s+[/+]\s+")


def _atomize_tech_token(tag: str) -> list[str]:
    """복합 표기 기술 라벨을 원자 토큰으로 분해한다.

    직무 데이터는 한 항목에 여러 기술을 묶어 적는다 (예: ``"AWS / GCP"``,
    ``"Docker + Kubernetes"``, ``"dbt (data build tool)"``). 트랙 토큰은 원자
    단위라, 분해하지 않으면 ``"AWS / GCP"`` 전체가 한 키가 돼 ``"AWS"`` 와
    매칭되지 않고 직무 커버율이 0 에 수렴한다.

    분리 기준은 **공백으로 둘러싸인** ``/`` ``+`` 뿐이다 — ``"C++"`` ``"C#"``
    ``"A/B"`` ``"CI/CD"`` ``"VR/AR"`` 처럼 구분자가 토큰 일부인 단일 기술은
    쪼개지 않는다. 괄호절(별칭·부연·도구 나열)은 제거하고 머리 토큰만 남긴다.

    Returns:
        분해된 토큰 리스트. 복합 표기가 아니면 ``[tag]`` 그대로.
    """
    no_paren = _PAREN_CLAUSE.sub("", tag)
    return [part for part in _COMPOUND_SEPARATOR.split(no_paren) if part.strip()]


def canonical_tech_keys(tags: Iterable[str]) -> set[str]:
    """기술 태그 목록을 정규 비교 키 집합으로 변환한다 (빈 키 제외).

    각 태그는 먼저 원자 토큰으로 분해된다 (``_atomize_tech_token``) — 직무
    데이터의 복합 표기(``"AWS / GCP"``)를 원자 단위 트랙 토큰과 같은 grain 으로
    맞춰, 교집합 비교가 표기 묶음 때문에 어긋나지 않게 한다.
    """
    keys: set[str] = set()
    for tag in tags:
        for atom in _atomize_tech_token(tag):
            key = canonical_tech_key(atom)
            if key:
                keys.add(key)
    return keys


def classify_job_category(title: str, text: str = "") -> str:
    """
    채용공고 제목에서 IT 직군 카테고리를 분류한다.
    제목에서 매칭되지 않으면 본문 텍스트로 재시도한다.

    Args:
        title: 채용공고 제목 문자열
        text: 채용공고 본문 텍스트 (기본값 ""). 제목 매칭 실패 시 사용.

    Returns:
        분류된 직군 이름. 매칭되지 않으면 "기타" 반환.

    Examples:
        >>> classify_job_category("시니어 백엔드 개발자")
        '백엔드'
        >>> classify_job_category("AI 연구원 (NLP)")
        'AI/ML'
        >>> classify_job_category("경영지원팀 사원")
        '기타'
        >>> classify_job_category("프롬프트 엔지니어")
        'AI/ML'
        >>> classify_job_category("SW 개발팀장", "Java Spring Boot 서버 개발")
        '백엔드'
    """
    if not title:
        return "기타"

    for pattern, category in _JOB_CATEGORY_PATTERNS:
        if pattern.search(title):
            return category

    if text:
        for pattern, category in _JOB_CATEGORY_PATTERNS:
            if pattern.search(text):
                return category

    return "기타"
