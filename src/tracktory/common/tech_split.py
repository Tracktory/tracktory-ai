"""
복합 기술 이름 분리 모듈

job_tech_stacks.json 등 큐레이션 소스는 "A / B", "A (B)", "A + B" 형태의
서술형·복합형 이름을 쓴다. 이 모듈은 그런 이름을 실제 채용 데이터
(원티드·사람인)에서 찾을 수 있는 단순 태그명 리스트로 분리한다.

분리된 태그는 normalize_tech_tag 를 거쳐 정규 이름으로 통합된다.

사용법:
    from tracktory.common.tech_split import split_tech_name

    split_tech_name("JavaScript (ES2024+)")   # → ["JavaScript"]
    split_tech_name("Docker + Kubernetes")    # → ["Docker", "Kubernetes"]
    split_tech_name("React")                  # → None  (등록 없음 → 단일 태그로 직접 조회)
    split_tech_name("A/B 테스팅 프레임워크")   # → []   (대응 태그 없음)
"""

from __future__ import annotations

# key   : 큐레이션/표시용 이름 (job_tech_stacks.json 등에 적힌 이름)
# value : 실제 채용 데이터에서 찾을 태그 이름 리스트
#         - 빈 리스트 : 대응 태그 없음 (wanted_count = 0 고정)
#         - 이 맵에 없는 이름 : 그대로 단일 태그로 조회 (None 반환)

TECH_SPLIT_MAP: dict[str, list[str]] = {
    # ── 언어 ──────────────────────────────────────────────────────────
    "JavaScript (ES2024+)": ["JavaScript"],
    "Python (Pandas, NumPy)": ["Python"],
    "Python / Bash": ["Python", "Bash"],
    "Go (Golang)": ["Go"],
    # ── 프레임워크 / 라이브러리 ─────────────────────────────────────────
    "Flutter (Dart)": ["Flutter", "Dart"],  # Flutter=프레임워크, Dart=언어
    "Node.js / Express": ["Node.js", "Express"],
    "Zustand / React Query": ["Zustand", "React Query"],
    "Storybook / Vitest": ["Storybook", "Vitest"],
    "TensorFlow / Keras": ["TensorFlow", "Keras"],
    "LangChain / LlamaIndex": ["LangChain", "LlamaIndex"],
    "Hugging Face Transformers": ["Hugging Face"],
    "Jupyter Notebook": ["Jupyter"],
    "GraphQL / REST API": ["GraphQL"],  # REST API는 아키텍처 개념 — 기술 스택 아님
    # ── 게임 엔진 (엔진 + 사용 언어가 묶인 경우) ───────────────────────
    "Unity (C#)": ["Unity", "C#"],
    "Unreal Engine 5 (C++)": ["Unreal Engine", "C++"],
    "HLSL / GLSL": ["HLSL", "GLSL"],  # 게임/그래픽 셰이더 언어
    "Blender / Maya": ["Blender", "Maya"],
    "게임 서버 (Go, Java, Node.js)": ["Go", "Java", "Node.js"],
    # ── 모바일 SDK / 빌드도구 ──────────────────────────────────────────
    "Kotlin Multiplatform (KMP)": ["Kotlin"],  # KMP=멀티플랫폼 SDK, Kotlin=언어
    "Android Jetpack": ["Android"],
    "Gradle / CocoaPods": ["Gradle", "CocoaPods"],
    "VR/AR SDK (ARCore, ARKit, OpenXR)": ["ARCore", "ARKit", "OpenXR"],
    # ── 인프라 / 클라우드 ───────────────────────────────────────────────
    "AWS / GCP": ["AWS", "GCP"],
    "AWS SageMaker / GCP Vertex AI": ["SageMaker", "Vertex AI"],
    "Docker + Kubernetes": ["Docker", "Kubernetes"],
    "Docker / Kubernetes": ["Docker", "Kubernetes"],
    "Kubernetes (k8s)": ["Kubernetes"],
    "GitHub Actions / GitLab CI": ["GitHub Actions", "GitLab CI"],
    "Prometheus + Grafana": ["Prometheus", "Grafana"],
    "Linux (Bash)": ["Linux", "Bash"],
    "Git + Perforce": ["Git", "Perforce"],
    # ── 데이터 ────────────────────────────────────────────────────────
    "SQL (+ BigQuery / Snowflake)": ["SQL", "BigQuery", "Snowflake"],
    "Snowflake / BigQuery": ["Snowflake", "BigQuery"],
    "Databricks / Delta Lake": ["Databricks", "Delta Lake"],
    "MLflow / Kubeflow": ["MLflow", "Kubeflow"],
    "Looker / Metabase": ["Looker", "Metabase"],
    "dbt (data build tool)": ["dbt"],
    "Apache Spark": ["Spark"],
    "Apache Kafka": ["Kafka"],
    "Apache Airflow": ["Airflow"],
    # ── 컴퓨팅 플랫폼 ─────────────────────────────────────────────────
    "CUDA / GPU 프로그래밍": ["CUDA", "GPU"],  # NVIDIA 병렬컴퓨팅 플랫폼
    # ── 보안 ──────────────────────────────────────────────────────────
    "SIEM 솔루션 (Splunk, QRadar, Sentinel)": ["SIEM", "Splunk", "QRadar", "Sentinel"],
    "SIEM 도구 (Splunk, QRadar, Sentinel)": ["SIEM", "Splunk", "QRadar", "Sentinel"],
    "DevSecOps (SonarQube, Snyk, OWASP ZAP)": ["SonarQube", "Snyk", "OWASP"],
    "IAM / Zero Trust (Okta, CrowdStrike)": ["Okta", "CrowdStrike"],
    "취약점 스캐너 (Nessus, OpenVAS)": ["Nessus", "OpenVAS"],
    "침투 테스팅 도구 (Metasploit, Burp Suite, Nmap)": ["Metasploit", "Burp Suite", "Nmap"],
    "클라우드 보안 (AWS Security Hub, Azure Defender)": ["AWS Security Hub", "Azure Defender"],
    "포렌식 도구 (Volatility, Autopsy)": ["Volatility", "Autopsy"],
    "Wireshark / tcpdump": ["Wireshark", "tcpdump"],
    "보안 인증 (CISSP, CEH, OSCP)": [],  # 자격증 — 도구/언어 아님
    # ── 기타 ──────────────────────────────────────────────────────────
    "A/B 테스팅 프레임워크": [],  # 방법론 개념 — 특정 도구 아님
    "Excel / Google Sheets": ["Excel", "Google Sheets"],
}


def split_tech_name(name: str) -> list[str] | None:
    """복합 기술 이름을 단순 태그명 리스트로 분리한다.

    TECH_SPLIT_MAP에 등록된 이름이면 해당 리스트를 반환한다.
    등록되지 않은 이름은 None을 반환한다 (단일 태그로 직접 조회해야 함).

    Args:
        name: 큐레이션 기술 이름 (job_tech_stacks.json 등의 표시용 이름).

    Returns:
        분리된 태그 리스트, 또는 None (맵에 없는 경우).
    """
    return TECH_SPLIT_MAP.get(name)
