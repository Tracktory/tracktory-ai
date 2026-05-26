from __future__ import annotations

from tracktory.common.models import JobPosting
from tracktory.rag.preprocessing.extract_rag_metadata import _parse_job
from tracktory.rag.preprocessing.jobs import build_jobs_from_csv, build_jobs_from_postings


def test_build_jobs_from_csv_removes_company_names(tmp_path):
    csv_path = tmp_path / "jobs.csv"
    output_dir = tmp_path / "rag" / "jobs"
    csv_path.write_text(
        "\n".join(
            [
                "source,source_id,company,title,category,tech_stacks,location,salary,experience,industry_name,deadline,url,responsibilities,requirements,preferred,benefits",
                'wanted,42,테스트회사(TestCo),테스트회사 AI 엔지니어,AI/ML,Python,"서울 강남구 테헤란로7길 22, 한국과학기술회관 2관 8층",,신입,IT,상시,,"테스트회사에서 TestCo 플랫폼 개발. 근무지: 서울 강남구 테헤란로7길 22, 8층",TestCo 경험,테스트회사 우대,테스트회사 복지',
                'wanted,43,다른회사,[BRAND] 다른회사 데이터 엔지니어,데이터엔지니어,SQL,"경기도 성남시 분당구 성남대로 331번길 8",,신입,IT,상시,,테스트회사와 다른회사 협업,TestCo 연동,다른회사 우대,테스트회사 복지',
            ]
        ),
        encoding="utf-8",
    )

    results = build_jobs_from_csv(str(csv_path), str(output_dir))

    files = list(output_dir.glob("*.txt"))
    assert len(files) == 2
    assert all("테스트회사" not in file.name for file in files)
    assert all("TestCo" not in file.name for file in files)
    assert all("다른회사" not in file.name for file in files)
    assert all("[BRAND]" not in file.name for file in files)
    assert "company" not in results[0]

    documents = "\n".join(file.read_text(encoding="utf-8") for file in files)
    assert "회사:" not in documents
    assert "테스트회사" not in documents
    assert "TestCo" not in documents
    assert "다른회사" not in documents
    assert "[BRAND]" not in documents
    assert "위치: 서울 강남구" in documents
    assert "위치: 경기 성남시 분당구" in documents
    assert "테헤란로7길" not in documents
    assert "성남대로" not in documents


def test_build_jobs_from_postings_removes_company_names(tmp_path):
    output_dir = tmp_path / "rag" / "jobs"
    posting = JobPosting(
        source="wanted",
        source_id="84",
        company="샘플랩(SampleLab)",
        title="SampleLab 백엔드 개발자",
        category="백엔드",
        tech_stacks=["Python"],
        location="서울 서초구 반포대로 306, 9층",
        salary="미공개",
        url="",
        experience="3년 이상",
        extra={
            "responsibilities": "샘플랩 서비스와 SampleLab API 개발. 근무지: 역삼역 부근 (서울시 강남구 역삼로 17길 16). 근무 장소: 서울 서초구 반포대로 306, 9층",
            "requirements": "샘플랩 도메인 이해",
        },
    )
    other_posting = JobPosting(
        source="wanted",
        source_id="85",
        company="외부회사",
        title="[UNEEKOR] 외부회사 데이터 엔지니어",
        category="데이터엔지니어",
        tech_stacks=["SQL"],
        location="리모트근무",
        salary="미공개",
        url="",
        experience="신입",
        extra={
            "responsibilities": "샘플랩과 외부회사 데이터 파이프라인 개발",
            "requirements": "SampleLab 연동 경험",
        },
    )

    results = build_jobs_from_postings([posting, other_posting], str(output_dir))

    files = list(output_dir.glob("*.txt"))
    assert len(files) == 2
    assert all("샘플랩" not in file.name for file in files)
    assert all("SampleLab" not in file.name for file in files)
    assert all("외부회사" not in file.name for file in files)
    assert all("[UNEEKOR]" not in file.name for file in files)
    assert "company" not in results[0]

    documents = "\n".join(file.read_text(encoding="utf-8") for file in files)
    assert "회사:" not in documents
    assert "샘플랩" not in documents
    assert "SampleLab" not in documents
    assert "외부회사" not in documents
    assert "[UNEEKOR]" not in documents
    assert "위치: 서울 서초구" in documents
    assert "위치: 원격" in documents
    assert "반포대로" not in documents
    assert "역삼로" not in documents
    assert "(서울 강남구)" in documents


def test_parse_job_metadata_does_not_extract_company():
    metadata = _parse_job(
        "채용공고_AI 엔지니어_테스트회사_42.txt",
        "직무: AI 엔지니어\n회사: 테스트회사\n카테고리: AI/ML",
    )

    assert metadata["doc_type"] == "job_posting"
    assert "company" not in metadata
