## ✨ PR 유형

<!-- 어떤 변경 사항이 있나요? (Feat / Fix / Refactor / Docs / Chore / Test / Design) -->

## 🛠️ 작업 내용

<!-- [ 작업한 내용을 작성해주세요 ] -->

## 📋 추후 진행 상황

<!-- [ 다음에 진행할 작업이 있다면 작성해주세요 ] -->

## 📌 리뷰 포인트

<!-- [ 어떤 부분을 잘 체크해야 하는지 작성해주세요 ] -->

## 🔗 관련 이슈

<!-- Closes #이슈번호 -->

---

## ✅ Checklist

PR이 다음 요구 사항을 충족하는지 확인해주세요.

- [ ] 커밋 메시지가 커밋 메시지 규칙을 따릅니다 (소문자 타입: `feat:`, `fix:`, `refactor:` ...)
- [ ] 커밋 전 루틴을 실행했습니다
  - [ ] `uv run ruff format .`
  - [ ] `uv run ruff check . --fix`
  - [ ] `uv run mypy src`
  - [ ] `uv run pytest`
- [ ] 공개 함수/클래스에 타입 힌트와 docstring을 작성했습니다
- [ ] LangGraph 관련 변경이라면 LangGraph 규약(단일 책임 노드 / 부분 상태 반환 / 부작용 격리)을 준수했습니다
