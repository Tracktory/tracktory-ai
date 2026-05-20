"""``ProfileEmbedNode`` 의 template 로더 분기 단위 테스트.

본 PR scope 의 missing line 분기 (template yaml 이 mapping 이 아닌 경우) 를 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tracktory.graph.nodes.profile_embed import ProfileEmbedNode


def test_constructor_raises_when_template_is_not_mapping(tmp_path: Path) -> None:
    """template yaml 의 top-level 이 mapping 이 아닌 경우 ValueError — 생성자에서 조기 감지."""
    path = tmp_path / "bad_template.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define a top-level mapping"):
        ProfileEmbedNode(template_path=path)
