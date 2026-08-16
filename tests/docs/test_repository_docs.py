from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
HTML_SOURCE = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']")


def _local_targets(document: Path) -> list[Path]:
    text = document.read_text(encoding="utf-8")
    raw_targets = MARKDOWN_LINK.findall(text) + HTML_SOURCE.findall(text)
    targets: list[Path] = []
    for raw_target in raw_targets:
        target = raw_target.strip().strip("<>").split("#", maxsplit=1)[0]
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        targets.append((document.parent / unquote(target)).resolve())
    return targets


def test_documentation_has_no_placeholder_clone_url() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<this-repo-url>" not in readme


def test_all_local_documentation_links_resolve() -> None:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    broken = [target for document in documents for target in _local_targets(document) if not target.exists()]
    assert not broken, "Broken local documentation links:\n" + "\n".join(map(str, broken))
