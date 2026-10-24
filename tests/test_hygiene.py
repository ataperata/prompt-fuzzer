"""Repo hygiene: no committed credentials or private keys."""

from __future__ import annotations

import re
from pathlib import Path

PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"sk-(?:live|test|proj)-[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
]

SKIP = {".git", ".venv", "venv", "runs", "__pycache__", ".pytest_cache", ".ruff_cache"}


def test_tree_has_no_credential_material() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in root.rglob("*"):
        if any(part in SKIP for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix in {".png", ".pyc"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(root)} matches {pattern.pattern}")
    assert offenders == []
