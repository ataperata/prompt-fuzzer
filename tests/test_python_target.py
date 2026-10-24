"""Python target plugin loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.targets import build_target


def test_python_target_from_a_local_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = tmp_path / "echo_target.py"
    module.write_text(
        textwrap.dedent(
            """
            class EchoTarget:
                def __init__(self, suffix=""):
                    self.suffix = suffix

                def generate(self, prompt):
                    from prompt_fuzzer.models import TargetResponse
                    return TargetResponse(text=prompt + self.suffix)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    target = build_target({"type": "python", "dotted": "echo_target:EchoTarget", "suffix": "!"})
    response = target.generate("hi")
    assert response.text == "hi!"


def test_python_target_rejects_relative_paths() -> None:
    with pytest.raises(ConfigError):
        build_target({"type": "python", "dotted": ".hidden:Target"})
