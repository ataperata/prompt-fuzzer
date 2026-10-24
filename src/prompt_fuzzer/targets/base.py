"""Target adapters turn a prompt string into a model response."""

from __future__ import annotations

from typing import Protocol

from prompt_fuzzer.models import TargetResponse


class Target(Protocol):
    def generate(self, prompt: str) -> TargetResponse:
        """Return the target's reply to ``prompt``."""
