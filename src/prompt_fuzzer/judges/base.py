"""Judges score a target response. They do not see adapter metadata."""

from __future__ import annotations

from typing import Protocol

from prompt_fuzzer.models import Judgment, Objective


class Judge(Protocol):
    def score(self, *, objective: Objective, prompt: str, response: str) -> Judgment:
        """Return component scores in ``[0, 1]``."""
