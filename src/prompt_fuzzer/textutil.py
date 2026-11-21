"""Small text helpers shared by the search loop and the mock target."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def word_list(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def jaccard_distance(a: str, b: str) -> float:
    """Token Jaccard distance in ``[0, 1]``. Identical text is ``0``."""

    return set_distance(words(a), words(b))


def set_distance(left: set[str], right: set[str]) -> float:
    """Jaccard distance between word sets that were already tokenized."""

    if not left and not right:
        return 0.0
    union = left | right
    return 1.0 - (len(left & right) / len(union))
