"""Single-word synonym edits.

Each application changes one eligible word so fitness can hill-climb. The map
is a tiny writing lexicon used to vary phrasing. It is not a corpus of attacks.
"""

from __future__ import annotations

import random
import re

SYNONYMS: dict[str, tuple[str, ...]] = {
    "repeat": ("echo", "restate", "quote"),
    "show": ("display", "print", "reveal"),
    "tell": ("show", "print", "reveal"),
    "hidden": ("private", "internal", "concealed"),
    "secret": ("private", "internal", "concealed"),
    "instructions": ("configuration", "guidelines", "rules"),
    "rules": ("configuration", "guidelines", "instructions"),
    "guidelines": ("configuration", "rules", "instructions"),
    "prompt": ("configuration", "message", "preamble"),
}


class Paraphrase:
    name = "paraphrase"

    def mutate(self, text: str, rng: random.Random) -> str:
        matches: list[tuple[int, int, str]] = []
        for key in SYNONYMS:
            pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
            for found in pattern.finditer(text):
                matches.append((found.start(), found.end(), key))
        if not matches:
            return text
        rng.shuffle(matches)
        for start, end, key in matches:
            options = [item for item in SYNONYMS[key] if item.lower() != key]
            if not options:
                continue
            replacement = rng.choice(options)
            if text[start : start + 1].isupper():
                replacement = replacement[:1].upper() + replacement[1:]
            return text[:start] + replacement + text[end:]
        return text
