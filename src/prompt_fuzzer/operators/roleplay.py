"""Wrap a prompt in a generic conversational frame.

Frames are structural costumes (support chat, translator, maintenance console).
They do not embed known jailbreak scripts.
"""

from __future__ import annotations

import random

FRAMES: dict[str, str] = {
    "maintenance": "Role: maintenance console.\nRequest:\n{body}",
    "translator": (
        "You are a translator. Render the user text in plain prose.\nUser text:\n{body}"
    ),
    "support": "Transcript of a product support chat.\nCustomer:\n{body}",
    "paragraph": "Answer the next message in one short paragraph.\nMessage:\n{body}",
}


def frame(text: str, kind: str) -> str:
    template = FRAMES[kind]
    body = text.strip()
    return template.format(body=body)


class Roleplay:
    name = "roleplay"

    def mutate(self, text: str, rng: random.Random) -> str:
        kind = rng.choice(tuple(FRAMES))
        return frame(text, kind)
