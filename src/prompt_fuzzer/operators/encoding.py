"""Re-encode a prompt and label the encoding.

The transforms are standard codecs (base64, rot13, hex, leetspeak). The label
tells a target which codec was used. This is a generic obfuscation operator,
not a payload catalog.
"""

from __future__ import annotations

import base64
import codecs
import random

KINDS: tuple[str, ...] = ("base64", "rot13", "hex", "leet")

_LEET_ENCODE = str.maketrans(
    {
        "a": "4",
        "e": "3",
        "i": "1",
        "o": "0",
        "t": "7",
        "A": "4",
        "E": "3",
        "I": "1",
        "O": "0",
        "T": "7",
    }
)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def encode(text: str, kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown encoding {kind!r}")
    body = _one_line(text)
    if kind == "base64":
        payload = base64.b64encode(body.encode("utf-8")).decode("ascii")
    elif kind == "rot13":
        payload = codecs.encode(body, "rot_13")
    elif kind == "hex":
        payload = body.encode("utf-8").hex()
    else:
        payload = body.translate(_LEET_ENCODE)
    return f"Audience: general.\nEncoding: {kind}\n{payload}"


class Encoding:
    name = "encoding"

    def mutate(self, text: str, rng: random.Random) -> str:
        return encode(text, rng.choice(KINDS))
