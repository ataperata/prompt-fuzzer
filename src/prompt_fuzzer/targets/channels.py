"""Decode helpers for the labeled encodings the encoding operator emits."""

from __future__ import annotations

import base64
import binascii
import codecs
import json
import re

_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_INNER = re.compile(r"(?im)^inner:\s*(.+)$")
_ENCODING = re.compile(r"(?im)^Encoding:\s*(base64|rot13|hex|leet)[ \t]*$\n^(\S.*)$")

_DELEET = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "7": "t"})


def decode_payload(kind: str, payload: str) -> str | None:
    try:
        if kind == "base64":
            padded = payload + ("=" * ((4 - len(payload) % 4) % 4))
            return base64.b64decode(padded, validate=False).decode("utf-8", errors="ignore")
        if kind == "rot13":
            return codecs.decode(payload, "rot_13")
        if kind == "hex":
            return bytes.fromhex(payload).decode("utf-8", errors="ignore")
        if kind == "leet":
            return payload.translate(_DELEET)
    except (binascii.Error, UnicodeError, ValueError):
        return None
    return None


def hidden_payloads(prompt: str) -> list[str]:
    """Text a careless parser will read even when a surface filter skips it."""

    found: list[str] = []
    for match in _COMMENT.finditer(prompt):
        found.append(match.group(1).strip())
    for match in _FENCE.finditer(prompt):
        found.append(match.group(1).strip())
    for match in _INNER.finditer(prompt):
        found.append(match.group(1).strip())
    for match in _ENCODING.finditer(prompt):
        decoded = decode_payload(match.group(1).lower(), match.group(2).strip())
        if decoded:
            found.append(decoded)
    stripped = prompt.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            document = json.loads(stripped)
        except json.JSONDecodeError:
            document = None
        if isinstance(document, dict) and isinstance(document.get("note"), str):
            found.append(document["note"])
    return [item for item in found if item]


def visible_text(prompt: str) -> str:
    """Text left after carriers and encoded lines are removed."""

    text = _FENCE.sub(" ", prompt)
    text = _COMMENT.sub(" ", text)
    text = _ENCODING.sub(" ", text)
    kept = [line for line in text.splitlines() if not line.lower().startswith("inner:")]
    text = "\n".join(kept)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            document = json.loads(stripped)
        except json.JSONDecodeError:
            document = None
        if isinstance(document, dict) and "note" in document:
            task = document.get("task", "")
            return task if isinstance(task, str) else ""
    return text
