"""HTTP helpers that never follow non-http URLs and never log credentials."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from prompt_fuzzer.errors import ConfigError

MAX_BYTES = 1_000_000


def check_http_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        raise ConfigError("url must be a non-empty string")
    lowered = url.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ConfigError("url must start with http:// or https://")
    return url


def post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str] | None,
    timeout_s: float,
) -> dict[str, object]:
    check_http_url(url)
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        request_headers.update({str(key): str(value) for key, value in headers.items()})
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            raw = response.read(MAX_BYTES + 1)
    except TimeoutError as exc:
        raise TimeoutError(f"timed out after {timeout_s}s") from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace")
        raise ConnectionError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"connection failed: {exc.reason}") from exc
    if len(raw) > MAX_BYTES:
        raise ConnectionError("response exceeded 1 MB")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConnectionError("response was not JSON") from exc
    if not isinstance(parsed, dict):
        raise ConnectionError("response JSON must be an object")
    return parsed


def dig(document: dict[str, object], path: str) -> str:
    """Read a dotted path. Numeric parts index lists (``choices.0.message.content``)."""

    if not path:
        raise ConfigError("response_path must not be empty")
    current: object = document
    for part in path.split("."):
        if isinstance(current, list):
            if not part.isdigit():
                raise ConnectionError(f"expected a list index at {part!r}")
            index = int(part)
            if index >= len(current):
                raise ConnectionError(f"index {index} out of range")
            current = current[index]
        elif isinstance(current, dict):
            if part not in current:
                raise ConnectionError(f"missing key {part!r}")
            current = current[part]
        else:
            raise ConnectionError(f"cannot walk into {type(current).__name__}")
    if not isinstance(current, str):
        raise ConnectionError("response path did not point at a string")
    return current
