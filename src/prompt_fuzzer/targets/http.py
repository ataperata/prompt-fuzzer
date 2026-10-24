"""POST a prompt to an HTTP endpoint you control and read a JSON field back."""

from __future__ import annotations

from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.models import TargetResponse
from prompt_fuzzer.targets.http_util import check_http_url, dig, post_json


class HttpTarget:
    def __init__(
        self,
        url: str,
        prompt_field: str = "message",
        response_path: str = "reply",
        timeout_s: float = 30,
        headers: dict[str, str] | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        self.url = check_http_url(url)
        if not prompt_field:
            raise ConfigError("prompt_field must not be empty")
        self.prompt_field = prompt_field
        self.response_path = response_path
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ConfigError("timeout_s must be positive")
        self.headers = dict(headers or {})
        self.extra_body = dict(extra_body or {})

    def generate(self, prompt: str) -> TargetResponse:
        payload = dict(self.extra_body)
        payload[self.prompt_field] = prompt
        try:
            document = post_json(self.url, payload, self.headers, self.timeout_s)
            text = dig(document, self.response_path)
        except (ConnectionError, TimeoutError, OSError, ConfigError) as exc:
            return TargetResponse(text="", error=str(exc))
        return TargetResponse(text=text, meta={"url": self.url})
