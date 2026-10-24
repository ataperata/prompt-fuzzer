"""OpenAI-compatible chat completions target.

The API key is read from the environment at construction time and is never
written into run artifacts by this adapter.
"""

from __future__ import annotations

import os

from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.models import TargetResponse
from prompt_fuzzer.targets.http_util import check_http_url, dig, post_json


class OpenAITarget:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        system_prompt: str = "",
        timeout_s: float = 60,
        temperature: float = 0,
    ) -> None:
        self.base_url = check_http_url(base_url.rstrip("/"))
        if not model:
            raise ConfigError("model must not be empty")
        if not api_key_env:
            raise ConfigError("api_key_env must not be empty")
        key = os.environ.get(api_key_env)
        if not key:
            raise ConfigError(
                f"environment variable {api_key_env} is not set; "
                "the openai target is optional and the offline demo does not need it"
            )
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self._headers = {"Authorization": f"Bearer {key}"}

    def generate(self, prompt: str) -> TargetResponse:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            document = post_json(url, payload, self._headers, self.timeout_s)
            text = dig(document, "choices.0.message.content")
        except (ConnectionError, TimeoutError, OSError) as exc:
            return TargetResponse(text="", error=str(exc))
        return TargetResponse(text=text, meta={"model": self.model})
