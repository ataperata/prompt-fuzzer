"""HTTP and OpenAI-compatible targets, including rejected URL schemes."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.targets.http import HttpTarget
from prompt_fuzzer.targets.http_util import dig
from prompt_fuzzer.targets.openai import OpenAITarget


def test_dig_walks_lists() -> None:
    document = {"choices": [{"message": {"content": "hello"}}]}
    assert dig(document, "choices.0.message.content") == "hello"


def test_http_target_posts_the_prompt() -> None:
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            seen["body"] = json.loads(self.rfile.read(length).decode())
            raw = json.dumps({"reply": "ack"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        target = HttpTarget(
            url=f"http://{host}:{port}/chat",
            prompt_field="message",
            response_path="reply",
            extra_body={"user": "red-team"},
        )
        response = target.generate("hello there")
    finally:
        server.shutdown()
        server.server_close()
    assert response.error == ""
    assert response.text == "ack"
    assert seen["body"] == {"user": "red-team", "message": "hello there"}


def test_non_http_urls_are_rejected() -> None:
    with pytest.raises(ConfigError):
        HttpTarget(url="file:///etc/passwd")
    with pytest.raises(ConfigError):
        HttpTarget(url="ftp://example.com/chat")


def test_openai_target_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            assert payload["messages"][-1]["content"] == "ping"
            assert self.headers.get("Authorization") == "Bearer test-token"
            raw = json.dumps(
                {"choices": [{"message": {"content": "pong"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("PROMPT_FUZZER_TEST_KEY", "test-token")
    try:
        target = OpenAITarget(
            base_url=f"http://{host}:{port}",
            model="local",
            api_key_env="PROMPT_FUZZER_TEST_KEY",
            system_prompt="be brief",
        )
        response = target.generate("ping")
    finally:
        server.shutdown()
        server.server_close()
    assert response.text == "pong"
    assert response.error == ""
