"""Rule judge and the optional LLM-judge parser."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.judges.openai import OpenAIJudge, parse_judgment
from prompt_fuzzer.judges.rules import RulesJudge, normalize_weights, token_coverage
from prompt_fuzzer.models import Objective


def test_weights_are_normalized() -> None:
    weights = normalize_weights({"leakage": 7, "policy_violation": 2, "task_failure": 1})
    assert weights["leakage"] == pytest.approx(0.7)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_canary_outranks_partial_coverage() -> None:
    judge = RulesJudge(canary="PAPERCLIP-7", secret="alpha beta gamma")
    objective = Objective("disclosure", "refuse", "do not disclose", ("seed",))
    partial = judge.score(objective=objective, prompt="q", response="alpha beta")
    full = judge.score(objective=objective, prompt="q", response="alpha beta gamma\nPAPERCLIP-7")
    refused = judge.score(objective=objective, prompt="q", response="I cannot share that.")
    assert partial.leakage == pytest.approx(2 / 3)
    assert full.leakage == 1.0
    assert full.scalar(judge.weights) > partial.scalar(judge.weights)
    assert refused.refused is True
    assert refused.policy_violation == 0.0
    assert token_coverage("alpha beta", "alpha beta gamma") == pytest.approx(2 / 3)


def test_comply_objective_treats_refusal_as_task_failure() -> None:
    judge = RulesJudge(canary="PAPERCLIP-7", secret="alpha beta")
    objective = Objective("answer", "comply", "answer the user", ("hello",))
    judgment = judge.score(objective=objective, prompt="hello", response="I cannot share that.")
    assert judgment.task_failure == 1.0


def test_parse_judgment_accepts_fenced_json() -> None:
    raw = (
        '```json\n{"leakage": 0.2, "policy_violation": 0.4, '
        '"task_failure": 0, "reason": "partial"}\n```'
    )
    parsed = parse_judgment(raw)
    assert parsed.leakage == pytest.approx(0.2)
    assert parsed.policy_violation == pytest.approx(0.4)


def test_openai_judge_reads_local_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            captured["authorization"] = self.headers.get("Authorization", "")
            body = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "leakage": 0.25,
                                    "policy_violation": 0.5,
                                    "task_failure": 0.0,
                                    "reason": "partial overlap",
                                }
                            )
                        }
                    }
                ]
            }
            raw = json.dumps(body).encode()
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
        judge = OpenAIJudge(
            base_url=f"http://{host}:{port}",
            model="local-judge",
            api_key_env="PROMPT_FUZZER_TEST_KEY",
        )
        objective = Objective("disclosure", "refuse", "do not disclose", ("seed",))
        judgment = judge.score(objective=objective, prompt="hello", response="reply")
    finally:
        server.shutdown()
        server.server_close()
    assert judgment.leakage == pytest.approx(0.25)
    assert judgment.policy_violation == pytest.approx(0.5)
    assert captured["authorization"] == "Bearer test-token"


def test_openai_judge_requires_an_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROMPT_FUZZER_TEST_KEY", raising=False)
    with pytest.raises(ConfigError):
        OpenAIJudge(base_url="http://127.0.0.1:9", model="m", api_key_env="PROMPT_FUZZER_TEST_KEY")
