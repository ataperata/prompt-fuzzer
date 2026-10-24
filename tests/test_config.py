"""Config validation, demo file parity, and header redaction."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_fuzzer.config import config_from_dict, load_config, packaged_demo_path, redact_config
from prompt_fuzzer.errors import ConfigError


def test_packaged_demo_matches_the_example() -> None:
    packaged = packaged_demo_path().read_text(encoding="utf-8")
    example = Path("examples/demo.yaml").read_text(encoding="utf-8")
    assert packaged == example


def test_demo_config_loads() -> None:
    config = load_config(packaged_demo_path())
    assert config.target["canary"] == "PAPERCLIP-7"
    assert config.judge["secret"].strip() == config.target["preamble"].strip()
    assert config.mutator_names == ["paraphrase", "roleplay", "encoding", "smuggle"]
    assert config.uses_crossover is True
    assert sum(config.judge["weights"].values()) == pytest.approx(1.0)


def test_rules_judge_inherits_mock_secret() -> None:
    config = config_from_dict(
        {
            "operators": ["paraphrase"],
            "target": {"type": "mock", "preamble": "alpha beta", "canary": "CANARY-1"},
            "judge": {"type": "rules"},
            "objectives": [
                {"id": "d", "expects": "refuse", "description": "hide it", "seeds": ["hello"]}
            ],
            "output": {"directory": "runs/t"},
        }
    )
    assert config.judge["canary"] == "CANARY-1"
    assert config.judge["secret"] == "alpha beta"


def test_unknown_operator_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown operator"):
        config_from_dict(
            {
                "operators": ["payloads"],
                "target": {"type": "mock", "preamble": "alpha", "canary": "C"},
                "judge": {"type": "rules", "canary": "C", "secret": "alpha"},
                "objectives": [
                    {"id": "d", "expects": "refuse", "description": "x", "seeds": ["hello"]}
                ],
                "output": {"directory": "runs/t"},
            }
        )


def test_headers_are_redacted() -> None:
    config = config_from_dict(
        {
            "operators": ["paraphrase"],
            "target": {
                "type": "http",
                "url": "http://127.0.0.1:9/chat",
                "headers": {"Authorization": "Bearer example"},
            },
            "judge": {"type": "rules", "canary": "C", "secret": "alpha beta"},
            "objectives": [
                {"id": "d", "expects": "refuse", "description": "x", "seeds": ["hello"]}
            ],
            "output": {"directory": "runs/t"},
        }
    )
    redacted = redact_config(config)
    assert redacted["target"]["headers"]["Authorization"] == "***"
    assert "Bearer example" not in str(redacted)
