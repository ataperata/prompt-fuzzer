"""Judge factory."""

from __future__ import annotations

from collections.abc import Mapping

from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.judges.openai import OpenAIJudge
from prompt_fuzzer.judges.rules import RulesJudge


def build_judge(spec: Mapping[str, object]):
    kind = spec.get("type")
    if kind == "rules":
        return _rules(spec)
    if kind == "openai":
        return _openai(spec)
    raise ConfigError(f"unknown judge type {kind!r}; expected rules or openai")


def _rules(spec: Mapping[str, object]) -> RulesJudge:
    allowed = {"type", "canary", "secret", "weights", "refusal_markers"}
    unknown = set(spec) - allowed
    if unknown:
        raise ConfigError(f"unknown judge options: {', '.join(sorted(unknown))}")
    weights = spec.get("weights")
    markers = spec.get("refusal_markers")
    if weights is not None and not isinstance(weights, dict):
        raise ConfigError("judge weights must be a mapping")
    if markers is not None and not isinstance(markers, list):
        raise ConfigError("refusal_markers must be a list")
    return RulesJudge(
        canary=str(spec.get("canary", "")),
        secret=str(spec.get("secret", "")),
        weights=dict(weights) if isinstance(weights, dict) else None,
        refusal_markers=markers,
    )


def _openai(spec: Mapping[str, object]) -> OpenAIJudge:
    allowed = {
        "type",
        "base_url",
        "model",
        "api_key_env",
        "canary",
        "secret",
        "timeout_s",
        "temperature",
        "weights",
    }
    unknown = set(spec) - allowed
    if unknown:
        raise ConfigError(f"unknown judge options: {', '.join(sorted(unknown))}")
    if "base_url" not in spec or "model" not in spec:
        raise ConfigError("openai judge requires base_url and model")
    weights = spec.get("weights")
    if weights is not None and not isinstance(weights, dict):
        raise ConfigError("judge weights must be a mapping")
    return OpenAIJudge(
        base_url=str(spec["base_url"]),
        model=str(spec["model"]),
        api_key_env=str(spec.get("api_key_env", "OPENAI_API_KEY")),
        canary=str(spec.get("canary", "")),
        secret=str(spec.get("secret", "")),
        timeout_s=float(spec.get("timeout_s", 60)),
        temperature=float(spec.get("temperature", 0)),
        weights=dict(weights) if isinstance(weights, dict) else None,
    )
