"""Build a target from a config mapping."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.targets.http import HttpTarget
from prompt_fuzzer.targets.mock import MockTarget
from prompt_fuzzer.targets.openai import OpenAITarget

_RESERVED = {"type", "dotted"}


def load_dotted(path: str) -> type:
    if not isinstance(path, str) or ":" not in path:
        raise ConfigError("python target dotted path must look like 'module:Class'")
    module_name, _, class_name = path.partition(":")
    parts = module_name.split(".")
    if not module_name or not class_name or module_name.startswith(".") or ".." in parts:
        raise ConfigError("python target module path must be absolute and importable")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"cannot import {module_name}") from exc
    try:
        loaded = getattr(module, class_name)
    except AttributeError as exc:
        raise ConfigError(f"{module_name} has no attribute {class_name}") from exc
    if not callable(loaded):
        raise ConfigError(f"{path} is not callable")
    return loaded


def build_target(spec: Mapping[str, object]):
    kind = spec.get("type")
    if kind == "mock":
        return _mock(spec)
    if kind == "http":
        return _http(spec)
    if kind == "openai":
        return _openai(spec)
    if kind == "python":
        return _python(spec)
    raise ConfigError(f"unknown target type {kind!r}; expected mock, http, openai, or python")


def _require(spec: Mapping[str, object], *keys: str, allowed: set[str]) -> None:
    missing = [key for key in keys if key not in spec]
    if missing:
        raise ConfigError(f"target is missing {', '.join(missing)}")
    unknown = set(spec) - allowed - {"type"}
    if unknown:
        raise ConfigError(f"unknown target options: {', '.join(sorted(unknown))}")


def _mock(spec: Mapping[str, object]) -> MockTarget:
    _require(spec, "preamble", "canary", allowed={"preamble", "canary"})
    preamble = spec["preamble"]
    canary = spec["canary"]
    if not isinstance(preamble, str) or not isinstance(canary, str):
        raise ConfigError("preamble and canary must be strings")
    try:
        return MockTarget(preamble=preamble, canary=canary)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _http(spec: Mapping[str, object]) -> HttpTarget:
    allowed = {"url", "prompt_field", "response_path", "timeout_s", "headers", "extra_body"}
    _require(spec, "url", allowed=allowed)
    headers = spec.get("headers") or {}
    extra = spec.get("extra_body") or {}
    if not isinstance(headers, dict) or not isinstance(extra, dict):
        raise ConfigError("headers and extra_body must be mappings")
    return HttpTarget(
        url=str(spec["url"]),
        prompt_field=str(spec.get("prompt_field", "message")),
        response_path=str(spec.get("response_path", "reply")),
        timeout_s=float(spec.get("timeout_s", 30)),
        headers={str(k): str(v) for k, v in headers.items()},
        extra_body=dict(extra),
    )


def _openai(spec: Mapping[str, object]) -> OpenAITarget:
    allowed = {"base_url", "model", "api_key_env", "system_prompt", "timeout_s", "temperature"}
    _require(spec, "base_url", "model", allowed=allowed)
    return OpenAITarget(
        base_url=str(spec["base_url"]),
        model=str(spec["model"]),
        api_key_env=str(spec.get("api_key_env", "OPENAI_API_KEY")),
        system_prompt=str(spec.get("system_prompt", "")),
        timeout_s=float(spec.get("timeout_s", 60)),
        temperature=float(spec.get("temperature", 0)),
    )


def _python(spec: Mapping[str, object]):
    if "dotted" not in spec:
        raise ConfigError("python target requires dotted")
    cls = load_dotted(str(spec["dotted"]))
    options = {key: value for key, value in spec.items() if key not in _RESERVED}
    try:
        target = cls(**options)
    except TypeError as exc:
        raise ConfigError(f"could not construct {spec['dotted']}: {exc}") from exc
    if not hasattr(target, "generate"):
        raise ConfigError(f"{spec['dotted']} must define generate(prompt)")
    return target
