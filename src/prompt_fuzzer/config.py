"""Load and validate YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.judges.rules import DEFAULT_WEIGHTS, normalize_weights
from prompt_fuzzer.models import Objective
from prompt_fuzzer.operators import OPERATOR_NAMES

_EXPECTS = frozenset({"refuse", "comply"})


@dataclass
class AppConfig:
    seed: int
    population_size: int
    generations: int
    elitism: int
    tournament_size: int
    crossover_rate: float
    mutation_rate: float
    novelty_weight: float
    failure_threshold: float
    max_prompt_chars: int
    stop_on_finding: bool
    operators: list[str]
    target: dict[str, Any]
    judge: dict[str, Any]
    objectives: list[Objective]
    output_directory: str
    bandit_exploration: float
    bandit_epsilon: float
    cache_responses: bool
    source_path: str
    raw: dict[str, Any]

    @property
    def mutator_names(self) -> list[str]:
        return [name for name in self.operators if name != "crossover"]

    @property
    def uses_crossover(self) -> bool:
        return "crossover" in self.operators


def packaged_demo_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "demo.yaml"


def load_config(path: Path) -> AppConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("config root must be a mapping")
    return config_from_dict(loaded, source_path=str(path))


def config_from_dict(data: dict[str, Any], source_path: str = "") -> AppConfig:
    target = _mapping(data.get("target"), "target")
    judge = _mapping(data.get("judge"), "judge")
    judge = _fill_judge_from_mock(target, judge)
    operators = _operators(data.get("operators"))
    objectives = _objectives(data.get("objectives"))
    output = _mapping(data.get("output"), "output")
    if "directory" not in output or not isinstance(output["directory"], str):
        raise ConfigError("output.directory must be a string")

    population = _int(data, "population_size", 24, minimum=4)
    elitism = _int(data, "elitism", 2, minimum=1)
    if elitism >= population:
        raise ConfigError("elitism must be smaller than population_size")
    tournament = _int(data, "tournament_size", 3, minimum=2)
    if tournament > population:
        raise ConfigError("tournament_size cannot exceed population_size")

    weights = judge.get("weights")
    if weights is None:
        judge = {**judge, "weights": dict(DEFAULT_WEIGHTS)}
    else:
        if not isinstance(weights, dict):
            raise ConfigError("judge.weights must be a mapping")
        normalized = normalize_weights({str(key): float(value) for key, value in weights.items()})
        judge = {**judge, "weights": normalized}

    return AppConfig(
        seed=_int(data, "seed", 7, minimum=0),
        population_size=population,
        generations=_int(data, "generations", 10, minimum=1),
        elitism=elitism,
        tournament_size=tournament,
        crossover_rate=_rate(data, "crossover_rate", 0.35),
        mutation_rate=_rate(data, "mutation_rate", 0.9),
        novelty_weight=_rate(data, "novelty_weight", 0.04),
        failure_threshold=_rate(data, "failure_threshold", 0.99),
        max_prompt_chars=_int(data, "max_prompt_chars", 2500, minimum=80),
        stop_on_finding=_bool(data, "stop_on_finding", False),
        operators=operators,
        target=target,
        judge=judge,
        objectives=objectives,
        output_directory=output["directory"],
        bandit_exploration=_positive_float(data, "bandit_exploration", 1.2),
        bandit_epsilon=_rate(data, "bandit_epsilon", 0.25),
        cache_responses=_bool(data, "cache_responses", True),
        source_path=source_path,
        raw=data,
    )


def redact_config(config: AppConfig) -> dict[str, Any]:
    """Drop header values so a saved snapshot cannot keep credentials."""

    target = dict(config.target)
    headers = target.get("headers")
    if isinstance(headers, dict):
        target["headers"] = {str(key): "***" for key in headers}
    return {
        "seed": config.seed,
        "population_size": config.population_size,
        "generations": config.generations,
        "elitism": config.elitism,
        "tournament_size": config.tournament_size,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "novelty_weight": config.novelty_weight,
        "failure_threshold": config.failure_threshold,
        "max_prompt_chars": config.max_prompt_chars,
        "stop_on_finding": config.stop_on_finding,
        "operators": list(config.operators),
        "target": target,
        "judge": config.judge,
        "objectives": [
            {
                "id": item.id,
                "expects": item.expects,
                "description": item.description,
                "seeds": list(item.seeds),
            }
            for item in config.objectives
        ],
        "output": {"directory": config.output_directory},
        "bandit_exploration": config.bandit_exploration,
        "bandit_epsilon": config.bandit_epsilon,
        "cache_responses": config.cache_responses,
        "source_path": config.source_path,
    }


def _fill_judge_from_mock(target: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    if judge.get("type") != "rules" or target.get("type") != "mock":
        return judge
    filled = dict(judge)
    if not filled.get("secret") and isinstance(target.get("preamble"), str):
        filled["secret"] = target["preamble"]
    if not filled.get("canary") and isinstance(target.get("canary"), str):
        filled["canary"] = target["canary"]
    return filled


def _operators(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError("operators must be a non-empty list")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in OPERATOR_NAMES:
            known = ", ".join(sorted(OPERATOR_NAMES))
            raise ConfigError(f"unknown operator {item!r}; expected one of {known}")
        if item not in names:
            names.append(item)
    return names


def _objectives(value: object) -> list[Objective]:
    if not isinstance(value, list) or not value:
        raise ConfigError("objectives must be a non-empty list")
    built: list[Objective] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigError(f"objectives[{index}] must be a mapping")
        name = item.get("id")
        expects = item.get("expects")
        description = item.get("description", "")
        seeds = item.get("seeds")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"objectives[{index}].id must be a string")
        if name in seen:
            raise ConfigError(f"duplicate objective id {name!r}")
        seen.add(name)
        if expects not in _EXPECTS:
            raise ConfigError(f"objective {name!r} expects must be 'refuse' or 'comply'")
        if not isinstance(description, str):
            raise ConfigError(f"objective {name!r} description must be a string")
        seeds_ok = isinstance(seeds, list) and bool(seeds)
        seeds_ok = seeds_ok and all(isinstance(seed, str) and seed.strip() for seed in seeds)
        if not seeds_ok:
            raise ConfigError(f"objective {name!r} needs at least one non-empty seed string")
        built.append(
            Objective(
                id=name.strip(),
                expects=str(expects),
                description=description.strip(),
                seeds=tuple(seed.strip() for seed in seeds),
            )
        )
    return built


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _int(data: dict[str, Any], key: str, default: int, minimum: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}")
    return value


def _rate(data: dict[str, Any], key: str, default: float) -> float:
    value = _positive_float(data, key, default, allow_zero=True)
    if value > 1:
        raise ConfigError(f"{key} must be between 0 and 1")
    return value


def _positive_float(
    data: dict[str, Any],
    key: str,
    default: float,
    allow_zero: bool = False,
) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    number = float(value)
    if allow_zero:
        if number < 0:
            raise ConfigError(f"{key} must be >= 0")
    elif number <= 0:
        raise ConfigError(f"{key} must be > 0")
    return number


def _bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value
