"""Datatypes passed between the engine, targets, judges, and the report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TargetResponse:
    text: str
    error: str = ""
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Objective:
    id: str
    expects: str
    description: str
    seeds: tuple[str, ...]


@dataclass
class Judgment:
    leakage: float
    policy_violation: float
    task_failure: float
    reason: str
    refused: bool

    def clamped(self) -> Judgment:
        def unit(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        return Judgment(
            leakage=unit(self.leakage),
            policy_violation=unit(self.policy_violation),
            task_failure=unit(self.task_failure),
            reason=self.reason,
            refused=self.refused,
        )

    def scalar(self, weights: dict[str, float]) -> float:
        scored = self.clamped()
        total = (
            weights["leakage"] * scored.leakage
            + weights["policy_violation"] * scored.policy_violation
            + weights["task_failure"] * scored.task_failure
        )
        # Binary fractions such as 0.7 + 0.2 + 0.1 undershoot 1. Keep scores stable.
        return round(total, 6)


@dataclass
class Individual:
    id: str
    text: str
    objective_id: str
    generation: int
    operator: str
    parent_ids: list[str]
    lineage: list[str]
    fitness: float | None = None
    leakage: float | None = None
    policy_violation: float | None = None
    task_failure: float | None = None
    reason: str = ""
    response: str = ""
    error: str = ""
    refused: bool | None = None
    channel: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class GenerationStats:
    generation: int
    best: float
    mean: float
    worst: float
    diversity: float
    failures: int
    operator_counts: dict[str, int]
    best_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RunResult:
    seed: int
    target_type: str
    judge_type: str
    failure_threshold: float
    weights: dict[str, float]
    generations: list[GenerationStats]
    hall_of_fame: list[Individual]
    by_id: dict[str, Individual]
    bandit: dict[str, dict[str, float]]
    evaluations: int
    cache_hits: int
    output_directory: str
    config_path: str
    stopped_early: bool = False

    @property
    def best(self) -> Individual | None:
        if not self.hall_of_fame:
            return None
        return self.hall_of_fame[0]

    @property
    def findings(self) -> list[Individual]:
        return [ind for ind in self.hall_of_fame if (ind.leakage or 0.0) >= self.failure_threshold]

    def to_summary(self) -> dict[str, object]:
        best = self.best
        return {
            "seed": self.seed,
            "target_type": self.target_type,
            "judge_type": self.judge_type,
            "failure_threshold": self.failure_threshold,
            "weights": self.weights,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "stopped_early": self.stopped_early,
            "best_fitness": None if best is None else best.fitness,
            "best_id": None if best is None else best.id,
            "findings": len(self.findings),
            "generations": [stats.to_dict() for stats in self.generations],
            "bandit": self.bandit,
            "output_directory": self.output_directory,
            "config_path": self.config_path,
        }


def ancestry(individual: Individual, by_id: dict[str, Individual]) -> list[Individual]:
    """Walk the primary parent link back to the seed."""

    chain: list[Individual] = []
    seen: set[str] = set()
    current: Individual | None = individual
    while current is not None and current.id not in seen:
        seen.add(current.id)
        chain.append(current)
        if not current.parent_ids:
            break
        current = by_id.get(current.parent_ids[0])
    chain.reverse()
    return chain
