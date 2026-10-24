"""Genetic search loop.

Each generation evaluates the population, records fitness, then builds the next
population from elites plus children. Children are produced by tournament
selection, optional crossover, and a mutation operator chosen by a bandit that
is credited with the fitness delta.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from prompt_fuzzer.bandit import OperatorBandit
from prompt_fuzzer.config import AppConfig, redact_config
from prompt_fuzzer.judges import build_judge
from prompt_fuzzer.judges.rules import normalize_weights
from prompt_fuzzer.models import (
    GenerationStats,
    Individual,
    RunResult,
    TargetResponse,
)
from prompt_fuzzer.operators import Crossover, build_mutators
from prompt_fuzzer.targets import build_target
from prompt_fuzzer.textutil import jaccard_distance

Progress = Callable[[GenerationStats], None]


def run(
    config: AppConfig,
    *,
    on_generation: Progress | None = None,
    target: object | None = None,
    judge: object | None = None,
) -> RunResult:
    rng = random.Random(config.seed)
    target = target or build_target(config.target)
    judge = judge or build_judge(config.judge)
    weights = _weights(judge, config)
    mutators = build_mutators(config.mutator_names)
    bandit = OperatorBandit(
        list(mutators),
        exploration=config.bandit_exploration,
        epsilon=config.bandit_epsilon,
    ) if mutators else None
    crossover = Crossover()
    objectives = {item.id: item for item in config.objectives}
    cache: dict[str, TargetResponse] = {}
    by_id: dict[str, Individual] = {}
    counter = {"n": 0}
    evaluations = {"n": 0}
    cache_hits = {"n": 0}

    population = _initial_population(config, rng, counter)
    for individual in population:
        by_id[individual.id] = individual

    stopped_early = False
    history: list[GenerationStats] = []
    fresh = list(population)
    total_rounds = config.generations

    for generation in range(total_rounds + 1):
        _evaluate(
            fresh,
            target=target,
            judge=judge,
            objectives=objectives,
            weights=weights,
            cache=cache,
            cache_responses=config.cache_responses,
            evaluations=evaluations,
            cache_hits=cache_hits,
        )
        if bandit is not None:
            _credit(fresh, by_id, bandit)
        stats = _stats(generation, population, config.failure_threshold)
        history.append(stats)
        if on_generation is not None:
            on_generation(stats)
        solved = any((ind.leakage or 0.0) >= config.failure_threshold for ind in population)
        if generation == total_rounds:
            break
        if solved and config.stop_on_finding:
            stopped_early = True
            break
        population, fresh = _breed(
            population,
            config=config,
            rng=rng,
            generation=generation + 1,
            counter=counter,
            mutators=mutators,
            crossover=crossover,
            bandit=bandit,
        )
        for individual in fresh:
            by_id[individual.id] = individual

    result = RunResult(
        seed=config.seed,
        target_type=str(config.target.get("type", "")),
        judge_type=str(config.judge.get("type", "")),
        failure_threshold=config.failure_threshold,
        weights=weights,
        generations=history,
        hall_of_fame=_hall_of_fame(by_id),
        by_id=by_id,
        bandit={} if bandit is None else bandit.stats(),
        evaluations=evaluations["n"],
        cache_hits=cache_hits["n"],
        output_directory=config.output_directory,
        config_path=config.source_path,
        stopped_early=stopped_early,
    )
    persist(result, config)
    return result


def persist(result: RunResult, config: AppConfig) -> None:
    from prompt_fuzzer.report.html import write_report

    directory = Path(config.output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(result.to_summary(), indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "history.json").write_text(
        json.dumps([item.to_dict() for item in result.generations], indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "hall_of_fame.json").write_text(
        json.dumps([item.to_dict() for item in result.hall_of_fame], indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "resolved.json").write_text(
        json.dumps(redact_config(config), indent=2) + "\n",
        encoding="utf-8",
    )
    ordered = sorted(result.by_id.values(), key=lambda ind: (ind.generation, ind.id))
    lines = [json.dumps(ind.to_dict(), sort_keys=True) for ind in ordered]
    (directory / "evaluations.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cases = directory / "cases"
    cases.mkdir(exist_ok=True)
    for index, individual in enumerate(result.hall_of_fame, start=1):
        payload = individual.to_dict()
        payload["run_seed"] = result.seed
        payload["config_path"] = result.config_path
        path = cases / f"{index:02d}-{individual.id}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(result, directory / "report.html")


def _weights(judge: object, config: AppConfig) -> dict[str, float]:
    attached = getattr(judge, "weights", None)
    if isinstance(attached, dict) and attached:
        return normalize_weights({str(key): float(value) for key, value in attached.items()})
    raw = config.judge.get("weights")
    if not isinstance(raw, dict):
        raise RuntimeError("judge weights are missing")
    return normalize_weights({str(key): float(value) for key, value in raw.items()})


def _next_id(rng: random.Random, generation: int, counter: dict[str, int]) -> str:
    counter["n"] += 1
    return f"g{generation}-{counter['n']:04d}-{rng.randrange(16**4):04x}"


def _make(
    *,
    rng: random.Random,
    counter: dict[str, int],
    text: str,
    objective_id: str,
    generation: int,
    operator: str,
    parent_ids: list[str],
    lineage: list[str],
) -> Individual:
    return Individual(
        id=_next_id(rng, generation, counter),
        text=text,
        objective_id=objective_id,
        generation=generation,
        operator=operator,
        parent_ids=list(parent_ids),
        lineage=list(lineage),
    )


def _initial_population(
    config: AppConfig,
    rng: random.Random,
    counter: dict[str, int],
) -> list[Individual]:
    mutators = build_mutators(config.mutator_names)
    names = list(mutators)
    seeds: list[tuple[str, str]] = []
    for objective in config.objectives:
        for text in objective.seeds:
            seeds.append((objective.id, text))
    population: list[Individual] = []
    for objective_id, text in seeds:
        if len(population) >= config.population_size:
            break
        population.append(
            _make(
                rng=rng,
                counter=counter,
                text=text,
                objective_id=objective_id,
                generation=0,
                operator="seed",
                parent_ids=[],
                lineage=["seed"],
            )
        )
    index = 0
    while len(population) < config.population_size:
        parent = population[index % len(population)]
        if names:
            name = names[index % len(names)]
            mutated = mutators[name].mutate(parent.text, rng)  # type: ignore[attr-defined]
            if mutated != parent.text and len(mutated) <= config.max_prompt_chars:
                population.append(
                    _make(
                        rng=rng,
                        counter=counter,
                        text=mutated,
                        objective_id=parent.objective_id,
                        generation=0,
                        operator=name,
                        parent_ids=[parent.id],
                        lineage=parent.lineage + [name],
                    )
                )
            else:
                population.append(_clone(parent, rng, counter, generation=0))
        else:
            population.append(_clone(parent, rng, counter, generation=0))
        index += 1
    return population


def _clone(
    parent: Individual,
    rng: random.Random,
    counter: dict[str, int],
    generation: int,
) -> Individual:
    return _make(
        rng=rng,
        counter=counter,
        text=parent.text,
        objective_id=parent.objective_id,
        generation=generation,
        operator="clone",
        parent_ids=[parent.id],
        lineage=parent.lineage + ["clone"],
    )


def _evaluate(
    fresh: list[Individual],
    *,
    target: object,
    judge: object,
    objectives: dict,
    weights: dict[str, float],
    cache: dict[str, TargetResponse],
    cache_responses: bool,
    evaluations: dict[str, int],
    cache_hits: dict[str, int],
) -> None:
    for individual in fresh:
        if individual.fitness is not None:
            continue
        response, was_cached = _respond(
            individual.text,
            target=target,
            cache=cache,
            cache_responses=cache_responses,
            evaluations=evaluations,
            cache_hits=cache_hits,
        )
        individual.cached = was_cached
        judgment = judge.score(  # type: ignore[attr-defined]
            objective=objectives[individual.objective_id],
            prompt=individual.text,
            response=response.text,
        )
        scored = judgment.clamped()
        individual.leakage = scored.leakage
        individual.policy_violation = scored.policy_violation
        individual.task_failure = scored.task_failure
        individual.reason = scored.reason
        individual.refused = scored.refused
        individual.fitness = scored.scalar(weights)
        individual.response = response.text[:20000]
        individual.error = response.error
        channel = response.meta.get("channel", "")
        individual.channel = str(channel)


def _respond(
    text: str,
    *,
    target: object,
    cache: dict[str, TargetResponse],
    cache_responses: bool,
    evaluations: dict[str, int],
    cache_hits: dict[str, int],
) -> tuple[TargetResponse, bool]:
    if cache_responses and text in cache:
        cache_hits["n"] += 1
        cached = cache[text]
        copy = TargetResponse(text=cached.text, error=cached.error, meta=dict(cached.meta))
        return copy, True
    response = target.generate(text)  # type: ignore[attr-defined]
    evaluations["n"] += 1
    if cache_responses and not response.error:
        cache[text] = response
    return response, False


def _credit(fresh: list[Individual], by_id: dict[str, Individual], bandit: OperatorBandit) -> None:
    for individual in fresh:
        if individual.fitness is None:
            continue
        name = individual.operator.split("+")[-1]
        if name not in bandit.updates:
            continue
        parent_scores = [
            by_id[parent_id].fitness
            for parent_id in individual.parent_ids
            if parent_id in by_id and by_id[parent_id].fitness is not None
        ]
        if not parent_scores:
            continue
        baseline = max(score for score in parent_scores if score is not None)
        bandit.update(name, individual.fitness - baseline)


def _stats(generation: int, population: list[Individual], threshold: float) -> GenerationStats:
    fitnesses = [ind.fitness or 0.0 for ind in population]
    best = max(population, key=lambda ind: (ind.fitness or 0.0, -ind.generation, ind.id))
    counts: dict[str, int] = {}
    for individual in population:
        if individual.generation != generation:
            continue
        counts[individual.operator] = counts.get(individual.operator, 0) + 1
    return GenerationStats(
        generation=generation,
        best=max(fitnesses),
        mean=sum(fitnesses) / len(fitnesses),
        worst=min(fitnesses),
        diversity=_diversity(population),
        failures=sum(1 for ind in population if (ind.leakage or 0.0) >= threshold),
        operator_counts=counts,
        best_id=best.id,
    )


def _diversity(population: list[Individual]) -> float:
    if len(population) < 2:
        return 0.0
    nearest: list[float] = []
    for index, individual in enumerate(population):
        distance = min(
            jaccard_distance(individual.text, other.text)
            for other_index, other in enumerate(population)
            if other_index != index
        )
        nearest.append(distance)
    return sum(nearest) / len(nearest)


def _hall_of_fame(by_id: dict[str, Individual], limit: int = 12) -> list[Individual]:
    ordered = sorted(
        by_id.values(),
        key=lambda ind: (-(ind.fitness or 0.0), ind.generation, ind.id),
    )
    selected: list[Individual] = []
    seen: set[str] = set()
    for individual in ordered:
        if individual.fitness is None or individual.text in seen:
            continue
        seen.add(individual.text)
        selected.append(individual)
        if len(selected) == limit:
            break
    return selected


def _breed(
    population: list[Individual],
    *,
    config: AppConfig,
    rng: random.Random,
    generation: int,
    counter: dict[str, int],
    mutators: dict,
    crossover: Crossover,
    bandit: OperatorBandit | None,
) -> tuple[list[Individual], list[Individual]]:
    elites = _elites(population, config.elitism)
    children: list[Individual] = []
    taken = {individual.text for individual in elites}
    keys = _selection_keys(population, config.novelty_weight)
    while len(elites) + len(children) < config.population_size:
        child = _birth(
            population,
            keys=keys,
            taken=taken,
            config=config,
            rng=rng,
            generation=generation,
            counter=counter,
            mutators=mutators,
            crossover=crossover,
            bandit=bandit,
        )
        children.append(child)
        taken.add(child.text)
    return elites + children, children


def _elites(population: list[Individual], count: int) -> list[Individual]:
    ordered = sorted(population, key=lambda ind: (-(ind.fitness or 0.0), ind.generation, ind.id))
    elites: list[Individual] = []
    seen: set[str] = set()
    for individual in ordered:
        if individual.text in seen:
            continue
        seen.add(individual.text)
        elites.append(individual)
        if len(elites) == count:
            break
    if len(elites) < count:
        for individual in ordered:
            if len(elites) == count:
                break
            elites.append(individual)
    return elites


def _selection_keys(population: list[Individual], novelty_weight: float) -> dict[str, float]:
    keys: dict[str, float] = {}
    for individual in population:
        if len(population) == 1:
            novelty = 0.0
        else:
            novelty = min(
                jaccard_distance(individual.text, other.text)
                for other in population
                if other.id != individual.id
            )
        keys[individual.id] = (individual.fitness or 0.0) + novelty_weight * novelty
    return keys


def _tournament(
    population: list[Individual],
    keys: dict[str, float],
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    group = rng.sample(population, k=min(tournament_size, len(population)))
    return max(group, key=lambda ind: keys[ind.id])


def _birth(
    population: list[Individual],
    *,
    keys: dict[str, float],
    taken: set[str],
    config: AppConfig,
    rng: random.Random,
    generation: int,
    counter: dict[str, int],
    mutators: dict,
    crossover: Crossover,
    bandit: OperatorBandit | None,
) -> Individual:
    fallback_parent = _tournament(population, keys, config.tournament_size, rng)
    for _ in range(8):
        text, operator, parents = _propose(
            population,
            keys=keys,
            config=config,
            rng=rng,
            mutators=mutators,
            crossover=crossover,
            bandit=bandit,
        )
        if len(text) > config.max_prompt_chars or text in taken:
            continue
        primary = parents[0]
        return _make(
            rng=rng,
            counter=counter,
            text=text,
            objective_id=primary.objective_id,
            generation=generation,
            operator=operator,
            parent_ids=[parent.id for parent in parents],
            lineage=primary.lineage + [operator],
        )
    return _clone(fallback_parent, rng, counter, generation)


def _propose(
    population: list[Individual],
    *,
    keys: dict[str, float],
    config: AppConfig,
    rng: random.Random,
    mutators: dict,
    crossover: Crossover,
    bandit: OperatorBandit | None,
) -> tuple[str, str, list[Individual]]:
    primary = _tournament(population, keys, config.tournament_size, rng)
    parents = [primary]
    text = primary.text
    applied: list[str] = []
    if config.uses_crossover and len(population) > 1 and rng.random() < config.crossover_rate:
        mate = _tournament(population, keys, config.tournament_size, rng)
        text = crossover.cross(primary.text, mate.text, rng)
        parents = [primary, mate]
        applied.append("crossover")
    if mutators and bandit is not None and rng.random() < config.mutation_rate:
        name = bandit.choose(rng)
        mutated = mutators[name].mutate(text, rng)
        if mutated != text:
            text = mutated
        applied.append(name)
    if not applied:
        applied.append("clone")
    return text, "+".join(applied), parents
