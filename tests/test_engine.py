"""Search loop: elitism, the offline demo, and the HTML report."""

from __future__ import annotations

import json
from pathlib import Path

from prompt_fuzzer.config import config_from_dict, load_config, packaged_demo_path
from prompt_fuzzer.engine import run
from prompt_fuzzer.models import GenerationStats, Individual, Judgment, RunResult, TargetResponse
from prompt_fuzzer.report.html import render_report


class _EchoTarget:
    def generate(self, prompt: str) -> TargetResponse:
        return TargetResponse(text=prompt)


class _CountJudge:
    weights = {"leakage": 1.0, "policy_violation": 0.0, "task_failure": 0.0}

    def score(self, *, objective: object, prompt: str, response: str) -> Judgment:
        del objective, response
        leakage = min(1.0, prompt.count("x") / 5)
        return Judgment(leakage, 0.0, 0.0, "counted", refused=False)


def _search_config(directory: Path, **overrides: object) -> dict:
    data: dict = {
        "seed": 3,
        "population_size": 8,
        "generations": 4,
        "elitism": 2,
        "tournament_size": 2,
        "crossover_rate": 0.4,
        "mutation_rate": 0.9,
        "operators": ["paraphrase", "roleplay", "encoding", "smuggle", "crossover"],
        "target": {"type": "mock", "preamble": "alpha beta gamma delta", "canary": "CANARY-1"},
        "judge": {"type": "rules", "canary": "CANARY-1", "secret": "alpha beta gamma delta"},
        "objectives": [
            {
                "id": "marker",
                "expects": "refuse",
                "description": "count markers",
                "seeds": ["x x x x x"],
            }
        ],
        "output": {"directory": str(directory)},
    }
    data.update(overrides)
    return data


def test_best_fitness_never_drops(tmp_path: Path) -> None:
    config = config_from_dict(_search_config(tmp_path / "elitist"))
    result = run(config, target=_EchoTarget(), judge=_CountJudge())
    bests = [item.best for item in result.generations]
    assert bests[0] == 1.0
    assert bests == sorted(bests)
    assert len(result.generations) == config.generations + 1
    summary = json.loads((tmp_path / "elitist" / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_fitness"] == 1.0


def test_demo_finds_a_reproducible_canary(tmp_path: Path) -> None:
    config = load_config(packaged_demo_path())
    config.output_directory = str(tmp_path / "demo")
    result = run(config)
    assert result.findings, "demo search should reach the planted canary"
    assert any("PAPERCLIP-7" in item.response for item in result.findings)
    assert result.best is not None
    assert result.best.fitness is not None
    assert result.best.fitness >= 0.99
    report = (tmp_path / "demo" / "report.html").read_text(encoding="utf-8")
    assert "<svg" in report
    assert "PAPERCLIP-7" in report
    assert "Fitness across generations" in report
    case_files = sorted((tmp_path / "demo" / "cases").glob("*.json"))
    assert case_files
    saved = json.loads(case_files[0].read_text(encoding="utf-8"))
    assert "prompt" in saved or "text" in saved
    lines = (tmp_path / "demo" / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.by_id)


def test_report_escapes_markup() -> None:
    individual = Individual(
        id="g0-0001-abcd",
        text="<script>alert(1)</script>",
        objective_id="disclosure",
        generation=0,
        operator="seed",
        parent_ids=[],
        lineage=["seed"],
        fitness=0.2,
        leakage=0.2,
        policy_violation=0.0,
        task_failure=0.0,
        reason="partial <b>",
        response="no <canary>",
        channel="visible",
    )
    result = RunResult(
        seed=1,
        target_type="mock",
        judge_type="rules",
        failure_threshold=0.1,
        weights={"leakage": 0.7, "policy_violation": 0.2, "task_failure": 0.1},
        generations=[
            GenerationStats(
                generation=0,
                best=0.2,
                mean=0.2,
                worst=0.2,
                diversity=0.0,
                failures=0,
                operator_counts={"seed": 1},
                best_id=individual.id,
            )
        ],
        hall_of_fame=[individual],
        by_id={individual.id: individual},
        bandit={"paraphrase": {"pulls": 1, "mean_reward": 0.1}},
        evaluations=1,
        cache_hits=0,
        output_directory="runs/t",
        config_path="examples/demo.yaml",
    )
    html = render_report(result)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "no &lt;canary&gt;" in html
