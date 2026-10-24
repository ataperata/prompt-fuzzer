"""Self-contained HTML report with an inline fitness chart."""

from __future__ import annotations

import html
from pathlib import Path

from prompt_fuzzer.models import GenerationStats, Individual, RunResult, ancestry

_WIDTH = 760
_HEIGHT = 280
_PAD_TOP = 18
_PAD_RIGHT = 16
_PAD_BOTTOM = 36
_PAD_LEFT = 46


def write_report(result: RunResult, path: Path) -> None:
    path.write_text(render_report(result), encoding="utf-8")


def render_report(result: RunResult) -> str:
    best = result.best
    best_fitness = 0.0 if best is None or best.fitness is None else best.fitness
    title = "prompt-fuzzer report"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #10151c;
  --card: #18222d;
  --ink: #e7eef6;
  --muted: #c5d0de;
  --line: #2c3b4c;
  --best: #5ee0b5;
  --mean: #f0c674;
  --div: #8ab4f8;
  --bad: #ff8b7a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.5 "Segoe UI", Helvetica, Arial, sans-serif;
}}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }}
h1 {{ font-size: 28px; margin: 0 0 8px; letter-spacing: -0.02em; }}
h2 {{ font-size: 18px; margin: 32px 0 12px; }}
p {{ color: var(--muted); }}
.lede {{ max-width: 68ch; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }}
.card span {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
.card strong {{ font-size: 22px; font-variant-numeric: tabular-nums; }}
.panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
.case {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; margin: 12px 0; }}
.meta {{ color: var(--muted); font-size: 13px; }}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  background: #0d131a;
  border-radius: 8px;
  padding: 10px 12px;
  overflow-x: auto;
  font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.legend {{ display: flex; gap: 16px; color: var(--muted); font-size: 13px; margin-top: 8px; }}
.swatch {{ display: inline-block; width: 18px; height: 3px; margin-right: 6px; vertical-align: middle; }}
ol.lineage {{ margin: 8px 0; padding-left: 18px; }}
footer {{ margin-top: 36px; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<main>
<h1>prompt-fuzzer</h1>
<p class="lede">Evolutionary search over prompt wording. Fitness is a weighted sum of leakage, policy violation, and task failure. Findings are replies whose leakage reached {result.failure_threshold:.2f}.</p>
<section class="cards">
{_card("Best fitness", f"{best_fitness:.3f}")}
{_card("Findings", str(len(result.findings)))}
{_card("Generations", str(len(result.generations)))}
{_card("Target calls", str(result.evaluations))}
{_card("Cache hits", str(result.cache_hits))}
{_card("Seed", str(result.seed))}
</section>
<h2>Fitness by generation</h2>
<div class="panel">
{_chart(result.generations)}
<div class="legend">
<span><i class="swatch" style="background: var(--best)"></i>best</span>
<span><i class="swatch" style="background: var(--mean)"></i>mean</span>
<span><i class="swatch" style="background: var(--div)"></i>diversity</span>
</div>
</div>
{_history_table(result.generations)}
<h2>Operator credit</h2>
<p>Mean reward is the average fitness change after that mutation, measured against the stronger parent. Untried operators stay available through epsilon-greedy exploration.</p>
{_bandit_table(result.bandit)}
<h2>Findings</h2>
{_findings(result)}
<h2>Leaderboard</h2>
{_leaderboard(result)}
<footer>
Target <code>{html.escape(result.target_type)}</code>,
judge <code>{html.escape(result.judge_type)}</code>,
weights leakage {result.weights.get("leakage", 0):.2f} /
policy {result.weights.get("policy_violation", 0):.2f} /
task {result.weights.get("task_failure", 0):.2f}.
Run only against applications you are allowed to test. Treat this report as sensitive: it stores prompts and replies.
</footer>
</main>
</body>
</html>
"""


def _card(label: str, value: str) -> str:
    return (
        f'<div class="card"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong></div>"
    )


def _chart(history: list[GenerationStats]) -> str:
    if not history:
        return "<p>No generations recorded.</p>"
    best = [item.best for item in history]
    mean = [item.mean for item in history]
    diversity = [item.diversity for item in history]
    inner_w = _WIDTH - _PAD_LEFT - _PAD_RIGHT
    inner_h = _HEIGHT - _PAD_TOP - _PAD_BOTTOM
    grid = []
    for tick in (0, 0.25, 0.5, 0.75, 1):
        y = _PAD_TOP + (1 - tick) * inner_h
        grid.append(
            f'<line x1="{_PAD_LEFT}" y1="{y:.1f}" x2="{_WIDTH - _PAD_RIGHT}" '
            f'y2="{y:.1f}" stroke="#2c3b4c" stroke-width="1"/>'
            f'<text x="4" y="{y + 4:.1f}" fill="#c5d0de" font-size="11">{tick:.2f}</text>'
        )
    labels = []
    for index, item in enumerate(history):
        x = _x(index, len(history), inner_w)
        labels.append(
            f'<text x="{x:.1f}" y="{_HEIGHT - 12}" fill="#c5d0de" font-size="11" '
            f'text-anchor="middle">{item.generation}</text>'
        )
    return (
        f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img" '
        f'aria-label="Best, mean, and diversity fitness across generations">'
        f"<title>Fitness across generations</title>"
        f"{''.join(grid)}"
        f'<polyline fill="none" stroke="#8ab4f8" stroke-width="2" '
        f'points="{_polyline(diversity, inner_w, inner_h)}"/>'
        f'<polyline fill="none" stroke="#f0c674" stroke-width="2" '
        f'points="{_polyline(mean, inner_w, inner_h)}"/>'
        f'<polyline fill="none" stroke="#5ee0b5" stroke-width="2.5" '
        f'points="{_polyline(best, inner_w, inner_h)}"/>'
        f"{''.join(labels)}"
        f"</svg>"
    )


def _x(index: int, count: int, inner_w: float) -> float:
    if count == 1:
        return _PAD_LEFT + inner_w / 2
    return _PAD_LEFT + inner_w * index / (count - 1)


def _polyline(values: list[float], inner_w: float, inner_h: float) -> str:
    points = []
    for index, value in enumerate(values):
        clamped = min(1.0, max(0.0, value))
        x = _x(index, len(values), inner_w)
        y = _PAD_TOP + (1 - clamped) * inner_h
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _history_table(history: list[GenerationStats]) -> str:
    rows = []
    for item in history:
        rows.append(
            "<tr>"
            f"<td>{item.generation}</td>"
            f"<td>{item.best:.3f}</td>"
            f"<td>{item.mean:.3f}</td>"
            f"<td>{item.worst:.3f}</td>"
            f"<td>{item.diversity:.3f}</td>"
            f"<td>{item.failures}</td>"
            f"<td>{html.escape(_counts(item.operator_counts))}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return f"""
<table>
<thead><tr><th>Gen</th><th>Best</th><th>Mean</th><th>Worst</th><th>Diversity</th><th>Findings</th><th>Births</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
"""


def _counts(counts: dict[str, int]) -> str:
    if not counts:
        return "—"
    parts = [f"{name} {count}" for name, count in sorted(counts.items())]
    return ", ".join(parts)


def _bandit_table(stats: dict[str, dict[str, float]]) -> str:
    if not stats:
        return "<p>No mutation operators were enabled.</p>"
    rows = []
    for name, payload in sorted(stats.items()):
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{int(payload.get('pulls', 0))}</td>"
            f"<td>{payload.get('mean_reward', 0.0):+.3f}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return f"""
<table>
<thead><tr><th>Operator</th><th>Scored uses</th><th>Mean reward</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
"""


def _findings(result: RunResult) -> str:
    if not result.findings:
        return "<p>No reply reached the leakage threshold. The leaderboard still lists the closest prompts.</p>"
    blocks = []
    for rank, individual in enumerate(result.findings, start=1):
        blocks.append(_case(result, individual, rank))
    return "\n".join(blocks)


def _leaderboard(result: RunResult) -> str:
    rows = []
    for rank, individual in enumerate(result.hall_of_fame, start=1):
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{_fmt(individual.fitness)}</td>"
            f"<td>{_fmt(individual.leakage)}</td>"
            f"<td>{individual.generation}</td>"
            f"<td>{html.escape(individual.operator)}</td>"
            f"<td>{html.escape(individual.channel)}</td>"
            f"<td><code>{html.escape(individual.id)}</code></td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return f"""
<table>
<thead><tr><th>#</th><th>Fitness</th><th>Leakage</th><th>Gen</th><th>Operator</th><th>Channel</th><th>Id</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
"""


def _case(result: RunResult, individual: Individual, rank: int) -> str:
    chain = ancestry(individual, result.by_id)
    items = []
    for step in chain:
        items.append(
            "<li><span class=\"meta\">"
            f"gen {step.generation} · {html.escape(step.operator)} · fitness {_fmt(step.fitness)}"
            "</span><pre>"
            f"{html.escape(step.text)}</pre></li>"
        )
    mate = ""
    if len(individual.parent_ids) > 1:
        other = result.by_id.get(individual.parent_ids[1])
        if other is not None:
            mate = (
                "<p class=\"meta\">Crossover mate</p><pre>"
                f"{html.escape(other.text)}</pre>"
            )
    case_name = _case_filename(result, individual, rank)
    replay = (
        f"prompt-fuzzer replay --config {result.config_path} --case "
        f"{result.output_directory}/{case_name}"
        if result.config_path
        else f"prompt-fuzzer replay --config <config.yaml> --case {result.output_directory}/{case_name}"
    )
    return f"""
<article class="case">
<h3>Finding {rank} · fitness {_fmt(individual.fitness)} · leakage {_fmt(individual.leakage)}</h3>
<p class="meta">id {html.escape(individual.id)} · objective {html.escape(individual.objective_id)} · generation {individual.generation} · operator {html.escape(individual.operator)} · channel {html.escape(individual.channel or "n/a")}</p>
<p class="meta">{html.escape(individual.reason)}</p>
<p class="meta">Scores · policy {_fmt(individual.policy_violation)} · task {_fmt(individual.task_failure)} · refused {individual.refused}</p>
<ol class="lineage">
{''.join(items)}
</ol>
{mate}
<p class="meta">Response</p>
<pre>{html.escape(individual.response)}</pre>
<p class="meta">Replay</p>
<pre>{html.escape(replay)}</pre>
</article>
"""


def _case_filename(result: RunResult, individual: Individual, rank: int) -> str:
    for index, item in enumerate(result.hall_of_fame, start=1):
        if item.id == individual.id:
            return f"cases/{index:02d}-{item.id}.json"
    return f"cases/{rank:02d}-{individual.id}.json"


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
