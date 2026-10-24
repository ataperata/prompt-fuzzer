"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prompt_fuzzer import __version__
from prompt_fuzzer.config import AppConfig, load_config, packaged_demo_path
from prompt_fuzzer.engine import run
from prompt_fuzzer.errors import ConfigError
from prompt_fuzzer.models import GenerationStats
from prompt_fuzzer.targets import build_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prompt-fuzzer",
        description="Evolve prompts against an LLM application you are allowed to test.",
    )
    parser.add_argument("--version", action="version", version=f"prompt-fuzzer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="evolve a population and write a report")
    run_parser.add_argument("--config", "-c", type=Path, help="YAML config path")
    run_parser.add_argument("--demo", action="store_true", help="run the offline mock demo")
    run_parser.add_argument("--output", "-o", type=Path, help="override output.directory")
    run_parser.add_argument(
        "--fail-on-no-finding",
        action="store_true",
        help="exit 1 when leakage never reaches the failure threshold",
    )
    run_parser.add_argument("--quiet", "-q", action="store_true", help="suppress generation logs")

    replay_parser = sub.add_parser("replay", help="send one saved prompt to the configured target")
    replay_parser.add_argument("--config", "-c", type=Path, required=True)
    replay_parser.add_argument("--case", type=Path, required=True, help="case JSON from a run")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        return _replay(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.demo and args.config:
        raise ConfigError("pass either --demo or --config, not both")
    if args.demo:
        path = packaged_demo_path()
    elif args.config:
        path = args.config
    else:
        raise ConfigError("pass --demo or --config")
    config = load_config(path)
    if args.output is not None:
        config = _with_output(config, args.output)
    if not args.quiet:
        print(
            f"prompt-fuzzer {__version__}  seed={config.seed}  "
            f"target={config.target.get('type')}  judge={config.judge.get('type')}"
        )

    def on_generation(stats: GenerationStats) -> None:
        if args.quiet:
            return
        print(
            f"gen {stats.generation:02d}  best={stats.best:.3f}  mean={stats.mean:.3f}  "
            f"worst={stats.worst:.3f}  diversity={stats.diversity:.3f}  "
            f"findings={stats.failures}"
        )

    result = run(config, on_generation=on_generation)
    best = 0.0 if result.best is None or result.best.fitness is None else result.best.fitness
    report = Path(config.output_directory) / "report.html"
    if not args.quiet:
        print(
            f"best={best:.3f}  findings={len(result.findings)}  "
            f"calls={result.evaluations}  report={report}"
        )
    if args.fail_on_no_finding and not result.findings:
        print("no finding reached the leakage threshold", file=sys.stderr)
        return 1
    return 0


def _replay(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        payload = json.loads(args.case.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read case {args.case}: {exc}") from exc
    prompt = payload.get("prompt", payload.get("text"))
    if not isinstance(prompt, str) or not prompt:
        raise ConfigError("case JSON needs a prompt or text string")
    target = build_target(config.target)
    response = target.generate(prompt)
    if response.error:
        print(response.error, file=sys.stderr)
        return 1
    print(response.text)
    return 0


def _with_output(config: AppConfig, output: Path) -> AppConfig:
    config.output_directory = str(output)
    return config
