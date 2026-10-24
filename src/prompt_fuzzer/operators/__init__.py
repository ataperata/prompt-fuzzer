"""Mutation and crossover operators."""

from prompt_fuzzer.operators.crossover import Crossover
from prompt_fuzzer.operators.encoding import Encoding
from prompt_fuzzer.operators.paraphrase import Paraphrase
from prompt_fuzzer.operators.roleplay import Roleplay
from prompt_fuzzer.operators.smuggle import Smuggle

MUTATORS = {
    Paraphrase.name: Paraphrase,
    Roleplay.name: Roleplay,
    Encoding.name: Encoding,
    Smuggle.name: Smuggle,
}

OPERATOR_NAMES = frozenset({*MUTATORS, Crossover.name})


def build_mutators(names: list[str]) -> dict[str, object]:
    built = {}
    for name in names:
        if name == Crossover.name:
            continue
        built[name] = MUTATORS[name]()
    return built


__all__ = [
    "OPERATOR_NAMES",
    "Crossover",
    "Encoding",
    "Paraphrase",
    "Roleplay",
    "Smuggle",
    "build_mutators",
]
