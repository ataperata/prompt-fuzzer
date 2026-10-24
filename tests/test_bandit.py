"""Operator bandit prefers rewarded arms after each arm has been scored."""

from __future__ import annotations

import random

from prompt_fuzzer.bandit import OperatorBandit


def test_untried_operators_are_all_eligible() -> None:
    bandit = OperatorBandit(["paraphrase", "encoding"])
    seen = {bandit.choose(random.Random(seed)) for seed in range(40)}
    assert seen == {"paraphrase", "encoding"}


def test_bandit_exploits_the_better_reward() -> None:
    bandit = OperatorBandit(["paraphrase", "encoding"], exploration=0.0, epsilon=0.0)
    for _ in range(4):
        bandit.update("paraphrase", 0.8)
        bandit.update("encoding", -0.4)
    choices = [bandit.choose(random.Random(seed)) for seed in range(15)]
    assert choices.count("paraphrase") > choices.count("encoding")
    assert bandit.mean("paraphrase") > bandit.mean("encoding")
