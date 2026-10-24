"""Upper-confidence-bound credit assignment for mutation operators.

Rewards are fitness deltas versus the fitter parent. Operators that have not
been scored yet are preferred, then a mix of UCB1 and uniform exploration keeps
every operator in the pool from starving.
"""

from __future__ import annotations

import math
import random


class OperatorBandit:
    def __init__(self, names: list[str], exploration: float = 1.2, epsilon: float = 0.25) -> None:
        if not names:
            raise ValueError("bandit requires at least one operator")
        self.names = list(names)
        self.exploration = exploration
        self.epsilon = epsilon
        self.updates = {name: 0 for name in names}
        self.reward_sum = {name: 0.0 for name in names}

    def choose(self, rng: random.Random) -> str:
        untried = [name for name in self.names if self.updates[name] == 0]
        if untried:
            return rng.choice(untried)
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return rng.choice(self.names)
        return max(self.names, key=self._ucb)

    def update(self, name: str, reward: float) -> None:
        if name not in self.updates:
            return
        clipped = max(-1.0, min(1.0, reward))
        self.updates[name] += 1
        self.reward_sum[name] += clipped

    def mean(self, name: str) -> float:
        count = self.updates[name]
        if count == 0:
            return 0.0
        return self.reward_sum[name] / count

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            name: {"pulls": float(self.updates[name]), "mean_reward": self.mean(name)}
            for name in self.names
        }

    def _ucb(self, name: str) -> float:
        total = sum(self.updates.values())
        count = self.updates[name]
        bonus = self.exploration * math.sqrt(math.log(total + 1) / count)
        return self.mean(name) + bonus
