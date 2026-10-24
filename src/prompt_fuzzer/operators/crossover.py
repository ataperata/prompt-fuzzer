"""Combine two prompts by splicing words or lines."""

from __future__ import annotations

import random


def join_prefix_suffix(left: list[str], right: list[str], cut_left: int, cut_right: int) -> str:
    return " ".join(left[:cut_left] + right[cut_right:]).strip()


def interleave_words(left: str, right: str, rng: random.Random) -> str:
    left_words = left.split()
    right_words = right.split()
    if not left_words:
        return right
    if not right_words:
        return left
    merged: list[str] = []
    i = 0
    j = 0
    while i < len(left_words) or j < len(right_words):
        take_left = i < len(left_words) and (j >= len(right_words) or rng.random() < 0.5)
        if take_left:
            merged.append(left_words[i])
            i += 1
        else:
            merged.append(right_words[j])
            j += 1
    return " ".join(merged)


def splice_lines(left: str, right: str, rng: random.Random) -> str:
    left_lines = left.splitlines() or [left]
    right_lines = right.splitlines() or [right]
    if len(left_lines) == 1 and len(right_lines) == 1:
        return interleave_words(left, right, rng)
    cut_left = rng.randint(1, len(left_lines)) if len(left_lines) > 1 else 1
    cut_right = rng.randint(0, max(0, len(right_lines) - 1))
    return "\n".join(left_lines[:cut_left] + right_lines[cut_right:]).strip()


class Crossover:
    name = "crossover"

    def cross(self, left: str, right: str, rng: random.Random) -> str:
        mode = rng.choice(("prefix_suffix", "interleave", "lines"))
        if mode == "interleave":
            return interleave_words(left, right, rng)
        if mode == "lines":
            return splice_lines(left, right, rng)
        left_words = left.split()
        right_words = right.split()
        if len(left_words) < 2 or len(right_words) < 2:
            return interleave_words(left, right, rng)
        cut_left = rng.randint(1, len(left_words) - 1)
        cut_right = rng.randint(1, len(right_words) - 1)
        child = join_prefix_suffix(left_words, right_words, cut_left, cut_right)
        return child or left
