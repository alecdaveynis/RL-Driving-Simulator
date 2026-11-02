"""Uniform-random driver (lower-bound baseline)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Policy


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, num_actions: int, seed: int = 0):
        self.num_actions = num_actions
        self._base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        # Re-seeding per episode (with the scenario seed) makes the random policy
        # independent of evaluation order, so paired comparisons are reproducible.
        self.rng = np.random.default_rng([self._base_seed, 0 if seed is None else int(seed)])

    def act(self, obs, mask=None) -> int:
        if mask is None:
            return int(self.rng.integers(self.num_actions))
        valid = np.flatnonzero(mask)
        return int(valid[self.rng.integers(len(valid))])
