"""Common policy interface shared by random, rule-based and DQN drivers."""
from __future__ import annotations

from typing import Optional

import numpy as np


class Policy:
    """A policy maps (observation, action mask) -> discrete action. `reset` is called per episode."""

    name = "policy"

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def q_values(self, obs: np.ndarray) -> Optional[np.ndarray]:
        return None
