"""Helpers to discover and load DQN checkpoints as ready-to-use policies."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from driving_sim.config import PROJECT_ROOT
from driving_sim.observations import OBS_DIM

from .dqn import (DQNAgent, DQNConfig, DQNPolicy, IncompatibleCheckpointError, check_compatibility, get_device,
                  load_checkpoint, read_checkpoint)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PREFERENCE = ("best.pt", "final.pt", "last.pt")


def list_checkpoints(root: Path = CHECKPOINT_DIR) -> list[Path]:
    """All loadable-looking checkpoints, best.pt files first, newest first within a kind."""
    found: list[Path] = []
    for name in PREFERENCE:
        found += sorted(root.glob(f"*/{name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def load_dqn_policy(path: str | Path, num_actions: Optional[int] = None, device: str = "cpu",
                    epsilon: float = 0.0, seed: int = 0) -> tuple[DQNPolicy, dict]:
    """Load a checkpoint as a greedy policy.

    If `num_actions` is given it must match the checkpoint, otherwise IncompatibleCheckpointError is
    raised. Returns (policy, checkpoint dict).
    """
    ck = read_checkpoint(path)
    n = ck["num_actions"] if num_actions is None else num_actions
    agent, ck = load_checkpoint(path, OBS_DIM, n, get_device(device), load_optimizer=False)
    agent.online.eval()
    label = f"dqn:{Path(path).parent.name}/{Path(path).name}"
    return DQNPolicy(agent, epsilon=epsilon, seed=seed, label=label), ck


def untrained_policy(num_actions: int, seed: int = 0) -> DQNPolicy:
    """A randomly-initialised network, for watching 'before training' behaviour. Clearly labelled."""
    agent = DQNAgent(OBS_DIM, num_actions, DQNConfig(), "cpu", seed)
    return DQNPolicy(agent, epsilon=0.0, seed=seed, label="dqn:UNTRAINED")
