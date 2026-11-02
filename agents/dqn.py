"""Double DQN in PyTorch: online/target MLPs, masked greedy selection, checkpointing.

Checkpoints record obs_dim, num_actions, observation version and architecture;
`load_checkpoint` refuses to load anything incompatible with the target environment.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from driving_sim.observations import OBS_VERSION

from .base import Policy
from .replay_buffer import ReplayBuffer

CHECKPOINT_FORMAT = 1
NEG_INF = -1e9


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint does not match the environment it is loaded for."""


@dataclass
class DQNConfig:
    hidden: Sequence[int] = (128, 128)
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 200_000
    learning_starts: int = 2_000
    train_freq: int = 1                 # env steps per gradient update
    target_sync_interval: int = 1_000   # gradient updates between hard target syncs
    grad_clip: float = 10.0
    huber_delta: float = 1.0
    double_dqn: bool = True

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DQNConfig":
        d = dict(d)
        if "hidden" in d:
            d["hidden"] = tuple(d["hidden"])
        return cls(**d)


@dataclass
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 60_000
    kind: str = "linear"      # linear | exponential

    def value(self, step: int) -> float:
        if self.kind == "linear":
            frac = min(1.0, max(0.0, step / max(1, self.decay_steps)))
            return self.start + (self.end - self.start) * frac
        if self.kind == "exponential":
            # reaches ~end at decay_steps: eps = end + (start-end) * exp(-5 * step / decay_steps)
            return self.end + (self.start - self.end) * float(np.exp(-5.0 * step / max(1, self.decay_steps)))
        raise ValueError(f"Unknown epsilon schedule '{self.kind}'")


def get_device(name: str = "auto") -> torch.device:
    """'auto' picks CPU: for this small MLP + batch 64, CPU beats MPS launch overhead. Use 'mps' to force it."""
    if name == "auto":
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available on this machine")
    if name == "cuda":
        raise RuntimeError("CUDA is not supported by this project (Apple Silicon target); use 'cpu' or 'mps'")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, num_actions: int, cfg: Optional[DQNConfig] = None,
                 device: str | torch.device = "cpu", seed: int = 0):
        self.cfg = cfg or DQNConfig()
        self.obs_dim, self.num_actions = obs_dim, num_actions
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.seed = seed
        seed_everything(seed)
        self.rng = np.random.default_rng(seed)   # exploration + replay sampling
        h = tuple(self.cfg.hidden)
        self.online = QNetwork(obs_dim, num_actions, h).to(self.device)
        self.target = QNetwork(obs_dim, num_actions, h).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.update_count = 0

    # ------------------------------------------------------------ acting ---
    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(x).squeeze(0).cpu().numpy()

    def select_action(self, obs: np.ndarray, mask: Optional[np.ndarray] = None, epsilon: float = 0.0,
                      rng: Optional[np.random.Generator] = None) -> int:
        """Epsilon-greedy over *valid* actions only (mask=True means allowed)."""
        rng = rng or self.rng
        if mask is None:
            mask = np.ones(self.num_actions, dtype=bool)
        if epsilon > 0.0 and rng.random() < epsilon:
            valid = np.flatnonzero(mask)
            return int(valid[rng.integers(len(valid))])
        q = self.q_values(obs)
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    # ---------------------------------------------------------- learning ---
    @torch.no_grad()
    def compute_targets(self, rewards: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor,
                        next_masks: torch.Tensor) -> torch.Tensor:
        """Double DQN: action chosen by the online net, evaluated by the target net.

        y = r + gamma * (1 - terminated) * Q_target(s', argmax_a' Q_online(s', a'))
        Invalid next actions are excluded from the argmax (and from the max, in vanilla mode).
        """
        q_target_next = self.target(next_obs)
        if self.cfg.double_dqn:
            q_online_next = self.online(next_obs).masked_fill(~next_masks, NEG_INF)
            best = q_online_next.argmax(dim=1, keepdim=True)
            next_q = q_target_next.gather(1, best).squeeze(1)
        else:
            next_q = q_target_next.masked_fill(~next_masks, NEG_INF).max(dim=1).values
        return rewards + self.cfg.gamma * (1.0 - terminated) * next_q

    def learn(self, batch: dict) -> dict:
        dev = self.device
        obs = torch.as_tensor(batch["obs"], device=dev)
        actions = torch.as_tensor(batch["actions"], device=dev).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], device=dev)
        next_obs = torch.as_tensor(batch["next_obs"], device=dev)
        terminated = torch.as_tensor(batch["terminated"], device=dev)
        next_masks = torch.as_tensor(batch["next_masks"], device=dev)
        targets = self.compute_targets(rewards, next_obs, terminated, next_masks)
        q = self.online(obs).gather(1, actions).squeeze(1)
        loss = F.huber_loss(q, targets, delta=self.cfg.huber_delta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.cfg.target_sync_interval == 0:
            self.sync_target()
        return {"loss": float(loss.item()), "q_mean": float(q.mean().item()), "grad_norm": float(gnorm)}

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------ checkpointing ---
    def save(self, path: str | Path, *, sim_config: Optional[dict] = None, sim_fingerprint: str = "",
             train_state: Optional[dict] = None, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CHECKPOINT_FORMAT,
            "obs_dim": self.obs_dim, "num_actions": self.num_actions, "obs_version": OBS_VERSION,
            "hidden": list(self.cfg.hidden), "dqn_config": self.cfg.to_dict(),
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "update_count": self.update_count,
            "sim_config": sim_config, "sim_fingerprint": sim_fingerprint,
            "train_state": train_state or {}, "extra": extra or {},
            "rng_state": {"agent_rng": self.rng.bit_generator.state, "torch": torch.get_rng_state(),
                          "python": random.getstate(), "numpy_global": np.random.get_state()},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def restore_rng(self, state: dict) -> None:
        self.rng.bit_generator.state = state["agent_rng"]
        torch.set_rng_state(state["torch"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy_global"])


def read_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt / not a checkpoint
        raise IncompatibleCheckpointError(f"Could not read checkpoint {path}: {exc}") from exc


def check_compatibility(ckpt: dict, obs_dim: int, num_actions: int, path: str = "") -> None:
    """Raise unless the checkpoint matches the environment's obs/action spaces."""
    where = f" ({path})" if path else ""
    if ckpt.get("format_version") != CHECKPOINT_FORMAT:
        raise IncompatibleCheckpointError(f"Unsupported checkpoint format{where}: {ckpt.get('format_version')}")
    if ckpt["num_actions"] != num_actions:
        raise IncompatibleCheckpointError(
            f"Checkpoint{where} was trained with {ckpt['num_actions']} actions but the environment has {num_actions}.")
    if ckpt["obs_dim"] != obs_dim:
        raise IncompatibleCheckpointError(
            f"Checkpoint{where} expects observation size {ckpt['obs_dim']} but the environment provides {obs_dim}.")
    if ckpt.get("obs_version") != OBS_VERSION:
        raise IncompatibleCheckpointError(
            f"Checkpoint{where} uses observation layout v{ckpt.get('obs_version')}, this code is v{OBS_VERSION}.")


def load_checkpoint(path: str | Path, obs_dim: int, num_actions: int, device: str | torch.device = "cpu",
                    load_optimizer: bool = True) -> tuple[DQNAgent, dict]:
    """Load an agent, verifying compatibility with (obs_dim, num_actions). Returns (agent, checkpoint dict)."""
    ckpt = read_checkpoint(path)
    check_compatibility(ckpt, obs_dim, num_actions, str(path))
    agent = DQNAgent(obs_dim, num_actions, DQNConfig.from_dict(ckpt["dqn_config"]), device=device)
    agent.online.load_state_dict(ckpt["online"])
    agent.target.load_state_dict(ckpt["target"])
    if load_optimizer:
        agent.optimizer.load_state_dict(ckpt["optimizer"])
    agent.update_count = int(ckpt.get("update_count", 0))
    return agent, ckpt


def checkpoint_info(path: str | Path) -> dict:
    """Light-weight metadata for UI/CLI (does not build a network)."""
    c = read_checkpoint(path)
    ts = c.get("train_state", {})
    return {"num_actions": c["num_actions"], "obs_dim": c["obs_dim"], "hidden": c["hidden"],
            "env_steps": ts.get("env_steps"), "epsilon": ts.get("epsilon"), "stage": ts.get("stage_name"),
            "extra": c.get("extra", {}), "sim_fingerprint": c.get("sim_fingerprint", "")}


class DQNPolicy(Policy):
    """Greedy (optionally epsilon-greedy) policy wrapper for evaluation and playback."""

    name = "dqn"

    def __init__(self, agent: DQNAgent, epsilon: float = 0.0, seed: int = 0, label: str = "dqn"):
        self.agent, self.epsilon = agent, epsilon
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.name = label
        self.last_q: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng([self._seed, 0 if seed is None else int(seed)])

    def q_values(self, obs):
        return self.agent.q_values(obs)

    def act(self, obs, mask=None) -> int:
        q = self.agent.q_values(obs)
        self.last_q = q
        if mask is None:
            mask = np.ones(self.agent.num_actions, dtype=bool)
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            valid = np.flatnonzero(mask)
            return int(valid[self.rng.integers(len(valid))])
        return int(np.argmax(np.where(mask, q, -np.inf)))
