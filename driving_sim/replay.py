"""Deterministic episode replay from a saved scenario spec + per-tick ego actions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import SimConfig, load_sim_config
from .environment import DrivingEnv

COMPARE_KEYS = ("collision", "success", "off_road", "stalled", "steps", "distance", "reward", "mean_speed")


def save_record(record: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def load_record(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def replay_episode(record: dict, cfg: Optional[SimConfig] = None) -> dict:
    """Re-simulate the recorded episode from scratch and return its metrics."""
    cfg = cfg or load_sim_config()
    cfg.num_actions = int(record["num_actions"])
    env = DrivingEnv(cfg)
    env.reset(options={"scenario": record["spec"]})
    for a in record["tick_actions"]:
        res = env.tick(int(a))
        if res is not None and (res[2] or res[3]):
            break
    return env.metrics.to_dict()


def verify_replay(record: dict, cfg: Optional[SimConfig] = None, tol: float = 1e-6) -> bool:
    got = replay_episode(record, cfg)
    ref = record["outcome"]
    return all(abs(float(got[k]) - float(ref[k])) <= tol for k in COMPARE_KEYS)
