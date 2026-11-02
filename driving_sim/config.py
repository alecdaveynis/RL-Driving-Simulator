"""Typed configuration for the simulator, loaded from YAML.

Every field has a default so the project runs without any config file, and a
YAML file only needs to list the values it overrides.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIM_CONFIG = PROJECT_ROOT / "configs" / "simulation.yaml"


@dataclass
class PhysicsCfg:
    dt: float = 0.05                    # fixed physics timestep [s]
    decision_substeps: int = 4          # physics ticks per agent decision (0.2 s)
    max_speed: float = 36.0             # hard speed bound [m/s]
    lane_change_duration: float = 3.0   # time a lane change takes [s]
    lane_change_min_speed: float = 3.0  # lane changes need at least this speed [m/s]
    crash_decel: float = 10.0           # wrecks brake with this deceleration [m/s^2]
    npc_max_jerk: float = 25.0
    npc_max_brake: float = 9.0


@dataclass
class RoadCfg:
    lane_width: float = 3.7


@dataclass
class EgoCfg:
    max_accel: float = 2.5      # "accelerate" action target acceleration
    max_brake: float = 6.0      # "brake" action target deceleration (positive number)
    coast_decel: float = 0.25   # drag when coasting
    max_jerk: float = 8.0       # |da/dt| bound [m/s^3]
    length: float = 4.7
    width: float = 1.9


@dataclass
class EpisodeCfg:
    max_steps: int = 400            # decision steps (80 s at 0.2 s)
    goal_distance: float = 900.0    # default goal (scenarios may override)
    stall_speed: float = 0.5        # below this speed ...
    stall_steps: int = 50           # ... for this many decisions (10 s) with a free road => stalled
    stall_free_gap: float = 30.0    # "free road" means no vehicle closer than this ahead


@dataclass
class SensorCfg:
    range: float = 100.0            # longitudinal sensing range [m]
    ttc_max: float = 10.0           # TTC clipped to this value [s]
    rel_speed_scale: float = 15.0   # relative speed normalisation [m/s]
    lane_overlap_min: float = 0.3   # lateral overlap [m] for a vehicle to occupy a lane


@dataclass
class RewardCfg:
    target_speed: float = 25.0
    speed_limit: float = 30.0
    w_progress: float = 1.0
    w_speed_limit: float = 0.2      # per m/s above the limit
    headway_safe: float = 1.5       # [s]
    w_headway: float = 1.0
    ttc_safe: float = 4.0           # [s]
    w_ttc: float = 1.0
    comfort_decel: float = 3.0      # braking harder than this is penalised [m/s^2]
    w_brake: float = 0.5
    w_jerk: float = 0.1
    w_lane_change: float = 0.4
    lane_change_cooldown: int = 25  # decisions; a second lane change inside this window costs extra
    w_repeat_lane_change: float = 2.0
    w_invalid: float = 0.2
    collision: float = -50.0
    off_road: float = -50.0
    stalled: float = -30.0
    completion: float = 25.0


@dataclass
class MetricsCfg:
    hard_brake_decel: float = 4.0       # m/s^2
    unsafe_headway: float = 1.0         # [s]
    unsafe_ttc: float = 2.5             # [s]
    unsafe_min_speed: float = 2.0       # [m/s] metrics ignore crawling


@dataclass
class ScenarioCfg:
    # (family, weight) pairs used by the default sampler
    families: dict = field(default_factory=lambda: {"mixed": 1.0})
    difficulty_range: tuple = (0.0, 1.0)


@dataclass
class RenderCfg:
    width: int = 1360
    height: int = 800
    panel_width: int = 330
    fps: int = 60
    pixels_per_meter: float = 13.0
    lateral_scale: float = 1.35      # widen lanes/cars on screen (readability; physics unaffected)
    dynamic_zoom: bool = True        # zoom in at low speed, out at high speed
    ego_screen_frac: float = 0.28


@dataclass
class SimConfig:
    num_actions: int = 5
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    road: RoadCfg = field(default_factory=RoadCfg)
    ego: EgoCfg = field(default_factory=EgoCfg)
    episode: EpisodeCfg = field(default_factory=EpisodeCfg)
    sensors: SensorCfg = field(default_factory=SensorCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)
    scenario: ScenarioCfg = field(default_factory=ScenarioCfg)
    render: RenderCfg = field(default_factory=RenderCfg)

    @property
    def decision_dt(self) -> float:
        return self.physics.dt * self.physics.decision_substeps

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self, sections: tuple[str, ...] = ("physics", "road", "ego", "episode", "sensors")) -> str:
        """Hash of the settings that define the environment dynamics/observations."""
        d = self.to_dict()
        payload = {k: d[k] for k in sections}
        payload["num_actions"] = self.num_actions
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _build(cls, data: dict | None):
    """Recursively build dataclass `cls` from a (partial) dict, rejecting unknown keys."""
    data = data or {}
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}; valid: {sorted(known)}")
        ftype = known[key].type
        sub = _resolve_dataclass(cls, key)
        if sub is not None and isinstance(value, dict):
            kwargs[key] = _build(sub, value)
        elif isinstance(getattr(cls(), key), tuple) and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve_dataclass(cls, key):
    default = cls().__dict__.get(key)
    if dataclasses.is_dataclass(default):
        return type(default)
    return None


def sim_config_from_dict(data: dict | None) -> SimConfig:
    return _build(SimConfig, data)


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_sim_config(path: str | Path | None = None, overrides: dict | None = None) -> SimConfig:
    """Load the simulation config (defaults < YAML file < overrides)."""
    data: dict = {}
    p = Path(path) if path else DEFAULT_SIM_CONFIG
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    elif path:
        raise FileNotFoundError(p)
    if overrides:
        data = deep_update(data, overrides)
    return sim_config_from_dict(data)
