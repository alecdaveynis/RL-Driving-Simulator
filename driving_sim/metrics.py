"""Per-episode driving metrics, computed from physical state (never from reward)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .rewards import REWARD_COMPONENTS


@dataclass
class EpisodeMetrics:
    family: str = ""
    scenario_seed: int = 0
    difficulty: float = 0.0
    steps: int = 0
    duration: float = 0.0
    distance: float = 0.0
    speed_sum: float = 0.0
    reward: float = 0.0
    collision: bool = False
    rear_ended: bool = False
    success: bool = False
    off_road: bool = False
    stalled: bool = False
    truncated: bool = False
    unsafe_following_steps: int = 0
    unsafe_following_events: int = 0
    hard_brake_steps: int = 0
    hard_brake_events: int = 0
    lane_changes: int = 0
    min_gap: float = math.inf
    min_ttc: float = math.inf
    components: dict = field(default_factory=lambda: {k: 0.0 for k in REWARD_COMPONENTS})
    _unsafe_prev: bool = False
    _brake_prev: bool = False

    def update(self, *, speed: float, reward: float, comps: dict, unsafe: bool, hard_brake: bool,
               lane_change: bool, gap: float, ttc: float, dt: float) -> None:
        self.steps += 1
        self.duration += dt
        self.speed_sum += speed
        self.reward += reward
        for k, v in comps.items():
            self.components[k] += v
        if unsafe:
            self.unsafe_following_steps += 1
            if not self._unsafe_prev:
                self.unsafe_following_events += 1
        self._unsafe_prev = unsafe
        if hard_brake:
            self.hard_brake_steps += 1
            if not self._brake_prev:
                self.hard_brake_events += 1
        self._brake_prev = hard_brake
        self.lane_changes += int(lane_change)
        self.min_gap = min(self.min_gap, gap)
        self.min_ttc = min(self.min_ttc, ttc)

    @property
    def mean_speed(self) -> float:
        return self.speed_sum / self.steps if self.steps else 0.0

    def to_dict(self) -> dict:
        d = {
            "family": self.family, "scenario_seed": self.scenario_seed, "difficulty": round(self.difficulty, 4),
            "steps": self.steps, "duration": round(self.duration, 3), "distance": round(self.distance, 3),
            "mean_speed": round(self.mean_speed, 4), "reward": round(self.reward, 4),
            "collision": int(self.collision), "rear_ended": int(self.rear_ended), "success": int(self.success),
            "off_road": int(self.off_road), "stalled": int(self.stalled), "truncated": int(self.truncated),
            "unsafe_following_steps": self.unsafe_following_steps,
            "unsafe_following_events": self.unsafe_following_events,
            "hard_brake_steps": self.hard_brake_steps, "hard_brake_events": self.hard_brake_events,
            "lane_changes": self.lane_changes,
            "min_gap": round(self.min_gap, 3) if math.isfinite(self.min_gap) else None,
            "min_ttc": round(self.min_ttc, 3) if math.isfinite(self.min_ttc) else None,
        }
        d.update({f"r_{k}": round(v, 4) for k, v in self.components.items()})
        return d
