"""Configurable, decomposed reward function.

Per-decision reward = sum of named components (all logged separately).

Anti-exploit design notes:
  * Progress is proportional to speed, capped at `target_speed`: speeding earns nothing extra,
    crawling earns little, standing still earns nothing and the stall detector ends the episode with a penalty.
  * Speed above the limit is penalised, so "drive recklessly fast" is not profitable.
  * A collision/off-road terminal penalty is much larger than any attainable per-episode gain.
  * Lane changes cost a fixed price, and a second one inside a cooldown window costs extra.
  * Gains only come from being alive *and moving*; the completion bonus requires reaching the goal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .config import RewardCfg

REWARD_COMPONENTS = [
    "progress", "speed_limit", "headway", "ttc", "brake", "jerk",
    "lane_change", "repeat_lane_change", "invalid_action",
    "collision", "off_road", "stalled", "completion",
]


@dataclass
class StepSignals:
    speed: float
    accel: float
    prev_accel: float
    peak_decel: float                 # largest braking deceleration during the interval (>= 0)
    decision_dt: float
    max_jerk: float
    headway: Optional[float] = None   # time headway to the lead vehicle [s]; None if none/ego near standstill
    ttc: Optional[float] = None       # time-to-collision [s]; None if not closing
    lane_change_started: bool = False
    steps_since_lane_change: int = 10**6
    invalid_action: bool = False
    collided: bool = False
    off_road: bool = False
    stalled: bool = False
    goal: bool = False


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_reward(sig: StepSignals, cfg: RewardCfg) -> tuple[float, dict]:
    c = {k: 0.0 for k in REWARD_COMPONENTS}
    c["progress"] = cfg.w_progress * min(max(sig.speed, 0.0), cfg.target_speed) / cfg.target_speed
    c["speed_limit"] = -cfg.w_speed_limit * max(0.0, sig.speed - cfg.speed_limit)
    if sig.headway is not None:
        c["headway"] = -cfg.w_headway * _clip01((cfg.headway_safe - sig.headway) / cfg.headway_safe) ** 2
    if sig.ttc is not None:
        c["ttc"] = -cfg.w_ttc * _clip01((cfg.ttc_safe - sig.ttc) / cfg.ttc_safe) ** 2
    c["brake"] = -cfg.w_brake * _clip01((sig.peak_decel - cfg.comfort_decel) / 3.0)
    c["jerk"] = -cfg.w_jerk * _clip01(abs(sig.accel - sig.prev_accel) / (sig.max_jerk * sig.decision_dt))
    if sig.lane_change_started:
        c["lane_change"] = -cfg.w_lane_change
        if sig.steps_since_lane_change < cfg.lane_change_cooldown:
            c["repeat_lane_change"] = -cfg.w_repeat_lane_change
    if sig.invalid_action:
        c["invalid_action"] = -cfg.w_invalid
    if sig.collided:
        c["collision"] = cfg.collision
    if sig.off_road:
        c["off_road"] = cfg.off_road
    if sig.stalled:
        c["stalled"] = cfg.stalled
    if sig.goal and not (sig.collided or sig.off_road):
        c["completion"] = cfg.completion
    return float(sum(c.values())), c


def time_headway(gap: float, speed: float, min_speed: float = 1.0) -> Optional[float]:
    if speed < min_speed or math.isinf(gap):
        return None
    return gap / speed


def time_to_collision(gap: float, ego_speed: float, lead_speed: float) -> Optional[float]:
    closing = ego_speed - lead_speed
    if math.isinf(gap) or closing <= 1e-3:
        return None
    return gap / closing
