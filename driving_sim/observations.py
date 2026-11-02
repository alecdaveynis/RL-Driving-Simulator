"""Sensor model and observation vector.

Information barrier: `perceive(world)` is the *only* place the agent's inputs
read the world, and it extracts nothing but kinematic quantities that a
radar/camera plus ego odometry could provide (positions, speeds, sizes).
NPC controllers, scripted events, vehicle ids and future behaviour are never
reachable from a `Percepts` object. `build_observation` is a pure function of
`Percepts`.

Observation layout (all features normalised, see `OBS_NAMES`):
  ego block (8)       speed, accel, lane_pos, lane_changing, left_exists, right_exists,
                      can_change_left, can_change_right
  neighbour blocks    for slot in (left, same, right) x (front, rear): present, gap, rel_speed
  ttc block (4)       time-to-collision for same-front, left-front, right-front, same-rear
  lateral block (2)   lateral speed *towards the ego lane* of the nearest front vehicle in the left /
                      right lane (radar-measurable; early cut-in cue). 0 when no vehicle is present.
Missing vehicles are encoded as present=0, gap=1.0 ("beyond sensor range"),
rel_speed=0, with the explicit presence flag disambiguating them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from gymnasium import spaces

from .config import SimConfig

OBS_VERSION = 2
LAT_SPEED_SCALE = 2.0   # m/s; a 3 s lane change peaks at ~1.85 m/s
SLOTS = ("left", "same", "right")
SIDES = ("front", "rear")

OBS_NAMES: list[str] = ["ego_speed", "ego_accel", "lane_pos", "lane_changing", "left_exists", "right_exists",
                        "can_change_left", "can_change_right"]
for _slot in SLOTS:
    for _side in SIDES:
        OBS_NAMES += [f"{_slot}_{_side}_present", f"{_slot}_{_side}_gap", f"{_slot}_{_side}_rel_speed"]
OBS_NAMES += ["ttc_same_front", "ttc_left_front", "ttc_right_front", "ttc_same_rear"]
OBS_NAMES += ["left_front_lat_toward", "right_front_lat_toward"]
OBS_DIM = len(OBS_NAMES)
_IDX = {n: i for i, n in enumerate(OBS_NAMES)}


@dataclass(frozen=True)
class OtherPercept:
    dx: float        # other.x - ego.x (centre to centre) [m]
    y: float         # lateral position [m]
    v: float         # speed [m/s]
    vy: float        # lateral speed [m/s] (positive = towards higher lane index / road right)
    length: float
    width: float


@dataclass(frozen=True)
class Percepts:
    ego_speed: float
    ego_accel: float
    ego_y: float
    ego_length: float
    ego_width: float
    ego_lane: int
    num_lanes: int
    lane_width: float
    lane_changing: bool
    can_change_left: bool
    can_change_right: bool
    max_brake: float
    max_speed: float
    others: tuple = field(default_factory=tuple)


def perceive(world) -> Percepts:
    """Extract what the ego sensors can see: kinematics of vehicles within range."""
    ego, road, rng = world.ego, world.road, world.cfg.sensors.range
    others = tuple(
        OtherPercept(o.x - ego.x, o.y, o.v, o.vy, o.length, o.width)
        for o in world.vehicles
        if o is not ego and abs(o.x - ego.x) - (o.length + ego.length) / 2 <= rng
    )
    return Percepts(
        ego_speed=ego.v, ego_accel=ego.a, ego_y=ego.y, ego_length=ego.length, ego_width=ego.width,
        ego_lane=road.lane_of(ego.y), num_lanes=road.num_lanes, lane_width=road.lane_width,
        lane_changing=ego.lc_active,
        can_change_left=world.can_change_lane(ego, -1), can_change_right=world.can_change_lane(ego, +1),
        max_brake=ego.max_brake, max_speed=world.cfg.physics.max_speed, others=others,
    )


def _slot_neighbours(p: Percepts, lane: int, min_overlap: float):
    """Nearest front/rear vehicle occupying `lane`: ((gap, rel_speed) | None, (gap, rel_speed) | None)."""
    lo, hi = lane * p.lane_width, (lane + 1) * p.lane_width
    front: Optional[tuple[float, float, float]] = None   # (gap, rel_speed, lateral_speed)
    rear: Optional[tuple[float, float, float]] = None
    for o in p.others:
        overlap = min(o.y + o.width / 2, hi) - max(o.y - o.width / 2, lo)
        if overlap <= min_overlap:
            continue
        half = (o.length + p.ego_length) / 2
        rel_v = o.v - p.ego_speed
        if o.dx >= 0:
            gap = max(0.0, o.dx - half)
            if front is None or gap < front[0]:
                front = (gap, rel_v, o.vy)
        else:
            gap = max(0.0, -o.dx - half)
            if rear is None or gap < rear[0]:
                rear = (gap, rel_v, o.vy)
    return front, rear


def _ttc(gap: float, closing: float, ttc_max: float) -> float:
    if closing <= 1e-3:
        return ttc_max
    return min(ttc_max, gap / closing)


def build_observation(p: Percepts, cfg: SimConfig) -> np.ndarray:
    s = cfg.sensors
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    n = p.num_lanes
    obs[_IDX["ego_speed"]] = np.clip(p.ego_speed / p.max_speed, 0.0, 1.0)
    obs[_IDX["ego_accel"]] = np.clip(p.ego_accel / p.max_brake, -1.0, 1.0)
    obs[_IDX["lane_pos"]] = p.ego_lane / (n - 1) if n > 1 else 0.0
    obs[_IDX["lane_changing"]] = 1.0 if p.lane_changing else 0.0
    obs[_IDX["left_exists"]] = 1.0 if p.ego_lane - 1 >= 0 else 0.0
    obs[_IDX["right_exists"]] = 1.0 if p.ego_lane + 1 < n else 0.0
    obs[_IDX["can_change_left"]] = 1.0 if p.can_change_left else 0.0
    obs[_IDX["can_change_right"]] = 1.0 if p.can_change_right else 0.0

    ttcs = {}
    lat = {"left": 0.0, "right": 0.0}
    for slot, off in zip(SLOTS, (-1, 0, 1)):
        lane = p.ego_lane + off
        exists = 0 <= lane < n
        front = rear = None
        if exists:
            front, rear = _slot_neighbours(p, lane, s.lane_overlap_min)
        for side, nb in (("front", front), ("rear", rear)):
            base = f"{slot}_{side}_"
            if nb is None:
                obs[_IDX[base + "present"]] = 0.0
                obs[_IDX[base + "gap"]] = 1.0
                obs[_IDX[base + "rel_speed"]] = 0.0
                ttcs[(slot, side)] = s.ttc_max
            else:
                gap, rel_v, vy = nb
                if side == "front" and slot != "same":
                    toward = vy if slot == "left" else -vy   # left lane has lower y: moving +y is towards ego
                    lat[slot] = float(np.clip(toward / LAT_SPEED_SCALE, -1.0, 1.0))
                obs[_IDX[base + "present"]] = 1.0
                obs[_IDX[base + "gap"]] = np.clip(gap / s.range, 0.0, 1.0)
                obs[_IDX[base + "rel_speed"]] = np.clip(rel_v / s.rel_speed_scale, -1.0, 1.0)
                closing = -rel_v if side == "front" else rel_v
                ttcs[(slot, side)] = _ttc(gap, closing, s.ttc_max)
    obs[_IDX["ttc_same_front"]] = ttcs[("same", "front")] / s.ttc_max
    obs[_IDX["ttc_left_front"]] = ttcs[("left", "front")] / s.ttc_max
    obs[_IDX["ttc_right_front"]] = ttcs[("right", "front")] / s.ttc_max
    obs[_IDX["ttc_same_rear"]] = ttcs[("same", "rear")] / s.ttc_max
    obs[_IDX["left_front_lat_toward"]] = lat["left"]
    obs[_IDX["right_front_lat_toward"]] = lat["right"]
    return obs


def observation_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)


def decode_observation(obs: np.ndarray, cfg: SimConfig) -> dict:
    """Invert the normalisation into physical units (used by the rule-based policy).

    Returns None for gap/speed entries whose presence flag is 0, never a fake zero.
    """
    s = cfg.sensors
    d = {
        "speed": float(obs[_IDX["ego_speed"]]) * cfg.physics.max_speed,
        "accel": float(obs[_IDX["ego_accel"]]) * cfg.ego.max_brake,
        "lane_changing": bool(obs[_IDX["lane_changing"]] > 0.5),
        "can_left": bool(obs[_IDX["can_change_left"]] > 0.5),
        "can_right": bool(obs[_IDX["can_change_right"]] > 0.5),
        "left_exists": bool(obs[_IDX["left_exists"]] > 0.5),
        "right_exists": bool(obs[_IDX["right_exists"]] > 0.5),
    }
    for slot in SLOTS:
        for side in SIDES:
            base = f"{slot}_{side}_"
            if obs[_IDX[base + "present"]] > 0.5:
                d[base[:-1]] = (float(obs[_IDX[base + "gap"]]) * s.range,
                                float(obs[_IDX[base + "rel_speed"]]) * s.rel_speed_scale)
            else:
                d[base[:-1]] = None  # (gap, rel_speed) or None
    d["ttc_same_front"] = float(obs[_IDX["ttc_same_front"]]) * s.ttc_max
    d["left_front_lat_toward"] = float(obs[_IDX["left_front_lat_toward"]]) * LAT_SPEED_SCALE
    d["right_front_lat_toward"] = float(obs[_IDX["right_front_lat_toward"]]) * LAT_SPEED_SCALE
    return d
