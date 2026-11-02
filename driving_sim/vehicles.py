"""Vehicles, discrete actions and NPC behaviour controllers.

Coordinates: x is metres along the road (forward), y is metres across the road
measured from the *left* road edge (lane 0 is the left-most lane, so "change
left" means lane - 1). Controllers only read vehicle kinematics through the
world API; they never touch the renderer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class Action(IntEnum):
    ACCELERATE = 0
    COAST = 1
    BRAKE = 2
    LANE_LEFT = 3
    LANE_RIGHT = 4


ACTION_NAMES = ["Accelerate", "Coast", "Brake", "Lane left", "Lane right"]

# action -> (longitudinal level in {+1, 0, -1}, lane-change direction in {-1, 0, +1})
ACTION_TABLE = {
    Action.ACCELERATE: (+1, 0),
    Action.COAST: (0, 0),
    Action.BRAKE: (-1, 0),
    Action.LANE_LEFT: (0, -1),
    Action.LANE_RIGHT: (0, +1),
}

NPC_COLORS = [
    (231, 76, 60),    # red
    (241, 196, 15),   # yellow
    (46, 204, 113),   # green
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (236, 240, 241),  # white
    (26, 188, 156),   # teal
    (127, 140, 141),  # grey
]
EGO_COLOR_INDEX = -1


@dataclass
class Vehicle:
    vid: int
    x: float
    y: float
    v: float
    length: float = 4.5
    width: float = 1.85
    is_ego: bool = False
    a: float = 0.0
    max_accel: float = 2.5
    max_brake: float = 6.0
    max_jerk: float = 8.0
    color: int = 0
    controller: Optional["Controller"] = None
    crashed: bool = False
    # lane-change state
    lc_active: bool = False
    lc_dir: int = 0
    lc_y0: float = 0.0
    lc_y1: float = 0.0
    lc_t: float = 0.0
    lc_dur: float = 3.0
    vy: float = 0.0
    # bookkeeping used by the swept collision test
    prev_x: float = 0.0
    prev_y: float = 0.0
    label: str = ""

    @property
    def heading(self) -> float:
        return math.atan2(self.vy, max(self.v, 1.0))

    @property
    def front(self) -> float:
        return self.x + self.length / 2

    @property
    def rear(self) -> float:
        return self.x - self.length / 2


# --------------------------------------------------------------------------- #
# Intelligent Driver Model                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class IDMParams:
    v0: float = 22.0     # desired speed
    T: float = 1.3       # desired time headway
    a: float = 1.8       # max acceleration
    b: float = 2.5       # comfortable deceleration
    s0: float = 2.5      # minimum gap

    @classmethod
    def from_dict(cls, d: dict) -> "IDMParams":
        return cls(**{k: float(d[k]) for k in ("v0", "T", "a", "b", "s0") if k in d})


def idm_accel(v: float, p: IDMParams, gap: Optional[float], v_lead: float = 0.0, v0: Optional[float] = None) -> float:
    v0 = p.v0 if v0 is None else v0
    free = 1.0 - (max(v, 0.0) / max(v0, 0.1)) ** 4
    if gap is None:
        return p.a * free
    dv = v - v_lead
    s_star = p.s0 + max(0.0, v * p.T + v * dv / (2.0 * math.sqrt(p.a * p.b)))
    return p.a * (free - (s_star / max(gap, 0.1)) ** 2)


# --------------------------------------------------------------------------- #
# Controllers                                                                 #
# --------------------------------------------------------------------------- #
class Controller:
    """NPC policy. `control` returns (commanded accel, lane-change direction)."""

    kind = "base"

    def control(self, veh: Vehicle, world) -> tuple[float, int]:  # pragma: no cover - interface
        raise NotImplementedError

    def idm_command(self, veh: Vehicle, world, params: IDMParams, v0: Optional[float] = None) -> float:
        gap, lead = world.front_in_path(veh)
        if lead is None:
            return idm_accel(veh.v, params, None, v0=v0)
        return idm_accel(veh.v, params, gap, lead.v, v0=v0)


class ConstantSpeedController(Controller):
    """Holds a fixed speed and ignores other traffic (stage-1 lead vehicle)."""

    kind = "constant"

    def __init__(self, v0: float):
        self.v0 = v0

    def control(self, veh, world):
        return float(np.clip(1.0 * (self.v0 - veh.v), -1.5, 1.5)), 0


class IDMController(Controller):
    kind = "idm"

    def __init__(self, params: IDMParams):
        self.p = params

    def control(self, veh, world):
        return self.idm_command(veh, world, self.p), 0


class BrakeEventController(IDMController):
    """IDM driver with scripted sudden-braking events.

    events: list of dicts {t, decel, v_target, hold} (t relative to `time_base`).
    The schedule is hidden from the agent: it only ever sees the resulting kinematics.
    """

    kind = "braker"

    def __init__(self, params: IDMParams, events: list[dict]):
        super().__init__(params)
        self.events = sorted(events, key=lambda e: e["t"])
        self.idx = 0
        self.phase = "idle"
        self.hold_until = 0.0

    def time_base(self, world) -> Optional[float]:
        return 0.0

    def control(self, veh, world):
        a_idm = self.idm_command(veh, world, self.p)
        base = self.time_base(world)
        if base is not None and self.idx < len(self.events):
            ev = self.events[self.idx]
            if self.phase == "idle" and world.time >= base + ev["t"]:
                self.phase = "braking"
            if self.phase == "braking":
                if veh.v <= ev["v_target"] + 0.1:
                    self.phase = "hold"
                    self.hold_until = world.time + ev["hold"]
                else:
                    return min(a_idm, -ev["decel"]), 0
            if self.phase == "hold":
                if world.time >= self.hold_until:
                    self.phase = "idle"
                    self.idx += 1
                else:
                    hold_a = float(np.clip(2.0 * (ev["v_target"] - veh.v), -ev["decel"], 1.0))
                    return min(a_idm, hold_a), 0
        return a_idm, 0


class CutInController(BrakeEventController):
    """Drives in an adjacent lane, then merges in front of the ego vehicle.

    Merge triggers when the bumper gap to the ego shrinks to `trigger_gap` (ego
    closing in) or at `t_force`, as long as the gap is at least `min_gap`.
    `direction` is the lane-index direction of the merge (+1 = towards higher lane).
    """

    kind = "cutin"

    def __init__(self, params: IDMParams, trigger_gap: float, t_force: float, direction: int,
                 min_gap: float = 5.0, events: Optional[list[dict]] = None):
        super().__init__(params, events or [])
        self.trigger_gap = trigger_gap
        self.t_force = t_force
        self.direction = direction
        self.min_gap = min_gap
        self.t_trigger: Optional[float] = None

    def time_base(self, world):
        return self.t_trigger

    def control(self, veh, world):
        lc = 0
        if self.t_trigger is None and not veh.lc_active:
            ego = world.ego
            gap = veh.rear - ego.front
            ready = gap >= self.min_gap and (gap <= self.trigger_gap or world.time >= self.t_force)
            if ready and world.can_change_lane(veh, self.direction):
                self.t_trigger = world.time
                lc = self.direction
        a, _ = super().control(veh, world)
        return a, lc


class RandomTrafficController(IDMController):
    """Background driver: IDM with drifting desired speed and opportunistic lane changes."""

    kind = "random"

    def __init__(self, params: IDMParams, seed: int, lc_prob: float = 0.25, interval: float = 2.5,
                 v_min: float = 14.0, v_max: float = 28.0):
        super().__init__(params)
        self.rng = np.random.default_rng(seed)
        self.lc_prob = lc_prob
        self.interval = interval
        self.v_min, self.v_max = v_min, v_max
        self.next_check = float(self.rng.uniform(0.5, interval))
        self.v0_now = params.v0

    def control(self, veh, world):
        lc = 0
        if world.time >= self.next_check:
            self.next_check = world.time + self.interval * float(self.rng.uniform(0.7, 1.3))
            # desired speed random walk
            self.v0_now = float(np.clip(self.v0_now + self.rng.normal(0.0, 2.0), self.v_min, self.v_max))
            if not veh.lc_active and self.rng.random() < self.lc_prob:
                lane = world.road.lane_of(veh.y)
                options = [d for d in (-1, 1) if world.can_change_lane(veh, d)]
                if options:
                    d = options[int(self.rng.integers(len(options)))]
                    fgap, fveh, rgap, rveh = world.lane_gaps(veh, lane + d)
                    rear_closing = max(0.0, (rveh.v - veh.v)) if rveh is not None else 0.0
                    front_ok = fveh is None or fgap > 20.0 + max(0.0, veh.v - fveh.v) * 2.0
                    rear_ok = rveh is None or rgap > 12.0 + rear_closing * 2.5
                    if front_ok and rear_ok:
                        lc = d
        return self.idm_command(veh, world, self.p, v0=self.v0_now), lc


def make_controller(behavior: str, params: dict, seed: int = 0):
    """Build a controller from a scenario NPC spec (behavior name + param dict)."""
    idm = IDMParams.from_dict(params)
    if behavior == "constant":
        return ConstantSpeedController(float(params["v0"]))
    if behavior == "idm":
        return IDMController(idm)
    if behavior == "braker":
        return BrakeEventController(idm, list(params.get("events", [])))
    if behavior == "cutin":
        return CutInController(idm, float(params["trigger_gap"]), float(params["t_force"]),
                               int(params["direction"]), float(params.get("min_gap", 5.0)),
                               list(params.get("events", [])))
    if behavior == "random":
        return RandomTrafficController(idm, seed, float(params.get("lc_prob", 0.25)),
                                       float(params.get("interval", 2.5)),
                                       float(params.get("v_min", 14.0)), float(params.get("v_max", 28.0)))
    raise ValueError(f"Unknown NPC behavior '{behavior}'")
