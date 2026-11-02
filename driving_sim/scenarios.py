"""Scenario families, deterministic generation, and world construction.

A `ScenarioSpec` is a fully-resolved, JSON-serialisable description of an
episode (road, ego start, every NPC and its hidden behaviour parameters). It is
generated deterministically from `(family, seed, difficulty)`, so any episode
can be replayed from the three numbers alone, or from the saved JSON spec.
"""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .config import SimConfig
from .physics import Road, World
from .vehicles import EGO_COLOR_INDEX, IDMParams, Vehicle, make_controller

# curriculum-ordered training families, plus mixed (random combinations) and
# unseen (out-of-distribution, held-out evaluation only)
BASE_FAMILIES = ["lead_constant", "lead_brake", "multi_lead", "multi_lane", "cut_in"]
FAMILIES = BASE_FAMILIES + ["mixed", "unseen"]
FAMILY_LABELS = {
    "lead_constant": "Constant-speed lead",
    "lead_brake": "Sudden lead braking",
    "multi_lead": "Multi-vehicle platoon",
    "multi_lane": "Multi-lane traffic",
    "cut_in": "Cut-ins",
    "mixed": "Mixed random traffic",
    "unseen": "Unseen (OOD) scenarios",
}


@dataclass
class NPCSpec:
    lane: int
    dx: float                   # initial centre offset from the ego start [m]
    speed: float
    behavior: str
    params: dict
    length: float = 4.5
    width: float = 1.85
    color: int = 0
    label: str = ""


@dataclass
class ScenarioSpec:
    family: str
    seed: int
    difficulty: float
    num_lanes: int
    goal_distance: float
    ego_lane: int
    ego_speed: float
    npcs: list = field(default_factory=list)
    description: str = ""
    sub_family: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        d = dict(d)
        d["npcs"] = [NPCSpec(**n) for n in d.get("npcs", [])]
        return cls(**d)


def _rng(seed: int, family: str) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(family.encode())])


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _idm(rng, v0: float, aggressive: float = 0.0) -> dict:
    return dict(
        v0=float(v0),
        T=float(rng.uniform(1.1, 1.7) - 0.3 * aggressive),
        a=float(rng.uniform(1.4, 2.2)),
        b=float(rng.uniform(2.0, 3.0)),
        s0=float(rng.uniform(2.0, 3.0)),
    )


def _vehicle_dims(rng, allow_truck: bool = True) -> tuple[float, float, bool]:
    if allow_truck and rng.random() < 0.12:
        return float(rng.uniform(7.5, 9.5)), 2.4, True
    return float(rng.uniform(4.3, 4.9)), float(rng.uniform(1.8, 1.95)), False


class _Builder:
    """Collects NPCs while enforcing non-overlapping initial placement."""

    MIN_GAP = 7.0

    def __init__(self, rng, spec: ScenarioSpec, ego_len: float = 4.7):
        self.rng, self.spec, self.ego_len = rng, spec, ego_len
        self._color = int(rng.integers(0, 8))

    def free(self, lane: int, dx: float, length: float) -> bool:
        s = self.spec
        if lane == s.ego_lane:
            if abs(dx) < (length + self.ego_len) / 2 + self.MIN_GAP:
                return False
        elif abs(dx) < 10.0 + length / 2:
            return False
        for n in s.npcs:
            if n.lane == lane and abs(n.dx - dx) < (n.length + length) / 2 + self.MIN_GAP:
                return False
        return True

    def add(self, lane: int, dx: float, speed: float, behavior: str, params: dict,
            label: str = "", length: Optional[float] = None, width: Optional[float] = None,
            truck_ok: bool = True) -> Optional[NPCSpec]:
        if length is None:
            length, width, is_truck = _vehicle_dims(self.rng, truck_ok)
        else:
            width, is_truck = width or 1.85, False
        if not (0 <= lane < self.spec.num_lanes) or not self.free(lane, dx, length):
            return None
        self._color = (self._color + 1 + int(self.rng.integers(0, 3))) % 8
        color = 7 if is_truck else self._color
        npc = NPCSpec(lane, float(dx), float(speed), behavior, params, float(length), float(width), color, label)
        self.spec.npcs.append(npc)
        return npc


def _gap_to_dx(gap: float, npc_len: float, ego_len: float = 4.7) -> float:
    return gap + (npc_len + ego_len) / 2


def _braker_events(rng, v0: float, d: float, n: int, t0=(3.0, 10.0), dv_frac=(0.15, 0.6),
                   decel_hi: float = 4.0, full_stop: bool = False) -> list[dict]:
    events, t = [], float(rng.uniform(*t0))
    for _ in range(n):
        events.append(dict(
            t=float(t),
            decel=float(rng.uniform(2.5, decel_hi + 3.0 * d)),
            v_target=0.0 if full_stop else float(v0 * rng.uniform(*dv_frac)),
            hold=float(rng.uniform(3.0, 6.0) if full_stop else rng.uniform(1.0, 4.0)),
        ))
        t += float(rng.uniform(10.0, 18.0))
    return events


# --------------------------------------------------------------------------- #
# Family generators                                                           #
# --------------------------------------------------------------------------- #
def _new_spec(family, seed, d, lanes, goal, ego_lane, ego_speed, desc="") -> ScenarioSpec:
    return ScenarioSpec(family, int(seed), float(d), lanes, float(goal), ego_lane, float(ego_speed), [], desc)


def _gen_lead_constant(rng, seed, d, goal) -> ScenarioSpec:
    gap = rng.uniform(_lerp(45, 25, d), _lerp(90, 50, d))
    v_lead = rng.uniform(14.0, 24.0)
    spec = _new_spec("lead_constant", seed, d, 3, goal, 1, rng.uniform(10.0, 22.0),
                     f"Lead at constant {v_lead:.1f} m/s, initial gap {gap:.0f} m")
    b = _Builder(rng, spec)
    b.add(1, _gap_to_dx(gap, 4.5), v_lead, "constant", dict(v0=v_lead), "lead", length=4.5, width=1.85)
    return spec


def _gen_lead_brake(rng, seed, d, goal) -> ScenarioSpec:
    v0 = rng.uniform(17.0, 25.0)
    gap = rng.uniform(_lerp(40, 25, d), _lerp(75, 45, d))
    n_ev = 1 + int(rng.integers(0, 2 + int(2 * d)))
    spec = _new_spec("lead_brake", seed, d, 3, goal, 1, v0 * rng.uniform(0.8, 1.0),
                     f"Lead cruises {v0:.1f} m/s and brakes suddenly x{n_ev}, gap {gap:.0f} m")
    b = _Builder(rng, spec)
    params = _idm(rng, v0)
    params["events"] = _braker_events(rng, v0, d, n_ev)
    b.add(1, _gap_to_dx(gap, 4.5), v0, "braker", params, "lead", length=4.5, width=1.85)
    return spec


def _gen_multi_lead(rng, seed, d, goal) -> ScenarioSpec:
    n = 3 + int(round(d * 2))
    v0 = rng.uniform(18.0, 24.0)
    spec = _new_spec("multi_lead", seed, d, 3, goal, 1, v0 * rng.uniform(0.85, 1.0),
                     f"Platoon of {n} vehicles in the ego lane, front vehicle brakes")
    b = _Builder(rng, spec)
    # build from the ego outward; the farthest vehicle is the braking leader
    x, specs_ = 0.0, []
    for i in range(n):
        length, width, _ = _vehicle_dims(rng, allow_truck=False)
        gap = rng.uniform(_lerp(25, 18, d), _lerp(50, 35, d))
        x += (length + 4.7) / 2 + gap if i == 0 else (length + prev_len) / 2 + gap
        prev_len = length
        specs_.append((x, length, width))
    for i, (x, length, width) in enumerate(specs_):
        speed = v0 * rng.uniform(0.9, 1.0)
        if i == n - 1:
            params = _idm(rng, v0)
            params["events"] = _braker_events(rng, v0, d, 1 + int(rng.integers(0, 2)), t0=(4.0, 10.0))
            b.add(1, x, speed, "braker", params, "leader", length=length, width=width)
        else:
            b.add(1, x, speed, "idm", _idm(rng, v0 + rng.uniform(0.0, 3.0)), f"follower{i}", length=length, width=width)
    return spec


def _gen_multi_lane(rng, seed, d, goal) -> ScenarioSpec:
    lanes = 4 if (d > 0.5 and rng.random() < 0.5) else 3
    ego_lane = int(rng.integers(0, lanes))
    spec = _new_spec("multi_lane", seed, d, lanes, goal, ego_lane, rng.uniform(14.0, 22.0),
                     f"{lanes}-lane traffic with a slow vehicle ahead; passing opportunities")
    b = _Builder(rng, spec)
    slow = rng.uniform(13.0, 17.0)
    lead_len = 4.5
    b.add(ego_lane, _gap_to_dx(rng.uniform(30, 60), lead_len), slow, "idm", _idm(rng, slow), "slow lead",
          length=lead_len, width=1.85, truck_ok=False)
    spacing = (_lerp(60, 35, d), _lerp(110, 70, d))
    for lane in range(lanes):
        v_lane = rng.uniform(15.0, 25.0) + 1.0 * (lanes - 1 - lane)
        x = rng.uniform(-110.0, -25.0)
        for _ in range(int(rng.integers(3, 5 + int(2 * d)))):
            x += rng.uniform(*spacing)
            b.add(lane, x, v_lane * rng.uniform(0.92, 1.0), "idm", _idm(rng, v_lane), f"lane{lane}")
    return spec


def _gen_cut_in(rng, seed, d, goal) -> ScenarioSpec:
    lanes = 4 if (d > 0.7 and rng.random() < 0.4) else 3
    ego_lane = 1
    ego_speed = rng.uniform(16.0, 24.0)
    spec = _new_spec("cut_in", seed, d, lanes, goal, ego_lane, ego_speed, "")
    b = _Builder(rng, spec)
    n_cut = 1 + int(rng.random() < 0.3 + 0.5 * d)
    sides = [int(rng.choice([-1, 1]))]
    sides.append(-sides[0])
    descr = []
    for k in range(n_cut):
        side = sides[k]
        lane = ego_lane + side
        gap0 = rng.uniform(20, 50) + 45 * k
        v = ego_speed * rng.uniform(0.6, 0.9)
        params = _idm(rng, v)
        params.update(trigger_gap=float(rng.uniform(_lerp(22, 10, d), _lerp(30, 16, d))),
                      t_force=float(rng.uniform(5, 14) + 6 * k), direction=-side, min_gap=5.0)
        if rng.random() < 0.3 + 0.4 * d:
            params["events"] = [dict(t=float(rng.uniform(2, 5)), decel=float(rng.uniform(2.5, 4.0 + 2.0 * d)),
                                     v_target=float(v * rng.uniform(0.4, 0.7)), hold=float(rng.uniform(1, 3)))]
        npc = b.add(lane, _gap_to_dx(gap0, 4.5), v, "cutin", params, "cutter", length=4.5, width=1.85)
        if npc is not None:
            descr.append(f"cut-in from {'left' if side < 0 else 'right'} (gap {gap0:.0f} m)")
    if rng.random() < 0.6:
        v_far = ego_speed * rng.uniform(0.9, 1.05)
        b.add(ego_lane, _gap_to_dx(rng.uniform(70, 130), 4.5), v_far, "idm", _idm(rng, v_far), "far lead",
              length=4.5, width=1.85)
    for _ in range(1 + int(2 * d)):
        lane = int(rng.integers(0, lanes))
        v = rng.uniform(15.0, 26.0)
        b.add(lane, rng.uniform(-90, 160), v, "random",
              {**_idm(rng, v), "lc_prob": 0.15, "v_min": 14.0, "v_max": 28.0}, "traffic")
    spec.description = "; ".join(descr) or "traffic"
    return spec


def _gen_unseen(rng, seed, d, goal) -> ScenarioSpec:
    """Out-of-distribution configurations that curriculum training never samples."""
    kind = ["stop_and_go", "aggressive_cutin", "dense_random"][int(rng.integers(0, 3))]
    if kind == "stop_and_go":
        v0 = rng.uniform(20.0, 27.0)
        spec = _new_spec("unseen", seed, d, 3, goal, 1, v0 * rng.uniform(0.9, 1.0),
                         "Unseen: lead brakes to a full stop twice")
        b = _Builder(rng, spec)
        params = _idm(rng, v0)
        params["events"] = _braker_events(rng, v0, 1.0, 2, t0=(5.0, 10.0), decel_hi=4.0, full_stop=True)
        b.add(1, _gap_to_dx(rng.uniform(35, 60), 4.5), v0, "braker", params, "lead", length=4.5, width=1.85)
        b.add(1, _gap_to_dx(rng.uniform(95, 140), 4.5), v0, "idm", _idm(rng, v0), "far lead", length=4.5, width=1.85)
    elif kind == "aggressive_cutin":
        ego_speed = rng.uniform(22.0, 28.0)
        spec = _new_spec("unseen", seed, d, 3, goal, 1, ego_speed, "Unseen: tight cut-in followed by hard braking")
        b = _Builder(rng, spec)
        side = int(rng.choice([-1, 1]))
        v = ego_speed * rng.uniform(0.5, 0.7)
        params = _idm(rng, v, aggressive=1.0)
        params.update(trigger_gap=float(rng.uniform(6, 10)), t_force=float(rng.uniform(5, 10)), direction=-side,
                      min_gap=4.0,
                      events=[dict(t=1.5, decel=float(rng.uniform(4.0, 7.0)), v_target=float(v * 0.4), hold=2.0)])
        b.add(1 + side, _gap_to_dx(rng.uniform(25, 40), 4.5), v, "cutin", params, "cutter", length=4.5, width=1.85)
        b.add(1 - side, rng.uniform(-60, -20), ego_speed, "idm", _idm(rng, ego_speed), "tail", length=4.5, width=1.85)
    else:
        ego_speed = rng.uniform(18.0, 26.0)
        spec = _new_spec("unseen", seed, d, 4, goal, int(rng.integers(0, 4)), ego_speed,
                         "Unseen: dense 4-lane traffic with aggressive random lane changes")
        b = _Builder(rng, spec)
        for _ in range(11):
            v = rng.uniform(12.0, 30.0)
            b.add(int(rng.integers(0, 4)), rng.uniform(-120, 220), v, "random",
                  {**_idm(rng, v, aggressive=1.0), "lc_prob": 0.6, "interval": 1.8, "v_min": 12.0, "v_max": 30.0},
                  "traffic")
    spec.sub_family = kind
    return spec


_GENERATORS = {
    "lead_constant": _gen_lead_constant,
    "lead_brake": _gen_lead_brake,
    "multi_lead": _gen_multi_lead,
    "multi_lane": _gen_multi_lane,
    "cut_in": _gen_cut_in,
    "unseen": _gen_unseen,
}


def _gen_mixed(rng, seed, d, goal) -> ScenarioSpec:
    sub = ["lead_brake", "multi_lead", "multi_lane", "cut_in"][int(rng.integers(0, 4))]
    sub_d = float(rng.uniform(0.3, 1.0))
    sub_seed = int(rng.integers(0, 2**31 - 1))
    spec = _GENERATORS[sub](_rng(sub_seed, sub), sub_seed, sub_d, goal)
    # overlay random background traffic on top of the base scenario
    b = _Builder(rng, spec)
    n_extra = int(rng.integers(0, 4))
    for _ in range(n_extra):
        lane = int(rng.integers(0, spec.num_lanes))
        v = rng.uniform(14.0, 27.0)
        b.add(lane, rng.uniform(-100, 220), v, "random",
              {**_idm(rng, v), "lc_prob": float(rng.uniform(0.1, 0.4)), "v_min": 13.0, "v_max": 28.0}, "traffic")
    spec.family, spec.seed, spec.difficulty, spec.sub_family = "mixed", int(seed), float(d), sub
    spec.description = f"Mixed ({sub}, d={sub_d:.2f}) + {n_extra} random vehicles: {spec.description}"
    return spec


_GENERATORS["mixed"] = _gen_mixed


def generate_scenario(family: str, seed: int, difficulty: float = 0.5, goal_distance: float = 900.0) -> ScenarioSpec:
    """Deterministically generate a scenario from (family, seed, difficulty)."""
    if family not in _GENERATORS:
        raise ValueError(f"Unknown scenario family '{family}'. Valid: {FAMILIES}")
    difficulty = float(min(1.0, max(0.0, difficulty)))
    return _GENERATORS[family](_rng(seed, family), int(seed), difficulty, goal_distance)


# --------------------------------------------------------------------------- #
# World construction                                                          #
# --------------------------------------------------------------------------- #
def build_world(spec: ScenarioSpec, cfg: SimConfig) -> World:
    road = Road(spec.num_lanes, cfg.road.lane_width)
    world = World(road, cfg, spec.goal_distance)
    e = cfg.ego
    world.add_vehicle(Vehicle(
        vid=0, x=0.0, y=road.lane_center(spec.ego_lane), v=float(spec.ego_speed), length=e.length, width=e.width,
        is_ego=True, max_accel=e.max_accel, max_brake=e.max_brake, max_jerk=e.max_jerk, color=EGO_COLOR_INDEX,
        lc_dur=cfg.physics.lane_change_duration, label="ego"))
    for i, n in enumerate(spec.npcs, start=1):
        ctrl = make_controller(n.behavior, n.params, seed=(spec.seed * 1009 + i) & 0x7FFFFFFF)
        world.add_vehicle(Vehicle(
            vid=i, x=n.dx, y=road.lane_center(n.lane), v=n.speed, length=n.length, width=n.width,
            max_accel=float(n.params.get("a", 2.0)) + 0.5, color=n.color, controller=ctrl,
            lc_dur=cfg.physics.lane_change_duration, label=n.label))
    return world
