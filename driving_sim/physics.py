"""Deterministic fixed-timestep physics: road, lane changes, swept collisions.

The world has no dependency on rendering. One call to `World.step` advances
exactly `cfg.physics.dt` seconds regardless of display frame rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .config import SimConfig
from .vehicles import ACTION_TABLE, Action, Vehicle

INF = float("inf")


# --------------------------------------------------------------------------- #
# Road                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Road:
    num_lanes: int
    lane_width: float = 3.7
    shoulder: float = 0.5   # body may overhang the outer lane lines by this much before "off-road"

    @property
    def width(self) -> float:
        return self.num_lanes * self.lane_width

    def lane_center(self, lane: int) -> float:
        return (lane + 0.5) * self.lane_width

    def lane_bounds(self, lane: int) -> tuple[float, float]:
        return lane * self.lane_width, (lane + 1) * self.lane_width

    def lane_of(self, y: float) -> int:
        return max(0, min(self.num_lanes - 1, int(y // self.lane_width)))

    def off_road(self, y: float, width: float) -> bool:
        return (y - width / 2 < -self.shoulder) or (y + width / 2 > self.width + self.shoulder)


# --------------------------------------------------------------------------- #
# Swept collision test                                                        #
# --------------------------------------------------------------------------- #
def swept_toi(a0: tuple[float, float], a1: tuple[float, float],
              b0: tuple[float, float], b1: tuple[float, float],
              hx: float, hy: float, eps: float = 1e-6) -> Optional[float]:
    """Earliest time-of-impact fraction in [0, 1] of two moving axis-aligned boxes.

    Boxes move linearly from (a0 -> a1) and (b0 -> b1); (hx, hy) are the summed
    half-extents. Works in the frame of B, so fast vehicles cannot tunnel through
    slow ones regardless of the timestep. Returns None when they never overlap.
    """
    px, py = a0[0] - b0[0], a0[1] - b0[1]
    dx = (a1[0] - a0[0]) - (b1[0] - b0[0])
    dy = (a1[1] - a0[1]) - (b1[1] - b0[1])
    if abs(px) < hx - eps and abs(py) < hy - eps:
        return 0.0
    tmin, tmax = 0.0, 1.0
    for p, d, h in ((px, dx, hx), (py, dy, hy)):
        if abs(d) < 1e-12:
            if abs(p) >= h - eps:
                return None
        else:
            t1, t2 = (-h - p) / d, (h - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax - 1e-12:
                return None
    return tmin


# --------------------------------------------------------------------------- #
# Events and snapshots                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    kind: str                      # collision | off_road | goal | lane_change_start | lane_change_end
    time: float
    vids: tuple = ()
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VehicleSnap:
    vid: int
    is_ego: bool
    x: float
    y: float
    v: float
    a: float
    heading: float
    length: float
    width: float
    crashed: bool
    color: int
    signal: int  # -1 left, +1 right, 0 none (active lane change)


@dataclass(frozen=True)
class WorldSnap:
    time: float
    tick: int
    vehicles: tuple


# --------------------------------------------------------------------------- #
# World                                                                       #
# --------------------------------------------------------------------------- #
class World:
    def __init__(self, road: Road, cfg: SimConfig, goal_distance: float):
        self.road = road
        self.cfg = cfg
        self.goal_distance = goal_distance
        self.vehicles: list[Vehicle] = []
        self.ego: Vehicle = None  # type: ignore[assignment]
        self.time = 0.0
        self.tick = 0
        self.events: list[Event] = []     # events from the most recent tick
        self.all_events: list[Event] = []  # events since reset

    # ---- construction ------------------------------------------------------
    def add_vehicle(self, veh: Vehicle) -> Vehicle:
        veh.prev_x, veh.prev_y = veh.x, veh.y
        self.vehicles.append(veh)
        if veh.is_ego:
            self.ego = veh
        return veh

    @property
    def npcs(self) -> list[Vehicle]:
        return [v for v in self.vehicles if not v.is_ego]

    # ---- queries -----------------------------------------------------------
    def lane_of(self, veh: Vehicle) -> int:
        return self.road.lane_of(veh.y)

    def occupies_lane(self, veh: Vehicle, lane: int) -> bool:
        lo, hi = self.road.lane_bounds(lane)
        overlap = min(veh.y + veh.width / 2, hi) - max(veh.y - veh.width / 2, lo)
        return overlap > self.cfg.sensors.lane_overlap_min

    def front_in_path(self, veh: Vehicle) -> tuple[float, Optional[Vehicle]]:
        """Nearest vehicle ahead whose body laterally overlaps `veh`'s path. (bumper gap, vehicle)"""
        best_gap, best = INF, None
        for o in self.vehicles:
            if o is veh or o.x < veh.x:
                continue
            if abs(o.y - veh.y) < (o.width + veh.width) / 2 + 0.2:
                gap = o.rear - veh.front
                if gap < best_gap:
                    best_gap, best = gap, o
        return (best_gap if best is not None else INF), best

    def lane_gaps(self, veh: Vehicle, lane: int):
        """(front_gap, front_vehicle, rear_gap, rear_vehicle) of `veh` relative to traffic in `lane`."""
        fgap, fveh, rgap, rveh = INF, None, INF, None
        for o in self.vehicles:
            if o is veh or not self.occupies_lane(o, lane):
                continue
            if o.x >= veh.x:
                gap = max(0.0, o.rear - veh.front)
                if gap < fgap:
                    fgap, fveh = gap, o
            else:
                gap = max(0.0, veh.rear - o.front)
                if gap < rgap:
                    rgap, rveh = gap, o
        return fgap, fveh, rgap, rveh

    def can_change_lane(self, veh: Vehicle, direction: int) -> bool:
        if direction == 0 or veh.crashed or veh.lc_active:
            return False
        if veh.v < self.cfg.physics.lane_change_min_speed:
            return False
        target = self.road.lane_of(veh.y) + direction
        return 0 <= target < self.road.num_lanes

    def snapshot(self) -> WorldSnap:
        vs = tuple(
            VehicleSnap(v.vid, v.is_ego, v.x, v.y, v.v, v.a, v.heading, v.length, v.width, v.crashed,
                        v.color, v.lc_dir if v.lc_active else 0)
            for v in self.vehicles
        )
        return WorldSnap(self.time, self.tick, vs)

    # ---- control helpers ---------------------------------------------------
    def start_lane_change(self, veh: Vehicle, direction: int) -> bool:
        if not self.can_change_lane(veh, direction):
            return False
        lane = self.road.lane_of(veh.y)
        veh.lc_active = True
        veh.lc_dir = direction
        veh.lc_y0 = veh.y
        veh.lc_y1 = self.road.lane_center(lane + direction)
        veh.lc_t = 0.0
        veh.lc_dur = self.cfg.physics.lane_change_duration
        self.events.append(Event("lane_change_start", self.time, (veh.vid,), {"dir": direction}))
        return True

    def ego_accel_command(self, level: int) -> float:
        e = self.cfg.ego
        if level > 0:
            return e.max_accel
        if level < 0:
            return -e.max_brake
        return -e.coast_decel if self.ego.v > 0 else 0.0

    # ---- stepping ----------------------------------------------------------
    def step(self, ego_action: int) -> list[Event]:
        """Advance exactly one physics tick. Returns the events produced."""
        p = self.cfg.physics
        dt = p.dt
        self.events = []

        # 1. decisions (all vehicles act on the state at the start of the tick)
        cmds: list[tuple[Vehicle, float, int]] = []
        for v in self.vehicles:
            if v.crashed:
                cmds.append((v, -p.crash_decel, 0))
            elif v.is_ego:
                level, lc = ACTION_TABLE[Action(int(ego_action))]
                cmds.append((v, self.ego_accel_command(level), lc))
            else:
                a_cmd, lc = v.controller.control(v, self) if v.controller else (0.0, 0)
                cmds.append((v, a_cmd, lc))

        # 2. lane-change requests
        for v, _, lc in cmds:
            if lc:
                self.start_lane_change(v, lc)

        # 3. integrate
        for v, a_cmd, _ in cmds:
            self._integrate(v, a_cmd, dt)

        # 4. collisions
        self._resolve_collisions(dt)

        # 5. boundaries / goal
        ego = self.ego
        if ego is not None:
            if self.road.off_road(ego.y, ego.width):
                self.events.append(Event("off_road", self.time + dt, (ego.vid,)))
            if ego.front >= self.goal_distance and not ego.crashed:
                self.events.append(Event("goal", self.time + dt, (ego.vid,)))

        self.time += dt
        self.tick += 1
        self.all_events.extend(self.events)
        return self.events

    def _integrate(self, v: Vehicle, a_cmd: float, dt: float) -> None:
        p = self.cfg.physics
        v.prev_x, v.prev_y = v.x, v.y
        jerk = v.max_jerk if v.is_ego else p.npc_max_jerk
        if v.crashed:
            jerk = INF
        a_lo, a_hi = -(p.npc_max_brake if not v.is_ego else v.max_brake), v.max_accel
        if v.crashed:
            a_lo = -p.crash_decel
        a_cmd = max(a_lo, min(a_hi, a_cmd))
        a_new = v.a + max(-jerk * dt, min(jerk * dt, a_cmd - v.a))
        v_new = v.v + a_new * dt
        if v_new < 0.0:
            v_new = 0.0
        elif v_new > p.max_speed:
            v_new = p.max_speed
        a_new = (v_new - v.v) / dt
        v.x += 0.5 * (v.v + v_new) * dt
        v.v, v.a = v_new, a_new

        if v.lc_active:
            v.lc_t += dt
            s = min(v.lc_t / v.lc_dur, 1.0)
            y_new = v.lc_y0 + (v.lc_y1 - v.lc_y0) * (s * s * (3.0 - 2.0 * s))
            v.vy = (y_new - v.y) / dt
            v.y = y_new
            if s >= 1.0:
                v.lc_active = False
                v.vy = 0.0
                v.y = v.lc_y1
                self.events.append(Event("lane_change_end", self.time + dt, (v.vid,), {"dir": v.lc_dir}))
        else:
            v.vy = 0.0

    def _resolve_collisions(self, dt: float) -> None:
        vs = sorted(self.vehicles, key=lambda q: min(q.x, q.prev_x) - q.length / 2)
        pairs: list[tuple[float, Vehicle, Vehicle]] = []
        n = len(vs)
        for i in range(n):
            a = vs[i]
            a_hi = max(a.x, a.prev_x) + a.length / 2
            for j in range(i + 1, n):
                b = vs[j]
                if min(b.x, b.prev_x) - b.length / 2 > a_hi:
                    break  # sorted by min x: nothing further can touch a
                if a.crashed and b.crashed:
                    continue
                toi = swept_toi((a.prev_x, a.prev_y), (a.x, a.y), (b.prev_x, b.prev_y), (b.x, b.y),
                                (a.length + b.length) / 2, (a.width + b.width) / 2)
                if toi is not None:
                    pairs.append((toi, a, b))
        pairs.sort(key=lambda t: (t[0], t[1].vid, t[2].vid))
        for toi, a, b in pairs:
            if a.crashed and b.crashed:
                continue
            for q in (a, b):
                if not q.crashed:
                    q.x = q.prev_x + (q.x - q.prev_x) * toi
                    q.y = q.prev_y + (q.y - q.prev_y) * toi
            closing = abs(a.v - b.v)
            vc = 0.5 * (a.v + b.v)
            for q in (a, b):
                q.crashed = True
                q.v = vc
                q.a = 0.0
                q.vy = 0.0
                q.lc_active = False
            self.events.append(Event("collision", self.time + toi * dt, (a.vid, b.vid),
                                     {"closing_speed": closing, "x": 0.5 * (a.x + b.x), "y": 0.5 * (a.y + b.y)}))
