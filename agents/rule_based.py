"""Deterministic rule-based driver used as the evaluation baseline.

It consumes the *same observation vector* (and action mask) as the DQN, so the
two policies are compared with identical information. Behaviour:

  * Longitudinal: Intelligent-Driver-Model acceleration towards a target speed,
    discretised into accelerate / coast / brake, with conservative parameters.
  * Safety override: brake whenever the gap is below a conservative stopping
    distance (reaction distance + own braking distance - lead's braking distance
    assuming the lead can stop at `lead_brake`), or TTC is short.
  * Lane change (5-action env only): only when stuck behind a clearly slower
    vehicle, and only into a lane with a safe front gap, no vehicle alongside,
    a safe rear gap given the rear vehicle's closing speed, and a real speed
    benefit. A cooldown prevents repeated lane switching.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from driving_sim.config import SimConfig
from driving_sim.observations import decode_observation
from driving_sim.vehicles import Action, IDMParams, idm_accel

from .base import Policy


class RuleBasedPolicy(Policy):
    name = "rule_based"

    def __init__(self, cfg: SimConfig, target_speed: float = 24.0, headway: float = 1.6, min_gap: float = 4.0,
                 brake_decel: float = 4.5, lead_brake: float = 6.0, reaction: float = 0.5, ttc_brake: float = 3.0,
                 lane_changes: bool = True, lane_cooldown: int = 60, encroach_speed: float = 0.3):
        self.cfg = cfg
        self.idm = IDMParams(v0=target_speed, T=headway, a=1.5, b=2.0, s0=min_gap)
        self.target_speed = target_speed
        self.min_gap = min_gap
        self.brake_decel, self.lead_brake, self.reaction = brake_decel, lead_brake, reaction
        self.ttc_brake = ttc_brake
        self.lane_changes = lane_changes
        self.lane_cooldown = lane_cooldown
        self.encroach_speed = encroach_speed
        self.cooldown = 0

    def reset(self, seed: Optional[int] = None) -> None:
        self.cooldown = 0

    # ------------------------------------------------------------------
    def required_gap(self, v: float, v_lead: float) -> float:
        """Conservative stopping-distance based minimum gap."""
        return (self.min_gap + self.reaction * v + v * v / (2 * self.brake_decel)
                - v_lead * v_lead / (2 * self.lead_brake))

    def _longitudinal(self, d: dict) -> tuple[int, bool]:
        """(action, emergency). `emergency` = safety override fired (stopping-distance or TTC rule)."""
        v = d["speed"]
        lead = d["same_front"]
        # early cut-in response: a vehicle in an adjacent lane drifting towards us counts as a lead
        for key in ("left", "right"):
            nb = d[f"{key}_front"]
            if nb is not None and d[f"{key}_front_lat_toward"] > self.encroach_speed:
                if lead is None or nb[0] < lead[0]:
                    lead = nb
        if lead is None:
            a = idm_accel(v, self.idm, None)
        else:
            gap, rel_v = lead
            v_lead = max(0.0, v + rel_v)
            if gap < self.required_gap(v, v_lead) and v > v_lead - 0.5:
                return int(Action.BRAKE), True
            closing = v - v_lead
            if closing > 0.5 and gap / closing < self.ttc_brake:
                return int(Action.BRAKE), True
            a = idm_accel(v, self.idm, gap, v_lead)
        if a > 0.5:
            return int(Action.ACCELERATE), False
        if a < -1.0:
            return int(Action.BRAKE), False
        return int(Action.COAST), False

    def _lane_change(self, d: dict, mask: Optional[np.ndarray]) -> Optional[int]:
        if not self.lane_changes or self.cooldown > 0 or d["lane_changing"]:
            return None
        if mask is None or len(mask) < 5:
            return None
        v, lead = d["speed"], d["same_front"]
        if lead is None or v < 5.0:
            return None
        gap, rel_v = lead
        v_lead = v + rel_v
        if v_lead > self.target_speed - 3.0 or gap > max(60.0, 3.0 * v):
            return None
        best, best_score = None, -1.0
        for act, key, flag in ((Action.LANE_LEFT, "left", "can_left"), (Action.LANE_RIGHT, "right", "can_right")):
            if not d[flag] or not mask[act]:
                continue
            front, rear = d[f"{key}_front"], d[f"{key}_rear"]
            if front is not None:
                fgap, frel = front
                if fgap < max(35.0, self.idm.T * v + 10.0) or (v + frel) < v_lead + 2.0:
                    continue
                score = min(fgap, 150.0) + 4.0 * (v + frel)
            else:
                score = 200.0 + 4.0 * self.target_speed
            if rear is not None:
                rgap, rrel = rear   # rrel > 0: rear vehicle is approaching
                if rgap < 12.0 + max(0.0, rrel) * 3.0:
                    continue
            if best is None or score > best_score:
                best, best_score = int(act), score
        return best

    def act(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        d = decode_observation(obs, self.cfg)
        if self.cooldown > 0:
            self.cooldown -= 1
        lc = self._lane_change(d, mask)
        longitudinal, emergency = self._longitudinal(d)
        if lc is not None and not emergency:
            self.cooldown = self.lane_cooldown
            return lc
        return longitudinal
