"""Interactive Pygame application: menu, manual / rule-based / DQN driving, scenario challenge,
results browser and crash replay.

The simulation advances on a fixed physics timestep (cfg.physics.dt) driven by an accumulator, so physics
is identical at any display FPS; the renderer interpolates between the last two physics states for
smooth 60 FPS motion.
"""
from __future__ import annotations

import datetime
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

from agents.dqn import IncompatibleCheckpointError
from agents.registry import list_checkpoints, load_dqn_policy, untrained_policy
from agents.rule_based import RuleBasedPolicy
from driving_sim.config import PROJECT_ROOT, SimConfig, load_sim_config
from driving_sim.environment import DrivingEnv
from driving_sim.physics import WorldSnap
from driving_sim.renderer import AMBER, GREEN, RED, Overlays, Renderer, SceneInfo
from driving_sim.scenarios import FAMILIES, FAMILY_LABELS, ScenarioSpec, generate_scenario
from driving_sim.vehicles import ACTION_NAMES, Action

from . import controls as ctl
from . import ui

REPLAY_DIR = PROJECT_ROOT / "results" / "replays"
MAX_TICKS_PER_FRAME = 48
POST_EPISODE_SECONDS = 3.0


# --------------------------------------------------------------------------- #
# Snapshot analysis (works for live state and replayed frames alike)          #
# --------------------------------------------------------------------------- #
def analyse_snap(snap: WorldSnap, num_lanes: int, lane_width: float, min_overlap: float = 0.3) -> dict:
    ego = next(v for v in snap.vehicles if v.is_ego)
    lane = max(0, min(num_lanes - 1, int(ego.y // lane_width)))

    def occupies(v, ln):
        lo, hi = ln * lane_width, (ln + 1) * lane_width
        return min(v.y + v.width / 2, hi) - max(v.y - v.width / 2, lo) > min_overlap

    detections, lead = [], None
    nearest = None
    for v in snap.vehicles:
        if v.is_ego:
            continue
        gx = max(0.0, abs(v.x - ego.x) - (v.length + ego.length) / 2)
        gy = max(0.0, abs(v.y - ego.y) - (v.width + ego.width) / 2)
        d = math.hypot(gx, gy)
        nearest = d if nearest is None else min(nearest, d)
    for off, slot in ((-1, "left"), (0, "same"), (1, "right")):
        ln = lane + off
        if not 0 <= ln < num_lanes:
            continue
        best = {"front": None, "rear": None}
        for v in snap.vehicles:
            if v.is_ego or not occupies(v, ln) or abs(v.x - ego.x) > 100 + 5:
                continue
            side = "front" if v.x >= ego.x else "rear"
            gap = max(0.0, abs(v.x - ego.x) - (v.length + ego.length) / 2)
            if best[side] is None or gap < best[side][0]:
                best[side] = (gap, v)
        for side, b in best.items():
            if b is not None:
                detections.append((b[1].vid, slot, side, b[0]))
                if slot == "same" and side == "front":
                    lead = (b[0], b[1])
    out = {"ego": ego, "lane": lane, "nearest": nearest, "detections": detections, "lead_gap": None,
           "lead_vid": None, "headway": None, "ttc": None}
    if lead is not None:
        gap, lv = lead
        out["lead_gap"], out["lead_vid"] = gap, lv.vid
        if ego.v > 1.0:
            out["headway"] = gap / ego.v
        closing = ego.v - lv.v
        if closing > 1e-3:
            out["ttc"] = gap / closing
    return out


# --------------------------------------------------------------------------- #
# Application                                                                 #
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, cfg: Optional[SimConfig] = None, *, headless: bool = False, mode: str = "rule",
                 family: str = "mixed", difficulty: float = 0.5, seed: int = 1, checkpoint: Optional[str] = None,
                 start_menu: bool = True, epsilon: float = 0.0, num_actions: Optional[int] = None,
                 record_dir: Path = REPLAY_DIR):
        self.cfg = cfg or load_sim_config()
        if num_actions:
            self.cfg.num_actions = num_actions
        self.headless = headless
        self.renderer = Renderer(self.cfg, headless=headless)
        self.pg = pygame
        self.record_dir = Path(record_dir)
        self.env = DrivingEnv(self.cfg)
        self.rule = RuleBasedPolicy(self.cfg)
        self.controls = ctl.ManualControls()
        self.overlays = Overlays(debug=True, trajectories=True, radar=True, zones=True)
        self.show_q = True
        self.show_help = False
        self.running = True
        self.screen = "menu" if start_menu else "sim"
        self.menu_sel = 0
        self.form = ui.ScenarioForm(FAMILIES)
        self.results: list[Path] = []
        self.results_idx = 0
        self.results_image = None
        self.message = ""
        self.message_until = 0.0

        # driving session
        self.mode = "rule" if start_menu else mode    # the menu background runs the rule-based driver
        self.family, self.difficulty = family, difficulty
        self.base_seed = seed
        self.cur_seed = seed
        self.fixed_scenario = False      # scenario challenge: R and auto-restart reuse the same seed
        self.episode = 0
        self.speed_idx = 2
        self.paused = False
        self.acc = 0.0
        self.fps = 60.0

        # policies
        self.ckpt_paths = list_checkpoints()
        self.ckpt_idx = 0
        self.explicit_ckpt = checkpoint
        self.dqn = None
        self.dqn_ckpt: dict = {}
        self.dqn_label = ""
        self.dqn_epsilon = epsilon
        self.recorded: Optional[dict] = None
        self.recorded_pos = 0
        self.recorded_verdict = ""

        self._reset_episode_state()
        if self.mode == "dqn" and not self.ensure_dqn():
            self.mode = "rule"
        self.start_episode(self.cur_seed)

    # ------------------------------------------------------------ utilities --
    def notify(self, text: str, seconds: float = 4.0) -> None:
        self.message, self.message_until = text, time.time() + seconds

    def _reset_episode_state(self) -> None:
        self.episode_over = False
        self.over_reason = ""
        self.over_timer = 0.0
        self.history: list[WorldSnap] = []
        self.crash_idx: Optional[int] = None
        self.last_comps: dict = {}
        self.hold_action = int(Action.COAST)
        self.current_action = int(Action.COAST)
        self.last_q: Optional[np.ndarray] = None
        self.replaying = False
        self.replay_pos = 0.0
        self.replay_paused = False
        self.replay_speed = 0.5
        self.saved_replay_path: Optional[Path] = None

    @property
    def sim_speed(self) -> float:
        return ctl.SPEED_STEPS[self.speed_idx]

    def active_policy(self):
        if self.mode == "rule":
            return self.rule
        if self.mode == "dqn":
            return self.dqn
        return None

    # ------------------------------------------------------------- policies --
    def set_num_actions(self, n: int) -> None:
        if n != self.env.num_actions:
            self.cfg.num_actions = n
            self.env = DrivingEnv(self.cfg)
            self.rule = RuleBasedPolicy(self.cfg)
            self.notify(f"Environment switched to {n} actions to match the checkpoint")

    def ensure_dqn(self, force_reload: bool = False) -> bool:
        if self.dqn is not None and not force_reload:
            return True
        path = self.explicit_ckpt or (self.ckpt_paths[self.ckpt_idx] if self.ckpt_paths else None)
        if path is None:
            self.notify("No trained checkpoint found in checkpoints/ -- train one (scripts/train.py) "
                        "or press U to watch an UNTRAINED network.", 8.0)
            return False
        try:
            policy, ck = load_dqn_policy(path, None, "cpu", epsilon=self.dqn_epsilon)
        except (IncompatibleCheckpointError, FileNotFoundError) as exc:
            self.notify(f"Cannot load checkpoint: {exc}", 10.0)
            self.explicit_ckpt = None
            return False
        self.set_num_actions(ck["num_actions"])
        self.dqn, self.dqn_ckpt, self.dqn_label = policy, ck, policy.name
        self.notify(f"Loaded {Path(path).parent.name}/{Path(path).name}  ({ck['num_actions']} actions, "
                    f"{ck.get('train_state', {}).get('env_steps', '?')} train steps)")
        return True

    def cycle_checkpoint(self) -> None:
        self.ckpt_paths = list_checkpoints()
        if not self.ckpt_paths:
            self.notify("No checkpoints found in checkpoints/")
            return
        self.explicit_ckpt = None
        self.ckpt_idx = (self.ckpt_idx + 1) % len(self.ckpt_paths)
        self.dqn = None
        if self.ensure_dqn(True):
            self.set_mode("dqn")

    def use_untrained(self) -> None:
        self.dqn = untrained_policy(self.env.num_actions)
        self.dqn_ckpt, self.dqn_label = {}, self.dqn.name
        self.notify("UNTRAINED network (random weights): behaviour is meaningless by design.", 6.0)
        self.set_mode("dqn")

    def set_mode(self, mode: str) -> None:
        if mode == "dqn" and self.dqn is None and not self.ensure_dqn():
            return
        if mode != "recorded":
            self.recorded = None
        self.mode = mode
        self.controls.reset()
        self.restart()

    # ------------------------------------------------------------- episodes --
    def start_episode(self, seed: int, spec: Optional[ScenarioSpec] = None) -> None:
        self.cur_seed = seed
        spec = spec or generate_scenario(self.family, seed, self.difficulty, self.cfg.episode.goal_distance)
        self.env.reset(seed=seed, options={"scenario": spec})
        self.episode += 1
        self._reset_episode_state()
        pol = self.active_policy()
        if pol is not None:
            pol.reset(seed)
        self.renderer.clear_effects()
        w = self.env.world
        self.curr_snap = self.prev_snap = w.snapshot()
        self.history.append(self.curr_snap)
        self.acc = 0.0
        self.recorded_pos = 0

    def restart(self) -> None:
        if self.recorded is not None and self.mode == "recorded":
            self.load_recording(self.recorded)
            return
        self.start_episode(self.cur_seed)

    def next_scenario(self) -> None:
        if self.mode == "recorded":
            return
        self.start_episode(self.cur_seed + 1)

    def load_recording(self, record: dict) -> None:
        """Re-simulate a saved scenario + action trace (deterministic replay)."""
        if record.get("num_actions") != self.env.num_actions:
            self.set_num_actions(int(record["num_actions"]))
        if record.get("config_fingerprint") != self.cfg.fingerprint():
            self.notify("Warning: simulation config differs from the recording; replay may diverge.", 8.0)
        self.recorded = record
        self.mode = "recorded"
        spec = ScenarioSpec.from_dict(record["spec"])
        self.family = spec.family
        self.start_episode(spec.seed, spec)
        self.recorded_pos = 0
        self.recorded_verdict = ""

    # ----------------------------------------------------------- simulation --
    def _decide(self) -> None:
        env = self.env
        pol = self.active_policy()
        if pol is None:
            return
        obs, mask = env.observe(), env.action_mask()
        self.hold_action = int(pol.act(obs, mask))
        self.current_action = self.hold_action
        self.last_q = getattr(pol, "last_q", None)

    def sim_tick(self) -> None:
        env = self.env
        if self.episode_over:
            env.world.step(int(Action.COAST))
            self._after_tick()
            return
        if self.mode == "manual":
            a = self.controls.action(self.pg.key.get_pressed())
        elif self.mode == "recorded":
            acts = self.recorded["tick_actions"]
            if self.recorded_pos >= len(acts):
                self._finish_recorded(None)
                return
            a = int(acts[self.recorded_pos])
            self.recorded_pos += 1
        else:
            if env.ticks_in_interval == 0:
                self._decide()
            first = env.ticks_in_interval == 0
            a = self.hold_action if (first or self.hold_action < Action.LANE_LEFT) else int(Action.COAST)
        result = env.tick(a)
        self.current_action = env.last_action if self.mode in ("manual", "recorded") else self.current_action
        self._after_tick()
        if result is not None:
            self._on_decision(result)

    def _after_tick(self) -> None:
        w = self.env.world
        self.prev_snap, self.curr_snap = self.curr_snap, w.snapshot()
        self.history.append(self.curr_snap)
        for e in w.events:
            if e.kind == "collision":
                strength = min(1.5, 0.3 + e.data.get("closing_speed", 5.0) / 12.0)
                self.renderer.trigger_collision(e.data["x"], e.data["y"], strength)
                if self.env.world.ego.vid in e.vids and self.crash_idx is None:
                    self.crash_idx = len(self.history) - 1

    def _on_decision(self, result) -> None:
        _, _, terminated, truncated, info = result
        self.last_comps = info.get("reward_components", {})
        if terminated or truncated:
            m = info["episode_metrics"]
            if info["collision"]:
                reason = "COLLISION"
            elif info["success"]:
                reason = "GOAL REACHED"
            elif info["off_road"]:
                reason = "OFF ROAD"
            elif info["stalled"]:
                reason = "STALLED"
            else:
                reason = "TIME LIMIT"
            self.episode_over, self.over_reason, self.over_timer = True, reason, 0.0
            if reason == "COLLISION":
                self.saved_replay_path = self.save_record()
            if self.mode == "recorded":
                self._finish_recorded(m)

    def _finish_recorded(self, metrics: Optional[dict]) -> None:
        if not self.recorded_verdict:
            ref = self.recorded.get("outcome", {})
            if metrics is None:
                metrics = self.env.metrics.to_dict()
            same = all(abs(float(metrics.get(k, 0)) - float(ref.get(k, 0))) < 1e-6
                       for k in ("collision", "success", "steps", "distance", "reward"))
            self.recorded_verdict = ("REPLAY VERIFIED: identical outcome to the recording" if same
                                     else "REPLAY MISMATCH: outcome differs from the recording")
        if not self.episode_over:
            self.episode_over, self.over_reason, self.over_timer = True, "REPLAY ENDED", 0.0

    def save_record(self) -> Optional[Path]:
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sp = self.env.scenario
            path = self.record_dir / f"crash_{stamp}_{sp.family}_s{sp.seed}_{self.mode}.json"
            rec = self.env.replay_record()
            rec["mode"] = self.mode
            path.write_text(json.dumps(rec))
            return path
        except OSError:
            return None

    # --------------------------------------------------------- crash replay --
    def start_crash_replay(self) -> None:
        if self.crash_idx is None or len(self.history) < 2:
            self.notify("No collision to replay in this episode.")
            return
        self.replaying = True
        self.replay_pos = float(max(0, self.crash_idx - 200))   # 10 s before impact
        self.replay_paused = False
        self.replay_speed = 0.5
        self.renderer.clear_effects()
        self._replay_effect_idx = -1

    def _advance_replay(self, dt: float) -> None:
        n = len(self.history)
        if not self.replay_paused:
            self.replay_pos = min(n - 1.0, self.replay_pos + dt * (1.0 / self.cfg.physics.dt) * self.replay_speed)
        i = int(self.replay_pos)
        if self.crash_idx is not None and i >= self.crash_idx and self._replay_effect_idx < self.crash_idx:
            self._replay_effect_idx = self.crash_idx
            v = [q for q in self.history[self.crash_idx].vehicles if q.crashed]
            if v:
                self.renderer.trigger_collision(sum(q.x for q in v) / len(v), sum(q.y for q in v) / len(v), 1.0)

    # --------------------------------------------------------------- events --
    def handle_event(self, e) -> None:
        pg = self.pg
        if e.type == pg.QUIT:
            self.running = False
            return
        if e.type == pg.MOUSEBUTTONDOWN and e.button == 1 and self.screen == "menu":
            for i, rect in enumerate(ui.menu_item_rects(self.renderer)):
                if rect.collidepoint(e.pos):
                    self.menu_sel = i
                    self.activate_menu(i)
            return
        if e.type == pg.MOUSEMOTION and self.screen == "menu":
            for i, rect in enumerate(ui.menu_item_rects(self.renderer)):
                if rect.collidepoint(e.pos):
                    self.menu_sel = i
            return
        if e.type != pg.KEYDOWN:
            return
        k = e.key
        if self.screen == "menu":
            self._menu_key(k)
        elif self.screen == "scenario":
            self._scenario_key(e)
        elif self.screen == "results":
            self._results_key(k)
        else:
            self._sim_key(e)

    def activate_menu(self, i: int) -> None:
        key = ui.MENU_ITEMS[i][0]
        if key in ("manual", "rule", "dqn"):
            if key == "dqn" and not self.ensure_dqn():
                return
            self.fixed_scenario = False
            self.screen = "sim"
            self.mode = key
            self.recorded = None
            self.controls.reset()
            self.start_episode(self.base_seed)
        elif key == "challenge":
            self.screen = "scenario"
        elif key == "results":
            self.results = ui.discover_results(PROJECT_ROOT / "results")
            self.results_idx = 0
            self._load_results_image()
            self.screen = "results"

    def _menu_key(self, k) -> None:
        pg = self.pg
        n = len(ui.MENU_ITEMS)
        if k == pg.K_UP:
            self.menu_sel = (self.menu_sel - 1) % n
        elif k == pg.K_DOWN:
            self.menu_sel = (self.menu_sel + 1) % n
        elif k in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
            self.activate_menu(self.menu_sel)
        elif k in (pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5):
            self.menu_sel = k - pg.K_1
            self.activate_menu(self.menu_sel)
        elif k == pg.K_h:
            self.show_help = not self.show_help
        elif k == pg.K_ESCAPE:
            if self.show_help:
                self.show_help = False
            else:
                self.running = False

    def _scenario_key(self, e) -> None:
        if e.key == self.pg.K_ESCAPE:
            self.screen = "menu"
            return
        if self.form.key(e) == "start":
            mode = self.form.drivers[self.form.driver_idx]
            if mode == "dqn" and not self.ensure_dqn():
                return
            self.family = self.form.families[self.form.fam_idx]
            self.difficulty = self.form.difficulty
            self.fixed_scenario = True
            self.mode, self.recorded = mode, None
            self.screen = "sim"
            self.controls.reset()
            self.start_episode(self.form.seed)

    def _load_results_image(self) -> None:
        self.results_image = ui.load_results_image(self.renderer, self.results[self.results_idx]) if self.results else None

    def _results_key(self, k) -> None:
        pg = self.pg
        if k == pg.K_ESCAPE:
            self.screen = "menu"
        elif k in (pg.K_LEFT, pg.K_RIGHT) and self.results:
            self.results_idx = (self.results_idx + (1 if k == pg.K_RIGHT else -1)) % len(self.results)
            self._load_results_image()

    def _sim_key(self, e) -> None:
        pg, k = self.pg, e.key
        if self.replaying:
            self._replay_key(k)
            return
        self.controls.handle_event(e)
        if k == pg.K_ESCAPE:
            if self.show_help:
                self.show_help = False
            else:
                self.go_menu()
        elif k == pg.K_m:
            self.go_menu()
        elif k in (pg.K_h, pg.K_F1):
            self.show_help = not self.show_help
        elif k == pg.K_SPACE:
            self.paused = not self.paused
            self.controls.reset()
        elif k == pg.K_r:
            self.restart()
        elif k == pg.K_n:
            self.next_scenario()
        elif k == pg.K_1:
            self.set_mode("manual")
        elif k == pg.K_2:
            self.set_mode("rule")
        elif k == pg.K_3:
            self.set_mode("dqn")
        elif k in ctl.PLUS_KEYS:
            self.speed_idx = min(len(ctl.SPEED_STEPS) - 1, self.speed_idx + 1)
        elif k in ctl.MINUS_KEYS:
            self.speed_idx = max(0, self.speed_idx - 1)
        elif k == pg.K_v:
            self.overlays.debug = not self.overlays.debug
        elif k == pg.K_t:
            self.overlays.trajectories = not self.overlays.trajectories
        elif k == pg.K_g:
            self.overlays.radar = not self.overlays.radar
        elif k == pg.K_z:
            self.overlays.zones = not self.overlays.zones
        elif k == pg.K_q:
            self.show_q = not self.show_q
        elif k == pg.K_k:
            self.cycle_checkpoint()
        elif k == pg.K_u:
            self.use_untrained()
        elif k == pg.K_b:
            self.start_crash_replay()

    def _replay_key(self, k) -> None:
        pg = self.pg
        n = len(self.history)
        if k in (pg.K_ESCAPE, pg.K_b):
            self.replaying = False
            self.renderer.clear_effects()
        elif k == pg.K_SPACE:
            self.replay_paused = not self.replay_paused
        elif k == pg.K_LEFT:
            self.replay_paused = True
            self.replay_pos = max(0.0, self.replay_pos - 1)
        elif k == pg.K_RIGHT:
            self.replay_paused = True
            self.replay_pos = min(n - 1.0, self.replay_pos + 1)
        elif k == pg.K_UP:
            self.replay_speed = min(4.0, self.replay_speed * 2)
        elif k == pg.K_DOWN:
            self.replay_speed = max(0.0625, self.replay_speed / 2)
        elif k == pg.K_HOME:
            self.replay_pos = float(max(0, (self.crash_idx or 0) - 200))
        elif k in (pg.K_h, pg.K_F1):
            self.show_help = not self.show_help

    def go_menu(self) -> None:
        self.screen = "menu"
        self.paused = False
        self.mode = "rule"
        self.recorded = None
        self.family, self.fixed_scenario = "mixed", False
        self.start_episode(int(time.time()) % 100000)

    # ------------------------------------------------------------ main loop --
    def update(self, dt_wall: float) -> None:
        if self.screen in ("scenario", "results"):
            return
        if self.replaying:
            self._advance_replay(dt_wall)
            return
        running_bg = self.screen == "menu"
        if self.episode_over:
            self.over_timer += dt_wall
            if (self.over_timer > POST_EPISODE_SECONDS and (self.over_reason != "COLLISION" or running_bg)
                    and self.mode != "recorded"):
                if self.fixed_scenario:
                    self.restart()
                else:
                    self.next_scenario()
                return
        if self.paused and not running_bg:
            return
        step = self.cfg.physics.dt
        self.acc += min(dt_wall, 0.1) * (1.0 if running_bg else self.sim_speed)
        n = 0
        while self.acc >= step and n < MAX_TICKS_PER_FRAME:
            self.sim_tick()
            self.acc -= step
            n += 1
        if n == MAX_TICKS_PER_FRAME:
            self.acc = 0.0

    def _scene_snaps(self):
        if self.replaying:
            i = int(self.replay_pos)
            i = max(0, min(len(self.history) - 1, i))
            cur = self.history[i]
            prev = self.history[max(0, i - 1)]
            alpha = 1.0 if self.replay_paused else self.replay_pos - i
            return cur, prev, alpha
        return self.curr_snap, self.prev_snap, min(1.0, self.acc / self.cfg.physics.dt)

    def panel_info(self, an: dict, snap: WorldSnap) -> dict:
        env, cfg = self.env, self.cfg
        sp = env.scenario
        ego = an["ego"]
        mode = "replay" if self.replaying else self.mode
        if self.mode == "dqn" or (self.replaying and self.mode == "dqn"):
            driver = (self.dqn_label or "dqn").replace("dqn:", "")
        elif self.mode == "rule":
            driver = "IDM + safety rules"
        elif self.mode == "recorded":
            driver = "recorded actions"
        else:
            driver = "keyboard"
        if self.mode == "dqn" and self.dqn is not None:
            tr_eps = self.dqn_ckpt.get("train_state", {}).get("epsilon")
            eps = f"{self.dqn.epsilon:.2f} (greedy)"
            if tr_eps is not None:
                eps += f" / trained->{tr_eps:.2f}"
            if self.dqn_label.endswith("UNTRAINED"):
                eps = f"{self.dqn.epsilon:.2f} (untrained)"
        else:
            eps = "n/a"
        status = "PAUSED" if (self.paused and not self.replaying) else "RUNNING"
        if self.replaying:
            status = "REPLAY"
        if self.episode_over:
            status = self.over_reason
        collided = any(v.crashed and v.is_ego for v in snap.vehicles)
        sp_label = FAMILY_LABELS.get(sp.family, sp.family)
        if sp.sub_family:
            sp_label += f" ({sp.sub_family})"
        action = ACTION_NAMES[int(self.current_action)] if not self.episode_over else "-"
        return {
            "mode": mode, "driver": driver, "speed": ego.v, "episode": self.episode, "scenario": sp_label[:22],
            "seed": sp.seed, "difficulty": sp.difficulty, "lane": an["lane"], "num_lanes": env.world.road.num_lanes,
            "action": action, "reward": env.metrics.reward, "distance": ego.x, "goal": sp.goal_distance,
            "nearest": an["nearest"], "lead_gap": an["lead_gap"], "ttc": an["ttc"], "headway": an["headway"],
            "collided": collided, "epsilon": eps, "status": status,
            "q_values": self.last_q if (self.mode == "dqn" and self.show_q and not self.replaying) else None,
            "show_q": self.show_q, "selected_action": int(self.current_action),
            "components": self.last_comps, "sim_speed": self.sim_speed, "fps": self.fps,
            "dt": cfg.physics.dt, "target_speed": cfg.reward.target_speed, "speed_limit": cfg.reward.speed_limit,
            "message": self.message if time.time() < self.message_until else "",
        }

    def draw(self, dt_wall: float) -> None:
        r = self.renderer
        env = self.env
        if self.screen == "results":
            r.view.fill((14, 17, 24))
            ui.draw_results_screen(r, self.results, self.results_idx, self.results_image,
                                   ui.results_summary_lines(self.results[self.results_idx]) if self.results else [])
            ui.draw_menu_side(r, self._system_lines())
            return
        snap, prev, alpha = self._scene_snaps()
        world = env.world
        an = analyse_snap(snap, world.road.num_lanes, world.road.lane_width)
        ego_target = world.ego.lc_y1 if (not self.replaying and world.ego.lc_active) else None
        info = SceneInfo(world.road.num_lanes, world.road.lane_width, world.goal_distance, snap.time,
                         an["lead_gap"], an["lead_vid"], an["headway"], an["ttc"], an["detections"],
                         ego_target_y=ego_target)
        banner = None
        if self.mode == "dqn" and self.dqn_label.endswith("UNTRAINED") and self.screen == "sim":
            banner = "UNTRAINED NETWORK (random weights)"
        r.draw_scene(snap, info, prev, alpha, self.overlays, dt_wall, banner)

        if self.screen == "menu":
            ui.draw_menu(r, self.menu_sel, self._menu_status())
            ui.draw_menu_side(r, self._system_lines())
        elif self.screen == "scenario":
            spec = generate_scenario(self.form.families[self.form.fam_idx], self.form.seed, self.form.difficulty,
                                     self.cfg.episode.goal_distance)
            ui.draw_scenario_screen(r, self.form, spec, FAMILY_LABELS)
            ui.draw_menu_side(r, self._system_lines())
        else:
            ui.draw_panel(r, self.panel_info(an, snap))
            if self.replaying:
                ct = self.crash_idx
                t_rel = (snap.tick - self.history[ct].tick) * self.cfg.physics.dt if ct is not None else 0.0
                ui.draw_replay_bar(r, int(self.replay_pos), len(self.history), t_rel, self.replay_paused,
                                   self.replay_speed, ct)
            elif self.episode_over:
                self._draw_end_banner()
        if self.show_help:
            ui.draw_help(r)

    def _draw_end_banner(self) -> None:
        r = self.renderer
        reason = self.over_reason
        m = self.env.metrics
        if reason == "COLLISION":
            ui.draw_banner(r, "COLLISION", "R: retry   N: next   B: replay the crash", RED)
            if self.saved_replay_path:
                r.text(r.view, f"replay saved: {self.saved_replay_path.name}", (r.view_w // 2, 206), 12, (150, 158, 175),
                       False, "midtop")
        elif reason == "GOAL REACHED":
            ui.draw_banner(r, "GOAL REACHED", f"reward {m.reward:+.1f}   avg speed {m.mean_speed * 3.6:.0f} km/h", GREEN)
        elif reason == "REPLAY ENDED" or self.recorded_verdict:
            ok = "VERIFIED" in self.recorded_verdict
            ui.draw_banner(r, reason, self.recorded_verdict, GREEN if ok else RED)
        else:
            ui.draw_banner(r, reason, f"reward {m.reward:+.1f}   distance {m.distance:.0f} m", AMBER)

    def _menu_status(self) -> dict:
        ck = self.explicit_ckpt or (self.ckpt_paths[0] if self.ckpt_paths else None)
        dqn = ("checkpoint ready", GREEN) if ck else ("no checkpoint yet", AMBER)
        res = len(ui.discover_results(PROJECT_ROOT / "results"))
        return {"dqn": dqn, "results": (f"{res} run(s)" if res else "none yet", GREEN if res else AMBER),
                "rule": ("ready", GREEN)}

    def _system_lines(self):
        import torch
        ck = self.explicit_ckpt or (self.ckpt_paths[0] if self.ckpt_paths else None)
        lines = [
            (f"physics dt: {self.cfg.physics.dt * 1000:.0f} ms (fixed)", (200, 206, 220)),
            (f"decision: every {self.cfg.decision_dt:.1f} s", (200, 206, 220)),
            (f"actions: {self.cfg.num_actions}", (200, 206, 220)),
            (f"torch {torch.__version__}  mps={torch.backends.mps.is_available()}", (140, 148, 164)),
            ("", (0, 0, 0)),
            ("DQN checkpoint:", AMBER),
            (str(Path(ck).parent.name + "/" + Path(ck).name) if ck else "none found in checkpoints/",
             GREEN if ck else RED),
            ("", (0, 0, 0)),
            (f"seed base: {self.base_seed}", (140, 148, 164)),
        ]
        return lines

    def run(self, max_frames: Optional[int] = None, screenshot: Optional[str] = None) -> None:
        pg, r = self.pg, self.renderer
        clock = r.clock
        frames = 0
        while self.running:
            dt = clock.tick(self.cfg.render.fps if not self.headless else 0) / 1000.0
            if self.headless:
                dt = 1.0 / 60.0
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-4)) if not self.headless else 60.0
            for e in pg.event.get():
                self.handle_event(e)
            self.update(dt)
            self.draw(dt)
            if not self.headless:
                pg.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        if screenshot:
            pg.image.save(r.screen, screenshot)
        r.close()
        pg.quit()
