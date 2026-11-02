"""Headless Double-DQN training loop with curriculum, evaluation, checkpoints and resume.

Nothing here imports pygame: training never touches the renderer.
"""
from __future__ import annotations

import collections
import copy
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

from driving_sim.config import PROJECT_ROOT, deep_update, load_sim_config
from driving_sim.environment import DrivingEnv
from driving_sim.evaluation import (EVAL_SEED_BASE, eval_score, evaluate_policy, make_suite, summarize,
                                    summarize_by_family)
from driving_sim.rewards import REWARD_COMPONENTS

from .dqn import DQNAgent, DQNConfig, DQNPolicy, EpsilonSchedule, get_device, load_checkpoint, read_checkpoint
from .replay_buffer import ReplayBuffer

DEFAULT_TRAIN_CONFIG = {
    "run_name": "dqn",
    "seed": 0,
    "device": "auto",
    "total_steps": 300_000,
    "reward_scale": 0.1,            # rewards are multiplied by this before entering the replay buffer
    "env": {"num_actions": 5, "simulation_config": "configs/simulation.yaml", "overrides": {}},
    "dqn": DQNConfig().to_dict(),
    "exploration": {"start": 1.0, "end": 0.05, "decay_steps": 60_000, "schedule": "linear",
                    "stage_restart": 0.25, "stage_restart_decay_steps": 15_000},
    "eval": {"interval_steps": 10_000, "episodes_per_family": 8,
             "families": ["lead_constant", "lead_brake", "multi_lead", "multi_lane", "cut_in", "mixed", "unseen"],
             "seed_base": EVAL_SEED_BASE},
    "checkpoint": {"interval_steps": 25_000, "keep_last": 2, "save_replay": True,
                   "dir": "checkpoints", "results_dir": "results"},
    "logging": {"interval_steps": 2_000, "episode_window": 50},
    "curriculum": {"enabled": False, "stages": []},
}


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotted `key.sub=value` overrides (values parsed as YAML) in place."""
    for ov in overrides:
        key, _, raw = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_train_config(path: Optional[str | Path] = None, overrides: Optional[list[str]] = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
    if path:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = PROJECT_ROOT / p
        with open(p) as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    apply_overrides(cfg, overrides or [])
    return cfg


class CsvLog:
    """Append-only CSV with a fixed header (written once, so resumed runs keep appending)."""

    def __init__(self, path: Path, fields: list[str]):
        self.path, self.fields = path, fields
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])


class Trainer:
    def __init__(self, cfg: dict, resume: Optional[str | Path] = None, log: Callable[[str], None] = print):
        self.log = log
        self.cfg = cfg
        self.resume_ckpt = None
        if resume:
            self.resume_ckpt = read_checkpoint(resume)
        self.run = cfg["run_name"]
        ck = cfg["checkpoint"]
        root = PROJECT_ROOT
        self.ckpt_dir = (root / ck["dir"] / self.run)
        self.res_dir = (root / ck["results_dir"] / self.run)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.res_dir.mkdir(parents=True, exist_ok=True)

        sim_path = cfg["env"].get("simulation_config")
        self.sim_cfg = load_sim_config(sim_path, cfg["env"].get("overrides") or None)
        self.sim_cfg.num_actions = int(cfg["env"]["num_actions"])
        self.env = DrivingEnv(self.sim_cfg)
        self.eval_env = DrivingEnv(self.sim_cfg)
        self.device = get_device(cfg["device"])

        self.dqn_cfg = DQNConfig.from_dict(cfg["dqn"])
        self.agent = DQNAgent(self.env.obs_dim, self.env.num_actions, self.dqn_cfg, self.device, cfg["seed"])
        self.buffer = ReplayBuffer(self.dqn_cfg.buffer_size, self.env.obs_dim, self.env.num_actions)
        ex = cfg["exploration"]
        self.eps_sched = EpsilonSchedule(ex["start"], ex["end"], ex["decay_steps"], ex["schedule"])

        # training state (restored on resume)
        self.env_steps = 0
        self.episodes = 0
        self.stage_idx = 0
        self.stage_start_step = 0
        self.stage_episodes: collections.deque = collections.deque(maxlen=500)
        self.best_score = -np.inf
        self.best_step = -1
        self.last_eval_step = 0
        self.resumes = 0

        self.curriculum = cfg["curriculum"]
        self.stages = self.curriculum["stages"] if self.curriculum.get("enabled") else []

        if self.resume_ckpt is not None:
            self._restore()

        self.episode_log = CsvLog(self.res_dir / "episodes.csv",
                                  ["step", "episode", "stage", "family", "difficulty", "scenario_seed", "length",
                                   "reward", "success", "collision", "off_road", "stalled", "epsilon"]
                                  + [f"r_{k}" for k in REWARD_COMPONENTS])
        self.metric_log = CsvLog(self.res_dir / "metrics.csv",
                                 ["step", "episodes", "stage", "epsilon", "loss", "q_mean", "grad_norm",
                                  "ep_reward", "collision_rate", "success_rate", "steps_per_sec"])
        fams = cfg["eval"]["families"]
        self.eval_log = CsvLog(self.res_dir / "eval.csv",
                               ["step", "stage", "score", "collision_rate", "success_rate", "mean_speed",
                                "mean_distance", "unsafe_events_per_ep", "hard_brakes_per_ep",
                                "lane_changes_per_ep", "mean_reward", "is_best"]
                               + [f"success_{f}" for f in fams] + [f"collision_{f}" for f in fams])

        with open(self.res_dir / "config.yaml", "w") as f:
            yaml.safe_dump({"train": cfg, "simulation": self.sim_cfg.to_dict(),
                            "sim_fingerprint": self.sim_cfg.fingerprint()}, f, sort_keys=False)

    # ------------------------------------------------------------ state ------
    def _train_state(self) -> dict:
        return {
            "env_steps": self.env_steps, "episodes": self.episodes, "stage_idx": self.stage_idx,
            "stage_name": self._stage_name(), "stage_start_step": self.stage_start_step,
            "stage_episodes": list(self.stage_episodes), "epsilon": self.epsilon_at(self.env_steps),
            "best_score": float(self.best_score), "best_step": self.best_step,
            "last_eval_step": self.last_eval_step, "resumes": self.resumes,
        }

    def _restore(self) -> None:
        ck = self.resume_ckpt
        from .dqn import check_compatibility
        check_compatibility(ck, self.env.obs_dim, self.env.num_actions, "resume checkpoint")
        self.agent.online.load_state_dict(ck["online"])
        self.agent.target.load_state_dict(ck["target"])
        self.agent.optimizer.load_state_dict(ck["optimizer"])
        self.agent.update_count = int(ck.get("update_count", 0))
        try:
            self.agent.restore_rng(ck["rng_state"])
        except Exception as exc:  # pragma: no cover
            self.log(f"[resume] could not restore RNG state ({exc}); continuing with fresh RNG")
        ts = ck["train_state"]
        self.env_steps, self.episodes = ts["env_steps"], ts["episodes"]
        self.stage_idx, self.stage_start_step = ts["stage_idx"], ts["stage_start_step"]
        self.stage_episodes.extend(ts.get("stage_episodes", []))
        self.best_score, self.best_step = ts["best_score"], ts["best_step"]
        self.last_eval_step = ts.get("last_eval_step", self.env_steps)
        self.resumes = ts.get("resumes", 0) + 1
        replay = self.ckpt_dir / "replay_last.npz"
        if replay.exists():
            try:
                self.buffer.load(replay)
                self.log(f"[resume] restored replay buffer with {len(self.buffer)} transitions")
            except Exception as exc:
                self.log(f"[resume] could not restore replay buffer: {exc}")
        else:
            self.log("[resume] no saved replay buffer; refilling from scratch (learning pauses until warm-up)")
        self.log(f"[resume] step {self.env_steps}, episode {self.episodes}, stage {self._stage_name()}, "
                 f"epsilon {self.epsilon_at(self.env_steps):.3f}")

    # --------------------------------------------------------- curriculum ----
    def _stage_name(self) -> str:
        return self.stages[self.stage_idx]["name"] if self.stages else "all"

    def _apply_stage(self) -> None:
        if not self.stages:
            self.env.set_sampler(dict(self.sim_cfg.scenario.families), self.sim_cfg.scenario.difficulty_range)
            return
        st = self.stages[self.stage_idx]
        self.env.set_sampler(st["families"], tuple(st.get("difficulty", (0.0, 1.0))))
        self.env.set_lane_changes_enabled(bool(st.get("lane_changes", True)))

    def _maybe_advance_stage(self) -> None:
        if not self.stages or self.stage_idx >= len(self.stages) - 1:
            return
        st = self.stages[self.stage_idx]
        stage_steps = self.env_steps - self.stage_start_step
        adv = st.get("advance", {})
        window = int(adv.get("window", 100))
        met = False
        if stage_steps >= st.get("min_steps", 0) and len(self.stage_episodes) >= window:
            recent = list(self.stage_episodes)[-window:]
            suc = np.mean([e[0] for e in recent])
            col = np.mean([e[1] for e in recent])
            met = suc >= adv.get("success_rate", 0.9) and col <= adv.get("collision_rate", 0.05)
        forced = stage_steps >= st.get("max_steps", 10**12)
        if met or forced:
            why = "criteria met" if met else "max_steps reached"
            self.log(f"[curriculum] leaving stage '{st['name']}' at step {self.env_steps} ({why})")
            self.stage_idx += 1
            self.stage_start_step = self.env_steps
            self.stage_episodes.clear()
            self._apply_stage()
            self.log(f"[curriculum] entering stage '{self._stage_name()}'")

    def epsilon_at(self, step: int) -> float:
        eps = self.eps_sched.value(step)
        ex = self.cfg["exploration"]
        if self.stages and self.stage_idx > 0 and ex.get("stage_restart", 0) > 0:
            age = step - self.stage_start_step
            decay = max(1, ex["stage_restart_decay_steps"])
            if age < decay:
                eps = max(eps, ex["stage_restart"] * (1.0 - age / decay))
        return float(eps)

    # ---------------------------------------------------------- evaluation ---
    def evaluate(self) -> dict:
        ec = self.cfg["eval"]
        suite = make_suite(ec["families"], ec["episodes_per_family"], ec["seed_base"])
        # evaluate under the same action restriction as training (masked lane changes in early stages)
        self.eval_env.set_lane_changes_enabled(self.env.lane_changes_enabled)
        policy = DQNPolicy(self.agent, epsilon=0.0, seed=self.cfg["seed"])
        rows = evaluate_policy(self.eval_env, policy, suite)
        overall = summarize(rows)
        by_fam = summarize_by_family(rows)
        score = eval_score(overall, self.sim_cfg.reward.target_speed)
        return {"overall": overall, "by_family": by_fam, "score": score}

    def _do_eval(self) -> None:
        t0 = time.time()
        res = self.evaluate()
        o, score = res["overall"], res["score"]
        final_stage = (not self.stages) or self.stage_idx == len(self.stages) - 1
        is_best = False
        if final_stage and score > self.best_score:
            self.best_score, self.best_step, is_best = score, self.env_steps, True
            self._save("best.pt", extra={"eval_score": score, "eval_overall": _jsonable(o)})
        row = {"step": self.env_steps, "stage": self._stage_name(), "score": round(score, 4),
               "collision_rate": round(o["collision_rate"], 4), "success_rate": round(o["success_rate"], 4),
               "mean_speed": round(o["mean_speed"], 3), "mean_distance": round(o["mean_distance"], 2),
               "unsafe_events_per_ep": round(o["unsafe_events_per_ep"], 3),
               "hard_brakes_per_ep": round(o["hard_brakes_per_ep"], 3),
               "lane_changes_per_ep": round(o["lane_changes_per_ep"], 3),
               "mean_reward": round(o["mean_reward"], 3), "is_best": int(is_best)}
        for f, s in res["by_family"].items():
            row[f"success_{f}"] = round(s["success_rate"], 4)
            row[f"collision_{f}"] = round(s["collision_rate"], 4)
        self.eval_log.write(row)
        self.last_eval_step = self.env_steps
        self.log(f"[eval] step {self.env_steps:>7d} stage={self._stage_name()} score={score:+.3f} "
                 f"collision={o['collision_rate']:.2f} complete={o['success_rate']:.2f} "
                 f"speed={o['mean_speed']:.1f} unsafe/ep={o['unsafe_events_per_ep']:.2f} "
                 f"({time.time() - t0:.0f}s){'  * new best' if is_best else ''}")

    # ---------------------------------------------------------- checkpoints --
    def _save(self, name: str, extra: Optional[dict] = None) -> Path:
        path = self.ckpt_dir / name
        ex = {"train_config": self.cfg, "reward_scale": self.cfg["reward_scale"], "run": self.run}
        if extra:
            ex.update(extra)
        self.agent.save(path, sim_config=self.sim_cfg.to_dict(), sim_fingerprint=self.sim_cfg.fingerprint(),
                        train_state=self._train_state(), extra=ex)
        return path

    def _checkpoint(self, final: bool = False) -> None:
        ck = self.cfg["checkpoint"]
        self._save("last.pt")
        if not final:
            self._save(f"step_{self.env_steps}.pt")
            old = sorted(self.ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            for p in old[:-ck["keep_last"]]:
                p.unlink(missing_ok=True)
        if ck.get("save_replay"):
            self.buffer.save(self.ckpt_dir / "replay_last.npz")

    # ------------------------------------------------------------ main loop --
    def run_training(self) -> dict:
        cfg, agent, env = self.cfg, self.agent, self.env
        total = int(cfg["total_steps"])
        log_int = cfg["logging"]["interval_steps"]
        win = cfg["logging"]["episode_window"]
        eval_int = cfg["eval"]["interval_steps"]
        ck_int = cfg["checkpoint"]["interval_steps"]
        rs = float(cfg["reward_scale"])
        self._apply_stage()
        seed = cfg["seed"] + 1000 * self.resumes
        obs, info = env.reset(seed=seed)
        recent = collections.deque(maxlen=win)   # (reward, success, collision)
        losses, qs, gns = [], [], []
        t_last, step_last = time.time(), self.env_steps
        self.log(f"[train] run '{self.run}'  actions={env.num_actions}  device={self.device}  "
                 f"steps {self.env_steps}->{total}  sim_fingerprint={self.sim_cfg.fingerprint()}")
        try:
            while self.env_steps < total:
                eps = self.epsilon_at(self.env_steps)
                action = agent.select_action(obs, info["action_mask"], eps)
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.buffer.add(obs, action, reward * rs, next_obs, terminated, info["action_mask"])
                self.env_steps += 1
                obs = next_obs

                if len(self.buffer) >= self.dqn_cfg.learning_starts and self.env_steps % self.dqn_cfg.train_freq == 0:
                    st = agent.learn(self.buffer.sample(self.dqn_cfg.batch_size, agent.rng))
                    losses.append(st["loss"]); qs.append(st["q_mean"]); gns.append(st["grad_norm"])

                if terminated or truncated:
                    m = info["episode_metrics"]
                    self.episodes += 1
                    recent.append((m["reward"], m["success"], m["collision"]))
                    self.stage_episodes.append((m["success"], m["collision"]))
                    row = {"step": self.env_steps, "episode": self.episodes, "stage": self._stage_name(),
                           "family": m["family"], "difficulty": m["difficulty"], "scenario_seed": m["scenario_seed"],
                           "length": m["steps"], "reward": m["reward"], "success": m["success"],
                           "collision": m["collision"], "off_road": m["off_road"], "stalled": m["stalled"],
                           "epsilon": round(eps, 4)}
                    row.update({f"r_{k}": m[f"r_{k}"] for k in REWARD_COMPONENTS})
                    self.episode_log.write(row)
                    self._maybe_advance_stage()
                    obs, info = env.reset()

                if self.env_steps % log_int == 0:
                    now = time.time()
                    sps = (self.env_steps - step_last) / max(1e-9, now - t_last)
                    t_last, step_last = now, self.env_steps
                    r = np.array(recent) if recent else np.zeros((1, 3))
                    row = {"step": self.env_steps, "episodes": self.episodes, "stage": self._stage_name(),
                           "epsilon": round(self.epsilon_at(self.env_steps), 4),
                           "loss": round(float(np.mean(losses)), 5) if losses else "",
                           "q_mean": round(float(np.mean(qs)), 4) if qs else "",
                           "grad_norm": round(float(np.mean(gns)), 4) if gns else "",
                           "ep_reward": round(float(r[:, 0].mean()), 3),
                           "success_rate": round(float(r[:, 1].mean()), 3),
                           "collision_rate": round(float(r[:, 2].mean()), 3), "steps_per_sec": round(sps, 1)}
                    self.metric_log.write(row)
                    self.log(f"[train] step {self.env_steps:>7d}/{total} ep {self.episodes:>5d} {self._stage_name():<14s} "
                             f"eps={row['epsilon']:.3f} loss={row['loss'] if row['loss'] == '' else format(row['loss'], '.4f')} "
                             f"R={row['ep_reward']:+7.1f} coll={row['collision_rate']:.2f} done={row['success_rate']:.2f} "
                             f"{sps:5.0f} sps")
                    losses, qs, gns = [], [], []

                if self.env_steps % eval_int == 0:
                    self._do_eval()
                if self.env_steps % ck_int == 0:
                    self._checkpoint()
        except KeyboardInterrupt:
            self.log("\n[train] interrupted: saving checkpoint (resume with --resume)")
            self._checkpoint(final=True)
            raise
        if self.last_eval_step != self.env_steps:
            self._do_eval()
        self._checkpoint(final=True)
        self._save("final.pt")
        try:
            from driving_sim.plotting import plot_training_curves
            out = plot_training_curves(self.res_dir)
            self.log(f"[train] wrote {out}")
        except Exception as exc:  # pragma: no cover
            self.log(f"[train] could not plot curves: {exc}")
        summary = {"run": self.run, "steps": self.env_steps, "episodes": self.episodes,
                   "best_score": float(self.best_score), "best_step": self.best_step,
                   "checkpoints": str(self.ckpt_dir), "results": str(self.res_dir)}
        with open(self.res_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self.log(f"[train] done. best eval score {self.best_score:+.3f} at step {self.best_step}")
        return summary


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d
