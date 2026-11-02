"""Policy evaluation on paired, held-out scenario suites.

Seed discipline
  * training scenario seeds are drawn from the training RNG and never fall in the reserved range
  * evaluation suite seeds start at EVAL_SEED_BASE (used for periodic eval / model selection)
  * test suite seeds start at TEST_SEED_BASE (used by scripts/evaluate.py for final comparisons)
  Every policy sees exactly the same (family, seed, difficulty) list: results are *paired*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .environment import DrivingEnv
from .scenarios import FAMILIES

EVAL_SEED_BASE = 10_000_000
TEST_SEED_BASE = 20_000_000


@dataclass(frozen=True)
class SuiteItem:
    family: str
    seed: int
    difficulty: float


def make_suite(families: Sequence[str], n_per_family: int, seed_base: int = TEST_SEED_BASE) -> list[SuiteItem]:
    """Deterministic scenario list. Difficulty is spread evenly over (0, 1) within each family."""
    items = []
    for fi, fam in enumerate(families):
        if fam not in FAMILIES:
            raise ValueError(f"Unknown family '{fam}'")
        fam_id = FAMILIES.index(fam)
        for i in range(n_per_family):
            items.append(SuiteItem(fam, seed_base + fam_id * 10_000 + i, (i + 0.5) / n_per_family))
    return items


def run_episode(env: DrivingEnv, policy, item: SuiteItem, trace: Optional[list] = None) -> dict:
    obs, info = env.reset(seed=item.seed, options={"family": item.family, "scenario_seed": item.seed,
                                                   "difficulty": item.difficulty})
    policy.reset(seed=item.seed)
    done = False
    while not done:
        action = policy.act(obs, info["action_mask"])
        if trace is not None:
            trace.append({"step": info["step"], "action": action, "speed": info["speed"], "gap": info["front_gap"]})
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    m = dict(info["episode_metrics"])
    return m


def evaluate_policy(env: DrivingEnv, policy, suite: Sequence[SuiteItem], progress: Optional[Callable] = None) -> list[dict]:
    rows = []
    for i, item in enumerate(suite):
        rows.append(run_episode(env, policy, item))
        if progress:
            progress(i + 1, len(suite))
    return rows


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


MIN_N_FOR_CI = 30


def summarize(rows: Sequence[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    k_col = int(sum(r["collision"] for r in rows))
    k_suc = int(sum(r["success"] for r in rows))
    out = {
        "n": n,
        "collision_rate": k_col / n, "collision_ci": wilson_interval(k_col, n),
        "success_rate": k_suc / n, "success_ci": wilson_interval(k_suc, n),
        "ci_reliable": n >= MIN_N_FOR_CI,
        "off_road_rate": mean("off_road"), "stalled_rate": mean("stalled"), "timeout_rate": mean("truncated"),
        "rear_ended_rate": mean("rear_ended"),
        "mean_distance": mean("distance"), "mean_speed": mean("mean_speed"),
        "unsafe_events_per_ep": mean("unsafe_following_events"),
        "unsafe_steps_per_ep": mean("unsafe_following_steps"),
        "hard_brakes_per_ep": mean("hard_brake_events"),
        "lane_changes_per_ep": mean("lane_changes"),
        "mean_reward": mean("reward"), "reward_ci": bootstrap_ci([r["reward"] for r in rows]),
        "mean_steps": mean("steps"),
    }
    return out


def summarize_by_family(rows: Sequence[dict]) -> dict:
    fams = sorted({r["family"] for r in rows}, key=lambda f: FAMILIES.index(f) if f in FAMILIES else 99)
    return {f: summarize([r for r in rows if r["family"] == f]) for f in fams}


def eval_score(s: dict, target_speed: float = 25.0) -> float:
    """Scalar model-selection criterion used for the 'best' checkpoint.

        score = success_rate - 2*collision_rate + 0.1*(mean_speed/target_speed) - 0.05*unsafe_events_per_episode

    Success and collisions dominate; speed and unsafe-following terms break ties between
    policies that are equally safe, discouraging both crawling and tailgating. It is computed
    on a fixed, held-out evaluation suite with exploration disabled, never from training reward.
    """
    return (s["success_rate"] - 2.0 * s["collision_rate"]
            + 0.1 * s["mean_speed"] / target_speed - 0.05 * s["unsafe_events_per_ep"])


def paired_comparison(rows_a: Sequence[dict], rows_b: Sequence[dict]) -> dict:
    """Compare policy A vs B on identical scenarios (rows must be aligned by suite order)."""
    assert len(rows_a) == len(rows_b)
    for a, b in zip(rows_a, rows_b):
        assert (a["family"], a["scenario_seed"]) == (b["family"], b["scenario_seed"]), "rows are not paired"
    n = len(rows_a)
    col_a = np.array([r["collision"] for r in rows_a])
    col_b = np.array([r["collision"] for r in rows_b])
    suc_a = np.array([r["success"] for r in rows_a])
    suc_b = np.array([r["success"] for r in rows_b])
    dr = np.array([a["reward"] - b["reward"] for a, b in zip(rows_a, rows_b)])
    return {
        "n": n,
        "collision_only_a": int(((col_a == 1) & (col_b == 0)).sum()),
        "collision_only_b": int(((col_a == 0) & (col_b == 1)).sum()),
        "collision_both": int(((col_a == 1) & (col_b == 1)).sum()),
        "success_only_a": int(((suc_a == 1) & (suc_b == 0)).sum()),
        "success_only_b": int(((suc_a == 0) & (suc_b == 1)).sum()),
        "success_both": int(((suc_a == 1) & (suc_b == 1)).sum()),
        "mean_reward_diff": float(dr.mean()), "reward_diff_ci": bootstrap_ci(dr),
    }
