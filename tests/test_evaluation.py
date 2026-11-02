import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from agents.random_agent import RandomPolicy
from agents.rule_based import RuleBasedPolicy
from driving_sim.environment import RESERVED_SEEDS, DrivingEnv
from driving_sim.evaluation import (EVAL_SEED_BASE, TEST_SEED_BASE, bootstrap_ci, eval_score, evaluate_policy,
                                    make_suite, paired_comparison, summarize, summarize_by_family, wilson_interval)
from driving_sim.replay import load_record, replay_episode, save_record, verify_replay

ROOT = Path(__file__).resolve().parent.parent
FAMS = ["lead_constant", "lead_brake", "multi_lane"]


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.963, abs=2e-3)
    assert np.isnan(wilson_interval(0, 0)[0])
    # interval shrinks with n
    assert (wilson_interval(50, 100)[1] - wilson_interval(50, 100)[0]) < (wilson_interval(5, 10)[1] - wilson_interval(5, 10)[0])


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(0).normal(5, 1, 200)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi and hi - lo < 0.6


def test_suite_is_deterministic_and_seed_ranges_are_disjoint():
    a = make_suite(FAMS, 5)
    assert a == make_suite(FAMS, 5)
    assert len(a) == 15 and len({(i.family, i.seed) for i in a}) == 15
    assert all(RESERVED_SEEDS[0] <= i.seed < RESERVED_SEEDS[1] for i in a)
    ev = make_suite(FAMS, 5, EVAL_SEED_BASE)
    assert not {i.seed for i in ev} & {i.seed for i in a}          # eval (model selection) vs test seeds
    assert all(0 < i.difficulty < 1 for i in a)


def test_headless_evaluation_and_paired_alignment():
    env = DrivingEnv(num_actions=5)
    suite = make_suite(FAMS, 4)
    rule_rows = evaluate_policy(env, RuleBasedPolicy(env.cfg), suite)
    rand_rows = evaluate_policy(env, RandomPolicy(5), suite)
    assert len(rule_rows) == len(rand_rows) == len(suite)
    for it, a, b in zip(suite, rule_rows, rand_rows):
        assert a["scenario_seed"] == b["scenario_seed"] == it.seed and a["family"] == b["family"] == it.family
    s = summarize(rule_rows)
    assert 0 <= s["collision_rate"] <= 1 and s["n"] == 12 and s["ci_reliable"] is False
    assert s["success_rate"] > summarize(rand_rows)["success_rate"]
    by = summarize_by_family(rule_rows)
    assert set(by) == set(FAMS) and all(v["n"] == 4 for v in by.values())
    pc = paired_comparison(rule_rows, rand_rows)
    assert pc["n"] == 12 and pc["success_only_a"] >= pc["success_only_b"]
    with pytest.raises(AssertionError):
        paired_comparison(rule_rows, rand_rows[::-1])               # misaligned rows are rejected
    assert eval_score(s) > eval_score(summarize(rand_rows))


def test_evaluation_is_reproducible():
    env = DrivingEnv()
    suite = make_suite(["multi_lane", "cut_in"], 3)
    r1 = evaluate_policy(env, RandomPolicy(5, seed=1), suite)
    r2 = evaluate_policy(DrivingEnv(), RandomPolicy(5, seed=1), suite)
    assert r1 == r2


def test_replay_reproduces_episode_exactly(tmp_path):
    env = DrivingEnv()
    rng = np.random.default_rng(3)
    env.reset(options={"family": "multi_lane", "scenario_seed": 12, "difficulty": 0.7})
    done = False
    while not done:
        _, _, te, tr, info = env.step(int(rng.integers(5)))
        done = te or tr
    rec = env.replay_record()
    assert len(rec["tick_actions"]) > 10
    assert verify_replay(rec)
    path = save_record(rec, tmp_path / "r.json")
    assert verify_replay(load_record(path))
    again = replay_episode(load_record(path))
    assert again == info["episode_metrics"] or all(again[k] == info["episode_metrics"][k] for k in again)
    bad = json.loads(json.dumps(rec))
    bad["tick_actions"] = [2] * len(bad["tick_actions"])           # different behaviour -> different outcome
    assert not verify_replay(bad)


def test_replay_of_mid_interval_collision_is_exact():
    env = DrivingEnv()
    env.reset(options={"family": "lead_constant", "scenario_seed": 2, "difficulty": 1.0})
    done = False
    while not done:
        _, _, te, tr, info = env.step(0)
        done = te or tr
    assert info["collision"]
    assert verify_replay(env.replay_record())


def test_evaluate_script_end_to_end():
    name = "pytest_tmp"
    out = ROOT / "results" / f"eval_{name}"
    shutil.rmtree(out, ignore_errors=True)
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--episodes-per-family", "3",
                            "--families", "lead_constant", "lead_brake", "--name", name, "--no-plot"],
                           capture_output=True, text=True, cwd=ROOT, timeout=300)
        assert r.returncode == 0, r.stderr
        data = json.loads((out / "evaluation.json").read_text())
        assert set(data["policies"]) == {"random", "rule_based"} and data["n_scenarios"] == 6
        assert (out / "report.md").exists() and (out / "per_episode.csv").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)
