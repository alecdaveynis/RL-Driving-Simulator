#!/usr/bin/env python
"""Evaluate random, rule-based and trained DQN policies on identical (paired) held-out scenarios.

Examples
  python scripts/evaluate.py                                           # random vs rule-based
  python scripts/evaluate.py --checkpoint checkpoints/dqn_5action/best.pt
  python scripts/evaluate.py --checkpoint checkpoints/dqn_3action/best.pt --episodes-per-family 50

Outputs (results/eval_<name>/): evaluation.json, per_episode.csv, report.md, comparison.png
Only numbers produced by the episodes actually run here are reported.
"""
import argparse
import csv
import datetime
import json
import time
from pathlib import Path

import _path  # noqa: F401

from agents.dqn import IncompatibleCheckpointError, read_checkpoint
from agents.random_agent import RandomPolicy
from agents.registry import load_dqn_policy
from agents.rule_based import RuleBasedPolicy
from driving_sim.config import PROJECT_ROOT, load_sim_config
from driving_sim.environment import DrivingEnv
from driving_sim.evaluation import (TEST_SEED_BASE, eval_score, evaluate_policy, make_suite, paired_comparison,
                                    summarize, summarize_by_family)
from driving_sim.scenarios import FAMILIES

DEFAULT_FAMILIES = ["lead_constant", "lead_brake", "multi_lead", "multi_lane", "cut_in", "mixed", "unseen"]


def pct(x):
    return f"{100 * x:5.1f}%"


def ci_str(rate, ci):
    return f"{pct(rate)} [{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]"


def write_report(path: Path, data: dict) -> None:
    L = []
    L.append(f"# Evaluation report\n")
    L.append(f"- generated: {data['timestamp']}")
    L.append(f"- scenarios: {data['n_scenarios']} paired episodes ({data['episodes_per_family']} per family), "
             f"seed base {data['seed_base']}, action space: {data['num_actions']}")
    L.append(f"- every policy is run on exactly the same (family, seed, difficulty) list; "
             f"difficulty is spread evenly over (0,1) inside each family")
    L.append("- 95% Wilson intervals shown in brackets; "
             + ("n per cell >= 30 for the 'ALL' rows, smaller per-family cells are indicative only"
                if data["episodes_per_family"] < 30 else "n >= 30 per family"))
    L.append(f"- model-selection score = completion - 2*collision + 0.1*speed/target - 0.05*unsafe_events/ep\n")
    L.append("## Overall\n")
    L.append("| policy | collision [95% CI] | completion [95% CI] | dist (m) | speed (m/s) | unsafe ev/ep | hard brakes/ep | lane chg/ep | mean reward | score |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, p in data["policies"].items():
        o = p["overall"]
        L.append(f"| {name} | {ci_str(o['collision_rate'], o['collision_ci'])} | {ci_str(o['success_rate'], o['success_ci'])} | "
                 f"{o['mean_distance']:.0f} | {o['mean_speed']:.1f} | {o['unsafe_events_per_ep']:.2f} | "
                 f"{o['hard_brakes_per_ep']:.2f} | {o['lane_changes_per_ep']:.2f} | {o['mean_reward']:.1f} | {p['score']:+.3f} |")
    fams = list(next(iter(data["policies"].values()))["by_family"])
    for title, key, ckey in (("Collision rate by scenario family", "collision_rate", "collision_ci"),
                             ("Completion rate by scenario family", "success_rate", "success_ci")):
        L.append(f"\n## {title}\n")
        L.append("| family | n | " + " | ".join(data["policies"]) + " |")
        L.append("|---|---|" + "---|" * len(data["policies"]))
        for f in fams:
            n = next(iter(data["policies"].values()))["by_family"][f]["n"]
            cells = [ci_str(p["by_family"][f][key], p["by_family"][f][ckey]) for p in data["policies"].values()]
            L.append(f"| {f} | {n} | " + " | ".join(cells) + " |")
    L.append("\n## Other driving metrics by family (mean per episode)\n")
    L.append("| family | policy | speed m/s | unsafe ev | hard brakes | lane chg | stalled | timeout | reward |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for f in fams:
        for name, p in data["policies"].items():
            s = p["by_family"][f]
            L.append(f"| {f} | {name} | {s['mean_speed']:.1f} | {s['unsafe_events_per_ep']:.2f} | "
                     f"{s['hard_brakes_per_ep']:.2f} | {s['lane_changes_per_ep']:.2f} | {pct(s['stalled_rate'])} | "
                     f"{pct(s['timeout_rate'])} | {s['mean_reward']:.1f} |")
    if data.get("paired"):
        L.append("\n## Paired comparisons (same scenarios)\n")
        for k, c in data["paired"].items():
            a, b = k.split(" vs ")
            L.append(f"- **{k}** (n={c['n']}): collisions only {a}: {c['collision_only_a']}, only {b}: "
                     f"{c['collision_only_b']}, both: {c['collision_both']}; completions only {a}: "
                     f"{c['success_only_a']}, only {b}: {c['success_only_b']}, both: {c['success_both']}; "
                     f"mean reward diff {c['mean_reward_diff']:+.1f} "
                     f"(95% bootstrap CI {c['reward_diff_ci'][0]:+.1f}, {c['reward_diff_ci'][1]:+.1f})")
    L.append("\nNotes: some scenarios (e.g. tight cut-ins or hard stops at high closing speed) may be physically "
             "unavoidable for any policy, so a non-zero collision rate is expected even for a good driver.")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", default=[], help="DQN checkpoint(s) to evaluate")
    ap.add_argument("--episodes-per-family", type=int, default=40)
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES, choices=FAMILIES)
    ap.add_argument("--seed-base", type=int, default=TEST_SEED_BASE)
    ap.add_argument("--actions", type=int, choices=(3, 5), help="env action space (defaults to the checkpoint's, else 5)")
    ap.add_argument("--config", help="simulation YAML")
    ap.add_argument("--name", help="output folder name suffix (default: timestamp)")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cks = [read_checkpoint(c) for c in args.checkpoint]
    n_acts = {c["num_actions"] for c in cks}
    if len(n_acts) > 1:
        raise SystemExit(f"Checkpoints have different action spaces {sorted(n_acts)}; evaluate them in separate runs "
                         f"(a 3-action and a 5-action policy cannot share one environment).")
    num_actions = args.actions or (n_acts.pop() if n_acts else 5)

    cfg = load_sim_config(args.config)
    cfg.num_actions = num_actions
    env = DrivingEnv(cfg)
    suite = make_suite(args.families, args.episodes_per_family, args.seed_base)

    policies = []
    if not args.no_random:
        policies.append(RandomPolicy(num_actions, seed=0))
    policies.append(RuleBasedPolicy(cfg))
    for path in args.checkpoint:
        try:
            pol, _ = load_dqn_policy(path, num_actions=num_actions)
        except IncompatibleCheckpointError as exc:
            raise SystemExit(f"error: {exc}")
        policies.append(pol)

    data = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "n_scenarios": len(suite),
            "episodes_per_family": args.episodes_per_family, "seed_base": args.seed_base, "num_actions": num_actions,
            "families": args.families, "sim_fingerprint": cfg.fingerprint(),
            "checkpoints": args.checkpoint, "policies": {}}
    all_rows = {}
    for pol in policies:
        t0 = time.time()
        rows = evaluate_policy(env, pol, suite)
        name = pol.name
        while name in all_rows:
            name += "'"
        all_rows[name] = rows
        overall = summarize(rows)
        data["policies"][name] = {"overall": overall, "by_family": summarize_by_family(rows),
                                  "score": eval_score(overall, cfg.reward.target_speed)}
        print(f"{name:<32s} collision {pct(overall['collision_rate'])}  completion {pct(overall['success_rate'])}  "
              f"speed {overall['mean_speed']:.1f} m/s  unsafe/ep {overall['unsafe_events_per_ep']:.2f}  "
              f"({time.time() - t0:.0f}s)")
    names = list(all_rows)
    data["paired"] = {}
    for n in names:
        if n.startswith("dqn"):
            data["paired"][f"{n} vs rule_based"] = paired_comparison(all_rows[n], all_rows["rule_based"])
            if "random" in all_rows:
                data["paired"][f"{n} vs random"] = paired_comparison(all_rows[n], all_rows["random"])

    out = PROJECT_ROOT / "results" / ("eval_" + (args.name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evaluation.json", "w") as f:
        json.dump(data, f, indent=2)
    with open(out / "per_episode.csv", "w", newline="") as f:
        cols = ["policy"] + list(next(iter(all_rows.values()))[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for n, rows in all_rows.items():
            for r in rows:
                w.writerow({"policy": n, **r})
    write_report(out / "report.md", data)
    if not args.no_plot:
        from driving_sim.plotting import plot_comparison
        plot_comparison(data, out / "comparison.png")
    print(f"\nWrote {out}/ (evaluation.json, per_episode.csv, report.md, comparison.png)")


if __name__ == "__main__":
    main()
