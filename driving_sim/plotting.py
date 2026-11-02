"""Matplotlib figures for training runs and policy comparisons (headless-safe)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def read_csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def _rolling(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = max(1, min(w, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def plot_training_curves(run_dir: str | Path, out: Optional[str | Path] = None) -> Optional[Path]:
    """Six-panel summary of a training run: reward, loss, epsilon, safety rates, eval score, eval by family."""
    run_dir = Path(run_dir)
    eps = read_csv(run_dir / "episodes.csv")
    met = read_csv(run_dir / "metrics.csv")
    ev = read_csv(run_dir / "eval.csv")
    if not eps and not met:
        return None
    plt = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(f"Training run: {run_dir.name}", fontsize=14, fontweight="bold")

    stage_changes = []
    if eps:
        steps = np.array([e["step"] for e in eps])
        stages = [e.get("stage", "") for e in eps]
        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                stage_changes.append((steps[i], stages[i]))

    def mark_stages(ax):
        for s, name in stage_changes:
            ax.axvline(s, color="gray", ls=":", lw=0.8)

    ax = axes[0, 0]
    if eps:
        rew = np.array([e["reward"] for e in eps])
        ax.plot(steps, rew, alpha=0.25, color="tab:blue", lw=0.6)
        ax.plot(steps, _rolling(rew, 50), color="tab:blue", label="episode reward (50-ep avg)")
        mark_stages(ax)
    ax.set_title("Episode reward (training, exploring)")
    ax.set_xlabel("env steps")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if met:
        ax.plot([m["step"] for m in met], [m["loss"] for m in met], color="tab:red")
        ax.set_yscale("log")
    ax.set_title("TD loss (Huber)")
    ax.set_xlabel("env steps")

    ax = axes[0, 2]
    if met:
        ax.plot([m["step"] for m in met], [m["epsilon"] for m in met], color="tab:green")
        mark_stages(ax)
    ax.set_title("Exploration epsilon (stage restarts visible)")
    ax.set_xlabel("env steps")

    ax = axes[1, 0]
    if eps:
        col = np.array([e["collision"] for e in eps])
        suc = np.array([e["success"] for e in eps])
        ax.plot(steps, _rolling(col, 50), color="tab:red", label="collision rate")
        ax.plot(steps, _rolling(suc, 50), color="tab:green", label="completion rate")
        mark_stages(ax)
        ax.set_ylim(-0.02, 1.02)
    ax.set_title("Training collision / completion rate (50-ep rolling)")
    ax.set_xlabel("env steps")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if ev:
        es = [e["step"] for e in ev]
        ax.plot(es, [e["collision_rate"] for e in ev], "o-", color="tab:red", label="collision")
        ax.plot(es, [e["success_rate"] for e in ev], "o-", color="tab:green", label="completion")
        ax2 = ax.twinx()
        ax2.plot(es, [e["score"] for e in ev], "s--", color="tab:purple", label="eval score")
        ax2.set_ylabel("eval score", color="tab:purple")
        best = [e for e in ev if e.get("is_best") == 1.0]
        if best:
            ax2.scatter([b["step"] for b in best], [b["score"] for b in best], marker="*", s=160, color="gold",
                        edgecolor="black", zorder=5, label="best.pt")
        ax.set_ylim(-0.02, 1.02)
        mark_stages(ax)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax.set_title("Held-out evaluation (greedy, fixed seeds)")
    ax.set_xlabel("env steps")

    ax = axes[1, 2]
    if ev:
        last = ev[-1]
        fams = sorted({k[len("success_"):] for k in last if k.startswith("success_") and k != "success_rate"})
        x = np.arange(len(fams))
        ax.bar(x - 0.2, [last[f"success_{f}"] for f in fams], 0.4, color="tab:green", label="completion")
        ax.bar(x + 0.2, [last[f"collision_{f}"] for f in fams], 0.4, color="tab:red", label="collision")
        ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        ax.set_title(f"Latest eval by scenario family (step {int(last['step'])})")
    else:
        ax.set_title("No evaluations yet")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out) if out else run_dir / "training_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_comparison(summary: dict, out: str | Path) -> Path:
    """Bar charts comparing policies per family. `summary` is the evaluation JSON structure."""
    plt = _mpl()
    policies = list(summary["policies"])
    fams = list(summary["policies"][policies[0]]["by_family"])
    fams_all = fams + ["ALL"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = {"random": "tab:gray", "rule_based": "tab:orange"}
    width = 0.8 / len(policies)
    for ax, key, title in zip(axes, ("success_rate", "collision_rate", "mean_speed"),
                              ("Completion rate", "Collision rate", "Mean speed [m/s]")):
        for i, p in enumerate(policies):
            vals, lo, hi = [], [], []
            for f in fams_all:
                s = summary["policies"][p]["overall"] if f == "ALL" else summary["policies"][p]["by_family"][f]
                vals.append(s[key])
                if key in ("success_rate", "collision_rate"):
                    ci = s[key.replace("_rate", "_ci")]
                    lo.append(max(0.0, s[key] - ci[0]))
                    hi.append(max(0.0, ci[1] - s[key]))
            x = np.arange(len(fams_all)) + (i - (len(policies) - 1) / 2) * width
            err = [lo, hi] if lo else None
            ax.bar(x, vals, width, label=p, color=colors.get(p, f"C{i + 2}"), yerr=err, capsize=2,
                   error_kw={"lw": 0.8})
        ax.set_xticks(np.arange(len(fams_all)))
        ax.set_xticklabels(fams_all, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        if key != "mean_speed":
            ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle("Policy comparison on identical held-out scenarios (95% Wilson CIs)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
