#!/usr/bin/env python
"""Headless Double-DQN training.

Examples
  python scripts/train.py                                  # 5-action, 6-stage curriculum (configs/training.yaml)
  python scripts/train.py --config configs/training_3action.yaml
  python scripts/train.py --steps 20000 --run-name smoke   # quick smoke run
  python scripts/train.py --resume checkpoints/dqn_5action/last.pt --steps 600000
  python scripts/train.py --set dqn.lr=0.001 --set exploration.end=0.1
"""
import argparse
import copy

import _path  # noqa: F401

from agents.dqn import read_checkpoint
from agents.trainer import Trainer, apply_overrides, load_train_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--run-name", help="override run_name (checkpoint/results directory name)")
    ap.add_argument("--steps", type=int, help="override total_steps")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", help="auto | cpu | mps")
    ap.add_argument("--actions", type=int, choices=(3, 5), help="override env.num_actions")
    ap.add_argument("--resume", help="path to a last.pt/step_N.pt checkpoint to continue from")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override")
    args = ap.parse_args()

    overrides = list(args.set)
    if args.run_name:
        overrides.append(f"run_name={args.run_name}")
    if args.steps:
        overrides.append(f"total_steps={args.steps}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.device:
        overrides.append(f"device={args.device}")
    if args.actions:
        overrides.append(f"env.num_actions={args.actions}")

    if args.resume:
        # continue with the exact configuration the run was started with (plus explicit overrides)
        cfg = apply_overrides(copy.deepcopy(read_checkpoint(args.resume)["extra"]["train_config"]), overrides)
    else:
        cfg = load_train_config(args.config, overrides)

    Trainer(cfg, resume=args.resume).run_training()


if __name__ == "__main__":
    main()
