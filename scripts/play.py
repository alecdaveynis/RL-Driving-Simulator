#!/usr/bin/env python
"""Launch the interactive simulator.

Examples
  python scripts/play.py                              # menu (manual / rule-based / DQN / challenge / results)
  python scripts/play.py --mode manual                # drive yourself, straight into the sim
  python scripts/play.py --mode rule                  # watch the rule-based controller
  python scripts/play.py --mode dqn --checkpoint checkpoints/dqn_5action/best.pt
  python scripts/play.py --mode rule --family cut_in --seed 7 --difficulty 0.8
  python scripts/play.py --replay results/replays/crash_xxx.json     # deterministic replay of a saved episode
"""
import argparse
import json

import _path  # noqa: F401

from driving_sim.config import load_sim_config
from driving_sim.scenarios import FAMILIES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("manual", "rule", "dqn"), help="skip the menu and start in this mode")
    ap.add_argument("--checkpoint", help="DQN checkpoint (.pt) for DQN mode")
    ap.add_argument("--family", default="mixed", choices=FAMILIES)
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1, help="scenario seed (episode n uses seed+n-1)")
    ap.add_argument("--fixed", action="store_true", help="restart the same seed after each episode")
    ap.add_argument("--actions", type=int, choices=(3, 5), help="action-space size (DQN checkpoints override this)")
    ap.add_argument("--epsilon", type=float, default=0.0, help="exploration epsilon during DQN playback")
    ap.add_argument("--config", help="simulation YAML (default configs/simulation.yaml)")
    ap.add_argument("--replay", help="saved replay JSON (from results/replays) to re-simulate deterministically")
    ap.add_argument("--headless", action="store_true", help="no window (for smoke tests)")
    ap.add_argument("--frames", type=int, help="exit after N frames (smoke tests)")
    ap.add_argument("--screenshot", help="save a screenshot on exit")
    args = ap.parse_args()

    from app.main import App

    cfg = load_sim_config(args.config)
    app = App(cfg, headless=args.headless, mode=args.mode or "rule", family=args.family,
              difficulty=args.difficulty, seed=args.seed, checkpoint=args.checkpoint,
              start_menu=(args.mode is None and not args.replay), epsilon=args.epsilon, num_actions=args.actions)
    if args.fixed:
        app.fixed_scenario = True
    if args.replay:
        with open(args.replay) as f:
            app.screen = "sim"
            app.load_recording(json.load(f))
    app.run(max_frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
