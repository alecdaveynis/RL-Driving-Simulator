# RL Driving Simulator

A standalone, lightweight 2D highway driving simulator with a Gymnasium environment, a Double DQN
agent, a deterministic rule-based baseline, a reproducible evaluation harness and an interactive
Pygame app. It is independent of any other project: no CARLA, no shared code with other repositories.

* **Physics**: fixed 50 ms timestep (4 ticks per 0.2 s decision), jerk-limited acceleration, smooth
  3 s lane changes, swept-AABB collision (no tunnelling), IDM-driven traffic.
* **Environment** (`driving_sim/environment.py`): `DrivingEnv`, a Gymnasium `Env` (`check_env` passes
  for the 3- and 5-action variants).
* **Agents**: Double DQN (`agents/dqn.py`), rule-based baseline (`agents/rule_based.py`), random policy.
* **App**: manual / rule-based / DQN / scenario challenge / training results, with crash replay.

## Install

```bash
cd RL-Driving-Simulator
python3.13 -m venv .venv            # 3.10+ should work; developed on 3.13 / Apple Silicon
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Commands

```bash
python scripts/play.py                                   # launch the simulator (menu)
python scripts/play.py --mode manual                     # drive yourself
python scripts/play.py --mode rule                       # watch the rule-based controller
python scripts/play.py --mode rule --family cut_in --seed 7 --difficulty 0.8

python scripts/train.py                                  # 5-action DQN + 6-stage curriculum (configs/training.yaml)
python scripts/train.py --config configs/training_3action.yaml   # separate 3-action model
python scripts/train.py --resume checkpoints/dqn_5action/last.pt --steps 600000

python scripts/play.py --mode dqn --checkpoint checkpoints/dqn_5action/best.pt   # watch a checkpoint
python scripts/play.py --replay results/replays/<crash>.json                     # deterministic replay

python scripts/evaluate.py --checkpoint checkpoints/dqn_5action/best.pt --name dqn_5action
python -m pytest                                         # test suite
```

Training is headless and never imports pygame. Training uses the CPU by default (fastest for this
small MLP); `--device mps` selects the Apple GPU.

### Keys

| key | action |
|---|---|
| W / Up | accelerate |
| S / Down | brake (neither: coast) |
| A / Left, D / Right | lane left / right |
| Space | pause / resume |
| R | restart the same scenario |
| N | next scenario (after an episode ends) |
| 1 / 2 / 3 | manual / rule-based / DQN |
| + / - | simulation speed |
| V | toggle overlays (safety zones, trajectories, radar) |
| B | crash replay (after a collision) |
| K / U | cycle checkpoints / use an untrained network (DQN mode) |
| H | help overlay, M menu |
| Esc | back / quit |

DQN mode with no checkpoint shows a message and does not pretend: press `U` to explicitly watch an
untrained network.

## Observation (`OBS_VERSION=2`, 32 floats, all normalised)

Built only from a `Percepts` object (kinematics a real sensor could provide); no scenario type, NPC
behaviour/intent, hidden brake timers, or future information reaches the agent
(`tests/test_observations.py` checks this).

Ego speed, acceleration, lane position, lane-changing flag; for left / same / right lane and
front / rear: presence flag, gap, relative speed (absent slot = presence 0 and neutral values);
lane existence and lane-change feasibility; time-to-collision for 4 slots; and lateral-approach cues
for neighbours that may cut in. Ordering is fixed (`OBS_NAMES`).

## Reward (`configs/simulation.yaml`, decomposed in `driving_sim/rewards.py`)

Progress (speed along the road, capped by a speed limit) minus penalties for short time headway,
low TTC, hard braking, jerk, lane changes (extra for repeated ones inside a cooldown), invalid
actions; terminal collision/off-road/stall penalties and a completion bonus. Every component is
logged per step (`r_*` columns in `episodes.csv`). Anti-exploit: standing still is penalised and
terminated as `stalled`, so stopping is not a safe local optimum; lane-change oscillation is taxed;
the speed limit is enforced in the reward.

## Double DQN

MLP 128-128 ReLU (configurable), online + target networks, replay buffer (stores action masks so the
masked max for the next state is correct), epsilon-greedy, Huber loss, gradient clipping, target
sync, Double-DQN targets `r + gamma * Q_target(s', argmax_a Q_online(s', a))` with terminal-only
bootstrap cutoff (timeouts bootstrap). Checkpoints store weights, optimiser, epsilon/step state, RNG
state, config and `obs_dim`/`num_actions`/`obs_version`; loading a checkpoint with a different
observation or action space raises `IncompatibleCheckpointError` (the app shows the message instead).
3-action and 5-action models are separate runs and are mutually incompatible by design.

## Training protocol

* 6-stage curriculum (`configs/training.yaml`): constant lead -> braking lead -> multi-lead ->
  multi-lane (lane changes unmasked) -> cut-ins -> mixed. Stages advance on success/collision
  criteria (or `max_steps`); epsilon is partly restarted per stage; earlier families stay mixed in.
* **Seed discipline**: training scenarios never use seeds in [10M, 30M). Periodic evaluation (model
  selection) uses seeds from 10M; the final test uses seeds from 20M, disjoint from both. The family
  `unseen` (different traffic mix/speeds) is never trained on.
* `best.pt` is selected by a score on the held-out eval seeds, only from the **final** curriculum stage
  (so it must be good on the full mixed distribution), `last.pt` / `final.pt` are also kept.
* Outputs: `checkpoints/<run>/*.pt`, `results/<run>/{config.yaml,episodes.csv,metrics.csv,eval.csv,
  train_summary.json,training_curves.png}`.

## Evaluation results (actually executed)

`scripts/evaluate.py`: 280 paired scenarios (40 per family x 7 families, difficulty spread over
(0,1)), test seeds 20,000,000+, every policy on exactly the same scenarios, greedy DQN
(epsilon = 0), 95% Wilson intervals. Single training seed (0) per model; no repeated training runs.
Reports: `results/eval_dqn_3action/`, `results/eval_dqn_5action/` (`report.md` has the per-family
tables, hard-brake/lane-change/unsafe-following metrics, and paired comparisons).

**3-action DQN** (accelerate/coast/brake; 250k steps, `best.pt` from step 160k) - no lane changes
for any policy in this table:

| policy | collision [95% CI] | completion [95% CI] | mean speed | unsafe ev/ep | mean reward |
|---|---|---|---|---|---|
| random | 37.1% [31.7, 42.9] | 2.1% [1.0, 4.6] | 11.1 m/s | 0.49 | 16.3 |
| rule-based | 4.3% [2.5, 7.3] | 92.9% [89.2, 95.3] | 16.4 m/s | 0.20 | 180.0 |
| DQN | 1.8% [0.8, 4.1] | 95.4% [92.2, 97.3] | 16.0 m/s | 0.24 | 183.8 |

**5-action DQN** (adds lane change left/right; 450k steps, `best.pt` from step 375k):

| policy | collision [95% CI] | completion [95% CI] | mean speed | unsafe ev/ep | hard brakes/ep | lane chg/ep | mean reward |
|---|---|---|---|---|---|---|---|
| random | 42.5% [36.8, 48.4] | 2.5% [1.2, 5.1] | 11.1 m/s | 0.58 | 3.31 | 8.91 | 5.6 |
| rule-based | 4.3% [2.5, 7.3] | 95.0% [91.8, 97.0] | 20.7 m/s | 0.19 | 0.79 | 0.97 | 184.9 |
| DQN | 5.0% [3.0, 8.2] | 95.0% [91.8, 97.0] | 24.2 m/s | 0.34 | 1.36 | 2.12 | 157.5 |

### How to read these (honest summary)

* Both DQNs are far better than random (paired: DQN avoids ~100+ collisions that random has).
* **The 5-action DQN does not beat the rule-based baseline.** Collision and completion are
  statistically indistinguishable (5.0% vs 4.3%; 12 vs 12 paired wins in completion), but it drives
  faster, has more unsafe-following events and hard brakes, and a significantly lower mean reward
  (paired difference -27.4, bootstrap 95% CI [-34.9, -19.9]). Its weak spots in this run are
  `multi_lane` (12.5% collisions vs 0% for the rule driver, n=40, intervals overlap) and `mixed` (10%).
  Its advantage is the `unseen` family (7.5% vs 30% collisions) an exploratory per-family
  observation (not pre-planned), and I did not investigate why.
* The **3-action DQN's apparent edge over the rule-based driver** (1.8% vs 4.3%, paired reward diff
  +3.8, CI [-0.3, +8.5]) is not statistically significant overall; it is concentrated in `unseen`
  (10% vs 30% collisions).
* The rule-based controller is deliberately strong, so "DQN ~ rule-based" is a meaningful result, not a
  weak-baseline artefact. Its remaining failures are mostly in `unseen`.
* Some scenarios (very tight cut-ins at high closing speed, hard stops) are physically unavoidable for
  any policy, so zero collisions is not expected.
* Training-time eval (8 episodes per family) is noisy, so `best.pt` selection is noisy
  (see the eval curve in `results/<run>/training_curves.png`: the 5-action final-stage eval fluctuated
  between about 5% and 16% collisions). The final-test numbers above are independent of that selection.
* Each number is one training seed; run-to-run variance was not measured.

## Tests

`python -m pytest` (113 tests): physics and braking distances, collisions, lane-change boundaries,
observation normalisation and absence of privileged information, rewards, Gymnasium API (`check_env`),
deterministic resets, DQN action selection with masks, Double-DQN targets, checkpoint
compatibility/resume, short end-to-end training run (curriculum advance, best checkpoint, CSVs, resume
restores optimiser and counters), headless evaluation and paired alignment, Wilson intervals, replay
reproducibility, and app smoke tests (headless menu/mode/crash-replay/no-checkpoint/FPS independence).

## Not tested / limitations

* The interactive window on a real display (keyboard focus, vsync/refresh-rate behaviour, macOS window
  handling) was **not** verified: the app was driven headlessly (SDL dummy driver) and
  inspected via screenshots and event injection.
* Apple GPU (MPS) training was not verified in the development sandbox (`mps` unavailable there);
  training and all reported results used the CPU.
* One training seed per model; no hyper-parameter search; the rule-based baseline and DQN share the
  same sensors, so conclusions apply to this simulator only (simplified 2D kinematics, no sensor noise).
* The rule-based controller and the NPC model (IDM) are hand-built; real-world transfer is not claimed.
