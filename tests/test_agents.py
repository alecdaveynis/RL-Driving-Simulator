import numpy as np
import pytest
import torch

from agents.dqn import (DQNAgent, DQNConfig, DQNPolicy, EpsilonSchedule, IncompatibleCheckpointError, QNetwork,
                        load_checkpoint)
from agents.random_agent import RandomPolicy
from agents.replay_buffer import ReplayBuffer
from agents.rule_based import RuleBasedPolicy
from conftest import add_ego, add_npc, make_world
from driving_sim.environment import DrivingEnv
from driving_sim.observations import OBS_DIM, build_observation, perceive
from driving_sim.vehicles import Action


def small_agent(n=5, **kw):
    return DQNAgent(OBS_DIM, n, DQNConfig(hidden=(16, 16), batch_size=8, learning_starts=0, **kw), "cpu", seed=0)


# ------------------------------------------------------------------ network --
def test_default_architecture_is_128_128_relu():
    net = QNetwork(OBS_DIM, 5)
    layers = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert [l.out_features for l in layers] == [128, 128, 5] and layers[0].in_features == OBS_DIM
    assert sum(isinstance(m, torch.nn.ReLU) for m in net.net) == 2
    assert [l.out_features for l in QNetwork(10, 3, (32,)).net if isinstance(l, torch.nn.Linear)] == [32, 3]


# ----------------------------------------------------------- action selection --
def test_greedy_action_selection_and_masking():
    ag = small_agent()
    obs = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
    q = ag.q_values(obs)
    assert ag.select_action(obs) == int(np.argmax(q))
    mask = np.ones(5, dtype=bool)
    mask[int(np.argmax(q))] = False
    a = ag.select_action(obs, mask)
    assert a != int(np.argmax(q)) and mask[a]
    assert a == int(np.argmax(np.where(mask, q, -np.inf)))


def test_epsilon_greedy_exploration_respects_mask_and_rate():
    ag = small_agent()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.array([True, False, True, False, False])
    rng = np.random.default_rng(0)
    acts = [ag.select_action(obs, mask, epsilon=1.0, rng=rng) for _ in range(300)]
    assert set(acts) == {0, 2}                            # never an invalid action, both valid ones explored
    greedy = ag.select_action(obs, mask, 0.0)
    acts = [ag.select_action(obs, mask, epsilon=0.3, rng=rng) for _ in range(2000)]
    frac_greedy = np.mean([a == greedy for a in acts])
    assert 0.7 < frac_greedy < 0.95


def test_epsilon_schedules():
    lin = EpsilonSchedule(1.0, 0.1, 100, "linear")
    assert lin.value(0) == 1.0 and lin.value(50) == pytest.approx(0.55) and lin.value(10_000) == pytest.approx(0.1)
    exp = EpsilonSchedule(1.0, 0.1, 100, "exponential")
    assert exp.value(0) == pytest.approx(1.0) and exp.value(100) < 0.2 and exp.value(10**6) == pytest.approx(0.1)
    assert all(a >= b for a, b in zip([exp.value(i) for i in range(0, 200, 10)], [exp.value(i) for i in range(10, 210, 10)]))


# --------------------------------------------------------- double DQN targets --
def test_double_dqn_target_calculation():
    ag = DQNAgent(OBS_DIM, 3, DQNConfig(hidden=(8,), gamma=0.9), "cpu", 0)
    # make the two networks disagree so double != vanilla
    for p in ag.target.parameters():
        torch.nn.init.normal_(p, std=0.5)
    next_obs = torch.randn(6, OBS_DIM)
    r = torch.arange(6, dtype=torch.float32)
    term = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.float32)
    masks = torch.ones(6, 3, dtype=torch.bool)
    y = ag.compute_targets(r, next_obs, term, masks)
    with torch.no_grad():
        qo, qt = ag.online(next_obs), ag.target(next_obs)
    best = qo.argmax(1)
    expected = r + 0.9 * (1 - term) * qt[torch.arange(6), best]
    assert torch.allclose(y, expected)
    assert torch.all(y[term == 1] == r[term == 1])                # terminal transitions do not bootstrap
    vanilla = r + 0.9 * (1 - term) * qt.max(1).values
    assert not torch.allclose(expected, vanilla)                  # it really is double-Q, not max over target


def test_double_dqn_target_excludes_masked_next_actions():
    ag = DQNAgent(OBS_DIM, 3, DQNConfig(hidden=(8,), gamma=1.0), "cpu", 0)
    next_obs = torch.randn(4, OBS_DIM)
    with torch.no_grad():
        best_online = ag.online(next_obs).argmax(1)
    masks = torch.ones(4, 3, dtype=torch.bool)
    masks[torch.arange(4), best_online] = False                   # forbid exactly the online-greedy action
    y = ag.compute_targets(torch.zeros(4), next_obs, torch.zeros(4), masks)
    with torch.no_grad():
        qo = ag.online(next_obs).masked_fill(~masks, -1e9)
        qt = ag.target(next_obs)
    assert torch.allclose(y, qt.gather(1, qo.argmax(1, keepdim=True)).squeeze(1))


def test_learning_reduces_loss_and_updates_target():
    ag = DQNAgent(OBS_DIM, 3, DQNConfig(hidden=(32, 32), lr=1e-2, target_sync_interval=5, learning_starts=0), "cpu", 0)
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(500, OBS_DIM, 3)
    for _ in range(500):
        o = rng.random(OBS_DIM).astype(np.float32)
        a = int(rng.integers(3))
        buf.add(o, a, float(o[0] if a == 0 else -o[0]), o, True)      # terminal: target is just the reward
    losses = [ag.learn(buf.sample(64, rng))["loss"] for _ in range(300)]
    assert np.mean(losses[-20:]) < 0.3 * np.mean(losses[:20])
    t_before = [p.clone() for p in ag.target.parameters()]
    for _ in range(5):
        ag.learn(buf.sample(64, rng))
    assert any(not torch.equal(a, b) for a, b in zip(t_before, ag.target.parameters()))   # synced after 5 updates


def test_gradient_clipping_bounds_update():
    ag = DQNAgent(OBS_DIM, 3, DQNConfig(hidden=(8,), grad_clip=0.5, learning_starts=0), "cpu", 0)
    batch = {"obs": np.random.rand(8, OBS_DIM).astype(np.float32), "actions": np.zeros(8, dtype=np.int64),
             "rewards": np.full(8, 1000.0, dtype=np.float32), "next_obs": np.random.rand(8, OBS_DIM).astype(np.float32),
             "terminated": np.ones(8, dtype=np.float32), "next_masks": np.ones((8, 3), dtype=bool)}
    stats = ag.learn(batch)
    assert stats["grad_norm"] > 0.5      # raw norm reported, update itself was clipped to 0.5
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in ag.online.parameters()))
    assert float(total) <= 0.5 + 1e-4


# ------------------------------------------------------------- checkpointing --
def test_checkpoint_roundtrip_preserves_policy_and_optimizer(tmp_path):
    ag = small_agent()
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(100, OBS_DIM, 5)
    for _ in range(100):
        o = rng.random(OBS_DIM).astype(np.float32)
        buf.add(o, int(rng.integers(5)), 1.0, o, False)
    for _ in range(10):
        ag.learn(buf.sample(8, rng))
    path = tmp_path / "ck.pt"
    ag.save(path, train_state={"env_steps": 123, "epsilon": 0.2})
    ag2, ck = load_checkpoint(path, OBS_DIM, 5)
    obs = rng.random(OBS_DIM).astype(np.float32)
    assert np.allclose(ag.q_values(obs), ag2.q_values(obs))
    assert ck["train_state"]["env_steps"] == 123
    assert ag2.update_count == ag.update_count
    s1, s2 = ag.optimizer.state_dict()["state"], ag2.optimizer.state_dict()["state"]
    assert s1.keys() == s2.keys() and len(s1) > 0
    # continuing from the loaded optimizer state reproduces the same next update
    b = buf.sample(8, np.random.default_rng(5))
    l1, l2 = ag.learn(b)["loss"], ag2.learn(b)["loss"]
    assert l1 == pytest.approx(l2, rel=1e-5)


def test_incompatible_checkpoints_are_rejected(tmp_path):
    ag3 = small_agent(3)
    path = tmp_path / "a3.pt"
    ag3.save(path)
    with pytest.raises(IncompatibleCheckpointError, match="3 actions"):
        load_checkpoint(path, OBS_DIM, 5)                                      # action-space mismatch
    with pytest.raises(IncompatibleCheckpointError, match="observation size"):
        load_checkpoint(path, OBS_DIM + 1, 3)                                  # obs-dim mismatch
    ck = torch.load(path, weights_only=False)
    ck["obs_version"] = 999
    torch.save(ck, tmp_path / "old.pt")
    with pytest.raises(IncompatibleCheckpointError, match="layout"):
        load_checkpoint(tmp_path / "old.pt", OBS_DIM, 3)
    (tmp_path / "junk.pt").write_text("not a checkpoint")
    with pytest.raises(IncompatibleCheckpointError):
        load_checkpoint(tmp_path / "junk.pt", OBS_DIM, 3)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", OBS_DIM, 3)


def test_registry_loads_policy_and_enforces_action_space(tmp_path):
    from agents.registry import load_dqn_policy
    ag = small_agent(3)
    ag.save(tmp_path / "best.pt")
    pol, ck = load_dqn_policy(tmp_path / "best.pt")
    assert ck["num_actions"] == 3
    env = DrivingEnv(num_actions=3)
    obs, info = env.reset(seed=0)
    assert pol.act(obs, info["action_mask"]) in (0, 1, 2)
    with pytest.raises(IncompatibleCheckpointError):
        load_dqn_policy(tmp_path / "best.pt", num_actions=5)


def test_seeded_agents_are_reproducible():
    a, b = small_agent(), small_agent()
    for pa, pb in zip(a.online.parameters(), b.online.parameters()):
        assert torch.equal(pa, pb)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    ra, rb = np.random.default_rng(1), np.random.default_rng(1)
    assert [a.select_action(obs, None, 0.5, ra) for _ in range(20)] == [b.select_action(obs, None, 0.5, rb) for _ in range(20)]


def test_replay_buffer_wraps_and_samples():
    buf = ReplayBuffer(10, 4, 3)
    for i in range(25):
        buf.add(np.full(4, i), i % 3, float(i), np.full(4, i + 1), i == 24, np.array([True, False, True]))
    assert len(buf) == 10
    assert set(buf.rewards.astype(int)) == set(range(15, 25))     # oldest overwritten
    s = buf.sample(5, np.random.default_rng(0))
    assert s["obs"].shape == (5, 4) and s["next_masks"].shape == (5, 3)
    assert not s["next_masks"][:, 1].any()


# --------------------------------------------------------------- policies -----
def obs_for(world, cfg):
    return build_observation(perceive(world), cfg)


def test_rule_based_brakes_for_close_slow_lead_and_accelerates_on_open_road(cfg):
    pol = RuleBasedPolicy(cfg)
    w = make_world(cfg)
    add_ego(w, v=20.0)
    assert pol.act(obs_for(w, cfg), np.ones(5, bool)) == Action.ACCELERATE      # empty road, below target
    add_npc(w, 1, 4.7 / 2 + 2.25 + 8.0, 1, 8.0)                                  # 8 m gap, lead much slower
    pol.reset()
    assert pol.act(obs_for(w, cfg), np.ones(5, bool)) == Action.BRAKE


def test_rule_based_holds_speed_at_target_and_follows(cfg):
    pol = RuleBasedPolicy(cfg, target_speed=24.0)
    w = make_world(cfg)
    add_ego(w, v=24.5)
    assert pol.act(obs_for(w, cfg), np.ones(5, bool)) == Action.COAST


def test_rule_based_never_makes_invalid_or_unsafe_lane_changes(cfg):
    pol = RuleBasedPolicy(cfg, lane_cooldown=0)
    # slow lead ahead; left lane has a vehicle alongside; right lane is free
    w = make_world(cfg)
    ego = add_ego(w, lane=1, v=22.0)
    add_npc(w, 1, 70.0, 1, 12.0)            # slow lead 63 m ahead: braking not yet mandatory
    add_npc(w, 2, 0.0, 0, 22.0)             # blocks the left
    mask = np.array([True, True, True, True, True])
    assert pol.act(obs_for(w, cfg), mask) == Action.LANE_RIGHT
    # block both sides -> no lane change
    add_npc(w, 3, -1.0, 2, 22.0)
    pol.reset()
    assert pol.act(obs_for(w, cfg), mask) not in (Action.LANE_LEFT, Action.LANE_RIGHT)
    # mask forbids lane changes (3-action env or masked stage) -> never selected
    w2 = make_world(cfg)
    add_ego(w2, lane=1, v=22.0)
    add_npc(w2, 1, 70.0, 1, 12.0)
    pol.reset()
    assert pol.act(obs_for(w2, cfg), np.array([True, True, True, False, False])) not in (3, 4)
    # edge lane: cannot go off the road
    w3 = make_world(cfg)
    add_ego(w3, lane=0, v=22.0)
    add_npc(w3, 1, 70.0, 0, 12.0)
    pol.reset()
    assert pol.act(obs_for(w3, cfg), np.ones(5, bool)) != Action.LANE_LEFT


def test_rule_based_reacts_to_encroaching_cut_in(cfg):
    pol = RuleBasedPolicy(cfg)
    w = make_world(cfg)
    add_ego(w, lane=1, v=22.0)
    c = add_npc(w, 1, 4.7 / 2 + 2.25 + 14.0, 0, 14.0)
    c.vy = 1.4                                # drifting into our lane, not yet overlapping it
    assert pol.act(obs_for(w, cfg), np.ones(5, bool)) == Action.BRAKE


def test_rule_based_is_deterministic(cfg):
    env = DrivingEnv()
    acts = []
    for _ in range(2):
        pol = RuleBasedPolicy(env.cfg)
        obs, info = env.reset(options={"family": "multi_lane", "scenario_seed": 4, "difficulty": 0.5})
        pol.reset()
        seq, done = [], False
        while not done:
            a = pol.act(obs, info["action_mask"])
            seq.append(a)
            obs, _, te, tr, info = env.step(a)
            done = te or tr
        acts.append(seq)
    assert acts[0] == acts[1]


def test_random_policy_respects_mask_and_is_reproducible():
    p = RandomPolicy(5, seed=3)
    mask = np.array([True, False, True, False, False])
    p.reset(seed=10)
    a = [p.act(None, mask) for _ in range(50)]
    p.reset(seed=10)
    b = [p.act(None, mask) for _ in range(50)]
    assert a == b and set(a) <= {0, 2}
