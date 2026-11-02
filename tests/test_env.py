import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from driving_sim.environment import DrivingEnv
from driving_sim.observations import OBS_DIM
from driving_sim.scenarios import FAMILIES, generate_scenario
from driving_sim.vehicles import Action


@pytest.mark.parametrize("n", [3, 5])
def test_gymnasium_env_checker(n):
    env = DrivingEnv(num_actions=n)
    check_env(env, skip_render_check=True)


def test_spaces():
    env = DrivingEnv(num_actions=5)
    assert env.action_space.n == 5
    assert env.observation_space.shape == (OBS_DIM,)
    assert DrivingEnv(num_actions=3).action_space.n == 3
    with pytest.raises(ValueError):
        DrivingEnv(num_actions=4)


def test_seeded_reset_is_deterministic():
    def rollout(seed):
        env = DrivingEnv()
        obs, info = env.reset(seed=seed)
        out = [obs.copy()]
        rng = np.random.default_rng(0)
        for _ in range(60):
            obs, r, te, tr, info = env.step(int(rng.integers(5)))
            out.append((obs.copy(), r, te, tr))
            if te or tr:
                break
        return info["scenario_seed"], out

    s1, a = rollout(7)
    s2, b = rollout(7)
    s3, c = rollout(8)
    assert s1 == s2 and s1 != s3
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, tuple):
            assert np.array_equal(x[0], y[0]) and x[1:] == y[1:]
        else:
            assert np.array_equal(x, y)


def test_reset_with_explicit_scenario_matches_family_options():
    env = DrivingEnv()
    o1, i1 = env.reset(options={"family": "cut_in", "scenario_seed": 9, "difficulty": 0.4})
    spec = generate_scenario("cut_in", 9, 0.4)
    o2, i2 = env.reset(options={"scenario": spec})
    assert np.array_equal(o1, o2)
    assert i1["scenario_seed"] == i2["scenario_seed"] == 9


@pytest.mark.parametrize("n", [3, 5])
def test_random_policy_completes_episodes_in_every_family(n):
    env = DrivingEnv(num_actions=n)
    rng = np.random.default_rng(1)
    for fam in FAMILIES:
        for seed in range(3):
            obs, info = env.reset(seed=seed, options={"family": fam, "scenario_seed": seed, "difficulty": 0.8})
            done, steps = False, 0
            while not done:
                a = int(rng.integers(n))
                obs, r, te, tr, info = env.step(a)
                assert np.isfinite(r) and np.all(np.isfinite(obs))
                done = te or tr
                steps += 1
                assert steps <= env.cfg.episode.max_steps
            m = info["episode_metrics"]
            assert m["steps"] == steps


def test_terminated_vs_truncated_semantics():
    env = DrivingEnv()
    # truncation: coast along an empty-ish road at modest speed until max_steps
    env.cfg.episode.max_steps = 5
    obs, info = env.reset(options={"family": "lead_constant", "scenario_seed": 1, "difficulty": 0.0})
    for i in range(5):
        obs, r, te, tr, info = env.step(Action.COAST)
    assert tr and not te
    # collision => terminated
    env2 = DrivingEnv()
    env2.reset(options={"family": "lead_constant", "scenario_seed": 2, "difficulty": 1.0})
    te = False
    for _ in range(400):
        _, _, te, tr, info = env2.step(Action.ACCELERATE)
        if te or tr:
            break
    assert te and info["collision"] and info["episode_metrics"]["collision"] == 1
    with pytest.raises(RuntimeError):
        env2.step(Action.COAST)


def test_goal_completion_terminates_with_success_and_bonus():
    env = DrivingEnv()
    env.cfg.episode.goal_distance = 150.0
    spec = generate_scenario("lead_constant", 5, 0.0, goal_distance=150.0)
    spec.npcs = []
    env.reset(options={"scenario": spec})
    done = False
    while not done:
        _, r, te, tr, info = env.step(Action.ACCELERATE)
        done = te or tr
    assert info["success"] and info["episode_metrics"]["success"] == 1
    assert info["reward_components"]["completion"] == env.cfg.reward.completion


def test_stalling_is_detected_and_penalised():
    env = DrivingEnv()
    spec = generate_scenario("lead_constant", 5, 0.0)
    spec.npcs = []
    spec.ego_speed = 0.0
    env.reset(options={"scenario": spec})
    for _ in range(200):
        _, r, te, tr, info = env.step(Action.BRAKE)
        if te:
            break
    assert te and info["stalled"]
    assert info["episode_metrics"]["distance"] < 1.0
    assert info["reward_components"]["stalled"] == env.cfg.reward.stalled


def test_action_mask_and_invalid_actions():
    env = DrivingEnv(num_actions=5)
    spec = generate_scenario("lead_constant", 1, 0.5)
    spec.ego_lane = 0
    env.reset(options={"scenario": spec})
    mask = env.action_mask()
    assert mask.tolist() == [True, True, True, False, True]      # leftmost lane: cannot go left
    _, r, te, tr, info = env.step(Action.LANE_LEFT)               # invalid -> coasts, flagged, penalised
    assert info["invalid_action"] and info["reward_components"]["invalid_action"] < 0
    assert not env.world.ego.lc_active
    # curriculum masking
    env.set_lane_changes_enabled(False)
    assert env.action_mask().tolist() == [True, True, True, False, False]
    # 3-action env mask has 3 entries
    assert DrivingEnv(num_actions=3).reset()[1]["action_mask"].shape == (3,)


def test_lane_change_action_executes_over_time_in_env():
    env = DrivingEnv()
    spec = generate_scenario("lead_constant", 1, 0.5)
    spec.npcs = []
    env.reset(options={"scenario": spec})
    lane0 = env.world.road.lane_of(env.world.ego.y)
    _, _, _, _, info = env.step(Action.LANE_LEFT)
    assert info["lane_change_started"] and env.world.ego.lc_active
    assert env.world.road.lane_of(env.world.ego.y) == lane0         # not teleported
    for _ in range(25):
        env.step(Action.COAST)
    assert env.world.road.lane_of(env.world.ego.y) == lane0 - 1 and not env.world.ego.lc_active


def test_reward_components_logged_and_metrics_independent_of_reward():
    env = DrivingEnv()
    env.reset(seed=3)
    done = False
    while not done:
        _, r, te, tr, info = env.step(Action.COAST)
        assert abs(sum(info["reward_components"].values()) - r) < 1e-9
        done = te or tr
    m = info["episode_metrics"]
    assert m["reward"] == pytest.approx(sum(v for k, v in m.items() if k.startswith("r_")), abs=1e-2)
    for key in ("distance", "mean_speed", "unsafe_following_events", "hard_brake_events", "lane_changes"):
        assert key in m


def test_hard_braking_and_unsafe_following_metrics():
    env = DrivingEnv()
    spec = generate_scenario("lead_constant", 1, 0.0)
    spec.npcs[0].dx = 4.7 / 2 + 4.5 / 2 + 12.0     # 12 m gap
    spec.npcs[0].speed = 10.0
    spec.npcs[0].params["v0"] = 10.0
    spec.ego_speed = 20.0
    env.reset(options={"scenario": spec})
    for _ in range(30):
        _, _, te, tr, info = env.step(Action.BRAKE)
        if te or tr:
            break
    m = env.metrics
    assert m.hard_brake_events >= 1
    assert m.unsafe_following_events >= 1 or m.collision


def test_render_rgb_array_headless():
    env = DrivingEnv(render_mode="rgb_array")
    env.reset(seed=1)
    for _ in range(5):
        env.step(Action.ACCELERATE)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.shape[0] > 100 and frame.dtype == np.uint8
    env.close()
