import copy

import numpy as np
import pytest

from conftest import add_ego, add_npc, make_world
from driving_sim.observations import (OBS_DIM, OBS_NAMES, LAT_SPEED_SCALE, Percepts, build_observation,
                                      decode_observation, observation_space, perceive)
from driving_sim.scenarios import build_world, generate_scenario
from driving_sim.vehicles import Action

IDX = {n: i for i, n in enumerate(OBS_NAMES)}


def obs_of(world, cfg):
    return build_observation(perceive(world), cfg)


def test_shape_dtype_and_bounds(cfg):
    for fam in ("lead_brake", "multi_lane", "cut_in", "unseen"):
        w = build_world(generate_scenario(fam, 1, 0.7), cfg)
        for t in range(300):
            w.step(Action.ACCELERATE if t % 40 < 20 else Action.BRAKE)
            o = obs_of(w, cfg)
            assert o.shape == (OBS_DIM,) and o.dtype == np.float32
            assert observation_space().contains(o)
            assert np.all(np.isfinite(o))


def test_missing_vehicles_use_explicit_presence_flags(cfg):
    w = make_world(cfg)
    add_ego(w, lane=1)
    o = obs_of(w, cfg)
    for slot in ("left", "same", "right"):
        for side in ("front", "rear"):
            assert o[IDX[f"{slot}_{side}_present"]] == 0.0
            assert o[IDX[f"{slot}_{side}_gap"]] == 1.0       # "beyond range", disambiguated by the flag
            assert o[IDX[f"{slot}_{side}_rel_speed"]] == 0.0
    assert o[IDX["ttc_same_front"]] == 1.0
    # a present vehicle at distance ~range edge has gap<1 and present=1
    add_npc(w, 1, 50.0, 1, 10.0)
    o = obs_of(w, cfg)
    assert o[IDX["same_front_present"]] == 1.0
    assert 0.0 < o[IDX["same_front_gap"]] < 1.0


def test_nonexistent_lane_flags(cfg):
    w = make_world(cfg, lanes=3)
    add_ego(w, lane=0)
    o = obs_of(w, cfg)
    assert o[IDX["left_exists"]] == 0.0 and o[IDX["right_exists"]] == 1.0
    assert o[IDX["can_change_left"]] == 0.0 and o[IDX["can_change_right"]] == 1.0


def test_normalisation_values(cfg):
    w = make_world(cfg)
    ego = add_ego(w, lane=1, v=18.0)
    add_npc(w, 1, 4.7 / 2 + 4.5 / 2 + 30.0, 1, 12.0)   # 30 m bumper gap, slower lead
    o = obs_of(w, cfg)
    assert o[IDX["ego_speed"]] == pytest.approx(18.0 / cfg.physics.max_speed)
    assert o[IDX["same_front_gap"]] == pytest.approx(30.0 / cfg.sensors.range, abs=1e-5)
    assert o[IDX["same_front_rel_speed"]] == pytest.approx(-6.0 / cfg.sensors.rel_speed_scale, abs=1e-5)
    assert o[IDX["ttc_same_front"]] == pytest.approx((30.0 / 6.0) / cfg.sensors.ttc_max, abs=1e-5)
    d = decode_observation(o, cfg)
    assert d["same_front"][0] == pytest.approx(30.0, abs=1e-3)
    assert d["same_front"][1] == pytest.approx(-6.0, abs=1e-3)
    assert d["speed"] == pytest.approx(18.0, abs=1e-4)


def test_stable_ordering_of_surrounding_vehicles(cfg):
    """Slot layout is fixed: reordering the vehicle list must not change the observation."""
    w = make_world(cfg)
    add_ego(w, lane=1)
    for i, (x, lane, v) in enumerate([(40, 1, 10), (-30, 1, 22), (25, 0, 15), (-15, 2, 20), (60, 2, 12)], start=1):
        add_npc(w, i, x, lane, v)
    o1 = obs_of(w, cfg)
    w.vehicles = [w.vehicles[0]] + list(reversed(w.vehicles[1:]))
    o2 = obs_of(w, cfg)
    assert np.array_equal(o1, o2)
    # nearest in each slot wins, regardless of insertion order
    add_npc(w, 9, 20.0, 1, 5.0)
    o3 = obs_of(w, cfg)
    assert o3[IDX["same_front_gap"]] < o1[IDX["same_front_gap"]]


def test_adjacent_lane_occupancy_and_relative_speed(cfg):
    w = make_world(cfg)
    add_ego(w, lane=1, v=20.0)
    add_npc(w, 1, 30.0, 0, 26.0)
    add_npc(w, 2, -25.0, 2, 24.0)
    o = obs_of(w, cfg)
    assert o[IDX["left_front_present"]] == 1.0 and o[IDX["left_front_rel_speed"]] > 0
    assert o[IDX["right_rear_present"]] == 1.0 and o[IDX["right_rear_rel_speed"]] > 0
    assert o[IDX["left_rear_present"]] == 0.0 and o[IDX["right_front_present"]] == 0.0


def test_lateral_cut_in_cue_sign(cfg):
    w = make_world(cfg)
    add_ego(w, lane=1, v=20.0)
    c = add_npc(w, 1, 25.0, 0, 15.0)    # left lane, will move right (towards ego lane)
    c.vy = +1.5
    o = obs_of(w, cfg)
    assert o[IDX["left_front_lat_toward"]] == pytest.approx(1.5 / LAT_SPEED_SCALE)
    c.vy = -1.5                          # moving away
    assert obs_of(w, cfg)[IDX["left_front_lat_toward"]] < 0
    w2 = make_world(cfg)
    add_ego(w2, lane=1)
    r = add_npc(w2, 1, 25.0, 2, 15.0)
    r.vy = -1.0                          # right lane moving left = towards ego
    assert obs_of(w2, cfg)[IDX["right_front_lat_toward"]] > 0


def test_sensor_range_limits_information(cfg):
    w = make_world(cfg)
    add_ego(w, lane=1)
    far = add_npc(w, 1, 400.0, 1, 5.0)
    o_far = obs_of(w, cfg)
    far.v = 30.0
    assert np.array_equal(o_far, obs_of(w, cfg))
    assert o_far[IDX["same_front_present"]] == 0.0


def test_no_privileged_information_in_observation(cfg):
    """Hidden NPC intent (scripted brake/cut-in schedules, controller params) must not leak."""
    spec = generate_scenario("cut_in", 11, 0.8)
    w1 = build_world(spec, cfg)
    spec2 = copy.deepcopy(spec)
    for n in spec2.npcs:                        # change only *hidden* behaviour parameters
        n.params = {**n.params, "trigger_gap": 1.0, "t_force": 999.0, "T": 3.0, "events": []}
    w2 = build_world(spec2, cfg)
    assert np.array_equal(obs_of(w1, cfg), obs_of(w2, cfg))       # identical kinematic state => identical obs
    # Percepts only carries kinematics: no ids, controllers, or future information
    fields = set(Percepts.__dataclass_fields__)
    assert not {"controller", "events", "vid", "params", "plan"} & fields
    from driving_sim.observations import OtherPercept
    assert set(OtherPercept.__dataclass_fields__) == {"dx", "y", "v", "vy", "length", "width"}


def test_observation_is_a_function_of_percepts_only(cfg):
    w = build_world(generate_scenario("multi_lane", 2, 0.5), cfg)
    p = perceive(w)
    for _ in range(20):
        w.step(Action.ACCELERATE)           # world moves on...
    assert np.array_equal(build_observation(p, cfg), build_observation(p, cfg))   # ...p is a frozen snapshot
