import json

import pytest

from conftest import make_world
from driving_sim.environment import RESERVED_SEEDS, DrivingEnv, ScenarioSampler
from driving_sim.scenarios import FAMILIES, ScenarioSpec, build_world, generate_scenario
from driving_sim.vehicles import Action

import numpy as np


@pytest.mark.parametrize("family", FAMILIES)
def test_generation_is_deterministic_and_serialisable(family):
    a = generate_scenario(family, 42, 0.6)
    b = generate_scenario(family, 42, 0.6)
    assert a.to_dict() == b.to_dict()
    c = ScenarioSpec.from_dict(json.loads(json.dumps(a.to_dict())))
    assert c.to_dict() == a.to_dict()
    assert generate_scenario(family, 43, 0.6).to_dict() != a.to_dict()


@pytest.mark.parametrize("family", FAMILIES)
def test_initial_placement_has_no_overlaps(family, cfg):
    for seed in range(40):
        spec = generate_scenario(family, seed, (seed % 10) / 9)
        w = build_world(spec, cfg)
        vs = w.vehicles
        for i, a in enumerate(vs):
            for b in vs[i + 1:]:
                overlap_x = abs(a.x - b.x) < (a.length + b.length) / 2
                overlap_y = abs(a.y - b.y) < (a.width + b.width) / 2
                assert not (overlap_x and overlap_y), (family, seed, a.label, b.label)
        assert 0 <= spec.ego_lane < spec.num_lanes
        assert all(0 <= n.lane < spec.num_lanes for n in spec.npcs)


def test_stage1_has_single_constant_lead():
    spec = generate_scenario("lead_constant", 1, 0.5)
    assert len(spec.npcs) == 1 and spec.npcs[0].behavior == "constant"


def test_lead_brake_scenario_actually_brakes(cfg):
    spec = generate_scenario("lead_brake", 3, 0.5)
    w = build_world(spec, cfg)
    lead = w.npcs[0]
    speeds = []
    for _ in range(1200):
        w.ego.v = 0.0   # park the ego far behind so only the lead's own behaviour shows
        w.ego.x = -1000.0
        w.step(Action.COAST)
        speeds.append(lead.v)
    v0 = spec.npcs[0].speed
    assert min(speeds) < 0.7 * v0


def test_cut_in_vehicle_merges_into_ego_lane(cfg):
    spec = generate_scenario("cut_in", 4, 0.8)
    w = build_world(spec, cfg)
    cutters = [v for v in w.npcs if v.label == "cutter"]
    assert cutters
    merged = False
    for _ in range(1500):
        ego = w.ego
        ego.v = 22.0   # keep closing so the trigger gap is reached
        w.step(Action.COAST)
        if any(c.lc_active for c in cutters):
            merged = True
        if any(e.kind == "collision" for e in w.events):
            break
    assert merged


def test_multi_lane_family_has_traffic_in_other_lanes():
    spec = generate_scenario("multi_lane", 7, 0.5)
    lanes = {n.lane for n in spec.npcs}
    assert len(lanes) >= 3


def test_unseen_family_is_out_of_training_distribution():
    kinds = {generate_scenario("unseen", s, 0.5).sub_family for s in range(30)}
    assert kinds == {"stop_and_go", "aggressive_cutin", "dense_random"}


def test_sampler_never_returns_reserved_eval_seeds():
    s = ScenarioSampler({"mixed": 1.0})
    rng = np.random.default_rng(0)
    # force draws into the reserved window by monkeypatching the generator output
    class R:
        def choice(self, n, p=None): return 0
        def integers(self, lo, hi=None): return RESERVED_SEEDS[0] + 5
        def uniform(self, a, b): return 0.5
    _, seed, _ = s.sample(R())
    assert not (RESERVED_SEEDS[0] <= seed < RESERVED_SEEDS[1])
