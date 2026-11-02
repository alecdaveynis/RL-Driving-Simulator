import math

import numpy as np
import pytest

from conftest import add_ego, add_npc, make_world
from driving_sim.physics import swept_toi
from driving_sim.vehicles import Action, ConstantSpeedController


def run(world, action, n):
    for _ in range(n):
        world.step(action)


def test_acceleration_is_bounded_and_jerk_limited(cfg):
    w = make_world(cfg)
    ego = add_ego(w, v=10.0)
    prev_a, max_jerk_seen = ego.a, 0.0
    for _ in range(200):
        w.step(Action.ACCELERATE)
        max_jerk_seen = max(max_jerk_seen, abs(ego.a - prev_a) / cfg.physics.dt)
        prev_a = ego.a
        assert ego.a <= cfg.ego.max_accel + 1e-9
    assert max_jerk_seen <= cfg.ego.max_jerk + 1e-6
    assert ego.a == pytest.approx(cfg.ego.max_accel, abs=1e-6)


def test_speed_is_bounded(cfg):
    w = make_world(cfg)
    ego = add_ego(w, v=cfg.physics.max_speed - 1)
    run(w, Action.ACCELERATE, 400)
    assert ego.v <= cfg.physics.max_speed + 1e-9
    run(w, Action.BRAKE, 600)
    assert ego.v == 0.0 and ego.a == 0.0   # never reverses


def test_stopping_distance_matches_analytic_bounds(cfg):
    w = make_world(cfg)
    ego = add_ego(w, v=20.0)
    x0, peak_decel = ego.x, 0.0
    while ego.v > 0:
        w.step(Action.BRAKE)
        peak_decel = max(peak_decel, -ego.a)
    d = ego.x - x0
    ideal = 20.0 ** 2 / (2 * cfg.ego.max_brake)
    # jerk-limited ramp adds distance: at most v * (a/j) extra
    assert ideal < d < ideal + 20.0 * cfg.ego.max_brake / cfg.ego.max_jerk
    assert peak_decel <= cfg.ego.max_brake + 1e-6


def test_longer_braking_distance_at_higher_speed(cfg):
    dists = []
    for v in (10.0, 20.0, 30.0):
        w = make_world(cfg)
        ego = add_ego(w, v=v)
        while ego.v > 0:
            w.step(Action.BRAKE)
        dists.append(ego.x)
    assert dists[0] < dists[1] < dists[2]
    assert dists[2] / dists[1] > 1.8   # roughly quadratic


def test_lane_change_takes_time_and_never_teleports(cfg):
    w = make_world(cfg)
    ego = add_ego(w, lane=1)
    y0 = ego.y
    target = w.road.lane_center(0)
    w.step(Action.LANE_LEFT)
    ys, ticks = [ego.y], 1
    while ego.lc_active:
        w.step(Action.COAST)
        ys.append(ego.y)
        ticks += 1
        assert ticks < 1000
    dur = ticks * cfg.physics.dt
    assert dur == pytest.approx(cfg.physics.lane_change_duration, abs=2 * cfg.physics.dt)
    assert ys[-1] == pytest.approx(target)
    max_step = max(abs(b - a) for a, b in zip([y0] + ys, ys))
    assert max_step < 0.12 * cfg.road.lane_width     # smooth, no jumps
    assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:]))   # monotone towards the target lane
    assert w.road.lane_of(ego.y) == 0


def test_lane_change_boundaries(cfg):
    w = make_world(cfg, lanes=3)
    ego = add_ego(w, lane=0)
    assert not w.can_change_lane(ego, -1)       # left of leftmost lane
    assert w.can_change_lane(ego, +1)
    w2 = make_world(cfg, lanes=3)
    ego2 = add_ego(w2, lane=2)
    assert not w2.can_change_lane(ego2, +1)
    # attempting an illegal change does nothing
    run(w2, Action.LANE_RIGHT, 5)
    assert not ego2.lc_active and ego2.y == pytest.approx(w2.road.lane_center(2))
    # single-lane road: neither direction
    w3 = make_world(cfg, lanes=1)
    ego3 = add_ego(w3, lane=0)
    assert not w3.can_change_lane(ego3, -1) and not w3.can_change_lane(ego3, +1)


def test_no_second_lane_change_while_changing_and_min_speed(cfg):
    w = make_world(cfg)
    ego = add_ego(w, lane=1)
    w.step(Action.LANE_LEFT)
    assert ego.lc_active
    assert not w.can_change_lane(ego, +1)
    w2 = make_world(cfg)
    slow = add_ego(w2, lane=1, v=1.0)
    assert not w2.can_change_lane(slow, -1)


def test_cannot_leave_road_via_lane_changes(cfg):
    w = make_world(cfg, lanes=2)
    ego = add_ego(w, lane=0, v=15.0)
    for _ in range(600):
        w.step(Action.LANE_LEFT)
        assert not w.road.off_road(ego.y, ego.width)
    assert w.road.lane_of(ego.y) == 0


def test_off_road_detection(cfg):
    w = make_world(cfg)
    ego = add_ego(w, lane=0)
    ego.y = -1.0
    assert w.road.off_road(ego.y, ego.width)
    ev = w.step(Action.COAST)
    assert any(e.kind == "off_road" for e in ev)


def test_rear_end_collision_detected_and_wreck_stops(cfg):
    w = make_world(cfg)
    ego = add_ego(w, v=25.0)
    lead = add_npc(w, 1, 30.0, 1, 5.0)
    hit = None
    for _ in range(100):
        ev = w.step(Action.ACCELERATE)
        col = [e for e in ev if e.kind == "collision"]
        if col:
            hit = col[0]
            break
    assert hit is not None and set(hit.vids) == {0, 1}
    assert ego.crashed and lead.crashed
    assert ego.front <= lead.rear + 1e-6          # boxes touch, never interpenetrate
    run(w, Action.COAST, 80)
    assert ego.v == 0.0 and lead.v == 0.0


def test_no_tunnelling_even_with_huge_timestep(cfg):
    import copy
    cfg2 = copy.deepcopy(cfg)
    cfg2.physics.dt = 1.0                    # 25 m per tick, far larger than a car length
    w = make_world(cfg2)
    add_ego(w, v=25.0)
    add_npc(w, 1, 40.0, 1, 0.0)
    events = []
    for _ in range(5):
        events += w.step(Action.COAST)
    assert any(e.kind == "collision" for e in events)


def test_swept_toi_basic_cases():
    # moving box passes straight through a static one within a tick
    assert swept_toi((-10, 0), (10, 0), (0, 0), (0, 0), 2.5, 1.0) == pytest.approx(0.375)
    # different lanes: never meet
    assert swept_toi((-10, 0), (10, 0), (0, 3.7), (0, 3.7), 2.5, 1.0) is None
    # already overlapping
    assert swept_toi((0, 0), (1, 0), (1, 0), (1, 0), 2.5, 1.0) == 0.0
    # moving apart from touching
    assert swept_toi((-5, 0), (-6, 0), (0, 0), (0, 0), 5.0, 1.0) is None


def test_adjacent_lane_vehicles_do_not_collide(cfg):
    w = make_world(cfg)
    add_ego(w, v=25.0)
    add_npc(w, 1, 0.5, 2, 10.0)
    ev = []
    for _ in range(100):
        ev += w.step(Action.COAST)
    assert not any(e.kind == "collision" for e in ev)


def test_lane_change_into_occupied_lane_collides(cfg):
    w = make_world(cfg)
    add_ego(w, v=20.0, lane=1)
    add_npc(w, 1, 0.0, 0, 20.0, controller=ConstantSpeedController(20.0))   # alongside, left lane
    ev = []
    ev += w.step(Action.LANE_LEFT)
    for _ in range(80):
        ev += w.step(Action.COAST)
    assert any(e.kind == "collision" for e in ev)


def test_world_is_deterministic(cfg):
    from driving_sim.scenarios import build_world, generate_scenario
    spec = generate_scenario("mixed", 123, 0.7)
    traces = []
    for _ in range(2):
        w = build_world(spec, cfg)
        for t in range(400):
            w.step(Action.ACCELERATE if t % 50 < 30 else Action.BRAKE)
        traces.append([(v.x, v.y, v.v, v.crashed) for v in w.vehicles])
    assert traces[0] == traces[1]


def test_physics_independent_of_render_rate(cfg):
    """The world only advances through step(); render FPS cannot alter trajectories."""
    from driving_sim.scenarios import build_world, generate_scenario
    spec = generate_scenario("lead_brake", 5, 0.5)
    a, b = build_world(spec, cfg), build_world(spec, cfg)
    for _ in range(300):
        a.step(Action.COAST)
    for _ in range(300):
        b.snapshot()         # "rendering" in between must not change anything
        b.step(Action.COAST)
    assert [(v.x, v.v) for v in a.vehicles] == [(v.x, v.v) for v in b.vehicles]
