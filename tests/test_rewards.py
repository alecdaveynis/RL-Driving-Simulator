import pytest

from driving_sim.config import RewardCfg
from driving_sim.rewards import REWARD_COMPONENTS, StepSignals, compute_reward, time_headway, time_to_collision

CFG = RewardCfg()


def sig(**kw):
    base = dict(speed=25.0, accel=0.0, prev_accel=0.0, peak_decel=0.0, decision_dt=0.2, max_jerk=8.0)
    base.update(kw)
    return StepSignals(**base)


def test_total_equals_sum_of_logged_components():
    total, comps = compute_reward(sig(headway=0.5, ttc=1.0, peak_decel=5.0, lane_change_started=True,
                                      steps_since_lane_change=3, invalid_action=True), CFG)
    assert set(comps) == set(REWARD_COMPONENTS)
    assert total == pytest.approx(sum(comps.values()))


def test_progress_capped_at_target_speed_no_reward_for_speeding():
    _, at_target = compute_reward(sig(speed=25.0), CFG)
    _, faster = compute_reward(sig(speed=33.0), CFG)
    assert at_target["progress"] == pytest.approx(1.0)
    assert faster["progress"] == pytest.approx(1.0)
    assert faster["speed_limit"] < 0                    # and speeding above the limit is penalised
    assert compute_reward(sig(speed=33.0), CFG)[0] < compute_reward(sig(speed=25.0), CFG)[0]


def test_standing_still_earns_nothing_and_crawling_earns_little():
    t0, c0 = compute_reward(sig(speed=0.0), CFG)
    assert c0["progress"] == 0.0 and t0 <= 0.0
    t5, _ = compute_reward(sig(speed=2.5), CFG)
    t25, _ = compute_reward(sig(speed=25.0), CFG)
    assert 0 < t5 < 0.2 * t25 + 1e-9


def test_stall_termination_is_penalised():
    total, c = compute_reward(sig(speed=0.0, stalled=True), CFG)
    assert c["stalled"] == CFG.stalled and total < 0


def test_following_distance_penalties():
    _, safe = compute_reward(sig(headway=2.0, ttc=8.0), CFG)
    _, tight = compute_reward(sig(headway=0.6, ttc=2.0), CFG)
    assert safe["headway"] == 0.0 and safe["ttc"] == 0.0
    assert tight["headway"] < -0.3 and tight["ttc"] < -0.2


def test_braking_and_jerk_penalties():
    _, soft = compute_reward(sig(peak_decel=2.0), CFG)
    _, hard = compute_reward(sig(peak_decel=6.0), CFG)
    assert soft["brake"] == 0.0 and hard["brake"] == pytest.approx(-CFG.w_brake)
    _, smooth = compute_reward(sig(accel=1.0, prev_accel=1.0), CFG)
    _, jerky = compute_reward(sig(accel=1.0, prev_accel=-0.6), CFG)
    assert smooth["jerk"] == 0.0 and jerky["jerk"] == pytest.approx(-CFG.w_jerk)


def test_lane_change_costs_and_repeat_penalty():
    _, none = compute_reward(sig(), CFG)
    _, one = compute_reward(sig(lane_change_started=True), CFG)
    _, repeat = compute_reward(sig(lane_change_started=True, steps_since_lane_change=5), CFG)
    assert none["lane_change"] == 0.0 and one["lane_change"] == -CFG.w_lane_change
    assert one["repeat_lane_change"] == 0.0 and repeat["repeat_lane_change"] == -CFG.w_repeat_lane_change


def test_terminal_events_dominate_per_step_gain():
    total_c, c = compute_reward(sig(collided=True), CFG)
    assert c["collision"] == CFG.collision and total_c < -40
    assert abs(CFG.collision) > 40 * CFG.w_progress           # one crash wipes out ~40 steps of perfect driving
    total_o, o = compute_reward(sig(off_road=True), CFG)
    assert o["off_road"] == CFG.off_road
    _, g = compute_reward(sig(goal=True), CFG)
    assert g["completion"] == CFG.completion
    _, g2 = compute_reward(sig(goal=True, collided=True), CFG)
    assert g2["completion"] == 0.0                              # cannot "complete" while crashing


def test_helpers():
    assert time_headway(30.0, 15.0) == pytest.approx(2.0)
    assert time_headway(30.0, 0.5) is None and time_headway(float("inf"), 20.0) is None
    assert time_to_collision(20.0, 20.0, 10.0) == pytest.approx(2.0)
    assert time_to_collision(20.0, 10.0, 20.0) is None
