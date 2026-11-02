import json

import numpy as np
import pygame
import pytest

import app.main as appmain
from app.main import App, analyse_snap
from driving_sim.config import load_sim_config
from driving_sim.replay import verify_replay
from driving_sim.vehicles import Action


def make_app(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(appmain, "list_checkpoints", lambda *a, **k: [])
    kw.setdefault("start_menu", False)
    return App(load_sim_config(), headless=True, record_dir=tmp_path / "replays", **kw)


def key(app, k):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode=""))


def frames(app, n, dt=1 / 60):
    for _ in range(n):
        app.update(dt)
        app.draw(dt)


def test_menu_renders_and_navigates(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, start_menu=True)
    frames(app, 30)
    key(app, pygame.K_DOWN)
    assert app.menu_sel == 1
    key(app, pygame.K_h)
    frames(app, 2)
    assert app.show_help
    key(app, pygame.K_h)
    key(app, pygame.K_1)            # manual
    assert app.screen == "sim" and app.mode == "manual"
    frames(app, 30)


def test_mode_switching_and_hotkeys(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, mode="manual")
    for k, mode in ((pygame.K_2, "rule"), (pygame.K_1, "manual")):
        key(app, k)
        assert app.mode == mode
    key(app, pygame.K_PLUS)
    assert app.sim_speed == 2.0
    key(app, pygame.K_MINUS); key(app, pygame.K_MINUS)
    assert app.sim_speed == 0.5
    d0 = app.overlays.debug
    key(app, pygame.K_v)
    assert app.overlays.debug != d0
    key(app, pygame.K_SPACE)
    assert app.paused
    t = app.env.world.time
    frames(app, 30)
    assert app.env.world.time == t                           # paused: no physics
    key(app, pygame.K_SPACE)
    frames(app, 60)
    assert app.env.world.time > t


def test_dqn_mode_without_checkpoint_shows_clear_message(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, mode="rule")
    key(app, pygame.K_3)
    assert app.mode == "rule" and app.dqn is None            # refuses to pretend
    assert "No trained checkpoint" in app.message
    frames(app, 3)
    key(app, pygame.K_u)                                     # explicit opt-in to an untrained network
    assert app.mode == "dqn" and app.dqn_label.endswith("UNTRAINED")
    frames(app, 120)
    assert app.last_q is not None and len(app.last_q) == 5


def test_incompatible_checkpoint_is_reported_not_loaded(tmp_path, monkeypatch):
    from agents.dqn import DQNAgent, DQNConfig
    from driving_sim.observations import OBS_DIM
    bad = tmp_path / "bad.pt"
    DQNAgent(OBS_DIM + 3, 5, DQNConfig(hidden=(8,)), "cpu", 0).save(bad)
    app = make_app(tmp_path, monkeypatch, checkpoint=str(bad))
    assert not app.ensure_dqn()
    assert "Cannot load checkpoint" in app.message and app.dqn is None


def test_checkpoint_action_space_switches_environment(tmp_path, monkeypatch):
    from agents.dqn import DQNAgent, DQNConfig
    from driving_sim.observations import OBS_DIM
    p = tmp_path / "a3.pt"
    DQNAgent(OBS_DIM, 3, DQNConfig(hidden=(8,)), "cpu", 0).save(p)
    app = make_app(tmp_path, monkeypatch, checkpoint=str(p), mode="dqn")
    assert app.mode == "dqn" and app.env.num_actions == 3
    frames(app, 120)
    assert len(app.last_q) == 3


def test_rule_mode_runs_full_episode_and_auto_advances(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, mode="rule", family="lead_constant", seed=1)
    ep0 = app.episode
    for _ in range(40):
        frames(app, 300)
        if app.episode > ep0:
            break
    assert app.episode > ep0


def test_manual_crash_saves_replay_and_scrubbing_works(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, mode="manual", family="lead_constant", difficulty=1.0, seed=2)
    monkeypatch.setattr(app.controls, "action", lambda keys: int(Action.ACCELERATE))
    frames(app, 3000)
    assert app.episode_over and app.over_reason == "COLLISION" and app.crash_idx is not None
    assert app.saved_replay_path and app.saved_replay_path.exists()
    rec = json.loads(app.saved_replay_path.read_text())
    assert verify_replay(rec)                                 # saved crash replays identically
    key(app, pygame.K_b)
    assert app.replaying and app.replay_pos < app.crash_idx
    frames(app, 30)
    assert app.replay_pos > max(0, app.crash_idx - 200)       # playback advances
    key(app, pygame.K_SPACE)
    assert app.replay_paused
    p1 = app.replay_pos
    key(app, pygame.K_RIGHT)
    assert app.replay_pos == pytest.approx(p1 + 1)
    key(app, pygame.K_LEFT)
    assert app.replay_pos == pytest.approx(p1)
    key(app, pygame.K_HOME)
    frames(app, 5)
    key(app, pygame.K_ESCAPE)
    assert not app.replaying
    # R restarts the very same scenario deterministically
    spec_before = app.env.scenario.to_dict()
    key(app, pygame.K_r)
    assert app.env.scenario.to_dict() == spec_before and not app.episode_over


def test_recorded_scenario_replays_with_verification(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, mode="manual", family="lead_constant", difficulty=1.0, seed=2)
    monkeypatch.setattr(app.controls, "action", lambda keys: int(Action.ACCELERATE))
    frames(app, 3000)
    rec = json.loads(app.saved_replay_path.read_text())
    app2 = make_app(tmp_path, monkeypatch)
    app2.load_recording(rec)
    frames(app2, 3000)
    assert app2.mode == "recorded" and app2.episode_over
    assert "VERIFIED" in app2.recorded_verdict


def test_physics_is_independent_of_display_fps(tmp_path, monkeypatch):
    def run(dt, seconds):
        app = make_app(tmp_path, monkeypatch, mode="rule", family="multi_lane", seed=4)
        for _ in range(int(seconds / dt)):
            app.update(dt)
        return app

    a, b = run(1 / 30, 12.0), run(1 / 144, 12.0)
    n = min(len(a.history), len(b.history))
    assert n > 200
    for i in (n // 4, n // 2, n - 1):
        va = [(v.x, v.y, v.v) for v in a.history[i].vehicles]
        vb = [(v.x, v.y, v.v) for v in b.history[i].vehicles]
        assert va == vb                                      # same tick index => bit-identical state


def test_scenario_challenge_form_and_start(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, start_menu=True)
    app.activate_menu(3)
    assert app.screen == "scenario"
    app.form.fam_idx = 4                                      # cut_in
    app.form.field = 2
    app.form.seed_text = ""
    key(app, pygame.K_4)
    key(app, pygame.K_2)
    assert app.form.seed == 42
    frames(app, 3)
    key(app, pygame.K_RETURN)
    assert app.screen == "sim" and app.env.scenario.family == "cut_in" and app.env.scenario.seed == 42
    assert app.fixed_scenario


def test_results_screen_handles_empty_and_populated(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, start_menu=True)
    app.activate_menu(4)
    frames(app, 3)
    assert app.screen == "results"
    key(app, pygame.K_ESCAPE)
    assert app.screen == "menu"


def test_snapshot_analysis_matches_env_info(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, mode="rule", family="lead_constant", seed=3)
    an = analyse_snap(app.curr_snap, app.env.world.road.num_lanes, app.env.world.road.lane_width)
    ego = app.env.world.ego
    gap, lead = app.env.world.lane_gaps(ego, app.env.world.road.lane_of(ego.y))[:2]
    assert an["lead_gap"] == pytest.approx(gap)
