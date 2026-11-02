import csv
import copy
from pathlib import Path

import numpy as np
import pytest

from agents.dqn import IncompatibleCheckpointError, load_checkpoint, read_checkpoint
from agents.trainer import Trainer, load_train_config
from driving_sim.observations import OBS_DIM


def tiny_cfg(tmp_path, steps=1600, actions=5, name="t"):
    cfg = load_train_config("configs/training_3action.yaml" if actions == 3 else "configs/training.yaml")
    cfg.update(run_name=name, total_steps=steps, device="cpu")
    cfg["env"]["num_actions"] = actions
    cfg["dqn"].update(learning_starts=200, buffer_size=5000, batch_size=32, target_sync_interval=100)
    cfg["exploration"].update(decay_steps=800, stage_restart_decay_steps=300)
    cfg["eval"].update(interval_steps=800, episodes_per_family=1, families=["lead_constant", "lead_brake"])
    cfg["checkpoint"].update(interval_steps=800, dir=str(tmp_path / "ck"), results_dir=str(tmp_path / "res"))
    cfg["logging"].update(interval_steps=400)
    cfg["curriculum"]["stages"] = [
        {"name": "a", "families": {"lead_constant": 1.0}, "lane_changes": False, "min_steps": 0, "max_steps": 500},
        {"name": "b", "families": {"lead_brake": 1.0}, "lane_changes": True, "min_steps": 0, "max_steps": 10**9},
    ]
    return cfg


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_short_training_run_produces_artifacts(tmp_path):
    cfg = tiny_cfg(tmp_path)
    logs = []
    summary = Trainer(cfg, log=logs.append).run_training()
    ck, res = tmp_path / "ck" / "t", tmp_path / "res" / "t"
    for f in ("last.pt", "final.pt", "best.pt"):
        assert (ck / f).exists(), f
    for f in ("config.yaml", "episodes.csv", "metrics.csv", "eval.csv", "train_summary.json", "training_curves.png"):
        assert (res / f).exists(), f
    assert summary["steps"] == 1600
    ep = read_rows(res / "episodes.csv")
    assert ep and {"reward", "success", "collision", "stage", "r_progress", "epsilon"} <= set(ep[0])
    assert {e["stage"] for e in ep} == {"a", "b"}                       # curriculum advanced
    ev = read_rows(res / "eval.csv")
    assert len(ev) >= 2 and "score" in ev[0] and "collision_lead_brake" in ev[0]
    assert any(l.startswith("[curriculum] leaving stage 'a'") for l in logs)
    ck_meta = read_checkpoint(ck / "best.pt")
    assert ck_meta["num_actions"] == 5 and ck_meta["obs_dim"] == OBS_DIM and ck_meta["extra"]["train_config"]["seed"] == 0
    best_rows = [r for r in ev if r["is_best"] == "1"]
    assert best_rows and all(r["stage"] == "b" for r in best_rows)      # best.pt only from the final stage


def test_curriculum_masks_lane_changes_in_early_stages(tmp_path):
    cfg = tiny_cfg(tmp_path, steps=400)
    t = Trainer(cfg, log=lambda s: None)
    t._apply_stage()
    obs, info = t.env.reset(seed=0)
    assert info["action_mask"].tolist() == [True, True, True, False, False]
    t.stage_idx = 1
    t._apply_stage()
    _, info = t.env.reset(seed=0)
    assert info["action_mask"][3] or info["action_mask"][4]


def test_resume_restores_optimizer_exploration_and_progress(tmp_path):
    cfg = tiny_cfg(tmp_path, steps=900, name="r")
    Trainer(cfg, log=lambda s: None).run_training()
    ck_path = tmp_path / "ck" / "r" / "last.pt"
    before = read_checkpoint(ck_path)
    assert before["train_state"]["env_steps"] == 900
    assert len(before["optimizer"]["state"]) > 0
    cfg2 = copy.deepcopy(before["extra"]["train_config"])
    cfg2["total_steps"] = 1300
    logs = []
    t2 = Trainer(cfg2, resume=ck_path, log=logs.append)
    assert t2.env_steps == 900 and t2.episodes == before["train_state"]["episodes"]
    assert t2.stage_idx == before["train_state"]["stage_idx"]
    assert t2.agent.update_count == before["update_count"]
    assert len(t2.buffer) > 0                                                   # replay buffer restored
    for p1, p2 in zip(before["optimizer"]["state"].values(), t2.agent.optimizer.state_dict()["state"].values()):
        assert np.allclose(p1["exp_avg"].numpy(), p2["exp_avg"].numpy())      # Adam moments restored
    assert t2.epsilon_at(900) == pytest.approx(before["train_state"]["epsilon"])
    summary = t2.run_training()
    assert summary["steps"] == 1300
    after = read_checkpoint(ck_path)
    assert after["train_state"]["env_steps"] == 1300 and after["update_count"] > before["update_count"]
    rows = read_rows(tmp_path / "res" / "r" / "metrics.csv")
    assert [int(r["step"]) for r in rows] == sorted(int(r["step"]) for r in rows)   # appended, not overwritten


def test_three_action_run_is_a_separate_incompatible_model(tmp_path):
    cfg = tiny_cfg(tmp_path, steps=600, actions=3, name="a3")
    Trainer(cfg, log=lambda s: None).run_training()
    path = tmp_path / "ck" / "a3" / "final.pt"
    assert read_checkpoint(path)["num_actions"] == 3
    with pytest.raises(IncompatibleCheckpointError):
        load_checkpoint(path, OBS_DIM, 5)
    # a 5-action trainer refuses to resume from it
    cfg5 = tiny_cfg(tmp_path, steps=600, actions=5, name="a5")
    with pytest.raises(IncompatibleCheckpointError):
        Trainer(cfg5, resume=path, log=lambda s: None)
