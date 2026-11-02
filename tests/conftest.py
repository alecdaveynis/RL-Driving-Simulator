import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from driving_sim.config import load_sim_config
from driving_sim.physics import Road, World
from driving_sim.vehicles import Vehicle


@pytest.fixture
def cfg():
    return load_sim_config()


def make_world(cfg, lanes=3, goal=10_000.0):
    return World(Road(lanes, cfg.road.lane_width), cfg, goal)


def add_ego(world, x=0.0, lane=1, v=20.0):
    e = world.cfg.ego
    return world.add_vehicle(Vehicle(0, x, world.road.lane_center(lane), v, e.length, e.width, is_ego=True,
                                     max_accel=e.max_accel, max_brake=e.max_brake, max_jerk=e.max_jerk,
                                     lc_dur=world.cfg.physics.lane_change_duration))


def add_npc(world, vid, x, lane, v, length=4.5, width=1.85, controller=None):
    return world.add_vehicle(Vehicle(vid, x, world.road.lane_center(lane), v, length, width,
                                     controller=controller, lc_dur=world.cfg.physics.lane_change_duration))
