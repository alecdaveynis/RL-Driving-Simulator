"""Keyboard mapping for manual driving and application hotkeys."""
from __future__ import annotations

import pygame

from driving_sim.vehicles import Action

SPEED_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

ACCEL_KEYS = (pygame.K_w, pygame.K_UP)
BRAKE_KEYS = (pygame.K_s, pygame.K_DOWN)
LEFT_KEYS = (pygame.K_a, pygame.K_LEFT)
RIGHT_KEYS = (pygame.K_d, pygame.K_RIGHT)
PLUS_KEYS = (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)
MINUS_KEYS = (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_UNDERSCORE)

HELP_ROWS = [
    ("W / Up", "Accelerate"),
    ("S / Down", "Brake"),
    ("A / Left", "Change lane left"),
    ("D / Right", "Change lane right"),
    ("Space", "Pause / resume"),
    ("R", "Restart this scenario (same seed)"),
    ("N", "Next scenario (seed + 1)"),
    ("1 / 2 / 3", "Manual / Rule-based / DQN driver"),
    ("+ / -", "Simulation speed"),
    ("V", "Toggle all debug overlays"),
    ("T / G / Z", "Trajectories / radar / safety zones"),
    ("Q", "Show Q-values (DQN)"),
    ("K", "Cycle DQN checkpoint"),
    ("U", "Use an UNTRAINED DQN (random weights)"),
    ("B", "Replay the last crash (Left/Right step, Space pause)"),
    ("H / F1", "Show / hide this help"),
    ("M", "Back to the menu"),
    ("Esc", "Back to menu / exit"),
]


class ManualControls:
    """Held keys give continuous accelerate/brake; lane changes are edge-triggered."""

    def __init__(self) -> None:
        self.lane_request = 0

    def handle_event(self, event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in LEFT_KEYS:
                self.lane_request = -1
            elif event.key in RIGHT_KEYS:
                self.lane_request = +1

    def reset(self) -> None:
        self.lane_request = 0

    def action(self, keys) -> int:
        """Action for the current physics tick (consumes a pending lane-change request)."""
        if self.lane_request:
            a = Action.LANE_LEFT if self.lane_request < 0 else Action.LANE_RIGHT
            self.lane_request = 0
            return int(a)
        accel = any(keys[k] for k in ACCEL_KEYS)
        brake = any(keys[k] for k in BRAKE_KEYS)
        if brake:
            return int(Action.BRAKE)
        if accel:
            return int(Action.ACCELERATE)
        return int(Action.COAST)
