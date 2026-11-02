"""Side panel, menus and overlay screens for the Pygame application."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

from driving_sim.renderer import AMBER, DIM, EGO_BLUE, GREEN, RED, TEXT, Renderer
from driving_sim.rewards import REWARD_COMPONENTS
from driving_sim.vehicles import ACTION_NAMES

from .controls import HELP_ROWS

PANEL_BG = (20, 24, 33)
PANEL_LINE = (44, 52, 70)
MODE_COLORS = {"manual": (250, 190, 60), "rule": (80, 220, 130), "dqn": (150, 125, 255), "replay": (255, 120, 120),
               "recorded": (120, 200, 255)}
MODE_LABELS = {"manual": "MANUAL", "rule": "RULE-BASED", "dqn": "DQN", "replay": "CRASH REPLAY",
               "recorded": "SCENARIO REPLAY"}


def _rr(r: Renderer, surf, rect, color, radius=8, width=0):
    r.pg.draw.rect(surf, color, rect, width=width, border_radius=radius)


def dim_overlay(r: Renderer, alpha: int = 160, rect=None):
    rect = rect or r.view.get_rect()
    s = r.pg.Surface(rect.size, r.pg.SRCALPHA)
    s.fill((8, 10, 16, alpha))
    r.view.blit(s, rect.topleft)


# --------------------------------------------------------------------------- #
# Side panel                                                                  #
# --------------------------------------------------------------------------- #
def draw_panel(r: Renderer, p: dict) -> None:
    pg, scr = r.pg, r.screen
    x0, w = r.view_w, r.panel_w
    scr.fill(PANEL_BG, (x0, 0, w, r.H))
    pg.draw.line(scr, PANEL_LINE, (x0, 0), (x0, r.H), 2)
    pad = 16
    x, y = x0 + pad, 14
    r.text(scr, "RL DRIVING LAB", (x, y), 20, TEXT, True)
    y += 30
    mode = p["mode"]
    col = MODE_COLORS.get(mode, TEXT)
    chip = pg.Rect(x, y, w - 2 * pad, 28)
    _rr(r, scr, chip, tuple(int(c * 0.28) for c in col), 14)
    _rr(r, scr, chip, col, 14, 2)
    r.text(scr, f"{MODE_LABELS.get(mode, mode)} | {p['driver']}"[:34], chip.center, 13, col, True, "center")
    y += 40

    # speed readout + gauge
    kmh = p["speed"] * 3.6
    r.text(scr, f"{kmh:5.1f}", (x, y), 44, TEXT, True)
    r.text(scr, "km/h", (x + 150, y + 26), 16, DIM)
    r.text(scr, f"{p['speed']:.1f} m/s", (x + w - 2 * pad, y + 8), 14, DIM, anchor="topright")
    y += 54
    gauge = pg.Rect(x, y, w - 2 * pad, 10)
    _rr(r, scr, gauge, (38, 44, 60), 5)
    vmax = 36.0
    fill = gauge.copy()
    fill.width = int(gauge.width * min(1.0, p["speed"] / vmax))
    _rr(r, scr, fill, EGO_BLUE, 5)
    for mark, c in ((p.get("target_speed", 25.0), GREEN), (p.get("speed_limit", 30.0), RED)):
        mx = gauge.left + gauge.width * mark / vmax
        pg.draw.line(scr, c, (mx, gauge.top - 3), (mx, gauge.bottom + 3), 2)
    y += 26

    status = p["status"]
    status_col = {"RUNNING": GREEN, "PAUSED": AMBER, "COLLISION": RED, "GOAL REACHED": GREEN}.get(status, AMBER)
    rows = [
        ("Episode", f"{p['episode']}"),
        ("Scenario", p["scenario"]),
        ("Seed / diff", f"{p['seed']} / {p['difficulty']:.2f}"),
        ("Lane", f"{p['lane'] + 1} / {p['num_lanes']}"),
        ("Action", p["action"]),
        ("Reward (ep)", f"{p['reward']:+8.1f}"),
        ("Distance", f"{p['distance']:6.0f} / {p['goal']:.0f} m"),
        ("Nearest veh", "none" if p["nearest"] is None else f"{p['nearest']:6.1f} m"),
        ("Lead gap", "none" if p["lead_gap"] is None else f"{p['lead_gap']:6.1f} m"),
        ("Est. TTC", "n/a" if p["ttc"] is None else f"{p['ttc']:5.1f} s"),
        ("Headway", "n/a" if p["headway"] is None else f"{p['headway']:5.2f} s"),
        ("Collision", "YES" if p["collided"] else "no"),
        ("Epsilon", p["epsilon"]),
        ("Status", status),
    ]
    for label, val in rows:
        c = TEXT
        if label == "Collision":
            c = RED if p["collided"] else GREEN
        elif label == "Status":
            c = status_col
        elif label == "Est. TTC" and p["ttc"] is not None:
            c = RED if p["ttc"] < 2.5 else AMBER if p["ttc"] < 4.0 else TEXT
        r.text(scr, label, (x, y), 14, DIM)
        r.text(scr, val, (x0 + w - pad, y), 14 if len(val) < 20 else 12, c, True, "topright")
        y += 22
    y += 6
    pg.draw.line(scr, PANEL_LINE, (x, y), (x0 + w - pad, y), 1)
    y += 10

    q = p.get("q_values")
    if q is not None and p.get("show_q", True):
        r.text(scr, "Q-values (selected = bright)", (x, y), 13, DIM)
        y += 20
        qs = np.asarray(q, dtype=float)
        lo, hi = float(qs.min()), float(qs.max())
        span = max(1e-6, hi - lo)
        for i, qv in enumerate(qs):
            sel = i == p.get("selected_action")
            bar_w = w - 2 * pad - 150
            r.text(scr, ACTION_NAMES[i][:11], (x, y), 12, TEXT if sel else DIM, sel)
            bx = x + 90
            _rr(r, scr, pg.Rect(bx, y + 2, bar_w, 12), (36, 42, 58), 4)
            frac = (qv - lo) / span
            _rr(r, scr, pg.Rect(bx, y + 2, max(3, int(bar_w * frac)), 12), (255, 214, 90) if sel else (90, 100, 140), 4)
            r.text(scr, f"{qv:+.2f}", (x0 + w - pad, y), 12, TEXT if sel else DIM, sel, "topright")
            y += 18
        y += 4
    elif p["mode"] == "dqn" and q is None:
        r.text(scr, "Q-values hidden (press Q)", (x, y), 12, DIM)
        y += 22

    # last-decision reward components
    comps = p.get("components") or {}
    shown = [(k, comps.get(k, 0.0)) for k in REWARD_COMPONENTS if abs(comps.get(k, 0.0)) > 1e-6]
    if shown:
        r.text(scr, "Reward components (last step)", (x, y), 13, DIM)
        y += 20
        for k, v in shown[:7]:
            c = GREEN if v > 0 else RED
            r.text(scr, k, (x, y), 12, DIM)
            bw = min(90, int(abs(v) * (90 if abs(v) <= 1 else 3)))
            _rr(r, scr, pg.Rect(x + 130, y + 2, max(2, bw), 10), c, 3)
            r.text(scr, f"{v:+.2f}", (x0 + w - pad, y), 12, c, False, "topright")
            y += 16

    # footer
    fy = r.H - 56
    pg.draw.line(scr, PANEL_LINE, (x, fy - 8), (x0 + w - pad, fy - 8), 1)
    r.text(scr, f"sim x{p['sim_speed']:g}   {p['fps']:.0f} fps   dt={p['dt']*1000:.0f}ms", (x, fy), 12, DIM)
    r.text(scr, "H: help   M: menu   V: overlays", (x, fy + 18), 12, DIM)
    if p.get("message"):
        r.text(scr, p["message"][:46], (x, fy + 34), 12, AMBER)


# --------------------------------------------------------------------------- #
# Overlays                                                                    #
# --------------------------------------------------------------------------- #
def draw_help(r: Renderer) -> None:
    dim_overlay(r, 200)
    pg, surf = r.pg, r.view
    w, h = 720, 120 + 26 * len(HELP_ROWS)
    rect = pg.Rect(0, 0, w, h)
    rect.center = (r.view_w // 2, r.H // 2)
    _rr(r, surf, rect, (24, 30, 44), 14)
    _rr(r, surf, rect, (80, 100, 150), 14, 2)
    r.text(surf, "Controls", (rect.centerx, rect.top + 16), 24, TEXT, True, "midtop")
    y = rect.top + 60
    for key, desc in HELP_ROWS:
        r.text(surf, key, (rect.left + 30, y), 15, AMBER, True)
        r.text(surf, desc, (rect.left + 190, y), 15, TEXT)
        y += 26
    r.text(surf, "press H to close", (rect.centerx, rect.bottom - 16), 13, DIM, anchor="midbottom")


def draw_banner(r: Renderer, title: str, subtitle: str, color) -> None:
    pg, surf = r.pg, r.view
    rect = pg.Rect(0, 0, 520, 96)
    rect.center = (r.view_w // 2, 150)
    s = pg.Surface(rect.size, pg.SRCALPHA)
    pg.draw.rect(s, (10, 12, 18, 215), s.get_rect(), border_radius=14)
    surf.blit(s, rect.topleft)
    _rr(r, surf, rect, color, 14, 3)
    r.text(surf, title, (rect.centerx, rect.top + 12), 34, color, True, "midtop")
    r.text(surf, subtitle, (rect.centerx, rect.top + 62), 15, TEXT, False, "midtop")


def draw_replay_bar(r: Renderer, idx: int, n: int, t_rel: float, paused: bool, speed: float,
                    crash_idx: Optional[int]) -> None:
    pg, surf = r.pg, r.view
    rect = pg.Rect(30, r.scene_h - 62, r.view_w - 60, 40)
    s = pg.Surface(rect.size, pg.SRCALPHA)
    pg.draw.rect(s, (10, 12, 18, 215), s.get_rect(), border_radius=10)
    surf.blit(s, rect.topleft)
    bar = pg.Rect(rect.left + 16, rect.centery - 4, rect.width - 32, 8)
    _rr(r, surf, bar, (50, 58, 78), 4)
    if n > 1:
        _rr(r, surf, pg.Rect(bar.left, bar.top, int(bar.width * idx / (n - 1)), bar.height), (255, 120, 120), 4)
        if crash_idx is not None:
            cx = bar.left + bar.width * crash_idx / (n - 1)
            pg.draw.line(surf, RED, (cx, bar.top - 8), (cx, bar.bottom + 8), 3)
        hx = bar.left + bar.width * idx / (n - 1)
        pg.draw.circle(surf, TEXT, (hx, bar.centery), 8)
    r.text(surf, f"t = {t_rel:+.2f}s vs impact" if crash_idx is not None else f"frame {idx}/{n}",
           (rect.left + 16, rect.top - 18), 14, (255, 160, 160), True)
    r.text(surf, f"{'PAUSED' if paused else 'PLAYING'} x{speed:g}   Left/Right: step  Space: pause  Up/Down: speed  B/Esc: exit",
           (rect.right - 16, rect.top - 18), 12, DIM, anchor="topright")


# --------------------------------------------------------------------------- #
# Menu screens                                                                #
# --------------------------------------------------------------------------- #
MENU_ITEMS = [
    ("manual", "Manual Driving", "Control the car directly and explore the simulation."),
    ("rule", "Rule-Based Driving", "Watch the deterministic baseline navigate traffic."),
    ("dqn", "DQN Driving", "Load a checkpoint and watch the learned agent drive."),
    ("challenge", "Scenario Challenge", "Pick a scenario family, difficulty and seed."),
    ("results", "Training Results", "Browse training curves and evaluation summaries."),
]


def menu_item_rects(r: Renderer):
    pg = r.pg
    rects = []
    for i in range(len(MENU_ITEMS)):
        rects.append(pg.Rect(80, 220 + i * 92, 560, 78))
    return rects


def draw_menu(r: Renderer, selected: int, status: dict) -> None:
    dim_overlay(r, 175)
    surf = r.view
    r.text(surf, "RL DRIVING SIMULATOR", (80, 70), 44, TEXT, True)
    r.text(surf, "Double DQN vs. rule-based driving in a 2D highway laboratory", (82, 128), 17, DIM)
    r.text(surf, "Up/Down + Enter, or click.   H = controls   Esc = quit", (82, 160), 14, DIM)
    for i, (rect, (key, title, desc)) in enumerate(zip(menu_item_rects(r), MENU_ITEMS)):
        sel = i == selected
        col = MODE_COLORS.get(key, (110, 180, 255))
        _rr(r, surf, rect, (30, 38, 56) if sel else (22, 27, 40), 12)
        _rr(r, surf, rect, col if sel else (52, 62, 84), 12, 3 if sel else 1)
        r.text(surf, f"{i + 1}" if i < 3 else "*", (rect.left + 22, rect.centery), 28, col, True, "midleft")
        r.text(surf, title, (rect.left + 66, rect.top + 14), 23, TEXT if sel else (200, 206, 220), True)
        r.text(surf, desc, (rect.left + 66, rect.top + 46), 14, DIM)
        note = status.get(key)
        if note:
            r.text(surf, note[0], (rect.right - 16, rect.top + 14), 13, note[1], True, "topright")


def draw_menu_side(r: Renderer, status_lines: list[tuple[str, tuple]]) -> None:
    pg, scr = r.pg, r.screen
    x0, w = r.view_w, r.panel_w
    scr.fill(PANEL_BG, (x0, 0, w, r.H))
    pg.draw.line(scr, PANEL_LINE, (x0, 0), (x0, r.H), 2)
    r.text(scr, "SYSTEM", (x0 + 16, 18), 18, TEXT, True)
    y = 56
    for line, col in status_lines:
        r.text(scr, line[:44], (x0 + 16, y), 13, col)
        y += 22


class ScenarioForm:
    """State of the Scenario Challenge selection screen."""

    def __init__(self, families: list[str]):
        self.families = families
        self.fam_idx = 0
        self.difficulty = 0.5
        self.seed_text = "1"
        self.driver_idx = 1
        self.field = 0       # 0 family, 1 difficulty, 2 seed, 3 driver
        self.drivers = ["manual", "rule", "dqn"]

    @property
    def seed(self) -> int:
        return int(self.seed_text) if self.seed_text else 0

    def key(self, event) -> Optional[str]:
        pg = pygame
        k = event.key
        if k in (pg.K_UP,):
            self.field = (self.field - 1) % 4
        elif k in (pg.K_DOWN, pg.K_TAB):
            self.field = (self.field + 1) % 4
        elif k in (pg.K_LEFT, pg.K_RIGHT):
            d = -1 if k == pg.K_LEFT else 1
            if self.field == 0:
                self.fam_idx = (self.fam_idx + d) % len(self.families)
            elif self.field == 1:
                self.difficulty = float(min(1.0, max(0.0, round(self.difficulty + 0.1 * d, 2))))
            elif self.field == 2:
                self.seed_text = str(max(0, self.seed + d))
            else:
                self.driver_idx = (self.driver_idx + d) % 3
        elif self.field == 2 and pg.K_0 <= k <= pg.K_9:
            self.seed_text = (self.seed_text + chr(k))[-9:]
        elif self.field == 2 and k == pg.K_BACKSPACE:
            self.seed_text = self.seed_text[:-1]
        elif k in (pg.K_RETURN, pg.K_KP_ENTER):
            return "start"
        return None


def draw_scenario_screen(r: Renderer, form: ScenarioForm, spec, labels: dict) -> None:
    dim_overlay(r, 200)
    pg, surf = r.pg, r.view
    r.text(surf, "Scenario Challenge", (80, 60), 38, TEXT, True)
    r.text(surf, "Up/Down: field   Left/Right: change   digits: type seed   Enter: start   Esc: back", (82, 112), 14, DIM)
    fam = form.families[form.fam_idx]
    fields = [
        ("Scenario family", labels.get(fam, fam)),
        ("Difficulty", f"{form.difficulty:.1f}   " + "#" * int(form.difficulty * 20) + "." * (20 - int(form.difficulty * 20))),
        ("Seed", form.seed_text + ("_" if form.field == 2 else "")),
        ("Driver", form.drivers[form.driver_idx].upper()),
    ]
    y = 170
    for i, (k, v) in enumerate(fields):
        rect = pg.Rect(80, y, 640, 56)
        sel = i == form.field
        _rr(r, surf, rect, (30, 38, 56) if sel else (22, 27, 40), 10)
        _rr(r, surf, rect, (110, 180, 255) if sel else (52, 62, 84), 10, 2 if sel else 1)
        r.text(surf, k, (rect.left + 20, rect.centery), 15, DIM, anchor="midleft")
        r.text(surf, v, (rect.right - 20, rect.centery), 19, TEXT, True, "midright")
        y += 68
    y += 10
    r.text(surf, "Preview (deterministic from family + seed + difficulty)", (82, y), 15, AMBER, True)
    y += 28
    lines = [
        f"{spec.description}",
        f"lanes: {spec.num_lanes}   ego lane: {spec.ego_lane + 1}   ego start: {spec.ego_speed * 3.6:.0f} km/h   "
        f"NPCs: {len(spec.npcs)}   goal: {spec.goal_distance:.0f} m",
    ]
    if spec.sub_family:
        lines.append(f"variant: {spec.sub_family}")
    behaviours = {}
    for n in spec.npcs:
        behaviours[n.behavior] = behaviours.get(n.behavior, 0) + 1
    lines.append("NPC behaviours: " + ", ".join(f"{k} x{v}" for k, v in sorted(behaviours.items())))
    for line in lines:
        r.text(surf, line[:110], (82, y), 14, TEXT)
        y += 22


# --------------------------------------------------------------------------- #
# Training results browser                                                    #
# --------------------------------------------------------------------------- #
def discover_results(results_dir: Path) -> list[Path]:
    out = []
    if not results_dir.exists():
        return out
    for d in sorted(results_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and any((d / n).exists() for n in ("metrics.csv", "evaluation.json", "training_curves.png",
                                                         "comparison.png")):
            out.append(d)
    return out


def results_summary_lines(d: Path) -> list[tuple[str, tuple]]:
    lines: list[tuple[str, tuple]] = []
    ev = d / "evaluation.json"
    if ev.exists():
        data = json.loads(ev.read_text())
        lines.append((f"Evaluation: {data.get('n_scenarios', '?')} paired scenarios, seeds from {data.get('seed_base')}",
                      AMBER))
        lines.append((f"{'policy':<28s}{'collision [95% CI]':<26s}{'completion [95% CI]':<26s}{'speed':>7s}", DIM))
        for name, p in data["policies"].items():
            o = p["overall"]
            cl, ch = o["collision_ci"]
            sl, sh = o["success_ci"]
            lines.append((f"{name[:27]:<28s}{o['collision_rate']*100:5.1f}% [{cl*100:4.1f},{ch*100:4.1f}]   "
                          f"{o['success_rate']*100:5.1f}% [{sl*100:4.1f},{sh*100:4.1f}]   {o['mean_speed']:6.1f}",
                          TEXT))
        return lines
    evc = d / "eval.csv"
    if evc.exists():
        with open(evc, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            best = [r_ for r_ in rows if r_.get("is_best") == "1"]
            lines.append((f"Latest eval @ step {last['step']} (stage {last['stage']}): score {last['score']}, "
                          f"collision {float(last['collision_rate']) * 100:.1f}%, "
                          f"completion {float(last['success_rate']) * 100:.1f}%, speed {last['mean_speed']} m/s", TEXT))
            if best:
                b = best[-1]
                lines.append((f"Best checkpoint: step {b['step']}, eval score {b['score']} "
                              f"(criterion: see README 'Model selection')", GREEN))
            else:
                lines.append(("No best.pt: training has not reached the final curriculum stage yet.", AMBER))
    ts = d / "train_summary.json"
    if ts.exists():
        s = json.loads(ts.read_text())
        lines.append((f"Run finished: {s['steps']} steps, {s['episodes']} episodes", DIM))
    if not lines:
        lines.append(("No evaluation recorded yet for this run.", AMBER))
    return lines


def load_results_image(r: Renderer, d: Path):
    pg = r.pg
    for name in ("training_curves.png", "comparison.png"):
        p = d / name
        if not p.exists() and name == "training_curves.png" and (d / "metrics.csv").exists():
            try:
                from driving_sim.plotting import plot_training_curves
                plot_training_curves(d)
            except Exception:
                pass
        if p.exists():
            try:
                img = pg.image.load(str(p))
                try:
                    return img.convert()        # needs a display surface; headless mode just keeps the raw surface
                except pg.error:
                    return img
            except Exception:
                return None
    return None


def draw_results_screen(r: Renderer, runs: list[Path], idx: int, image, lines) -> None:
    dim_overlay(r, 235)
    pg, surf = r.pg, r.view
    r.text(surf, "Training Results", (40, 24), 32, TEXT, True)
    if not runs:
        r.text(surf, "No results found yet.", (42, 100), 20, AMBER, True)
        r.text(surf, "Train a model first:  python scripts/train.py", (42, 136), 16, TEXT)
        r.text(surf, "then evaluate:         python scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt", (42, 162), 16, TEXT)
        r.text(surf, "Esc: back", (42, 220), 14, DIM)
        return
    r.text(surf, f"{runs[idx].name}   ({idx + 1}/{len(runs)})     Left/Right: switch run   Esc: back", (42, 66), 15, AMBER)
    avail = pg.Rect(30, 96, r.view_w - 60, r.H - 96 - 190)
    if image is not None:
        iw, ih = image.get_size()
        sc = min(avail.width / iw, avail.height / ih)
        img = pg.transform.smoothscale(image, (int(iw * sc), int(ih * sc)))
        surf.blit(img, img.get_rect(midtop=(avail.centerx, avail.top)))
    else:
        r.text(surf, "(no figure available for this run)", (avail.left, avail.top + 20), 16, DIM)
    y = r.H - 180
    for text, col in lines[:8]:
        r.text(surf, text, (42, y), 14, col)
        y += 22
