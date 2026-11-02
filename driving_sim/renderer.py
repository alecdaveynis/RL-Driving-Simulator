"""Pygame renderer. Purely a view: it reads `WorldSnap`s and never mutates simulation state.

World -> screen: the road runs left-to-right, lane 0 at the top. The camera follows the
(interpolated) ego vehicle, so rendering at any FPS gives smooth motion while physics
stays on its fixed timestep.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import SimConfig
from .physics import VehicleSnap, WorldSnap
from .vehicles import NPC_COLORS

# --- palette ---------------------------------------------------------------
GRASS = (30, 58, 42)
GRASS_B = (35, 66, 48)
SHOULDER = (74, 76, 84)
ASPHALT_A = (47, 50, 58)
ASPHALT_B = (51, 54, 62)
LINE = (226, 229, 236)
EDGE = (240, 200, 70)
RAIL = (150, 156, 168)
EGO_BLUE = (52, 120, 246)
EGO_GLOW = (90, 160, 255)
TEXT = (232, 236, 244)
DIM = (140, 148, 164)
GREEN = (80, 220, 130)
AMBER = (250, 190, 60)
RED = (240, 80, 70)


@dataclass
class Overlays:
    debug: bool = True
    trajectories: bool = True
    radar: bool = True
    zones: bool = True


@dataclass
class SceneInfo:
    num_lanes: int
    lane_width: float
    goal_distance: float
    time: float = 0.0
    lead_gap: Optional[float] = None       # bumper gap to lead in ego lane [m]
    lead_vid: Optional[int] = None
    headway: Optional[float] = None
    ttc: Optional[float] = None
    detections: list = field(default_factory=list)   # (vid, slot, side, gap)
    selected_vid: Optional[int] = None
    ego_target_y: Optional[float] = None


def _lerp(a, b, t):
    return a + (b - a) * t


def _hash(i: int) -> int:
    i = (i * 2654435761) & 0xFFFFFFFF
    i ^= i >> 15
    return (i * 2246822519) & 0xFFFFFFFF


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "color")

    def __init__(self, x, y, vx, vy, life, size, color):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.size, self.color = size, color


class Renderer:
    def __init__(self, cfg: SimConfig, headless: bool = False, window: bool = True, caption: str = "RL Driving Simulator"):
        if headless and "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame

        self.pg = pygame
        pygame.init()
        self.cfg = cfg
        rc = cfg.render
        self.W, self.H = rc.width, rc.height
        self.panel_w = rc.panel_width
        self.view_w = self.W - self.panel_w
        self.minimap_h = 96
        self.scene_h = self.H - self.minimap_h
        self.base_ppm = rc.pixels_per_meter
        self.lat = rc.lateral_scale
        self.zoom = 1.0
        self.ppm = self.base_ppm
        self.ego_sx = int(self.view_w * rc.ego_screen_frac)
        if headless or not window:
            self.screen = pygame.Surface((self.W, self.H))
        else:
            self.screen = pygame.display.set_mode((self.W, self.H), pygame.DOUBLEBUF)
            pygame.display.set_caption(caption)
        self.view = self.screen.subsurface(pygame.Rect(0, 0, self.view_w, self.H))
        self.fonts = {}
        self._car_cache: dict = {}
        self.particles: list[Particle] = []
        self.rings: list[list] = []   # [x, y, age]
        self.shake = 0.0
        self.flash = 0.0
        self.cam_x = 0.0
        self._rng = random.Random(7)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------ utils
    def font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = self.pg.font.SysFont("menlo,monaco,consolas,dejavusansmono,arial", size, bold=bold)
        return self.fonts[key]

    def text(self, surf, s, pos, size=14, color=TEXT, bold=False, anchor="topleft"):
        img = self.font(size, bold).render(str(s), True, color)
        r = img.get_rect()
        setattr(r, anchor, pos)
        surf.blit(img, r)
        return r

    def close(self):
        try:
            self.pg.display.quit()
        except Exception:
            pass

    # ------------------------------------------------------------- collisions
    def trigger_collision(self, x: float, y: float, strength: float = 1.0) -> None:
        self.rings.append([x, y, 0.0])
        self.shake = max(self.shake, 10.0 * min(1.5, strength))
        self.flash = 1.0
        for _ in range(int(40 * min(1.5, strength))):
            ang = self._rng.uniform(0, math.tau)
            sp = self._rng.uniform(40, 360)
            col = self._rng.choice([(255, 210, 90), (255, 140, 50), (255, 90, 60), (250, 250, 250), (90, 90, 90)])
            self.particles.append(Particle(x, y, math.cos(ang) * sp, math.sin(ang) * sp,
                                           self._rng.uniform(0.4, 1.4), self._rng.uniform(2, 6), col))

    def clear_effects(self):
        self.particles.clear()
        self.rings.clear()
        self.shake = 0.0
        self.flash = 0.0

    def _update_effects(self, dt: float):
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= 0.96
            p.vy *= 0.96
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0]
        for r in self.rings:
            r[2] += dt
        self.rings = [r for r in self.rings if r[2] < 0.9]
        self.shake = max(0.0, self.shake - 30.0 * dt)
        self.flash = max(0.0, self.flash - 2.5 * dt)

    # --------------------------------------------------------- vehicle sprite
    def _car_surface(self, snap: VehicleSnap, braking: bool, blink: int):
        pg = self.pg
        L = int(round(snap.length * self.base_ppm))
        Wd = int(round(snap.width * self.base_ppm * self.lat))
        key = (snap.color, L, Wd, braking, blink, snap.crashed, snap.is_ego)
        s = self._car_cache.get(key)
        if s is not None:
            return s
        pad = 6
        s = pg.Surface((L + 2 * pad, Wd + 2 * pad), pg.SRCALPHA)
        body = EGO_BLUE if snap.is_ego else NPC_COLORS[snap.color % len(NPC_COLORS)]
        if snap.crashed:
            body = tuple(int(c * 0.45 + 40) for c in body)
        dark = tuple(int(c * 0.62) for c in body)
        light = tuple(min(255, int(c * 1.18 + 12)) for c in body)
        r = pg.Rect(pad, pad, L, Wd)
        is_truck = snap.length > 7.0
        pg.draw.rect(s, dark, r, border_radius=int(Wd * 0.34))
        inner = r.inflate(-3, -3)
        pg.draw.rect(s, body, inner, border_radius=int(Wd * 0.3))
        if is_truck:
            cab_w = int(L * 0.24)
            cargo = pg.Rect(pad + 2, pad + 2, L - cab_w - 4, Wd - 4)
            pg.draw.rect(s, (205, 210, 218) if not snap.crashed else (120, 120, 120), cargo, border_radius=3)
            pg.draw.rect(s, (160, 166, 176), cargo, width=1, border_radius=3)
            cab = pg.Rect(pad + L - cab_w, pad + 1, cab_w - 1, Wd - 2)
            pg.draw.rect(s, body, cab, border_radius=int(Wd * 0.28))
            pg.draw.rect(s, (24, 30, 42), (cab.x + cab_w * 0.5, pad + 4, cab_w * 0.4, Wd - 8), border_radius=3)
        else:
            # roof + glass
            roof = pg.Rect(pad + int(L * 0.30), pad + 3, int(L * 0.34), Wd - 6)
            pg.draw.rect(s, light, roof, border_radius=int(Wd * 0.2))
            glass = (26, 34, 50) if not snap.crashed else (60, 60, 60)
            pg.draw.polygon(s, glass, [(pad + L * 0.64, pad + 3), (pad + L * 0.79, pad + Wd * 0.2),
                                       (pad + L * 0.79, pad + Wd * 0.8), (pad + L * 0.64, pad + Wd - 3)])
            pg.draw.polygon(s, glass, [(pad + L * 0.30, pad + 3), (pad + L * 0.17, pad + Wd * 0.22),
                                       (pad + L * 0.17, pad + Wd * 0.78), (pad + L * 0.30, pad + Wd - 3)])
        # head / tail lights (front is +x = right side of the sprite)
        pg.draw.rect(s, (255, 250, 205), (pad + L - 4, pad + 2, 4, 4), border_radius=2)
        pg.draw.rect(s, (255, 250, 205), (pad + L - 4, pad + Wd - 6, 4, 4), border_radius=2)
        tail = (255, 40, 40) if braking else (130, 20, 24)
        pg.draw.rect(s, tail, (pad, pad + 2, 4, 5), border_radius=2)
        pg.draw.rect(s, tail, (pad, pad + Wd - 7, 4, 5), border_radius=2)
        if braking:
            glow = pg.Surface((24, Wd + 8), pg.SRCALPHA)
            pg.draw.ellipse(glow, (255, 40, 40, 70), glow.get_rect())
            s.blit(glow, (pad - 14, pad - 4))
        if blink:  # -1 left (screen up), +1 right (screen down)
            yy = pad + 1 if blink < 0 else pad + Wd - 5
            for xx in (pad + 1, pad + L - 6):
                pg.draw.rect(s, (255, 170, 20), (xx, yy, 5, 4), border_radius=2)
        if snap.is_ego:
            pg.draw.rect(s, EGO_GLOW, r, width=2, border_radius=int(Wd * 0.34))
        self._car_cache[key] = s
        return s

    def _vis_heading(self, h: float) -> float:
        """Heading as drawn: lateral exaggeration also steepens the visual path angle."""
        return math.atan(math.tan(h) * self.lat)

    # ----------------------------------------------------------------- scene
    def _sx(self, x: float) -> float:
        return self.ego_sx + (x - self.cam_x) * self.ppm

    def _interp(self, snap: WorldSnap, prev: Optional[WorldSnap], alpha: float):
        pm = {v.vid: v for v in prev.vehicles} if prev is not None else {}
        out = []
        for v in snap.vehicles:
            p = pm.get(v.vid)
            if p is None or alpha >= 1.0:
                out.append((v, v.x, v.y, v.heading))
            else:
                out.append((v, _lerp(p.x, v.x, alpha), _lerp(p.y, v.y, alpha), _lerp(p.heading, v.heading, alpha)))
        return out

    def draw_scene(self, snap: WorldSnap, info: SceneInfo, prev: Optional[WorldSnap] = None, alpha: float = 1.0,
                   overlays: Optional[Overlays] = None, dt_wall: float = 1 / 60, banner: Optional[str] = None):
        pg = self.pg
        ov = overlays or Overlays()
        self._update_effects(dt_wall)
        surf = self.view
        items = self._interp(snap, prev, alpha)
        ego_item = next((it for it in items if it[0].is_ego), items[0])
        self.cam_x = ego_item[1]
        if self.cfg.render.dynamic_zoom:
            target = 1.22 - 0.48 * min(ego_item[0].v / 30.0, 1.0)
            self.zoom += (target - self.zoom) * min(1.0, dt_wall * 2.5)
        else:
            self.zoom = 1.0
        self.ppm = self.base_ppm * self.zoom
        ppm = self.ppm
        lat = self.lat
        road_w = info.num_lanes * info.lane_width
        road_h_px = road_w * ppm * lat
        top = (self.scene_h - road_h_px) / 2
        sh = (self._rng.uniform(-1, 1) * self.shake, self._rng.uniform(-1, 1) * self.shake) if self.shake > 0.5 else (0, 0)
        top += sh[1]

        def sy(y):
            return top + y * ppm * lat

        x0w = self.cam_x - self.ego_sx / ppm - 5
        x1w = x0w + self.view_w / ppm + 10

        # background
        surf.fill(GRASS, (0, 0, self.view_w, self.scene_h))
        i0, i1 = int(x0w // 24), int(x1w // 24) + 1
        for i in range(i0, i1):  # grass mowing stripes
            if i % 2:
                xa = self._sx(i * 24)
                surf.fill(GRASS_B, (xa, 0, 24 * ppm + 1, self.scene_h))
        for i in range(int(x0w // 16) - 1, int(x1w // 16) + 2):  # trees
            h = _hash(i)
            for side in (0, 1):
                hh = _hash(h + side * 7919)
                if hh % 5 == 0:
                    continue
                off = 4.5 + (hh % 1000) / 1000 * 12
                wy = -off if side == 0 else road_w + off
                rad = 8 + (hh >> 10) % 11
                px, py = self._sx(i * 16 + (hh >> 4) % 12), sy(wy)
                if -40 < py < self.scene_h + 40:
                    pg.draw.circle(surf, (18, 38, 28), (px + 4, py + 5), rad)
                    pg.draw.circle(surf, (28, 84, 50), (px, py), rad)
                    pg.draw.circle(surf, (44, 112, 66), (px - rad * 0.25, py - rad * 0.25), rad * 0.62)

        # shoulder + asphalt
        sh_px = self.cfg.road.lane_width * 0.2 * ppm * lat
        surf.fill(SHOULDER, (0, sy(0) - sh_px, self.view_w, road_h_px + 2 * sh_px))
        surf.fill(ASPHALT_A, (0, sy(0), self.view_w, road_h_px))
        for i in range(int(x0w // 10), int(x1w // 10) + 1):  # scroll cue: faint alternating tiles
            if i % 2:
                surf.fill(ASPHALT_B, (self._sx(i * 10), sy(0), 10 * ppm + 1, road_h_px))
        # edge lines
        for yy in (0.0, road_w):
            pg.draw.line(surf, EDGE, (0, sy(yy)), (self.view_w, sy(yy)), 3)
        # lane dashes
        dash, period = 3.0, 12.0
        for k in range(1, info.num_lanes):
            yy = sy(k * info.lane_width)
            for i in range(int(x0w // period) - 1, int(x1w // period) + 2):
                xa = self._sx(i * period)
                pg.draw.line(surf, LINE, (xa, yy), (xa + dash * ppm, yy), 3)
        # guardrails + posts + km markers
        for side, yy in ((0, -sh_px - 10), (1, road_w * ppm * lat + sh_px + 10)):
            ypx = sy(0) + yy
            pg.draw.line(surf, RAIL, (0, ypx), (self.view_w, ypx), 4)
            for i in range(int(x0w // 4) - 1, int(x1w // 4) + 2):
                xa = self._sx(i * 4)
                pg.draw.rect(surf, (96, 100, 112), (xa - 2, ypx - 5, 4, 10))
        for m in range(int(x0w // 100), int(x1w // 100) + 1):
            xa = self._sx(m * 100)
            if m * 100 >= 0:
                pg.draw.rect(surf, (230, 230, 240), (xa - 1, sy(road_w) + sh_px + 16, 3, 14))
                self.text(surf, f"{m * 100}m", (xa, sy(road_w) + sh_px + 32), 12, DIM, anchor="midtop")
        # goal line (checkered)
        gx = self._sx(info.goal_distance)
        if -60 < gx < self.view_w + 60:
            cell = 10
            for r in range(int(road_h_px // cell) + 1):
                for c in range(2):
                    col = (245, 245, 245) if (r + c) % 2 == 0 else (20, 20, 20)
                    pg.draw.rect(surf, col, (gx + c * cell, sy(0) + r * cell, cell, min(cell, road_h_px - r * cell)))
            self.text(surf, "FINISH", (gx + cell, sy(0) - sh_px - 30), 16, (255, 255, 255), True, "midtop")

        ego_v = ego_item[0]
        # safety zones & radar under the vehicles
        if ov.debug and ov.zones:
            self._draw_zones(surf, ego_item, info, sy)
        # shadows
        for v, x, y, h in items:
            sp = pg.Surface((int(v.length * self.base_ppm) + 10, int(v.width * self.base_ppm * lat) + 10), pg.SRCALPHA)
            pg.draw.rect(sp, (0, 0, 0, 70), sp.get_rect(), border_radius=8)
            sp = pg.transform.rotozoom(sp, -math.degrees(self._vis_heading(h)), self.zoom)
            surf.blit(sp, sp.get_rect(center=(self._sx(x) + 3, sy(y) + 4)))
        # vehicles (far ones first so overlaps look natural)
        for v, x, y, h in sorted(items, key=lambda it: (it[0].is_ego, it[1])):
            sxv = self._sx(x)
            if sxv < -80 or sxv > self.view_w + 80:
                continue
            if v.is_ego:
                glow = pg.Surface((int(v.length * self.base_ppm) + 40, int(v.width * self.base_ppm * lat) + 40), pg.SRCALPHA)
                pg.draw.ellipse(glow, (60, 130, 255, 55), glow.get_rect())
                glow = pg.transform.rotozoom(glow, 0, self.zoom)
                surf.blit(glow, glow.get_rect(center=(sxv, sy(y))))
            braking = v.a < -0.7
            blink = v.signal if (v.signal and int(info.time * 3.0) % 2 == 0) else 0
            car = self._car_surface(v, braking, blink)
            car = pg.transform.rotozoom(car, -math.degrees(self._vis_heading(h)), self.zoom)
            surf.blit(car, car.get_rect(center=(sxv, sy(y))))
            if v.crashed:
                pg.draw.rect(surf, RED, car.get_rect(center=(sxv, sy(y))).inflate(-6, -6), 1, border_radius=6)
                self.text(surf, "X", (sxv, sy(y)), 16, RED, True, "center")
        if ov.debug:
            self._draw_debug(surf, items, ego_item, info, sy, ov)
        # effects
        for rg in self.rings:
            t = rg[2] / 0.9
            rad = 8 + 140 * t
            s = pg.Surface((int(rad * 2) + 4, int(rad * 2) + 4), pg.SRCALPHA)
            pg.draw.circle(s, (255, 170, 70, int(220 * (1 - t))), (s.get_width() // 2, s.get_height() // 2), int(rad), 4)
            surf.blit(s, s.get_rect(center=(self._sx(rg[0]), sy(rg[1]))))
        for p in self.particles:
            a = max(0.0, p.life / p.max_life)
            col = tuple(int(c * (0.4 + 0.6 * a)) for c in p.color)
            pg.draw.circle(surf, col, (self._sx(p.x), sy(p.y)), max(1, p.size * a))
        if self.flash > 0.01:
            fl = pg.Surface((self.view_w, self.scene_h), pg.SRCALPHA)
            fl.fill((255, 80, 60, int(90 * self.flash)))
            surf.blit(fl, (0, 0))
        if banner:
            self.text(surf, banner, (self.view_w // 2, 22), 22, AMBER, True, "midtop")
        self._draw_minimap(surf, items, info)

    # ---------------------------------------------------------------- overlays
    def _draw_zones(self, surf, ego_item, info, sy):
        pg = self.pg
        v, x, y, _ = ego_item
        lane = min(info.num_lanes - 1, max(0, int(y // info.lane_width)))
        y0, y1 = lane * info.lane_width, (lane + 1) * info.lane_width
        front = x + v.length / 2
        speed = v.v
        d_brake = speed * speed / (2 * self.cfg.ego.max_brake)
        d_unsafe = d_brake + speed * 1.0
        d_cautious = d_brake + speed * 2.0
        bands = [(0, d_brake, (240, 70, 60, 70)), (d_brake, d_unsafe, (250, 190, 60, 55)),
                 (d_unsafe, d_cautious, (80, 220, 130, 40))]
        for a, b, col in bands:
            if b - a < 0.2:
                continue
            rect = pg.Rect(self._sx(front + a), sy(y0) + 2, (b - a) * self.ppm, sy(y1) - sy(y0) - 4)
            s = pg.Surface(rect.size, pg.SRCALPHA)
            s.fill(col)
            surf.blit(s, rect)

    def _draw_debug(self, surf, items, ego_item, info, sy, ov):
        pg = self.pg
        ego, ex, ey, _ = ego_item
        by_id = {v.vid: (v, x, y, h) for v, x, y, h in items}
        # following distance
        if info.lead_vid is not None and info.lead_gap is not None and info.lead_vid in by_id:
            lv, lx, ly, _ = by_id[info.lead_vid]
            xa, xb = self._sx(ex + ego.length / 2), self._sx(lx - lv.length / 2)
            ya = sy(ey)
            col = GREEN
            if info.headway is not None:
                col = RED if info.headway < 1.0 else AMBER if info.headway < 1.5 else GREEN
            if xb > xa:
                pg.draw.line(surf, col, (xa, ya), (xb, ya), 2)
                for xx in (xa, xb):
                    pg.draw.line(surf, col, (xx, ya - 7), (xx, ya + 7), 2)
                label = f"{info.lead_gap:.1f} m"
                if info.headway is not None:
                    label += f"  {info.headway:.1f}s"
                self.text(surf, label, ((xa + xb) / 2, ya - 12), 13, col, True, "midbottom")
        # trajectories
        if ov.trajectories:
            pts = []
            for k in range(0, 31, 3):
                t = k * 0.1
                pts.append((self._sx(ex + ego.v * t + 0.5 * ego.a * t * t * 0.0), sy(ey)))
            if info.ego_target_y is not None and abs(info.ego_target_y - ey) > 0.05:
                pts = [(px, sy(ey + (info.ego_target_y - ey) * min(1.0, (i / (len(pts) - 1)) * 1.0)))
                       for i, (px, _) in enumerate(pts)]
            if len(pts) > 1:
                for i in range(0, len(pts) - 1, 2):
                    pg.draw.line(surf, (120, 190, 255), pts[i], pts[i + 1], 2)
            for v, x, y, h in items:
                if v.is_ego or v.crashed:
                    continue
                a = (self._sx(x), sy(y))
                b = (self._sx(x + v.v * 2.0), sy(y + v.v * math.tan(h) * 2.0))
                if -50 < a[0] < self.view_w + 50:
                    pg.draw.line(surf, (255, 255, 255, 90), a, b, 1)
        # radar / perception slots
        if ov.radar:
            ex_px, ey_px = self._sx(ex), sy(ey)
            for r_m in (25, 50, 75, 100):
                for sgn in (-1, 1):
                    xx = self._sx(ex + sgn * r_m)
                    if 0 < xx < self.view_w:
                        pg.draw.line(surf, (70, 130, 190), (xx, sy(0) - 6), (xx, sy(0) + 2), 1)
                        self.text(surf, f"{r_m}", (xx, sy(0) - 20), 10, (90, 150, 210), anchor="midbottom")
            for vid, slot, side, gap in info.detections:
                if vid not in by_id:
                    continue
                v, x, y, h = by_id[vid]
                cx, cy = self._sx(x), sy(y)
                col = (70, 200, 255) if slot == "same" else (180, 140, 255)
                pg.draw.line(surf, (*col, 90), (ex_px, ey_px), (cx, cy), 1)
                rr = pg.Rect(0, 0, v.length * self.ppm + 8, v.width * self.ppm * self.lat + 8)
                rr.center = (cx, cy)
                ln = 7
                for (px, py, dx_, dy_) in ((rr.left, rr.top, 1, 1), (rr.right, rr.top, -1, 1),
                                           (rr.left, rr.bottom, 1, -1), (rr.right, rr.bottom, -1, -1)):
                    pg.draw.line(surf, col, (px, py), (px + dx_ * ln, py), 2)
                    pg.draw.line(surf, col, (px, py), (px, py + dy_ * ln), 2)
                self.text(surf, f"{gap:.0f}m", (cx, cy - v.width * self.ppm * self.lat / 2 - 14), 11, col, anchor="midbottom")
        # speed tags
        for v, x, y, h in items:
            if v.is_ego:
                continue
            cx = self._sx(x)
            if 0 < cx < self.view_w:
                self.text(surf, f"{v.v * 3.6:.0f}", (cx, sy(y) + v.width * self.ppm * self.lat / 2 + 4), 11, (200, 205, 215), anchor="midtop")

    def _draw_minimap(self, surf, items, info):
        pg = self.pg
        area = pg.Rect(10, self.scene_h + 8, self.view_w - 20, self.minimap_h - 16)
        pg.draw.rect(surf, (18, 22, 30), area, border_radius=8)
        pg.draw.rect(surf, (50, 58, 74), area, 1, border_radius=8)
        ego = next((it for it in items if it[0].is_ego), items[0])
        back, ahead = 120.0, 360.0
        span = back + ahead
        inner = area.inflate(-16, -26)
        lane_h = inner.height / info.num_lanes

        def mx(x):
            return inner.left + (x - (ego[1] - back)) / span * inner.width

        for k in range(info.num_lanes):
            r = pg.Rect(inner.left, inner.top + k * lane_h + 1, inner.width, lane_h - 2)
            pg.draw.rect(surf, (36, 42, 54), r, border_radius=3)
        gx = mx(info.goal_distance)
        if inner.left <= gx <= inner.right:
            pg.draw.line(surf, (245, 245, 245), (gx, inner.top - 3), (gx, inner.bottom + 3), 3)
        for v, x, y, h in items:
            cx = mx(x)
            if cx < inner.left or cx > inner.right:
                continue
            cy = inner.top + (y / info.lane_width) * lane_h
            w = max(4, v.length / span * inner.width)
            col = EGO_BLUE if v.is_ego else NPC_COLORS[v.color % len(NPC_COLORS)]
            if v.crashed:
                col = RED
            pg.draw.rect(surf, col, (cx - w / 2, cy - lane_h * 0.28, w, lane_h * 0.56), border_radius=2)
        prog = max(0.0, min(1.0, ego[1] / info.goal_distance))
        self.text(surf, f"minimap  -{back:.0f}m .. +{ahead:.0f}m", (area.left + 10, area.top + 2), 10, DIM)
        self.text(surf, f"route {prog * 100:.0f}%", (area.right - 10, area.top + 2), 10, DIM, anchor="topright")

    # --------------------------------------------------------- gym rendering
    def render_env(self, env, mode: str):
        world = env.world
        ego = world.ego
        lane = world.road.lane_of(ego.y)
        gap, lead = world.lane_gaps(ego, lane)[:2]
        info = SceneInfo(world.road.num_lanes, world.road.lane_width, world.goal_distance, world.time,
                         lead_gap=gap if lead is not None else None, lead_vid=lead.vid if lead else None,
                         headway=(gap / ego.v if lead is not None and ego.v > 1 else None))
        self.draw_scene(world.snapshot(), info, overlays=Overlays(debug=True))
        s = self.view
        self.text(s, f"{ego.v * 3.6:5.1f} km/h   step {env.step_count}", (14, 12), 16, TEXT, True)
        if mode == "human":
            self.pg.display.flip()
            self.pg.event.pump()
            self.clock.tick(env.metadata["render_fps"])
            return None
        arr = self.pg.surfarray.array3d(self.view)
        return np.transpose(arr, (1, 0, 2))
