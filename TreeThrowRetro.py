#!/usr/bin/env python3
"""
TreeThrowRetro.py — 1980s pixel-art animation of TreeThrow.py simulations
=========================================================================
Runs the tree-throw model while recording what actually happens (surface,
soil, every tree, every uprooting, snags, fires and storms, running
statistics) and draws each frame as a 320 x 200 EGA-palette picture:

    +--------------------------------------------------------------+
    | title bar: species, year                                     |
    | SIDE VIEW: hillslope transect, soil over bedrock, the trees  |
    |   standing on it, uprootings toppling (root wad + log), fire |
    |   and storms.  Pit-mound relief and soil thickness are       |
    |   vertically exaggerated (factor shown on screen).           |
    +-------------------------------------+------------------------+
    | PLAN VIEW: pit-mound relief map      | HUD: year, trees,      |
    |   (pits blue/grey, mounds yellow),   |   uprootings, rate,    |
    |   tree canopies, new uprootings      |   soil, sediment yield,|
    |   flashing with their fall direction |   flux split, roughness|
    +-------------------------------------+------------------------+

Everything on screen is taken from the simulation: tree positions and sizes,
uprooting positions, sizes and fall directions, the surface and soil, and the
statistics.  Only the sprites, colours and effects are decoration.

Command line (no GUI needed):

    python TreeThrowRetro.py --species spruce --years 2000 --gif spruce.gif
    python TreeThrowRetro.py --species pine --disturbance fire --first-disturbance 600 \\
        --years 1200 --gif fire.gif --scale 3

The GUI (TreeThrowGUI.py) uses the same engine to play the animation.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

import TreeThrow as tt

W_PX, H_PX = 320, 200

# ----------------------------------------------------------------------------
# EGA palette, dithering and a 5x7 bitmap font
# ----------------------------------------------------------------------------
EGA = np.array([
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170), (170, 0, 0), (170, 0, 170), (170, 85, 0),
    (170, 170, 170), (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255), (255, 85, 85),
    (255, 85, 255), (255, 255, 85), (255, 255, 255)], dtype=np.uint8)
(BLACK, BLUE, GREEN, CYAN, RED, MAGENTA, BROWN, LGRAY, DGRAY, LBLUE, LGREEN, LCYAN, LRED, LMAGENTA,
 YELLOW, WHITE) = range(16)
CLEAR = 255                                    # transparent sprite pixel

BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16.0
BAYER_FULL = np.tile(BAYER, (H_PX // 4 + 1, W_PX // 4 + 1))[:H_PX, :W_PX]

_F = {
    "A": ".###.#...##...#######...##...##...#", "B": "####.#...##...#####.#...##...#####.",
    "C": ".###.#...##....#....#....#...#.###.", "D": "####.#...##...##...##...##...#####.",
    "E": "######....#....####.#....#....#####", "F": "######....#....####.#....#....#....",
    "G": ".###.#...##....#.####...##...#.####", "H": "#...##...##...#######...##...##...#",
    "I": ".###...#....#....#....#....#...###.", "J": "..###...#....#....#....#.#..#..##..",
    "K": "#...##..#.#.#..##...#.#..#..#.#...#", "L": "#....#....#....#....#....#....#####",
    "M": "#...###.###.#.##.#.##...##...##...#", "N": "#...##...###..##.#.##..###...##...#",
    "O": ".###.#...##...##...##...##...#.###.", "P": "####.#...##...#####.#....#....#....",
    "Q": ".###.#...##...##...##.#.##..#..##.#", "R": "####.#...##...#####.#.#..#..#.#...#",
    "S": ".#####....#.....###.....#....#####.", "T": "#####..#....#....#....#....#....#..",
    "U": "#...##...##...##...##...##...#.###.", "V": "#...##...##...##...##...#.#.#...#..",
    "W": "#...##...##...##.#.##.#.##.#.#.#.#.", "X": "#...##...#.#.#...#...#.#.#...##...#",
    "Y": "#...##...#.#.#...#....#....#....#..", "Z": "#####....#...#...#...#...#....#####",
    "0": ".###.#...##..###.#.###..##...#.###.", "1": "..#...##....#....#....#....#...###.",
    "2": ".###.#...#....#...#...#...#...#####", "3": "####.....#....#.###.....#....#####.",
    "4": "...#...##..#.#.#..#.#####...#....#.", "5": "######....####.....#....##...#.###.",
    "6": "..##..#...#....####.#...##...#.###.", "7": "#####....#...#...#...#....#....#...",
    "8": ".###.#...##...#.###.#...##...#.###.", "9": ".###.#...##...#.####....#...#..##..",
    " ": "." * 35, ".": "." * 25 + ".##...##..", ",": "." * 20 + ".##....#...#...",
    ":": "......##...##........##...##.......", "-": "." * 15 + "#####" + "." * 15,
    "/": "....#....#...#...#...#...#....#....", "%": "##..###..#...#...#...#...#..###..##",
    "+": ".......#....#..#####..#....#.......", "(": "...#...#...#....#....#.....#.....#.",
    ")": ".#.....#.....#....#....#...#...#...", "=": "..........#####.....#####..........",
    "!": "..#....#....#....#....#.........#..", "?": ".###.#...#....#...#...#.........#..",
    ">": ".#.....#.....#.....#...#...#...#...", "<": "...#...#...#...#.....#.....#.....#.",
    "'": "..#....#...#.......................", "*": ".....#.#.#.###.#####.###.#.#.#.....",
    "#": ".#.#.#####.#.#..#.#..#.#.#####.#.#.", "_": "." * 30 + "#####", "|": "..#.." * 7,
}
FONT = {k: np.array([c == "#" for c in v], dtype=bool).reshape(7, 5) for k, v in _F.items()}


def draw_text(img, x, y, text, color, shadow=None):
    """Draw upper-case text with the 5x7 font (6 px advance); returns the end x."""
    for ch in str(text).upper():
        g = FONT.get(ch, FONT["?"])
        if shadow is not None:
            _stamp(img, g, x + 1, y + 1, shadow)
        _stamp(img, g, x, y, color)
        x += 6
    return x


def _stamp(img, mask, x, y, color):
    h, w = mask.shape
    y0, x0 = max(y, 0), max(x, 0)
    y1, x1 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
    if y0 >= y1 or x0 >= x1:
        return
    sub = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    img[y0:y1, x0:x1][sub] = color


def dither(c0, c1, frac, ys, xs):
    """Ordered dither between two palette colours; frac may be an array (0 = c0, 1 = c1)."""
    return np.where(BAYER_FULL[ys % H_PX, xs % W_PX] < frac, c1, c0).astype(np.uint8)


def text_width(text):
    return 6 * len(str(text))


# ----------------------------------------------------------------------------
# Frame capture
# ----------------------------------------------------------------------------


@dataclass
class Frame:
    t: float
    prof: np.ndarray            # mean elevation along x (m)
    tz: np.ndarray              # transect elevation along x (m)
    th: np.ndarray              # transect soil thickness (m)
    relief: np.ndarray          # detrended pit-mound relief, downsampled (m)
    trees: Dict[str, np.ndarray]
    snags: Dict[str, np.ndarray]
    events: np.ndarray          # uprootings since the previous frame: t, x, y, R, H, phi, vol, disp, exp, cause
    disturbances: List[tuple]
    stats: Dict[str, float]


@dataclass
class RunMeta:
    species_key: str
    species_name: str
    kind: str
    length: float               # hillslope length incl. channel cell (m)
    width: float
    x: np.ndarray
    years: float
    frame_every: float
    area_ha: float
    slope: float
    mature_R: float
    extras: Dict[str, float] = field(default_factory=dict)


def species_kind(sp) -> str:
    k = (sp.key + " " + sp.name).lower()
    for kind in ("spruce", "aspen", "pine"):
        if kind in k:
            return kind
    if "fir" in k or "conifer" in k:
        return "spruce"
    return "deciduous"


def _downsample(a, max_x=320, max_y=180):
    kx = max(int(math.ceil(a.shape[0] / max_x)), 1)
    ky = max(int(math.ceil(a.shape[1] / max_y)), 1)
    nx, ny = (a.shape[0] // kx) * kx, (a.shape[1] // ky) * ky
    return a[:nx, :ny].reshape(nx // kx, kx, ny // ky, ky).mean(axis=(1, 3)).astype(np.float32)


class FrameRecorder:
    """Model-step callback that captures a Frame every `frame_every` years."""

    def __init__(self, model, frame_every, progress: Optional[Callable[[float, str], None]] = None,
                 stop=None):
        self.m, self.dt = model, float(frame_every)
        self.frames: List[Frame] = []
        self.n_logged = 0
        self.n_dist = 0
        self.next_t = 0.0
        self.progress, self.stop = progress, stop
        self.years = model.cfg.years
        self._last_pct = -1
        self.extra: List[float] = []     # extra capture times after a disturbance
        self.capture()

    def capture(self):
        m = self.m
        z = m.z
        j = m.ny // 2
        rel = z[:-1] - z[:-1].mean(axis=1, keepdims=True)
        live = m.age >= 0
        size = tt.growth_factor(m.age[live], m.sp.maturity_age)
        ev = np.array(m.event_log[self.n_logged:], dtype=float) if len(m.event_log) > self.n_logged \
            else np.zeros((0, 10))
        self.n_logged = len(m.event_log)
        dist = list(m.disturbance_log[self.n_dist:])
        self.n_dist = len(m.disturbance_log)
        area = m.L_active * m.W
        T = max(m.time, 1e-9)
        q_tt = m.moment_tt + m.moment_fresh
        tot = q_tt + m.moment_diff
        stats = {
            "trees": int(live.sum()), "snags": int(m.snags["x"].size), "uprootings": int(m.n_events),
            "deaths": int(m.n_deaths),
            "rate": m.n_events / (area / 1e4) / T if m.time > 0 else 0.0,
            "share": q_tt / tot if tot > 0 else 0.0,
            "roughness_cm": 100 * m.roughness(),
            "soil_m": float(m.h[:-1].mean()) if m.cfg.soil_model else float("nan"),
            "yield_m3_m": (m.export_tt + m.export_diff) / m.W,
            "relief_drop_m": float(m.z_initial[0].mean() - z[0].mean()),
        }
        self.frames.append(Frame(
            t=m.time, prof=z.mean(axis=1).astype(np.float32), tz=z[:, j].astype(np.float32),
            th=m.h[:, j].astype(np.float32), relief=_downsample(rel),
            trees={"x": m.tx[live].astype(np.float32), "y": m.ty[live].astype(np.float32),
                   "size": size.astype(np.float32)},
            snags={"x": m.snags["x"].astype(np.float32), "y": m.snags["y"].astype(np.float32)},
            events=ev, disturbances=dist, stats=stats))
        self.next_t = m.time + self.dt

    def __call__(self, model):
        if len(model.disturbance_log) > self.n_dist:
            # Always show a fire or storm, and the decades after it, whatever the frame spacing.
            self.capture()
            self.extra = [model.time + d for d in (2, 5, 10, 20, 35) if d < self.dt * 6]
        elif self.extra and model.time >= self.extra[0] - 1e-9:
            while self.extra and model.time >= self.extra[0] - 1e-9:
                self.extra.pop(0)
            self.capture()
        elif model.time >= self.next_t - 1e-9:
            self.capture()
        if self.progress is not None:
            pct = int(100 * model.time / self.years)
            if pct != self._last_pct:
                self._last_pct = pct
                self.progress(model.time / self.years, f"year {model.time:.0f} / {self.years:.0f}  "
                                                       f"uprootings {model.n_events}")
        if self.stop is not None and self.stop.is_set():
            return False
        return True


def run_with_frames(sp, cfg, n_frames=200, frame_every=None, progress=None, stop=None):
    """Run the model and capture animation frames.  Returns (model, frames, meta)."""
    model = tt.TreeThrowModel(sp, cfg)
    fe = frame_every if frame_every else max(cfg.years / max(n_frames, 2), cfg.dt)
    rec = FrameRecorder(model, fe, progress, stop)
    model.run(progress=False, callback=rec)
    if rec.frames[-1].t != model.time:
        rec.capture()
    meta = RunMeta(species_key=sp.key, species_name=sp.name, kind=species_kind(sp),
                   length=float(model.x[-1] + 0.5 * cfg.dx), width=float(model.W), x=model.x.copy(),
                   years=float(model.time), frame_every=fe, area_ha=model.L_active * model.W / 1e4,
                   slope=cfg.slope, mature_R=0.5 * sp.root_plate_diameter)
    return model, rec.frames, meta


# ----------------------------------------------------------------------------
# Sprites
# ----------------------------------------------------------------------------
HMAX = {"pine": 22, "spruce": 26, "deciduous": 20, "aspen": 18}
TREE_COLOURS = {   # near: (canopy, highlight, trunk); far versions are darker
    "pine": (GREEN, LGREEN, BROWN), "spruce": (GREEN, CYAN, BROWN),
    "deciduous": (GREEN, LGREEN, BROWN), "aspen": (LGREEN, YELLOW, WHITE),
}
MAP_COLOURS = {"pine": (GREEN, LGREEN), "spruce": (GREEN, CYAN), "deciduous": (LGREEN, GREEN),
               "aspen": (YELLOW, LGREEN)}


def make_tree_sprite(kind, h, far=False):
    """Pixel-art tree of height h px.  Returns an index array (CLEAR = transparent)."""
    h = max(int(h), 3)
    canopy, hi, trunk = TREE_COLOURS[kind]
    if far:
        canopy, hi, trunk = (GREEN, YELLOW, LGRAY) if kind == "aspen" else (GREEN, GREEN, DGRAY)
    if kind in ("pine", "spruce"):
        w = max(3, int(round(h * (0.42 if kind == "spruce" else 0.6))) | 1)
        spr = np.full((h, w), CLEAR, np.uint8)
        cx = w // 2
        th = max(1, h // 6)
        crown = h - th
        for r in range(crown):
            frac = (r + 1) / crown
            tier = (r % 4) / 4.0 if kind == "spruce" else (r % 3) / 3.0
            half = int(round(frac * (w / 2) * (0.75 + 0.25 * tier)))
            spr[r, max(cx - half, 0):cx + half + 1] = canopy
            spr[r, cx + 1:cx + half + 1] = hi if (r % 2 == 0 or far) else canopy
            spr[r, max(cx - half, 0)] = BLACK
            spr[r, min(cx + half, w - 1)] = BLACK
        spr[0, cx] = BLACK if h > 6 else canopy
        spr[crown:, cx] = trunk
        if h >= 16:
            spr[crown:, cx - 1] = trunk if trunk != WHITE else BLACK
        return spr
    if kind == "aspen":
        w = max(3, int(round(h * 0.5)) | 1)
        ch = max(2, int(round(h * 0.62)))
        spr = np.full((h, w), CLEAR, np.uint8)
        cx = w // 2
        yy, xx = np.mgrid[0:ch, 0:w]
        ell = ((xx - cx) / (w / 2)) ** 2 + ((yy - ch / 2) / (ch / 2)) ** 2 <= 1.0
        light = (xx - cx) + (ch / 2 - yy) > 0
        spr[:ch][ell] = np.where(light[ell] & (BAYER[yy[ell] % 4, xx[ell] % 4] < 0.6), hi, canopy)
        edge = ell & ~np.roll(ell, 1, 1) | ell & ~np.roll(ell, -1, 1)
        spr[:ch][edge] = BLACK if not far else DGRAY
        spr[ch - 1:, cx] = trunk
        for r in range(ch, h, 3):
            spr[r, cx] = BLACK
        return spr
    # deciduous: round crown, forked trunk
    w = max(4, int(round(h * 0.85)))
    ch = max(3, int(round(h * 0.62)))
    spr = np.full((h, w), CLEAR, np.uint8)
    cx = (w - 1) / 2
    yy, xx = np.mgrid[0:ch, 0:w]
    blob = ((xx - cx) / (w / 2)) ** 2 + ((yy - ch / 2) / (ch / 2)) ** 2 <= 1.0
    if w >= 8:   # lumpy crown
        blob |= ((xx - cx * 0.55) / (w / 3.2)) ** 2 + ((yy - ch * 0.62) / (ch / 2.6)) ** 2 <= 1.0
        blob |= ((xx - cx * 1.45) / (w / 3.2)) ** 2 + ((yy - ch * 0.62) / (ch / 2.6)) ** 2 <= 1.0
    light = (xx - cx) * 0.6 + (ch / 2 - yy) > 0
    spr[:ch][blob] = np.where(light[blob] & (BAYER[yy[blob] % 4, xx[blob] % 4] < 0.55), hi, canopy)
    edge = blob & (~np.roll(blob, 1, 1) | ~np.roll(blob, -1, 1) | ~np.roll(blob, 1, 0))
    spr[:ch][edge] = BLACK if not far else DGRAY
    c = int(round(cx))
    spr[ch - 2:, c] = trunk
    if h >= 12:
        spr[ch - 2:, c + 1] = trunk
        spr[ch - 3, c - 1] = trunk
        spr[ch - 3, c + 2] = trunk
    return spr


def make_snag_sprite(h):
    h = max(int(h * 0.85), 3)
    w = 5
    spr = np.full((h, w), CLEAR, np.uint8)
    spr[:, 2] = DGRAY
    spr[h // 3:, 1] = LGRAY if h > 8 else DGRAY
    for r in range(2, h - 3, 4):
        spr[r, 3 if (r // 4) % 2 else 1] = DGRAY
    spr[0, 2] = BLACK
    return spr


def blit(img, spr, cx, base_y, y_min=0, y_max=H_PX, x_min=0, x_max=W_PX):
    """Paste a sprite with its bottom-centre at (cx, base_y), clipped to a box."""
    h, w = spr.shape
    x0, y0 = int(cx) - w // 2, int(base_y) - h + 1
    xs0, ys0 = max(x0, x_min), max(y0, y_min)
    xs1, ys1 = min(x0 + w, x_max), min(y0 + h, y_max)
    if xs0 >= xs1 or ys0 >= ys1:
        return
    sub = spr[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    dst = img[ys0:ys1, xs0:xs1]
    mask = sub != CLEAR
    dst[mask] = sub[mask]


def line(img, x0, y0, x1, y1, color, box=(0, 0, W_PX, H_PX)):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.round(np.linspace(x0, x1, n)).astype(int)
    ys = np.round(np.linspace(y0, y1, n)).astype(int)
    ok = (xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3])
    img[ys[ok], xs[ok]] = color


def disc(img, cx, cy, r, color, box=(0, 0, W_PX, H_PX), outline=None):
    r = max(float(r), 0.5)
    x0, x1 = int(math.floor(cx - r)), int(math.ceil(cx + r))
    y0, y1 = int(math.floor(cy - r)), int(math.ceil(cy + r))
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    ok = inside & (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])
    img[yy[ok], xx[ok]] = color
    if outline is not None and r >= 2:
        ring = inside & ((xx - cx) ** 2 + (yy - cy) ** 2 > (r - 1) ** 2)
        ok = ring & (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])
        img[yy[ok], xx[ok]] = outline


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------
SIDE = (0, 11, W_PX, 100)           # x0, y0, x1, y1 of the side view
MAPB = (2, 111, 202, 198)           # plan view box
HUD = (206, 102, 318, 198)
RAMP = [BLUE, DGRAY, BROWN, GREEN, LGREEN, YELLOW]   # deep pit -> high mound


class RetroRenderer:
    def __init__(self, frames: List[Frame], meta: RunMeta):
        self.frames, self.meta = frames, meta
        self.kind = meta.kind
        self._sprites: Dict[tuple, np.ndarray] = {}
        f0 = frames[0]
        # Vertical scale of the side view: fit the whole run, leave headroom for trees.
        zmax = max(float(f.tz.max()) for f in frames)
        zmin = min(float((f.tz - f.th).min()) for f in frames) - 0.5
        self.y_ground_top = SIDE[1] + HMAX.get(self.kind, 22) + 6
        self.y_ground_bot = SIDE[3] - 4
        self.vpx_per_m = (self.y_ground_bot - self.y_ground_top) / max(zmax - zmin, 1e-6)
        self.zmax = zmax
        self.hpx_per_m = (W_PX - 1) / meta.length
        # Micro-relief exaggeration so that typical pit-mounds show as a few pixels.
        dev = np.concatenate([np.abs(f.tz - f.prof) for f in frames[len(frames) // 2:]])
        p99 = float(np.percentile(dev, 99)) if dev.size else 0.2
        self.exag = int(np.clip(round(3.0 / max(p99 * self.vpx_per_m, 1e-6)), 1, 25))
        self.soil_exag = min(self.exag, 3)        # soil band: modest exaggeration so bedrock stays visible
        # Relief colour scale for the plan view.
        allrel = np.concatenate([np.abs(f.relief).ravel() for f in frames[len(frames) // 3:]])
        self.vrel = max(float(np.percentile(allrel, 99.5)) if allrel.size else 0.1, 0.05)
        # Plan view geometry.
        bw, bh = MAPB[2] - MAPB[0], MAPB[3] - MAPB[1]
        s = min(bw / meta.length, bh / meta.width)
        self.map_s = s
        self.map_w, self.map_h = max(int(meta.length * s), 1), max(int(meta.width * s), 1)
        self.map_x0 = MAPB[0] + (bw - self.map_w) // 2
        self.map_y0 = MAPB[1] + (bh - self.map_h) // 2
        nx, ny = f0.relief.shape
        self.map_i = np.minimum((np.arange(self.map_w) + 0.5) / self.map_w * nx, nx - 1).astype(int)
        self.map_j = np.minimum((np.arange(self.map_h) + 0.5) / self.map_h * ny, ny - 1).astype(int)
        self.rough_max = max(max(f.stats["roughness_cm"] for f in frames), 1.0)
        self.fire_times = [d[0] for f in frames for d in f.disturbances if d[1] == "fire"]
        self.storm_times = [d[0] for f in frames for d in f.disturbances if d[1] == "windstorm"]

    # ---------------------------------------------------------------- helpers
    def sprite(self, key, *args):
        k = (key,) + args
        if k not in self._sprites:
            self._sprites[k] = make_snag_sprite(*args) if key == "snag" else make_tree_sprite(*args)
        return self._sprites[k]

    def col(self, x):
        return np.round(x * self.hpx_per_m).astype(int)

    def zpix(self, z):
        return np.round(self.y_ground_top + (self.zmax - z) * self.vpx_per_m).astype(int)

    def _state(self, t):
        fire = [t - tf for tf in self.fire_times if 0 <= t - tf <= 25]
        storm = [t - ts for ts in self.storm_times if 0 <= t - ts <= self.meta.frame_every * 1.01]
        return (min(fire) if fire else None), (min(storm) if storm else None)

    # ---------------------------------------------------------------- panels
    def _sky(self, img, i, fire_age, storm_age):
        x0, y0, x1, y1 = SIDE
        ys, xs = np.mgrid[y0:y1, x0:x1]
        frac = (ys - y0) / (y1 - y0)
        if fire_age is not None:
            k = fire_age / 25.0
            img[y0:y1, x0:x1] = np.where(frac < 0.5 + 0.4 * k, dither(BLACK, RED, frac * 1.6 * (1 - k) + 0.2, ys, xs),
                                         dither(RED, BROWN, frac, ys, xs))
        elif storm_age is not None:
            img[y0:y1, x0:x1] = dither(DGRAY, BLUE, frac * 0.9, ys, xs)
        else:
            top = dither(BLUE, LBLUE, np.clip(frac * 1.8, 0, 1), ys, xs)
            low = dither(LBLUE, LCYAN, np.clip((frac - 0.55) * 2.2, 0, 1), ys, xs)
            img[y0:y1, x0:x1] = np.where(frac < 0.55, top, low)
            disc(img, 292, y0 + 13, 7, YELLOW, box=SIDE, outline=WHITE)        # sun
            for k, (cx, cy) in enumerate(((60, 18), (150, 24), (230, 16))):  # drifting pixel clouds
                cxx = (cx + i * (1 + k % 2)) % (W_PX + 40) - 20
                for dxc, dyc, r in ((0, 0, 4), (5, 1, 3), (-5, 1, 3), (2, -2, 3)):
                    disc(img, cxx + dxc, y0 + cy + dyc, r, WHITE, box=SIDE)

    def _terrain(self, img, f):
        x0, y0, x1, y1 = SIDE
        cols = np.arange(W_PX)
        xm = cols / self.hpx_per_m
        prof = np.interp(xm, self.meta.x, f.prof)
        tz = np.interp(xm, self.meta.x, f.tz)
        th = np.interp(xm, self.meta.x, f.th)
        zs = prof + self.exag * (tz - prof)
        ys = self.zpix(zs)
        yb = np.maximum(ys + np.round(th * self.soil_exag * self.vpx_per_m).astype(int), ys + 1)
        rows = np.arange(y0, y1)[:, None]
        soil = (rows >= ys[None, :]) & (rows < yb[None, :])
        rock = rows >= yb[None, :]
        R, C = np.mgrid[y0:y1, 0:W_PX]
        soil_col = dither(BROWN, DGRAY, 0.18, R, C)
        rock_col = dither(DGRAY, LGRAY, 0.22, R, C)
        panel = img[y0:y1]
        panel[soil] = soil_col[soil]
        panel[rock] = rock_col[rock]
        grass = rows == ys[None, :]
        panel[grass] = np.broadcast_to(np.where(cols % 3 == 0, LGREEN, GREEN), panel.shape)[grass]
        # Channel at the foot of the slope.
        for c in range(W_PX - 6, W_PX):
            ya, yb = max(ys[c] - 1, y0), min(ys[c] + 3, y1)
            img[ya:yb, c] = BLUE
            img[ya, c] = LBLUE
        self._ys = ys
        return ys

    def _side_trees(self, img, f, i, fire_age):
        tr = f.trees
        n = tr["x"].size
        if n == 0 and f.snags["x"].size == 0:
            return
        W = self.meta.width
        order = np.argsort(-np.abs(tr["y"] - W / 2))          # far trees first
        if n > 450:
            order = order[np.linspace(0, n - 1, 450).astype(int)]
        hmax = HMAX.get(self.kind, 22)
        for k in order:
            c = int(np.clip(round(tr["x"][k] * self.hpx_per_m), 0, W_PX - 1))
            far = abs(tr["y"][k] - W / 2) > W / 4
            h = int(round(3 + (hmax - 3) * tr["size"][k] * (0.8 if far else 1.0)))
            blit(img, self.sprite("tree", self.kind, h, far), c, self._ys[c], y_min=SIDE[1], y_max=SIDE[3])
        for xs in f.snags["x"]:
            c = int(np.clip(round(xs * self.hpx_per_m), 0, W_PX - 1))
            blit(img, self.sprite("snag", hmax), c, self._ys[c], y_min=SIDE[1], y_max=SIDE[3])
            if fire_age is not None and fire_age < 8 and (i + c) % 2 == 0:    # flames flicker on fresh snags
                for dy in range(1, 4):
                    yy = self._ys[c] - hmax // 2 - dy * 2 + (i % 3)
                    if SIDE[1] <= yy < SIDE[3]:
                        img[yy, max(c - 1, 0):c + 2] = (YELLOW, LRED, RED)[dy - 1]

    def _side_events(self, img, i):
        hmax = HMAX.get(self.kind, 22)
        timber_done = False
        for age in (2, 1, 0):
            if i - age < 0:
                continue
            ev = self.frames[i - age].events
            for e in ev:
                c = int(np.clip(round(e[1] * self.hpx_per_m), 0, W_PX - 1))
                g = self._ys[c]
                R = e[3]
                cos = math.cos(math.radians(e[5]))
                sgn = 1 if cos >= 0 else -1
                size = min(R / max(self.meta.mature_R, 1e-6), 1.3)
                L = max(int(round(hmax * size * max(abs(cos), 0.35))), 3)
                rw = max(1.5, 0.25 * hmax * size)      # root wad drawn to the same scale as the sprites
                log_c = BROWN if age == 0 else DGRAY
                line(img, c, g - 1, c + sgn * L, g - 1, log_c, box=SIDE)
                if age == 0:
                    line(img, c, g - 2, c + sgn * (L - 1), g - 2, BROWN, box=SIDE)
                    disc(img, c + sgn * L, g - 2, 2, LGREEN if self.kind != "aspen" else YELLOW, box=SIDE)
                    disc(img, c - sgn * 1, g - rw, rw, BROWN, box=SIDE, outline=BLACK)        # upturned root wad
                    rng = np.random.default_rng(int(e[0] * 1000 + e[1] * 10))
                    for _ in range(6):                                             # flying soil
                        px = int(c + rng.integers(-5, 6))
                        py = int(g - rw * 2 - rng.integers(1, 7))
                        if SIDE[1] <= py < SIDE[3] and 0 <= px < W_PX:
                            img[py, px] = (YELLOW, BROWN)[rng.integers(0, 2)]
                    if e[9] == 1 and not timber_done and SIDE[1] + 20 <= g - rw * 2 - 9 < SIDE[3]:
                        draw_text(img, max(min(c - 18, W_PX - 44), 0), int(g - rw * 2 - 9), "TIMBER!", WHITE, BLACK)
                        timber_done = True
                else:
                    disc(img, c - sgn * 1, g - rw * 0.4, rw * (0.8 if age == 1 else 0.6), BROWN, box=SIDE)

    def _side_effects(self, img, i, storm_age):
        if storm_age is None:
            return
        rng = np.random.default_rng(i)
        for _ in range(140):                        # rain
            x = int(rng.integers(0, W_PX))
            y = int(rng.integers(SIDE[1], SIDE[3] - 4))
            line(img, x, y, x - 2, y + 3, LBLUE, box=SIDE)
        if storm_age < 1e-6 or i % 2 == 0:          # lightning
            x = int(rng.integers(40, 280))
            y = SIDE[1]
            pts = [(x, y)]
            while y < SIDE[1] + 40:
                x += int(rng.integers(-6, 7))
                y += int(rng.integers(5, 10))
                pts.append((x, y))
            for (a, b), (cc, d) in zip(pts[:-1], pts[1:]):
                line(img, a, b, cc, d, YELLOW, box=SIDE)
                line(img, a + 1, b, cc + 1, d, WHITE, box=SIDE)

    def _side_labels(self, img, f):
        lab = f"RELIEF X{self.exag} SOIL X{self.soil_exag}"
        img[SIDE[1] + 1:SIDE[1] + 19, 1:max(text_width(lab), 54) + 5] = BLACK
        draw_text(img, 3, SIDE[1] + 2, "SIDE VIEW", WHITE)
        draw_text(img, 3, SIDE[1] + 11, lab, LGRAY)
        draw_text(img, 3, SIDE[3] - 9, "RIDGE", WHITE, BLACK)
        draw_text(img, W_PX - text_width("CHANNEL") - 8, SIDE[3] - 9, "CHANNEL", WHITE, BLACK)

    def _map(self, img, i, f):
        x0, y0 = self.map_x0, self.map_y0
        img[MAPB[1] - 1:MAPB[3] + 1, MAPB[0] - 1:MAPB[2] + 1] = BLACK
        rel = f.relief[self.map_i][:, self.map_j].T        # rows = y, cols = x
        r = np.clip(rel / self.vrel, -1, 1)
        level = np.where(r < 0, 3 + 3 * r, 3 + 2 * r)       # 0 .. 5 on RAMP
        lo = np.clip(np.floor(level).astype(int), 0, 4)
        frac = level - lo
        ys, xs = np.mgrid[y0:y0 + self.map_h, x0:x0 + self.map_w]
        ramp = np.array(RAMP, np.uint8)
        img[y0:y0 + self.map_h, x0:x0 + self.map_w] = np.where(BAYER_FULL[ys, xs] < frac, ramp[lo + 1], ramp[lo])
        box = (x0, y0, x0 + self.map_w, y0 + self.map_h)
        s = self.map_s
        canopy, hi = MAP_COLOURS.get(self.kind, (GREEN, LGREEN))
        tr = f.trees
        big = tr["size"] > 0.5
        for k in np.flatnonzero(~big):
            px, py = x0 + tr["x"][k] * s, y0 + tr["y"][k] * s
            if box[0] <= px < box[2] and box[1] <= py < box[3]:
                img[int(py), int(px)] = canopy
        rc = max(self.meta.mature_R * 1.4 * s, 1.0)
        for k in np.flatnonzero(big):
            px, py = x0 + tr["x"][k] * s, y0 + tr["y"][k] * s
            disc(img, px, py, rc * tr["size"][k], canopy, box=box)
            if box[0] <= px + 1 < box[2] and box[1] <= py - 1 < box[3]:
                img[int(py - 1), int(px + 1)] = hi
        for xs_, ys_ in zip(f.snags["x"], f.snags["y"]):
            px, py = x0 + xs_ * s, y0 + ys_ * s
            if box[0] <= px < box[2] and box[1] <= py < box[3]:
                img[int(py), int(px)] = LGRAY
        for age in (2, 1, 0):
            if i - age < 0:
                continue
            for e in self.frames[i - age].events:
                px, py = x0 + e[1] * s, y0 + e[2] * s
                L = max(e[3] * 2.2 * s, 2)
                a = math.radians(e[5])
                colr = YELLOW if age == 0 else BROWN
                line(img, px, py, px + L * math.cos(a), py + L * math.sin(a), colr, box=box)
                if age == 0:
                    disc(img, px, py, 1.6, WHITE if i % 2 == 0 else LRED, box=box)
        # frame + labels
        img[MAPB[1] - 2, MAPB[0] - 1:MAPB[2] + 1] = LGRAY
        img[MAPB[3] + 1, MAPB[0] - 1:MAPB[2] + 1] = LGRAY
        img[MAPB[1] - 2:MAPB[3] + 2, MAPB[0] - 2] = LGRAY
        img[MAPB[1] - 2:MAPB[3] + 2, MAPB[2] + 1] = LGRAY

    def _hud(self, img, i, f):
        x0, y0, x1, y1 = HUD
        img[y0 - 1:y1 + 1, x0 - 2:x1 + 2] = BLACK
        img[y0 - 1, x0 - 2:x1 + 2] = LGRAY
        img[y1, x0 - 2:x1 + 2] = LGRAY
        img[y0 - 1:y1 + 1, x0 - 2] = LGRAY
        img[y0 - 1:y1 + 1, x1 + 1] = LGRAY
        s = f.stats
        lines = [
            ("TREES", f"{s['trees']}", LGREEN),
            ("SNAGS", f"{s['snags']}", LGRAY),
            ("UPROOT", f"{s['uprootings']}", YELLOW),
            ("RATE", f"{s['rate']:.2f}/HA", YELLOW),
            ("SOIL", "--" if not np.isfinite(s['soil_m']) else f"{s['soil_m']:.2f}M", BROWN),
            ("YIELD", f"{s['yield_m3_m']:.2f}M3/M", LCYAN),
        ]
        y = y0 + 2
        for lab, val, c in lines:
            draw_text(img, x0 + 1, y, lab, LGRAY)
            draw_text(img, x1 - text_width(val), y, val, c)
            y += 9
        # flux split bar: tree throw vs creep
        draw_text(img, x0 + 1, y, "THROW", YELLOW)
        pct = f"{100 * s['share']:.0f}%"
        draw_text(img, x1 - text_width(pct), y, pct, YELLOW)
        y += 9
        bw = x1 - x0 - 2
        k = int(round(bw * s["share"]))
        img[y:y + 4, x0 + 1:x0 + 1 + bw] = BROWN
        img[y:y + 4, x0 + 1:x0 + 1 + k] = YELLOW
        y += 7
        # roughness sparkline
        draw_text(img, x0 + 1, y, "ROUGH", LGRAY)
        rv = f"{s['roughness_cm']:.1f}CM"
        draw_text(img, x1 - text_width(rv), y, rv, LCYAN)
        y += 9
        gh = y1 - y - 2
        if gh >= 6:
            img[y:y + gh, x0 + 1:x1] = BLACK
            img[y + gh - 1, x0 + 1:x1] = DGRAY
            vals = np.array([fr.stats["roughness_cm"] for fr in self.frames[:i + 1]])
            n = len(self.frames)
            xs = x0 + 1 + np.round(np.arange(vals.size) / max(n - 1, 1) * (x1 - x0 - 3)).astype(int)
            yv = y + gh - 2 - np.round(vals / self.rough_max * (gh - 3)).astype(int)
            for a in range(1, vals.size):
                line(img, xs[a - 1], yv[a - 1], xs[a], yv[a], LCYAN)
            if vals.size:
                img[yv[-1], xs[-1]] = WHITE

    def _title(self, img, f, fire_age, storm_age, i):
        img[0:10, :] = BLUE
        img[10, :] = LBLUE
        name = f"TREE-THROW  {self.meta.species_key[:12]}"
        draw_text(img, 3, 2, name, YELLOW, BLACK)
        yr = f"YEAR {int(round(f.t)):05d}"
        draw_text(img, W_PX - text_width(yr) - 3, 2, yr, WHITE, BLACK)
        tag = None
        if fire_age is not None and fire_age < 3:
            tag, c = "FIRE!", LRED
        elif storm_age is not None:
            tag, c = "STORM!", LCYAN
        if tag and i % 2 == 0:
            draw_text(img, 175, 2, tag, c, BLACK)
        draw_text(img, 4, 102, "PLAN VIEW", WHITE)
        draw_text(img, 66, 102, "PIT", LBLUE)
        draw_text(img, 90, 102, "MOUND", YELLOW)
        draw_text(img, 128, 102, f"X/Y {self.meta.length:.0f}X{self.meta.width:.0f}M", LGRAY)

    def render(self, i) -> np.ndarray:
        """Return frame i as a (200, 320, 3) uint8 RGB array."""
        i = int(np.clip(i, 0, len(self.frames) - 1))
        f = self.frames[i]
        img = np.zeros((H_PX, W_PX), np.uint8)
        fire_age, storm_age = self._state(f.t)
        self._sky(img, i, fire_age, storm_age)
        self._terrain(img, f)
        self._side_trees(img, f, i, fire_age)
        self._side_events(img, i)
        self._side_effects(img, i, storm_age)
        self._side_labels(img, f)
        img[SIDE[3]:, :] = BLACK
        self._map(img, i, f)
        self._hud(img, i, f)
        self._title(img, f, fire_age, storm_age, i)
        if i == len(self.frames) - 1:
            msg = "SIMULATION COMPLETE"
            w = text_width(msg) + 8
            xb = (W_PX - w) // 2
            img[44:58, xb:xb + w] = BLACK
            img[44, xb:xb + w] = YELLOW
            img[57, xb:xb + w] = YELLOW
            draw_text(img, xb + 4, 47, msg, YELLOW)
        return EGA[img]


def upscale(rgb: np.ndarray, scale: int = 3, scanlines: bool = True) -> np.ndarray:
    """Nearest-neighbour upscale with optional CRT scanlines."""
    out = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    if scanlines and scale >= 2:
        out = out.copy()
        out[scale - 1::scale] = (out[scale - 1::scale].astype(np.uint16) * 5 // 9).astype(np.uint8)
    return out


def to_ppm(rgb: np.ndarray) -> bytes:
    """Binary PPM (Tk's PhotoImage reads this without Pillow)."""
    h, w, _ = rgb.shape
    return b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb).tobytes()


def save_gif(renderer: RetroRenderer, path, scale=2, scanlines=True, fps=12, progress=None):
    from PIL import Image
    imgs = []
    n = len(renderer.frames)
    for k in range(n):
        imgs.append(Image.fromarray(upscale(renderer.render(k), scale, scanlines)))
        if progress:
            progress((k + 1) / n, f"rendering frame {k + 1}/{n}")
    pal = Image.new("P", (1, 1))
    palette = list(EGA.ravel()) + list((EGA * 5 // 9).astype(np.uint8).ravel())
    pal.putpalette(palette + [0] * (768 - len(palette)))
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    durations = [int(1000 / fps)] * (n - 1) + [2500]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=False)
    return path


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------


def main(argv=None):
    p = tt.build_parser()
    p.description = "Run TreeThrow.py and render a retro pixel animation (GIF)."
    g = p.add_argument_group("animation")
    g.add_argument("--gif", default="tree_throw_retro.gif", help="output GIF path")
    g.add_argument("--frames", type=int, default=160, help="number of animation frames")
    g.add_argument("--scale", type=int, default=2, help="pixel scale factor")
    g.add_argument("--fps", type=int, default=12)
    g.add_argument("--no-scanlines", action="store_true")
    g.add_argument("--png", help="also save the final frame as PNG")
    a = p.parse_args(argv)
    cfg = tt.config_from_args(a)
    base = tt.load_species_file(a.species_file) if a.species_file else tt.SPECIES[a.species]
    sp = tt.apply_overrides(base, a)

    def prog(frac, msg):
        if not a.quiet:
            print(f"\r  {100 * frac:5.1f} %  {msg}      ", end="", flush=True)

    model, frames, meta = run_with_frames(sp, cfg, n_frames=a.frames, progress=prog)
    if not a.quiet:
        print()
    r = RetroRenderer(frames, meta)
    save_gif(r, a.gif, scale=a.scale, scanlines=not a.no_scanlines, fps=a.fps, progress=prog)
    if a.png:
        from PIL import Image
        Image.fromarray(upscale(r.render(len(frames) - 1), a.scale, not a.no_scanlines)).save(a.png)
    if not a.quiet:
        print(f"\nSaved {a.gif} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
