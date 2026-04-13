#!/usr/bin/env python3
"""
HillslopeSim.py — Geomorphic Slope Stability via Discrete-Element Tilting
==========================================================================
A physics-based hillslope stability model that fractures a cohesive slope
into discrete particles (rectangles or spheres), then progressively tilts
the domain (rotates gravity by 90°) to test slope failure.

Two physics backends:
  - pymunk  (2D): fast Chipmunk-based rigid-body engine
  - pychrono (3D): full Chrono engine with native cohesion support

The default analysis perspective is through the Y-plane (Z vertical,
X longitudinal downslope).  A quasi-2D mode for pychrono automatically
sizes the Y dimension to ~1 particle width.

Diagnostic metrics reported:
  - Cumulative displacement vs. rotation angle
  - Coordination number (avg. contacts per particle)
  - Touching-surface fraction (contacting material surface area / total material surface area)
  - Bulk density evolution
  - Mobilised friction angle
  - Kinetic energy (dynamic instability indicator)
  - Fabric tensor anisotropy (directional contact distribution)
  - Factor of Safety estimate (driving vs. resisting moments)

Run
---
    python HillslopeSim.py
Outputs:
    hillslope_sim.mp4 — animated slope cross-section + diagnostics inset
"""

from __future__ import annotations
import sys, os, math, warnings, subprocess, time
from typing import List, Tuple, Dict, Optional

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Auto-install dependencies ──────────────────────────────────────
_REQUIRED = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "pymunk": "pymunk",
    "shapely": "shapely",
}
for _mod, _pkg in _REQUIRED.items():
    try:
        __import__(_mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle as MplCircle, Rectangle as MplRect, Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib import cm

# =====================================================================
# 1. USER CONTROL DASHBOARD
# =====================================================================
# All hard-coded parameters the user should tune before running.

# -------------------------------------------------------------------
# 1.1  ENGINE SELECTION
# -------------------------------------------------------------------
ENGINE = "pychrono"     # "pymunk" (2D) or "pychrono" (3D)
QUASI_2D = True         # For pychrono: auto-size Y to ~1 particle width

# -------------------------------------------------------------------
# 1.2  DOMAIN GRID
# -------------------------------------------------------------------
DOMAIN_X = 2.0          # X-extent in metres (longitudinal, downslope)
DOMAIN_Y = 0.2          # Y-extent in metres (lateral — auto-overridden in quasi-2D)
DOMAIN_Z = 2.0          # Z-extent in metres (vertical)
CELL_SIZE = 0.05        # Reference grid cell edge (m) — used for diagnostics

# -------------------------------------------------------------------
# 1.3  HILLSLOPE GEOMETRY
# -------------------------------------------------------------------
# The slope occupies the LEFT side of the domain.  Its top-left corner
# is at (0, SLOPE_HEIGHT) and it descends linearly to (SLOPE_LENGTH, 0).
SLOPE_HEIGHT = 1.0      # Height of the crest above domain floor (m)
SLOPE_LENGTH = 1.5      # Horizontal run from crest to toe (m)
SLOPE_BASE   = 0.0      # Elevation of slope toe (m)

# -------------------------------------------------------------------
# 1.4  FRACTURE / PARTICLE PARAMETERS
# -------------------------------------------------------------------
PARTICLE_SHAPE = "rectangle" # "circle" or "rectangle"
PARTICLE_SIZE  = 0.06        # a-axis: length (m) of a circle diameter or
                             #   rectangle long edge (X-dimension)
PARTICLE_B_AXIS = 0.04       # b-axis: rectangle width (Y or in-plane short
                             #   edge, m). Ignored for circles.
                             #   For quasi-2D, b-axis is the cross-domain
                             #   (Y) extent of the block; must be ≤ domain Y.
PARTICLE_C_AXIS = None       # c-axis: rectangle height (Z-dimension, m).
                             #   None  → same as b-axis (square cross-section).
                             #   For quasi-2D the c-axis equals the smaller of
                             #   a and b (block is thin across Y).
RECT_ARRANGEMENT = "uniform" # "uniform" → all a/b/c-axes share one orientation
                             # "mixed"   → 2D/quasi-2D: a/b random in XZ plane,
                             #              c always perpendicular to Y-plane.
                             #              3D: all axes fully randomised.
FRACTURE_MODE = "DFN"       # "grid" (legacy packing) or "DFN" (joint-set block generator)
TARGET_POROSITY = 0.15       # Desired void fraction (0–1); controls packing tightness

# DFN controls (used when FRACTURE_MODE == "DFN")
# Each joint set is defined by dip angle (deg from +X in XZ section),
# mean spacing (m), and persistence (trace length in m).
DFN_JOINT_SETS = [
    {"dip_deg": 35.0, "spacing": 0.18, "persistence": 0.90, "persistence_std": 0.20},
    {"dip_deg": 70.0, "spacing": 0.24, "persistence": 0.65, "persistence_std": 0.15},
]
DFN_SEED = 42
DFN_LINE_JITTER = 0.10       # Fraction of spacing used as random offset jitter
DFN_MIN_BLOCK_AREA = 2.5e-4  # m^2; tiny slivers are removed for numerical robustness

# -------------------------------------------------------------------
# 1.5  MATERIAL PROPERTIES
# -------------------------------------------------------------------
MATERIAL_TYPE    = "granular" # "cohesive" (soils, weak rock) or "granular" (sand/gravel/cobbles)
PARTICLE_DENSITY = 2000.0   # kg/m³ (equivalent bulk density of rock/sediment)
#
# Material presets (auto-set below; override manually here if desired):
#   COHESIVE:
#     - Friction angle φ ≈ 25–35°  → tan(φ) ≈ 0.47–0.70  [FRICTION_COEFF = 0.6 default]
#     - Cohesion c' ≈ 1–50 kPa      [COHESION_FORCE 5–20 N for 0.06 m particles]
#   GRANULAR (sand/gravel/cobbles):
#     - Friction angle φ ≈ 30–50°  → tan(φ) ≈ 0.58–1.19  [FRICTION_COEFF = 0.75–0.85]
#     - Cohesion c' = 0 kPa         [COHESION_FORCE = 0 N, auto-enforced]
#
FRICTION_COEFF   = 0.6      # Coulomb friction tan(φ); auto-adjusted by MATERIAL_TYPE
COHESION_FORCE   = 5.0      # Cohesive bond strength (N); set to 0 for MATERIAL_TYPE="granular"
                             # Realistic cohesive range: 1–20 N (≈ 0.3–6 kPa contact stress).
                             # pymunk: breakable DampedSpring bonds (set up post-settle).
                             # pychrono: native ChContactMaterialNSC.SetCohesion().
BOND_BREAK_MULT  = 1.5      # Bonds break when gap ≥ BOND_BREAK_MULT × PARTICLE_SIZE.
                             # Lower = more brittle; higher = more ductile.
RESTITUTION      = 0.01     # Coefficient of restitution (bounciness; 0.01=inelastic cohesive)

# -------------------------------------------------------------------
# 1.6  TILTING / ROTATION PARAMETERS
# -------------------------------------------------------------------
TOTAL_ROTATION   = 90.0     # Total clockwise rotation of domain (degrees)
ROTATION_RATE    = 2.0      # Degrees per second of simulation time
SETTLE_TIME      = 2.0      # Seconds of settling before rotation starts

# -------------------------------------------------------------------
# 1.7  SIMULATION CONTROLS
# -------------------------------------------------------------------
DT               = 0.005    # Physics time-step (s)
SOLVER_ITERS     = 30       # Constraint solver iterations per step
GRAVITY_MAG      = 9.81     # m/s²

# -------------------------------------------------------------------
# 1.8  OUTPUT
# -------------------------------------------------------------------
FPS              = 30       # Animation frames per second
FRAME_INTERVAL   = 5        # Save a frame every N physics steps
VIDEO_FILE       = "hillslope_sim.mp4"
DISPLACEMENT_THRESHOLD = 2.0 * PARTICLE_SIZE  # "significant movement" threshold (2 diameters)

# -------------------------------------------------------------------
# 1.9  MATERIAL PRESETS (auto-applied if MATERIAL defaults are used)
# -------------------------------------------------------------------
# Adjust FRICTION_COEFF and COHESION_FORCE based on MATERIAL_TYPE.
# Uncomment to use presets; otherwise manually set above.
if MATERIAL_TYPE == "granular":
    FRICTION_COEFF = 0.80      # φ ≈ 39° (sand: 30–45°, gravel: 35–50°)
    COHESION_FORCE = 0.0       # Non-cohesive; zero bonding
elif MATERIAL_TYPE == "cohesive":
    # FRICTION_COEFF and COHESION_FORCE retain dashboard values (user-set)
    # Default: FRICTION_COEFF = 0.6 (φ ≈ 31°), COHESION_FORCE = 5 N
    pass

# =====================================================================
# 2. DERIVED QUANTITIES
# =====================================================================
_rotation_duration = TOTAL_ROTATION / max(ROTATION_RATE, 1e-6)
SIM_DURATION = SETTLE_TIME + _rotation_duration + 2.0   # extra 2 s post-rotation
N_STEPS = int(round(SIM_DURATION / DT))

# ── Resolve rectangle axes ────────────────────────────────────────
# PARTICLE_SIZE is always the a-axis (longest in-plane dimension).
# For circles these are unused; shapes use PARTICLE_SIZE as diameter.
_is_quasi2d = (ENGINE == "pychrono" and QUASI_2D)
_is_3d_full  = (ENGINE == "pychrono" and not QUASI_2D)

# b-axis defaults to a-axis if not specified (square blocks)
_rect_b = PARTICLE_B_AXIS if PARTICLE_B_AXIS is not None else PARTICLE_SIZE

# c-axis resolution:
#   quasi-2D → c is the Y-thickness = smaller of a and b (fills one 'slot')
#   3D full  → user value or b-axis default
#   2D pymunk → c irrelevant (2D), but store it for documentation
if PARTICLE_C_AXIS is not None:
    _rect_c = PARTICLE_C_AXIS
elif _is_quasi2d:
    _rect_c = min(PARTICLE_SIZE, _rect_b)
else:
    _rect_c = _rect_b  # square cross-section default

# Validate: a >= b >= c is conventional; warn if not
if PARTICLE_SHAPE == "rectangle":
    _a, _b, _c = PARTICLE_SIZE, _rect_b, _rect_c
    _axes_sorted = sorted([_a, _b, _c], reverse=True)
    if [_a, _b, _c] != _axes_sorted:
        warnings.warn(
            f"Rectangle axes a={_a:.3f} b={_b:.3f} c={_c:.3f} are not in "
            f"descending order (a \u2265 b \u2265 c).  Physics is unaffected but "
            f"labelling may be confusing."
        )

# Scale for physics units (pymunk works in pixels; we use 1 m = 100 px)
PX_PER_M = 100.0

def m2px(m):
    return m * PX_PER_M

def px2m(px):
    return px / PX_PER_M

# =====================================================================
# 3. HILLSLOPE GENERATION + FRACTURE INTO PARTICLES
# =====================================================================

def _slope_surface_z(x: float) -> float:
    """Return the slope surface elevation at horizontal position x."""
    if x < 0:
        return SLOPE_HEIGHT
    if x > SLOPE_LENGTH:
        return SLOPE_BASE
    frac = x / SLOPE_LENGTH
    return SLOPE_HEIGHT - frac * (SLOPE_HEIGHT - SLOPE_BASE)


_DFN_BLOCKS_CACHE: Optional[List[Dict[str, object]]] = None


def _fracture_domain_with_dfn() -> List[Dict[str, object]]:
    """
    Build a 2D DFN (Discrete Fracture Network) in XZ and return fractured blocks.

    Workflow:
      1) Create a solid slope polygon domain.
      2) Generate intersecting joint traces from user-defined joint sets
         (dip, spacing, persistence).
      3) Split domain polygon along traces.
      4) Filter sliver blocks and compute block descriptors used by backends.
    """
    global _DFN_BLOCKS_CACHE
    if _DFN_BLOCKS_CACHE is not None:
        return _DFN_BLOCKS_CACHE

    from shapely.geometry import Polygon, LineString
    from shapely.ops import split

    rng = np.random.default_rng(DFN_SEED)

    # Cross-section slope polygon (X horizontal, Z vertical)
    slope_poly = Polygon([
        (0.0, SLOPE_BASE),
        (0.0, SLOPE_HEIGHT),
        (SLOPE_LENGTH, SLOPE_BASE),
    ]).buffer(0)

    if slope_poly.is_empty or slope_poly.area <= 0:
        _DFN_BLOCKS_CACHE = []
        return _DFN_BLOCKS_CACHE

    minx, minz, maxx, maxz = slope_poly.bounds
    diag = math.hypot(maxx - minx, maxz - minz)
    centre = np.array([(minx + maxx) * 0.5, (minz + maxz) * 0.5])

    blocks = [slope_poly]
    trace_count = 0

    for js in DFN_JOINT_SETS:
        dip_deg = float(js.get("dip_deg", 45.0))
        spacing = max(float(js.get("spacing", PARTICLE_SIZE * 2.0)), PARTICLE_SIZE * 0.5)
        persistence = max(float(js.get("persistence", diag * 0.7)), PARTICLE_SIZE)
        pstd = max(float(js.get("persistence_std", 0.0)), 0.0)

        theta = math.radians(dip_deg)
        t_dir = np.array([math.cos(theta), math.sin(theta)])
        n_dir = np.array([-math.sin(theta), math.cos(theta)])

        half_span = 0.75 * diag
        offsets = np.arange(-half_span, half_span + spacing, spacing)

        for off in offsets:
            off += rng.uniform(-DFN_LINE_JITTER * spacing, DFN_LINE_JITTER * spacing)
            mid = centre + off * n_dir
            trace_len = max(PARTICLE_SIZE, persistence + rng.normal(0.0, pstd))
            p0 = mid - 0.5 * trace_len * t_dir
            p1 = mid + 0.5 * trace_len * t_dir
            line = LineString([tuple(p0), tuple(p1)])

            next_blocks = []
            for poly in blocks:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.area < DFN_MIN_BLOCK_AREA:
                    continue
                if not poly.intersects(line):
                    next_blocks.append(poly)
                    continue
                try:
                    pieces = split(poly, line)
                    for g in pieces.geoms:
                        if g.area >= DFN_MIN_BLOCK_AREA:
                            next_blocks.append(g)
                    trace_count += 1
                except Exception:
                    next_blocks.append(poly)
            blocks = next_blocks if next_blocks else blocks

    block_desc: List[Dict[str, object]] = []
    for poly in blocks:
        poly = poly.buffer(0)
        if poly.is_empty or poly.area < DFN_MIN_BLOCK_AREA:
            continue
        c = poly.centroid
        mrr = poly.minimum_rotated_rectangle
        corners = list(mrr.exterior.coords)[:-1]
        if len(corners) < 4:
            continue

        e0 = np.array(corners[1]) - np.array(corners[0])
        e1 = np.array(corners[2]) - np.array(corners[1])
        L0 = float(np.hypot(e0[0], e0[1]))
        L1 = float(np.hypot(e1[0], e1[1]))
        if L0 >= L1:
            a_len = L0
            c_len = L1
            ang = math.atan2(e0[1], e0[0])
        else:
            a_len = L1
            c_len = L0
            ang = math.atan2(e1[1], e1[0])

        verts = [(float(x), float(z)) for x, z in list(poly.exterior.coords)[:-1]]
        block_desc.append({
            "centroid": (float(c.x), float(c.y)),
            "angle": float(ang),
            "a": max(a_len, PARTICLE_SIZE * 0.35),
            "c": max(c_len, PARTICLE_SIZE * 0.25),
            "area": float(poly.area),
            "verts": verts,
        })

    if block_desc:
        areas = np.array([b["area"] for b in block_desc], dtype=float)
        print("\n--- DFN Fracture Preprocessor ---")
        print(f"  Joint sets             : {len(DFN_JOINT_SETS)}")
        print(f"  Successful trace cuts  : {trace_count}")
        print(f"  DFN blocks generated   : {len(block_desc)}")
        print(f"  Block area mean±std    : {areas.mean():.4e} ± {areas.std():.4e} m²")
        print(f"  Block area range       : [{areas.min():.4e}, {areas.max():.4e}] m²")
        print("-------------------------------\n")

    _DFN_BLOCKS_CACHE = block_desc
    return _DFN_BLOCKS_CACHE


def _generate_particle_centres() -> List[Tuple[float, float, float, float]]:
    """
    Generate particle centre positions filling the hillslope cross-section.

    Returns list of (x, z, angle_xz, size) where:
      - x, z  : centre coordinates in metres
      - angle_xz: in-plane rotation of the a-axis from the +X direction (radians).
                  Always 0 for circles or RECT_ARRANGEMENT == "uniform".
                  Random in XZ plane for RECT_ARRANGEMENT == "mixed".
      - size : particle diameter/edge length (m). For rectangles this is
               usually PARTICLE_SIZE; for DFN mode it is the block a-axis.
               For pychrono spheres this may include smaller infill spheres
               if target porosity is very low.
    """
    if FRACTURE_MODE.upper() == "DFN":
        blocks = _fracture_domain_with_dfn()
        return [
            (b["centroid"][0], b["centroid"][1], b["angle"], b["a"])
            for b in blocks
        ]

    rng = np.random.default_rng(42)

    # Characteristic packing size — use the largest footprint edge of each
    # particle for spacing calculations so they never overlap when uniform.
    if PARTICLE_SHAPE == "circle":
        pack_size = PARTICLE_SIZE            # diameter
        solid_area_ratio = math.pi / 4.0    # circle packing factor
    else:
        # In uniform layout the long a-axis is aligned with X, so footprint
        # in the XZ plane is a × c (a-axis horizontal, c-axis vertical).
        pack_size = max(PARTICLE_SIZE, _rect_c)
        solid_area_ratio = 1.0               # rectangle: full footprint

    solid_fraction = max(1.0 - TARGET_POROSITY, 0.1)
    spacing = pack_size * math.sqrt(solid_area_ratio / solid_fraction)
    spacing = max(spacing, pack_size * 1.01)  # never overlap

    # Half-size used to keep particles inside the slope boundary
    half = pack_size / 2.0

    centres: List[Tuple[float, float, float, float]] = []
    x = half
    while x < SLOPE_LENGTH + pack_size:
        z = half
        z_surf = _slope_surface_z(x)
        while z < z_surf - half * 0.5:
            centres.append((x, z, 0.0, PARTICLE_SIZE))  # angle filled in below
            z += spacing
        x += spacing

    # Random jitter (small) to break regularity
    jitter = 0.05 * pack_size
    centres = [
        (cx + rng.uniform(-jitter, jitter),
         cz + rng.uniform(-jitter, jitter),
         ang,
         size)
        for cx, cz, ang, size in centres
    ]

    # Apply orientation arrangement for rectangles
    if PARTICLE_SHAPE == "rectangle" and RECT_ARRANGEMENT == "mixed":
        # 2D / quasi-2D: randomise in-plane (a/b rotate in XZ plane).
        # The c-axis always remains perpendicular to the XZ analysis plane
        # (i.e. parallel to Y), so we only rotate around the Y-axis.
        centres = [
            (cx, cz, rng.uniform(0.0, math.pi), size)
            for cx, cz, _, size in centres
        ]

    # For pychrono spheres, optionally add smaller infill spheres when the
    # requested porosity is lower than monodisperse circles can physically reach.
    # This keeps the requested trend while warning the user that a polydisperse
    # assembly is being generated.
    if ENGINE == "pychrono" and PARTICLE_SHAPE == "circle":
        slope_area = 0.5 * (SLOPE_HEIGHT + SLOPE_BASE) * SLOPE_LENGTH
        slope_area = max(slope_area, 1e-12)
        base_area = sum(math.pi * (size / 2.0) ** 2 for _, _, _, size in centres)
        current_porosity = max(0.0, 1.0 - base_area / slope_area)

        if current_porosity > TARGET_POROSITY + 1e-3:
            diam_factors = [0.50, 0.35, 0.25]
            added_sizes = []

            for fac in diam_factors:
                if current_porosity <= TARGET_POROSITY + 1e-3:
                    break

                d_small = PARTICLE_SIZE * fac
                r_small = d_small / 2.0
                step_small = d_small * 1.05
                x = r_small

                while x < SLOPE_LENGTH + d_small and current_porosity > TARGET_POROSITY + 1e-3:
                    z = r_small
                    z_surf = _slope_surface_z(x)
                    while z < z_surf - r_small * 0.5 and current_porosity > TARGET_POROSITY + 1e-3:
                        ok = True
                        for ex, ez, _, es in centres:
                            min_dist = 0.5 * (es + d_small) * 1.02
                            dx = x - ex
                            dz = z - ez
                            if (dx * dx + dz * dz) < (min_dist * min_dist):
                                ok = False
                                break
                        if ok:
                            centres.append((x, z, 0.0, d_small))
                            added_sizes.append(d_small)
                            base_area += math.pi * r_small ** 2
                            current_porosity = max(0.0, 1.0 - base_area / slope_area)
                        z += step_small
                    x += step_small

            if added_sizes:
                mono_limit = max(0.0, 1.0 - math.pi / 4.0)
                print(
                    "  [WARN] Sphere porosity target is below monodisperse packing capability; "
                    f"added {len(added_sizes)} smaller infill spheres "
                    f"(d={min(added_sizes):.4f}–{max(added_sizes):.4f} m) "
                    f"to approach TARGET_POROSITY={TARGET_POROSITY:.3f}. "
                    f"Monodisperse limit is approximately {mono_limit:.3f}."
                )

    return centres


# =====================================================================
# 3b. STABILIZE INITIAL SLOPE
# =====================================================================
# Calculates the minimum cohesion required to keep the parameterised
# hillslope stable (FS \u2265 1) under self-weight alone (0\u00b0 tilt, no tilting).
# If COHESION_FORCE is set lower than this minimum, it is automatically
# raised and a warning is printed.
#
# Method \u2014 Infinite-Slope Coulomb Analysis on the slope face
# ---------------------------------------------------------------
# For a planar slope inclined at angle beta (= arctan(H/L)):
#
#   FS = [c' / (\u03b3 \u00b7 H \u00b7 sin\u03b2\u00b7cos\u03b2)] + [tan\u03c6 / tan\u03b2]
#
# where:
#   c'  = cohesion per unit area (Pa) = COHESION_FORCE / contact_area
#   \u03b3   = PARTICLE_DENSITY \u00d7 (1 \u2212 TARGET_POROSITY)  (bulk unit weight, kg/m\u00b3)
#   H   = SLOPE_HEIGHT
#   \u03b2   = slope angle from horizontal = arctan(SLOPE_HEIGHT / SLOPE_LENGTH)
#   \u03c6   = friction angle = arctan(FRICTION_COEFF)
#
# Rearranging for the minimum c' at FS = 1:
#   c'_min = \u03b3 \u00b7 H \u00b7 sin\u03b2 \u00b7 cos\u03b2 \u00d7 (1 \u2212 tan\u03c6/tan\u03b2)
#          (if this is \u2264 0 the slope is stable by friction alone)
#
# Converting c'_min (Pa) to force-per-bond (N):
#   contact_area = PARTICLE_SIZE\u00b2  (one contact face)
#   coh_force_min = c'_min \u00d7 contact_area

def _stabilize_initial_slope() -> float:
    """
    Compute the minimum cohesion force (N) for FS \u2265 1 on the initial slope.
    Updates the global COHESION_FORCE if it is insufficient, and returns
    the required value.
    """
    global COHESION_FORCE

    if MATERIAL_TYPE == "granular":
        COHESION_FORCE = 0.0

    if SLOPE_HEIGHT <= 0 or SLOPE_LENGTH <= 0:
        return COHESION_FORCE

    beta = math.atan2(SLOPE_HEIGHT - SLOPE_BASE, SLOPE_LENGTH)
    phi  = math.atan(FRICTION_COEFF)
    # Bulk unit weight accounting for porosity
    gamma_bulk = PARTICLE_DENSITY * (1.0 - TARGET_POROSITY)   # kg/m\u00b3
    unit_weight = gamma_bulk * GRAVITY_MAG                     # N/m\u00b3

    # Frictional contribution to FS on an infinite slope
    fs_friction = math.tan(phi) / math.tan(beta) if math.tan(beta) > 1e-12 else 99.0

    if fs_friction >= 1.0:
        # Friction alone is sufficient \u2014 zero cohesion needed
        c_prime_min = 0.0
    else:
        # Cohesion required to raise FS to unity
        # Infinite-slope formula: FS = c'/(\u03b3 H sin\u03b2 cos\u03b2) + tan\u03c6/tan\u03b2
        c_prime_min = (unit_weight * SLOPE_HEIGHT
                       * math.sin(beta) * math.cos(beta)
                       * (1.0 - fs_friction))

    # Contact area: one face of the characteristic particle
    if PARTICLE_SHAPE == "circle":
        contact_area = math.pi * (PARTICLE_SIZE / 2.0) ** 2   # projected end-face
    else:
        contact_area = PARTICLE_SIZE * _rect_c                  # a \u00d7 c face

    coh_force_min = c_prime_min * contact_area

    print("\n--- Stabilize Initial Slope ---")
    print(f"  Material type           : {MATERIAL_TYPE.upper()}")
    print(f"  Slope angle (β)          : {math.degrees(beta):.1f}°")
    print(f"  Friction angle (φ)       : {math.degrees(phi):.1f}°")
    print(f"  Bulk unit weight (γ)     : {unit_weight:.1f} N/m³")
    print(f"  FS (friction only)       : {fs_friction:.3f}")
    print(f"  Min cohesion (c'_min)    : {c_prime_min:.1f} Pa")
    print(f"  Min cohesion force/bond  : {coh_force_min:.4f} N")

    if MATERIAL_TYPE == "granular":
        print(f"  [GRANULAR MODE] Cohesion disabled; slope stability depends on friction alone.")
        if fs_friction < 1.0:
            print(f"    ⚠ WARNING: Friction alone (FS={fs_friction:.3f}) is INSUFFICIENT for stability.")
            print(f"    Consider reducing slope angle or increasing friction coefficient.")
    elif coh_force_min > COHESION_FORCE:
        print(f"  [AUTO-ADJUST] COHESION_FORCE raised: "
              f"{COHESION_FORCE:.4f} → {coh_force_min:.4f} N "
              f"(parameterised value was insufficient to hold slope at FS = 1)")
        COHESION_FORCE = coh_force_min
    else:
        fs_with_coh = (COHESION_FORCE / max(contact_area, 1e-15) /
                       max(unit_weight * SLOPE_HEIGHT *
                           math.sin(beta) * math.cos(beta), 1e-15)
                       + fs_friction)
        print(f"  [COHESIVE MODE] Cohesion is sufficient  "
              f"(FS ≈ {fs_with_coh:.2f} with COHESION_FORCE = {COHESION_FORCE:.4f} N).")
    print("-------------------------------\n")
    return COHESION_FORCE


# =====================================================================
# 4. PHYSICS BACKEND — PYMUNK (2D)
# =====================================================================

class PymunkBackend:
    """2D rigid-body simulation using pymunk (Chipmunk2D wrapper)."""

    def __init__(self):
        import pymunk
        self.pymunk = pymunk
        self.space = pymunk.Space()
        self.space.gravity = (0, -m2px(GRAVITY_MAG))
        self.space.iterations = SOLVER_ITERS
        self.space.damping = 0.95  # slight global damping for stability

        self.particle_bodies: List[pymunk.Body] = []
        self.particle_shapes = []
        self.particle_sizes: List[float] = []
        self.particle_c_sizes: List[float] = []
        self.particle_planform_areas: List[float] = []
        self.particle_draw_local_vertices: List[Optional[np.ndarray]] = []
        self.initial_positions: List[Tuple[float, float]] = []  # in metres
        # Breakable bond springs (DampedSpring); populated by setup_bonds()
        # which is called externally after the settle phase.
        self.bond_springs: List = []
        self._break_dist_px: float = PARTICLE_SIZE * BOND_BREAK_MULT * PX_PER_M

        self._build_walls()
        self._build_particles()
        # Note: cohesion bonds are NOT applied during settling so particles
        # can compact naturally.  Call setup_bonds() after settling.

    # ── Walls ──────────────────────────────────────────────────────
    def _build_walls(self):
        pm = self.pymunk
        sb = self.space.static_body
        W = m2px(DOMAIN_X)
        H = m2px(DOMAIN_Z)
        thickness = m2px(0.02)

        walls = [
            pm.Segment(sb, (0, 0), (W, 0), thickness),           # floor
            pm.Segment(sb, (0, 0), (0, H), thickness),           # left wall
            pm.Segment(sb, (W, 0), (W, H), thickness),           # right wall
            pm.Segment(sb, (0, H), (W, H), thickness),           # ceiling
        ]
        for w in walls:
            w.friction = FRICTION_COEFF
            w.elasticity = RESTITUTION
            w.collision_type = 2  # WALL
        self.space.add(*walls)

    # ── Particles ──────────────────────────────────────────────────
    def _build_particles(self):
        pm = self.pymunk
        centres = _generate_particle_centres()

        if FRACTURE_MODE.upper() == "DFN" and PARTICLE_SHAPE == "rectangle":
            blocks = _fracture_domain_with_dfn()
            for b in blocks:
                cx, cz = b["centroid"]
                verts_world = b["verts"]
                verts_local = [(vx - cx, vz - cz) for vx, vz in verts_world]

                body = pm.Body()
                body.position = (m2px(cx), m2px(cz))

                verts_px = [(m2px(vx), m2px(vz)) for vx, vz in verts_local]
                shape = pm.Poly(body, verts_px)
                shape.density = PARTICLE_DENSITY / (PX_PER_M ** 2)
                shape.friction = FRICTION_COEFF
                shape.elasticity = RESTITUTION
                shape.collision_type = 1

                self.space.add(body, shape)
                self.particle_bodies.append(body)
                self.particle_shapes.append(shape)
                self.particle_sizes.append(float(b["a"]))
                self.particle_c_sizes.append(float(b["c"]))
                self.particle_planform_areas.append(float(b["area"]))
                self.particle_draw_local_vertices.append(np.array(verts_local, dtype=float))
                self.initial_positions.append((cx, cz))

            print(f"  Created {len(self.particle_bodies)} particles "
                  f"(shape=DFN-polygons, sets={len(DFN_JOINT_SETS)})")
            return

        r_px  = m2px(PARTICLE_SIZE / 2.0)
        # For rectangles use resolved a-axis (PARTICLE_SIZE) × c-axis (_rect_c)
        # as the 2D footprint visible in the XZ analysis plane.
        a_px  = m2px(PARTICLE_SIZE)
        c_px  = m2px(_rect_c)

        for cx, cz, ang, _size in centres:
            body = pm.Body()
            body.position = (m2px(cx), m2px(cz))

            if PARTICLE_SHAPE == "circle":
                shape = pm.Circle(body, r_px)
                shape.density = PARTICLE_DENSITY / (PX_PER_M ** 2)
            else:
                shape = pm.Poly.create_box(body, size=(a_px, c_px))
                shape.density = PARTICLE_DENSITY / (PX_PER_M ** 2)
                body.angle = ang   # apply in-plane orientation

            shape.friction = FRICTION_COEFF
            shape.elasticity = RESTITUTION
            shape.collision_type = 1  # PARTICLE

            self.space.add(body, shape)
            self.particle_bodies.append(body)
            self.particle_shapes.append(shape)
            if PARTICLE_SHAPE == "circle":
                self.particle_sizes.append(PARTICLE_SIZE)
                self.particle_c_sizes.append(PARTICLE_SIZE)
                self.particle_planform_areas.append(math.pi * (PARTICLE_SIZE / 2.0) ** 2)
                self.particle_draw_local_vertices.append(None)
            else:
                self.particle_sizes.append(PARTICLE_SIZE)
                self.particle_c_sizes.append(_rect_c)
                self.particle_planform_areas.append(PARTICLE_SIZE * _rect_c)
                rect_local = np.array([
                    [-PARTICLE_SIZE * 0.5, -_rect_c * 0.5],
                    [ PARTICLE_SIZE * 0.5, -_rect_c * 0.5],
                    [ PARTICLE_SIZE * 0.5,  _rect_c * 0.5],
                    [-PARTICLE_SIZE * 0.5,  _rect_c * 0.5],
                ], dtype=float)
                self.particle_draw_local_vertices.append(rect_local)
            self.initial_positions.append((cx, cz))

        _shape_desc = (f"d={PARTICLE_SIZE:.3f} m" if PARTICLE_SHAPE == "circle"
                       else f"a={PARTICLE_SIZE:.3f} b={_rect_b:.3f} c={_rect_c:.3f} m, "
                            f"arr={RECT_ARRANGEMENT}")
        print(f"  Created {len(self.particle_bodies)} particles "
              f"(shape={PARTICLE_SHAPE}, {_shape_desc})")

    # ── Breakable cohesive bonds (DampedSpring) ──────────────────
    def setup_bonds(self):
        """
        Create breakable DampedSpring bonds between particles that are in
        contact after settling.  Call this ONCE after the settling phase.

        Bond model (brittle cohesive zone):
          - Rest length  = particle separation at bond formation
          - Stiffness k  = COHESION_FORCE / stretch-at-break  [N/m]
          - Bonds are removed permanently when stretched > BOND_BREAK_MULT *
            PARTICLE_SIZE (simulating irreversible tensile fracture).

        This is physically correct: no cohesion during settling (natural
        compaction), then cemented contacts that fracture under tension.
        """
        if COHESION_FORCE <= 0:
            return
        pm = self.pymunk
        pos = self.get_positions()  # metres after settling
        n = len(pos)
        bond_search = PARTICLE_SIZE * 1.05  # only truly touching pairs bond
        break_ext = max((BOND_BREAK_MULT - 1.0) * PARTICLE_SIZE, 1e-4)  # m
        # Spring stiffness: F rises linearly from 0 at contact to
        # COHESION_FORCE at break extension (see unit analysis: k_pymunk = k_SI)
        k = COHESION_FORCE / break_ext  # N/m
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dz = pos[j, 1] - pos[i, 1]
                dist = math.sqrt(dx * dx + dz * dz)
                if dist < bond_search:
                    ba = self.particle_bodies[i]
                    bb = self.particle_bodies[j]
                    spring = pm.DampedSpring(
                        ba, bb, (0, 0), (0, 0),
                        rest_length=dist * PX_PER_M,
                        stiffness=k,
                        damping=0.02,
                    )
                    self.space.add(spring)
                    self.bond_springs.append(spring)
                    count += 1
        print(f"  Breakable bonds: {count} formed  "
              f"(k={k:.1f} N/m, break at {BOND_BREAK_MULT:.1f}× d)")

    def _break_springs(self):
        """Remove bonds whose length has exceeded the break distance."""
        if not self.bond_springs:
            return
        bd_sq = self._break_dist_px ** 2
        to_remove = [
            s for s in self.bond_springs
            if ((s.b.position.x - s.a.position.x) ** 2 +
                (s.b.position.y - s.a.position.y) ** 2) > bd_sq
        ]
        for s in to_remove:
            self.bond_springs.remove(s)
            try:
                self.space.remove(s)
            except Exception:
                pass

    # ── Gravity rotation ──────────────────────────────────────────
    def set_rotation_angle(self, angle_deg: float):
        """Rotate gravity clockwise by angle_deg from original vertical."""
        rad = math.radians(angle_deg)
        gx = GRAVITY_MAG * math.sin(rad)
        gz = -GRAVITY_MAG * math.cos(rad)
        self.space.gravity = (m2px(gx), m2px(gz))

    # ── Step ──────────────────────────────────────────────────────
    def step(self):
        self._break_springs()  # remove broken cohesive bonds before stepping
        self.space.step(DT)

    # ── Queries ───────────────────────────────────────────────────
    def get_positions(self) -> np.ndarray:
        """Return (N, 2) array of particle positions in metres."""
        pos = np.array([(b.position.x, b.position.y) for b in self.particle_bodies])
        return pos / PX_PER_M

    def get_velocities(self) -> np.ndarray:
        vel = np.array([(b.velocity.x, b.velocity.y) for b in self.particle_bodies])
        return vel / PX_PER_M

    def get_kinetic_energy(self) -> float:
        ke = 0.0
        for b in self.particle_bodies:
            v = b.velocity
            ke += 0.5 * b.mass * (v.x ** 2 + v.y ** 2)
        return ke / (PX_PER_M ** 2)   # convert to SI (J/m for 2D)

    def get_angles(self) -> np.ndarray:
        """Return particle in-plane orientation angles (rad)."""
        return np.array([b.angle for b in self.particle_bodies], dtype=float)

    def get_contact_info(self) -> Tuple[float, np.ndarray]:
        """
        Returns:
            coord_number: average contacts per particle
            contact_normals: (M, 2) array of contact normal vectors
        """
        contact_count = np.zeros(len(self.particle_bodies))
        normals = []
        body_id_set = {id(b) for b in self.particle_bodies}

        for i, body in enumerate(self.particle_bodies):
            def _count_arb(arb, idx=i):
                contact_count[idx] += 1
                n = arb.contact_point_set.normal
                normals.append((n.x, n.y))
            body.each_arbiter(_count_arb)

        # Each contact is counted twice (once per body), normalise
        coord_num = contact_count.mean() if len(contact_count) > 0 else 0.0
        # Deduplicate normals (each pair counted from both sides)
        if len(normals) > 1:
            normals = normals[::2]
        normals_arr = np.array(normals) if normals else np.zeros((0, 2))
        return coord_num, normals_arr

    def num_particles(self) -> int:
        return len(self.particle_bodies)


# =====================================================================
# 5. PHYSICS BACKEND — PYCHRONO (3D)
# =====================================================================

class PychronoBackend:
    """3D rigid-body simulation using pychrono."""

    def __init__(self):
        import pychrono as chrono
        self.chrono = chrono

        # ── Shrink Bullet collision envelope BEFORE creating any collision shape.
        # Default envelope (0.01 m) is close to particle radius (0.03 m), causing
        # false contacts and tunneling through thin walls.
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.001)   # 1 mm
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0005)    # 0.5 mm

        self.sys = chrono.ChSystemNSC()
        self.sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY_MAG))
        self.sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

        # Solver settings — increase iterations for tight quasi-2D packing.
        _solver = self.sys.GetSolver()
        _it_solver = _solver.AsIterative()
        if _it_solver is not None:
            _it_solver.SetMaxIterations(max(SOLVER_ITERS, 80))
        self.sys.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)

        self.particle_bodies = []
        self.particle_sizes = []
        self.particle_c_sizes = []
        self.particle_planform_areas = []
        self.particle_draw_local_vertices = []
        self.initial_positions = []

        # Determine Y extent for quasi-2D
        self._y_extent = DOMAIN_Y
        if QUASI_2D:
            self._y_extent = PARTICLE_SIZE * 1.5  # just wide enough for 1 particle
            print(f"  Quasi-2D mode: Y extent = {self._y_extent:.4f} m")

        self._build_walls()
        self._build_particles()

    def _make_material(self, is_wall=False):
        chrono = self.chrono
        mat = chrono.ChContactMaterialNSC()
        mat.SetFriction(FRICTION_COEFF)
        mat.SetRestitution(RESTITUTION)
        mat.SetCohesion(COHESION_FORCE if not is_wall else 0.0)
        return mat

    def _build_walls(self):
        chrono = self.chrono
        mat = self._make_material(is_wall=True)
        W = DOMAIN_X
        H = DOMAIN_Z
        Y = self._y_extent
        # Wall thickness must exceed Bullet's tunneling distance.
        # Rule: t >= 6 × particle_radius AND t >= 0.40 m so fast-moving
        # particles cannot traverse the full wall in one timestep.
        t = max(PARTICLE_SIZE * 6, 0.40)

        # Walls are built with overlapping extents so there are NO corner
        # gaps.  Floor and ceiling span the full X range INCLUDING the
        # side-wall thickness, and side walls span the full Z range
        # INCLUDING the floor/ceiling thickness.  This creates a fully
        # closed box with no escape path at edges or corners.
        walls = [
            # (body-centre position,  full box dimensions)
            # Floor  — inner face at z = 0
            ((W / 2,       0,           -t / 2),       (W + 2 * t,  Y + 2 * t,  t)),
            # Ceiling — inner face at z = H
            ((W / 2,       0,            H + t / 2),   (W + 2 * t,  Y + 2 * t,  t)),
            # Left wall  — inner face at x = 0
            ((-t / 2,      0,            H / 2),       (t,          Y + 2 * t,  H + 2 * t)),
            # Right wall — inner face at x = W
            ((W + t / 2,   0,            H / 2),       (t,          Y + 2 * t,  H + 2 * t)),
        ]
        if QUASI_2D:
            # Front Y-wall — inner face at y = -Y/2
            walls.append(((W / 2, -Y / 2 - t / 2, H / 2),
                          (W + 2 * t, t, H + 2 * t)))
            # Back Y-wall  — inner face at y = +Y/2
            walls.append(((W / 2,  Y / 2 + t / 2, H / 2),
                          (W + 2 * t, t, H + 2 * t)))

        for pos, sz in walls:
            body = chrono.ChBodyEasyBox(sz[0], sz[1], sz[2], 1000, True, True, mat)
            body.SetPos(chrono.ChVector3d(pos[0], pos[1], pos[2]))
            body.SetFixed(True)
            self.sys.Add(body)

    def _build_particles(self):
        chrono = self.chrono
        mat = self._make_material(is_wall=False)
        centres = _generate_particle_centres()
        y_pos = 0.0  # centre of Y-extent

        # For quasi-2D the Y thickness is the c-axis so the block fits
        # exactly in the Y-confined channel.  In full 3D use the b-axis.
        y_size = _rect_c if (_is_quasi2d and PARTICLE_SHAPE == "rectangle") else _rect_b

        if FRACTURE_MODE.upper() == "DFN" and PARTICLE_SHAPE == "rectangle":
            blocks = _fracture_domain_with_dfn()
            y_half = max(y_size * 0.5, PARTICLE_SIZE * 0.2)

            for b in blocks:
                cx, cz = b["centroid"]
                ang = b["angle"]
                a_len = b["a"]
                c_len = b["c"]

                body = None
                verts = b["verts"]
                if hasattr(chrono, "ChBodyEasyConvexHull") and hasattr(chrono, "vector_ChVector3d"):
                    try:
                        cloud = chrono.vector_ChVector3d()
                        for vx, vz in verts:
                            cloud.push_back(chrono.ChVector3d(vx, -y_half, vz))
                            cloud.push_back(chrono.ChVector3d(vx,  y_half, vz))
                        body = chrono.ChBodyEasyConvexHull(cloud, PARTICLE_DENSITY, True, True, mat)
                    except Exception:
                        body = None

                # Robust fallback: oriented box from minimum rotated rectangle.
                if body is None:
                    body = chrono.ChBodyEasyBox(
                        a_len, y_size, c_len,
                        PARTICLE_DENSITY, True, True, mat
                    )
                    q = chrono.ChQuaterniond()
                    q.SetFromAngleAxis(ang, chrono.ChVector3d(0, 1, 0))
                    body.SetRot(q)

                body.SetPos(chrono.ChVector3d(cx, y_pos, cz))
                try:
                    body.SetLinDamping(0.15)
                    body.SetAngDamping(0.05)
                except AttributeError:
                    pass

                self.sys.Add(body)
                self.particle_bodies.append(body)
                self.particle_sizes.append(a_len)
                self.particle_c_sizes.append(c_len)
                self.particle_planform_areas.append(float(b["area"]))
                self.particle_draw_local_vertices.append(np.array([
                    [-a_len * 0.5, -c_len * 0.5],
                    [ a_len * 0.5, -c_len * 0.5],
                    [ a_len * 0.5,  c_len * 0.5],
                    [-a_len * 0.5,  c_len * 0.5],
                ], dtype=float))
                self.initial_positions.append((cx, cz))

            print(f"  Created {len(self.particle_bodies)} particles "
                  f"(shape=DFN-blocks, sets={len(DFN_JOINT_SETS)})")
            return

        for cx, cz, ang, size in centres:
            if PARTICLE_SHAPE == "circle":
                r = size / 2.0
                body = chrono.ChBodyEasySphere(r, PARTICLE_DENSITY, True, True, mat)
                # Circles/spheres have no orientation
            else:
                # Box dims: a-axis (X), b/c-axis (Y), c-axis (Z)
                body = chrono.ChBodyEasyBox(
                    size, y_size, _rect_c,
                    PARTICLE_DENSITY, True, True, mat
                )
                if RECT_ARRANGEMENT == "mixed" and abs(ang) > 1e-9:
                    # Rotate around Y-axis by ang (in-plane XZ rotation)
                    # ChQuaterniond(axis, angle) uses angle in radians
                    q = chrono.ChQuaterniond()
                    q.SetFromAngleAxis(ang, chrono.ChVector3d(0, 1, 0))
                    body.SetRot(q)
                elif _is_3d_full and RECT_ARRANGEMENT == "mixed":
                    # Full 3D: randomise all axes using a random axis angle
                    # (ang was drawn uniform [0, pi], re-use as polar angle)
                    # Generate a random unit axis via spherical coordinates
                    import random as _random
                    phi3   = _random.uniform(0, 2 * math.pi)
                    theta3 = _random.uniform(0, math.pi)
                    ax = math.sin(theta3) * math.cos(phi3)
                    ay = math.sin(theta3) * math.sin(phi3)
                    az = math.cos(theta3)
                    q = chrono.ChQuaterniond()
                    q.SetFromAngleAxis(ang, chrono.ChVector3d(ax, ay, az))
                    body.SetRot(q)

            body.SetPos(chrono.ChVector3d(cx, y_pos, cz))
            try:
                body.SetLinDamping(0.15)
                body.SetAngDamping(0.05)
            except AttributeError:
                pass
            self.sys.Add(body)
            self.particle_bodies.append(body)
            self.particle_sizes.append(size)
            self.particle_c_sizes.append(size if PARTICLE_SHAPE == "circle" else _rect_c)
            if PARTICLE_SHAPE == "circle":
                self.particle_planform_areas.append(math.pi * (size / 2.0) ** 2)
                self.particle_draw_local_vertices.append(None)
            else:
                self.particle_planform_areas.append(size * _rect_c)
                self.particle_draw_local_vertices.append(np.array([
                    [-size * 0.5, -_rect_c * 0.5],
                    [ size * 0.5, -_rect_c * 0.5],
                    [ size * 0.5,  _rect_c * 0.5],
                    [-size * 0.5,  _rect_c * 0.5],
                ], dtype=float))
            self.initial_positions.append((cx, cz))

        _shape_desc = (f"d={PARTICLE_SIZE:.3f} m" if PARTICLE_SHAPE == "circle"
                       else f"a={PARTICLE_SIZE:.3f} b={_rect_b:.3f} c={_rect_c:.3f} m, "
                            f"arr={RECT_ARRANGEMENT}")
        print(f"  Created {len(self.particle_bodies)} particles "
              f"(shape={PARTICLE_SHAPE}, {_shape_desc})")

    def set_rotation_angle(self, angle_deg: float):
        chrono = self.chrono
        rad = math.radians(angle_deg)
        gx = GRAVITY_MAG * math.sin(rad)
        gz = -GRAVITY_MAG * math.cos(rad)
        self.sys.SetGravitationalAcceleration(chrono.ChVector3d(gx, 0, gz))

    def step(self):
        self.sys.DoStepDynamics(DT)

    def get_positions(self) -> np.ndarray:
        chrono = self.chrono
        pos = np.zeros((len(self.particle_bodies), 2))
        for i, b in enumerate(self.particle_bodies):
            p = b.GetPos()
            pos[i, 0] = p.x
            pos[i, 1] = p.z
        return pos

    def get_velocities(self) -> np.ndarray:
        pos = np.zeros((len(self.particle_bodies), 2))
        for i, b in enumerate(self.particle_bodies):
            v = b.GetPosDt()
            pos[i, 0] = v.x
            pos[i, 1] = v.z
        return pos

    def get_kinetic_energy(self) -> float:
        ke = 0.0
        for b in self.particle_bodies:
            v = b.GetPosDt()
            m = b.GetMass()
            ke += 0.5 * m * (v.x ** 2 + v.y ** 2 + v.z ** 2)
        return ke

    def get_angles(self) -> np.ndarray:
        """Return particle in-plane orientation angles (rad), projected to XZ."""
        ang = np.zeros(len(self.particle_bodies), dtype=float)
        for i, b in enumerate(self.particle_bodies):
            try:
                q = b.GetRot()
                # Yaw (rotation about Y) from quaternion (w, x, y, z).
                w = float(q.e0)
                x = float(q.e1)
                y = float(q.e2)
                z = float(q.e3)
                ang[i] = math.atan2(2.0 * (w * y + x * z),
                                    1.0 - 2.0 * (y * y + z * z))
            except Exception:
                ang[i] = 0.0
        return ang

    def get_contact_info(self) -> Tuple[float, np.ndarray]:
        """Approximate coordination number from collision system."""
        # pychrono contact iteration is complex; estimate from proximity
        pos = self.get_positions()
        n = len(pos)
        if n == 0:
            return 0.0, np.zeros((0, 2))

        contacts = 0
        normals = []
        sizes = self.particle_sizes
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dz = pos[j, 1] - pos[i, 1]
                dist = math.sqrt(dx * dx + dz * dz)
                contact_thresh = (sizes[i] + sizes[j]) / 2.0 * 1.15
                if dist < contact_thresh and dist > 1e-12:
                    contacts += 2  # both particles get a contact
                    normals.append((dx / dist, dz / dist))

        coord_num = contacts / max(n, 1)
        normals_arr = np.array(normals) if normals else np.zeros((0, 2))
        return coord_num, normals_arr

    def num_particles(self) -> int:
        return len(self.particle_bodies)


# =====================================================================
# 6. STABILITY DIAGNOSTICS
# =====================================================================

def compute_displacement(current_pos: np.ndarray, initial_pos: np.ndarray) -> np.ndarray:
    """Euclidean displacement of each particle from initial position (metres)."""
    return np.sqrt(np.sum((current_pos - initial_pos) ** 2, axis=1))


def compute_pct_moved(displacements: np.ndarray,
                     particle_sizes: Optional[np.ndarray] = None) -> float:
    """Percentage of particles exceeding a size-scaled significant movement threshold."""
    if particle_sizes is None or len(particle_sizes) != len(displacements):
        thr = np.full(len(displacements), DISPLACEMENT_THRESHOLD)
    else:
        thr = 2.0 * particle_sizes
    return 100.0 * np.sum(displacements > thr) / max(len(displacements), 1)


def compute_bulk_density(positions: np.ndarray,
                         particle_planform_areas: Optional[np.ndarray] = None) -> float:
    """
    Estimate bulk density from particle bounding volume.
    Uses the convex hull area (2D) of particle positions.
    """
    n = len(positions)
    if n < 3:
        return 0.0
    x_range = positions[:, 0].max() - positions[:, 0].min()
    z_range = positions[:, 1].max() - positions[:, 1].min()
    area = max(x_range * z_range, 1e-12)
    if particle_planform_areas is not None and len(particle_planform_areas) == n:
        particle_area = float(np.sum(particle_planform_areas))
    elif PARTICLE_SHAPE == "circle":
        particle_area = n * math.pi * (PARTICLE_SIZE / 2.0) ** 2
    else:
        particle_area = n * PARTICLE_SIZE * _rect_c
    vol_fraction = min(particle_area / area, 1.0)
    return vol_fraction * PARTICLE_DENSITY


def compute_touching_surface_fraction(positions: np.ndarray, particle_sizes: np.ndarray,
                                      shape: str,
                                      particle_c_sizes: Optional[np.ndarray] = None) -> float:
    """
    Fraction of total material surface area that is touching other material.

    touching_surface_fraction = (sum contacted surface area over all particles) /
                                (sum total surface area over all particles)

    Contact area is estimated from pair proximity and capped per particle so
    the contacted area cannot exceed that particle's total surface area.
    """
    n = len(positions)
    if n < 2:
        return 0.0

    if shape == "circle":
        radii = particle_sizes / 2.0
        total_areas = 4.0 * math.pi * radii * radii
    else:
        # Rectangle particles are treated as 3D boxes:
        # a-axis = particle_sizes[i], b-axis = _rect_b, c-axis = _rect_c
        a = particle_sizes
        b = np.full(n, _rect_b)
        c = particle_c_sizes if particle_c_sizes is not None else np.full(n, _rect_c)
        total_areas = 2.0 * (a * b + b * c + a * c)

    total_area = float(np.sum(total_areas))
    if total_area < 1e-15:
        return 0.0

    contacted_area = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dz = positions[j, 1] - positions[i, 1]
            dist = math.sqrt(dx * dx + dz * dz)

            # Keep consistency with other contact diagnostics.
            contact_thresh = (particle_sizes[i] + particle_sizes[j]) / 2.0 * 1.15
            if dist >= contact_thresh:
                continue

            if shape == "circle":
                # Hertz-style contact patch proxy using an effective indentation
                # from the proximity threshold (no force state available here).
                ri = particle_sizes[i] / 2.0
                rj = particle_sizes[j] / 2.0
                r_eff = (ri * rj) / max(ri + rj, 1e-12)
                delta = max(contact_thresh - dist, 0.0)
                a_contact = math.sqrt(max(r_eff * delta, 0.0))
                pair_area = math.pi * a_contact * a_contact
                # Prevent unrealistic patches from consuming large sphere areas.
                pair_area = min(pair_area, 0.25 * math.pi * min(ri, rj) ** 2)
            else:
                # Box-face contact proxy scaled by approach within threshold.
                ai = particle_sizes[i]
                aj = particle_sizes[j]
                ci = (particle_c_sizes[i] if particle_c_sizes is not None else _rect_c)
                cj = (particle_c_sizes[j] if particle_c_sizes is not None else _rect_c)
                min_face_i = min(ai * _rect_b, _rect_b * ci, ai * ci)
                min_face_j = min(aj * _rect_b, _rect_b * cj, aj * cj)
                proximity = (contact_thresh - dist) / max(contact_thresh, 1e-12)
                pair_area = min(min_face_i, min_face_j) * max(0.0, min(1.0, proximity))

            contacted_area[i] += pair_area
            contacted_area[j] += pair_area

    contacted_area = np.minimum(contacted_area, total_areas)
    return float(np.sum(contacted_area)) / total_area


def compute_fabric_anisotropy(contact_normals: np.ndarray) -> float:
    """
    Fabric tensor anisotropy 'a' from contact normal distribution.
    a = 0 → isotropic;  a → 1 → highly anisotropic.
    Defined as a = (F1 - F2) / (F1 + F2) where F1, F2 are eigenvalues
    of the 2D fabric tensor F_ij = (1/Nc) * sum( n_i * n_j ).
    """
    nc = len(contact_normals)
    if nc < 2:
        return 0.0
    F = np.zeros((2, 2))
    for n in contact_normals:
        F[0, 0] += n[0] * n[0]
        F[0, 1] += n[0] * n[1]
        F[1, 0] += n[1] * n[0]
        F[1, 1] += n[1] * n[1]
    F /= nc
    eigvals = np.linalg.eigvalsh(F)
    eigvals = np.sort(eigvals)[::-1]
    denom = eigvals[0] + eigvals[1]
    if denom < 1e-15:
        return 0.0
    return (eigvals[0] - eigvals[1]) / denom


def compute_mobilised_friction_angle(contact_normals: np.ndarray, positions: np.ndarray,
                                      angle_deg: float) -> float:
    """
    Estimate mobilised friction angle from the ratio of shear to normal
    contact forces implied by the fabric tensor orientation relative to
    the current gravity direction.  Standard geotechnical metric:
      phi_mob = arctan(tau / sigma_n)
    Approximated here from the fabric tensor principal direction vs gravity.
    """
    nc = len(contact_normals)
    if nc < 3:
        return 0.0
    F = np.zeros((2, 2))
    for n in contact_normals:
        F[0, 0] += n[0] * n[0]
        F[0, 1] += n[0] * n[1]
        F[1, 0] += n[1] * n[0]
        F[1, 1] += n[1] * n[1]
    F /= nc
    eigvals, eigvecs = np.linalg.eigh(F)
    # Major principal direction
    major_dir = eigvecs[:, np.argmax(eigvals)]
    # Gravity direction
    rad = math.radians(angle_deg)
    g_dir = np.array([math.sin(rad), -math.cos(rad)])
    # Angle between major fabric direction and gravity
    cos_theta = abs(np.dot(major_dir, g_dir))
    cos_theta = min(cos_theta, 1.0)
    theta = math.acos(cos_theta)
    # Mobilised friction ≈ deviation of fabric from gravity alignment
    return math.degrees(theta)


def estimate_factor_of_safety(positions: np.ndarray, angle_deg: float,
                              coord_number: float,
                              particle_planform_areas: Optional[np.ndarray] = None) -> float:
    """
    Simplified Factor of Safety estimate for the tilting hillslope.

    Uses a moment-balance approach about the slope toe:
      FS = (Resisting Moment) / (Driving Moment)

    Resisting moment comes from:
      - Inter-particle friction × normal contact pressure
      - Cohesive bonding
    Driving moment comes from:
      - Gravity component along the tilted slope acting on particles above toe

    This is analogous to the Bishop simplified method adapted for a
    discrete-element assembly.
    """
    if len(positions) < 2:
        return 99.0

    rad = math.radians(angle_deg)
    # Gravity direction in domain frame (tilted)
    gx = math.sin(rad)
    gz = -math.cos(rad)

    # Toe position (lowest, most rightward point)
    toe_x = positions[:, 0].min()
    toe_z = positions[:, 1].min()

    # Driving moment: sum of (gravity tangential component × distance from toe)
    driving = 0.0
    resisting = 0.0
    if particle_planform_areas is not None and len(particle_planform_areas) == len(positions):
        particle_weights = PARTICLE_DENSITY * particle_planform_areas * GRAVITY_MAG
    elif PARTICLE_SHAPE == "circle":
        w = PARTICLE_DENSITY * math.pi * (PARTICLE_SIZE / 2.0) ** 2 * GRAVITY_MAG
        particle_weights = np.full(len(positions), w)
    else:
        w = PARTICLE_DENSITY * (PARTICLE_SIZE * _rect_c) * GRAVITY_MAG
        particle_weights = np.full(len(positions), w)

    for i in range(len(positions)):
        dx = positions[i, 0] - toe_x
        dz = positions[i, 1] - toe_z
        arm = math.sqrt(dx ** 2 + dz ** 2)
        if arm < 1e-12:
            continue
        # Component of weight driving rotation about toe
        # Cross product (gravity vector × lever arm) gives moment
        particle_weight = particle_weights[i]
        moment_drive = particle_weight * abs(gx * dz - gz * dx)
        driving += moment_drive

        # Resisting: friction × normal force approx + cohesion
        # Normal force ≈ weight component perpendicular to slope
        normal_comp = particle_weight * abs(gx * dx + gz * dz) / arm
        resist = (normal_comp * math.tan(math.radians(math.degrees(math.atan(FRICTION_COEFF))))
              + COHESION_FORCE)
        resisting += resist * arm

    if driving < 1e-12:
        return 99.0
    return resisting / driving


# =====================================================================
# 7. MAIN SIMULATION LOOP
# =====================================================================

def run_simulation():
    global PARTICLE_SHAPE

    if FRACTURE_MODE.upper() == "DFN" and PARTICLE_SHAPE != "rectangle":
        print("  [WARN] FRACTURE_MODE='DFN' requires polygonal blocks; forcing PARTICLE_SHAPE='rectangle'.")
        PARTICLE_SHAPE = "rectangle"

    print("=" * 65)
    print("HillslopeSim — Geomorphic Slope Stability via Tilting")
    print("=" * 65)
    print(f"Engine       : {ENGINE}" + (" (quasi-2D)" if ENGINE == "pychrono" and QUASI_2D else ""))
    print(f"Fracture     : {FRACTURE_MODE}")
    print(f"Domain       : {DOMAIN_X} x {DOMAIN_Y} x {DOMAIN_Z} m")
    print(f"Slope        : H={SLOPE_HEIGHT} m, L={SLOPE_LENGTH} m")
    if FRACTURE_MODE.upper() == "DFN":
        _shape_info = (f"DFN blocks, sets={len(DFN_JOINT_SETS)}, "
                       f"min_area={DFN_MIN_BLOCK_AREA:.2e} m2, seed={DFN_SEED}")
    else:
        _shape_info = (f"d={PARTICLE_SIZE} m" if PARTICLE_SHAPE == "circle"
                       else f"a={PARTICLE_SIZE} b={_rect_b:.3f} c={_rect_c:.3f} m, arr={RECT_ARRANGEMENT}")
    print(f"Particles    : shape={PARTICLE_SHAPE}, {_shape_info}, "
          f"porosity={TARGET_POROSITY}")
    print(f"Rotation     : {TOTAL_ROTATION} deg @ {ROTATION_RATE} deg/s "
          f"(settle={SETTLE_TIME} s)")
    print(f"Sim          : T={SIM_DURATION:.1f} s, dt={DT} s, "
          f"{N_STEPS} steps")
    print(f"Output       : {VIDEO_FILE}")
    print("-" * 65)

    # ── Run the slope stability pre-check; may raise COHESION_FORCE ──
    _stabilize_initial_slope()

    print(f"Material     : type={MATERIAL_TYPE}, rho={PARTICLE_DENSITY} kg/m3, "
          f"mu={FRICTION_COEFF}, coh={COHESION_FORCE:.4f} N, e={RESTITUTION}")

    # ── Initialise backend ────────────────────────────────────────
    if ENGINE == "pychrono":
        try:
            backend = PychronoBackend()
        except Exception as e:
            print(f"  [WARN] pychrono init failed ({e}); falling back to pymunk")
            backend = PymunkBackend()
    else:
        backend = PymunkBackend()

    n_particles = backend.num_particles()
    init_pos = np.array(backend.initial_positions)  # (N, 2) in metres
    sizes_static = (np.array(getattr(backend, "particle_sizes", []), dtype=float)
                    if hasattr(backend, "particle_sizes") else np.full(n_particles, PARTICLE_SIZE))
    c_sizes_static = (np.array(getattr(backend, "particle_c_sizes", []), dtype=float)
                      if hasattr(backend, "particle_c_sizes") else np.full(n_particles, _rect_c))
    planform_static = (np.array(getattr(backend, "particle_planform_areas", []), dtype=float)
                       if hasattr(backend, "particle_planform_areas") else None)

    # ── Storage for frame data ────────────────────────────────────
    frames_pos = []         # particle positions per frame
    frames_angle = []       # rotation angle per frame
    frames_time = []
    frames_pct_moved = []
    frames_coord_num = []
    frames_bulk_dens = []
    frames_ke = []
    frames_fabric_a = []
    frames_mob_phi = []
    frames_fs = []
    frames_touch_frac = []
    frames_vel = []
    frames_block_ang = []
    frames_mob_small = []
    frames_mob_medium = []
    frames_mob_large = []

    # DFN class masks for geometric-control diagnostics (terciles by a-axis).
    if len(sizes_static) > 0:
        q1, q2 = np.quantile(sizes_static, [1.0 / 3.0, 2.0 / 3.0])
        mask_small = sizes_static <= q1
        mask_medium = (sizes_static > q1) & (sizes_static <= q2)
        mask_large = sizes_static > q2
    else:
        mask_small = np.zeros(0, dtype=bool)
        mask_medium = np.zeros(0, dtype=bool)
        mask_large = np.zeros(0, dtype=bool)

    # ── Time-stepping ─────────────────────────────────────────────
    wall_t0 = time.time()
    current_angle = 0.0
    settled_positions = None  # will be captured at end of settle phase

    for step in range(N_STEPS + 1):
        t = step * DT

        # Determine rotation angle
        if t < SETTLE_TIME:
            current_angle = 0.0
        elif t < SETTLE_TIME + _rotation_duration:
            current_angle = ROTATION_RATE * (t - SETTLE_TIME)
        else:
            current_angle = TOTAL_ROTATION

        # Capture settled positions at end of settling phase, then form bonds
        if settled_positions is None and t >= SETTLE_TIME:
            settled_positions = backend.get_positions().copy()
            init_pos = settled_positions  # use post-settle positions as baseline
            if hasattr(backend, 'setup_bonds'):
                backend.setup_bonds()

        backend.set_rotation_angle(current_angle)
        backend.step()

        # Save frame
        if step % FRAME_INTERVAL == 0:
            pos = backend.get_positions()
            if hasattr(backend, "particle_sizes") and len(getattr(backend, "particle_sizes", [])) == len(pos):
                sizes = np.array(backend.particle_sizes, dtype=float)
            else:
                sizes = np.full(len(pos), PARTICLE_SIZE, dtype=float)

            if hasattr(backend, "particle_c_sizes") and len(getattr(backend, "particle_c_sizes", [])) == len(pos):
                c_sizes = np.array(backend.particle_c_sizes, dtype=float)
            else:
                c_sizes = np.full(len(pos), _rect_c if PARTICLE_SHAPE == "rectangle" else PARTICLE_SIZE, dtype=float)

            disp = compute_displacement(pos, init_pos)
            pct = compute_pct_moved(disp, sizes)
            coord, normals = backend.get_contact_info()
            vel = backend.get_velocities()
            ke = backend.get_kinetic_energy()
            bd = compute_bulk_density(pos, planform_static)
            fa = compute_fabric_anisotropy(normals)
            mp = compute_mobilised_friction_angle(normals, pos, current_angle)
            fs = estimate_factor_of_safety(pos, current_angle, coord, planform_static)
            touch_frac = compute_touching_surface_fraction(pos, sizes, PARTICLE_SHAPE, c_sizes)
            ang_arr = backend.get_angles() if hasattr(backend, "get_angles") else np.zeros(len(pos))

            def _class_pct(mask: np.ndarray) -> float:
                if len(mask) != len(disp) or not np.any(mask):
                    return 0.0
                return compute_pct_moved(disp[mask], sizes[mask])

            mob_small = _class_pct(mask_small)
            mob_medium = _class_pct(mask_medium)
            mob_large = _class_pct(mask_large)

            frames_pos.append(pos.copy())
            frames_angle.append(current_angle)
            frames_time.append(t)
            frames_pct_moved.append(pct)
            frames_coord_num.append(coord)
            frames_bulk_dens.append(bd)
            frames_ke.append(ke)
            frames_fabric_a.append(fa)
            frames_mob_phi.append(mp)
            frames_fs.append(fs)
            frames_touch_frac.append(touch_frac)
            frames_vel.append(vel.copy())
            frames_block_ang.append(ang_arr.copy())
            frames_mob_small.append(mob_small)
            frames_mob_medium.append(mob_medium)
            frames_mob_large.append(mob_large)

        # Progress
        if step % max(1, N_STEPS // 10) == 0:
            pos = backend.get_positions()
            disp = compute_displacement(pos, init_pos)
            pct = compute_pct_moved(disp, sizes_static)
            elapsed = time.time() - wall_t0
            print(f"  t={t:6.2f}s  angle={current_angle:5.1f}°  "
                  f"moved={pct:5.1f}%  KE={backend.get_kinetic_energy():.2e}  "
                  f"[{elapsed:.1f}s wall]")

    print(f"\nSimulation complete ({time.time() - wall_t0:.1f}s wall time).\n")

    # =====================================================================
    # 8. VISUALISATION
    # =====================================================================
    n_frames = len(frames_time)
    print(f"Rendering {n_frames} frames → {VIDEO_FILE} ...")

    # Convert lists to arrays for plotting
    arr_time = np.array(frames_time)
    arr_angle = np.array(frames_angle)
    arr_pct = np.array(frames_pct_moved)
    arr_coord = np.array(frames_coord_num)
    arr_bd = np.array(frames_bulk_dens)
    arr_ke = np.array(frames_ke)
    arr_fa = np.array(frames_fabric_a)
    arr_mp = np.array(frames_mob_phi)
    arr_fs = np.array(frames_fs)
    arr_touch = np.array(frames_touch_frac)
    arr_vel = np.array(frames_vel) if len(frames_vel) > 0 else np.zeros((0, n_particles, 2))
    arr_mob_small = np.array(frames_mob_small)
    arr_mob_medium = np.array(frames_mob_medium)
    arr_mob_large = np.array(frames_mob_large)

    # ── Figure layout: main particle view + several inset diagnostics ──
    fig = plt.figure(figsize=(16, 10))

    # Main particle view (left)
    ax_main = fig.add_axes([0.05, 0.14, 0.55, 0.74])

    # Right-side diagnostics in a clean 2x3 grid
    ax_disp    = fig.add_axes([0.66, 0.66, 0.145, 0.22])
    ax_fs_plot = fig.add_axes([0.825, 0.66, 0.145, 0.22])
    ax_coord   = fig.add_axes([0.66, 0.40, 0.145, 0.22])
    ax_ke_plot = fig.add_axes([0.825, 0.40, 0.145, 0.22])
    ax_fab     = fig.add_axes([0.66, 0.14, 0.145, 0.22])
    ax_touch   = fig.add_axes([0.825, 0.14, 0.145, 0.22])
    ax_mobility = fig.add_axes([0.05, 0.02, 0.55, 0.10])
    diag_axes = [ax_disp, ax_fs_plot, ax_coord, ax_ke_plot, ax_fab, ax_touch]

    # Publication-style diagnostics: subtle grids, lighter frames.
    for ax in diag_axes:
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("0.35")
        ax.grid(True, alpha=0.18, ls=":", lw=0.8)

    # ── Initial slope outline ──
    outline_x = np.linspace(0, SLOPE_LENGTH, 50)
    outline_z = np.array([_slope_surface_z(x) for x in outline_x])

    # ── Colour map for the primary activity field ──
    is_dfn_mode = (FRACTURE_MODE.upper() == "DFN")
    cmap_disp = plt.colormaps.get_cmap("viridis" if is_dfn_mode else "YlOrRd")
    max_displacement = max(float(np.max(sizes_static)) * 5.0 if len(sizes_static) else PARTICLE_SIZE * 5, 0.3)
    vel_mag_max = (float(np.percentile(np.linalg.norm(arr_vel.reshape(-1, 2), axis=1), 98))
                   if arr_vel.size > 0 else 1.0)
    vel_mag_max = max(vel_mag_max, 1e-3)

    # ── Static elements for main axes ──
    cx, cz = DOMAIN_X / 2.0, DOMAIN_Z / 2.0
    R = math.hypot(DOMAIN_X / 2.0, DOMAIN_Z / 2.0)
    ax_main.set_xlim(cx - R - 0.1, cx + R + 0.1)
    ax_main.set_ylim(cz - R - 0.1, cz + R + 0.1)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("Domain X (m)")
    ax_main.set_ylabel("Domain Z (m)")
    
    slope_line1, = ax_main.plot(outline_x, outline_z, "k--", lw=1, alpha=0.4, label="Initial slope")
    slope_line2, = ax_main.plot([0, 0], [0, SLOPE_HEIGHT], "k--", lw=1, alpha=0.4)
    slope_line3, = ax_main.plot([0, SLOPE_LENGTH], [0, 0], "k--", lw=1, alpha=0.4)
    ax_main.legend(loc="upper right", fontsize=7)

    # Draw walls
    wall_color = "0.3"
    wall_line1, = ax_main.plot([0, DOMAIN_X], [0, 0], color=wall_color, lw=3)         # floor
    wall_line2, = ax_main.plot([0, 0], [0, DOMAIN_Z], color=wall_color, lw=3)         # left
    wall_line3, = ax_main.plot([DOMAIN_X, DOMAIN_X], [0, DOMAIN_Z], color=wall_color, lw=3)  # right

    rotating_elements = [slope_line1, slope_line2, slope_line3, wall_line1, wall_line2, wall_line3]

    # DFN fracture traces (initial geometry) to show mapped fracture network.
    dfn_trace_lines = []
    if is_dfn_mode:
        blocks_vis = _fracture_domain_with_dfn()
        for b in blocks_vis:
            verts = b.get("verts", [])
            if len(verts) < 3:
                continue
            xs = [v[0] for v in verts] + [verts[0][0]]
            zs = [v[1] for v in verts] + [verts[0][1]]
            ln, = ax_main.plot(xs, zs, color="0.45", lw=0.6, alpha=0.45)
            dfn_trace_lines.append(ln)
        rotating_elements.extend(dfn_trace_lines)

    # ── Color Legend ──
    color_max = vel_mag_max if is_dfn_mode else max_displacement
    sm = plt.cm.ScalarMappable(cmap=cmap_disp, norm=plt.Normalize(vmin=0, vmax=color_max))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.61, 0.14, 0.015, 0.74])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Block Speed (m/s)" if is_dfn_mode else "Particle Displacement (m)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ── Displacement vs angle inset ──
    ax_disp.set_xlim(0, TOTAL_ROTATION + 5)
    ax_disp.set_ylim(0, 105)
    ax_disp.set_xlabel("", fontsize=8)
    ax_disp.set_ylabel("% particles moved", fontsize=8)
    ax_disp.set_title("Cumulative Movement", fontsize=8.5)
    ax_disp.axhline(50, color="red", ls="--", lw=0.8, alpha=0.5)
    ax_disp.text(TOTAL_ROTATION * 0.6, 52, "50% failure", fontsize=7, color="red", alpha=0.7)
    disp_line, = ax_disp.plot([], [], "#0b5fa5", lw=2.2)

    # ── Factor of Safety ──
    ax_fs_plot.set_xlim(0, SIM_DURATION)
    ax_fs_plot.set_ylim(0, max(5, max(arr_fs) * 1.1) if len(arr_fs) > 0 else 5)
    ax_fs_plot.set_xlabel("", fontsize=7)
    ax_fs_plot.set_ylabel("FS", fontsize=7)
    ax_fs_plot.set_title("Factor of Safety", fontsize=8.5)
    ax_fs_plot.axhline(1.0, color="red", ls="--", lw=0.8)
    fs_line, = ax_fs_plot.plot([], [], "#2e7d32", lw=2.0)

    # ── Coordination number ──
    ax_coord.set_xlim(0, SIM_DURATION)
    ax_coord.set_ylim(0, max(8, max(arr_coord) * 1.2) if len(arr_coord) > 0 else 8)
    ax_coord.set_xlabel("", fontsize=7)
    ax_coord.set_ylabel("Contacts/particle", fontsize=7)
    ax_coord.set_title("Coordination Number", fontsize=8.5)
    coord_line, = ax_coord.plot([], [], "#8e24aa", lw=2.0)

    # ── Kinetic energy ──
    ax_ke_plot.set_xlim(0, SIM_DURATION)
    ke_max = max(arr_ke) * 1.2 if len(arr_ke) > 0 and max(arr_ke) > 0 else 1.0
    ax_ke_plot.set_ylim(0, ke_max)
    ax_ke_plot.set_xlabel("", fontsize=7)
    ax_ke_plot.set_ylabel("KE (J)", fontsize=7)
    ax_ke_plot.set_title("Kinetic Energy", fontsize=8.5)
    ke_line, = ax_ke_plot.plot([], [], "#c62828", lw=2.0)

    # ── Fabric anisotropy ──
    ax_fab.set_xlim(0, SIM_DURATION)
    ax_fab.set_ylim(0, 1.05)
    ax_fab.set_xlabel("Time (s)", fontsize=7)
    ax_fab.set_ylabel("Anisotropy", fontsize=7)
    ax_fab.set_title("Fabric Anisotropy", fontsize=8.5)
    fab_line, = ax_fab.plot([], [], "#ef6c00", lw=2.0)

    # ── Touching-surface fraction ──
    ax_touch.set_xlim(0, SIM_DURATION)
    ax_touch.set_ylim(0, 105)
    ax_touch.set_xlabel("Time (s)", fontsize=7)
    ax_touch.set_ylabel("Contacted surface (%)", fontsize=7)
    ax_touch.set_title("Touching Fraction", fontsize=8.5)
    touch_line, = ax_touch.plot([], [], "#00838f", lw=2.0)

    # Consistent readability for compact diagnostic panels.
    for ax in diag_axes:
        ax.tick_params(labelsize=7, width=0.8, length=3)

    # ── DFN block-size class mobility panel ──
    ax_mobility.set_xlim(0, SIM_DURATION)
    ax_mobility.set_ylim(0, 105)
    ax_mobility.set_xlabel("Time (s)", fontsize=8)
    ax_mobility.set_ylabel("Class moved (%)", fontsize=8)
    ax_mobility.set_title("DFN Block-Class Mobility", fontsize=9)
    ax_mobility.grid(True, alpha=0.20, ls=":", lw=0.8)
    for sp in ax_mobility.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("0.35")
    ax_mobility.tick_params(labelsize=7, width=0.8, length=3)
    mob_small_line, = ax_mobility.plot([], [], color="#1f77b4", lw=1.8, label="Small")
    mob_med_line, = ax_mobility.plot([], [], color="#ff7f0e", lw=1.8, label="Medium")
    mob_large_line, = ax_mobility.plot([], [], color="#2ca02c", lw=1.8, label="Large")
    ax_mobility.legend(loc="upper left", ncol=3, fontsize=7, frameon=False)
    ax_mobility.set_visible(is_dfn_mode)

    # ── Title ──
    title_text = fig.suptitle("", fontsize=12, fontweight="bold")

    # ── Particle patches (initial) ──
    r_draw = PARTICLE_SIZE / 2.0
    rect_w = PARTICLE_SIZE
    rect_h = _rect_c
    draw_local_vertices = getattr(backend, "particle_draw_local_vertices", None)
    particle_patches = []
    for i in range(n_particles):
        if is_dfn_mode and draw_local_vertices is not None and draw_local_vertices[i] is not None:
            p = MplPolygon(np.zeros((4, 2)), closed=True, lw=0.4, ec="0.2")
        elif PARTICLE_SHAPE == "circle":
            p = MplCircle((0, 0), r_draw, lw=0.3, ec="0.2")
        else:
            p = MplRect((0, 0), rect_w, rect_h, lw=0.3, ec="0.2")
        ax_main.add_patch(p)
        particle_patches.append(p)

    # DFN process overlay: velocity vectors of fastest-moving blocks.
    dfn_quiver_n = min(25, max(n_particles, 1))
    _q0 = np.zeros(dfn_quiver_n)
    _qxy0 = np.column_stack((_q0, _q0))
    dfn_quiver = ax_main.quiver(_qxy0[:, 0], _qxy0[:, 1], _q0, _q0,
                                color="#08306b", alpha=0.70,
                                width=0.0035, scale_units="xy", scale=1.0)
    dfn_quiver.set_visible(is_dfn_mode)

    # ── Static Gravity arrow ──
    ax_main.annotate(
        "g", xy=(cx + R, cz - R + 0.2),
        xytext=(cx + R, cz - R + 0.7),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2.5),
        fontsize=12, color="blue", fontweight="bold",
        ha="center", va="bottom"
    )

    def _update_frame(idx):
        pos = frames_pos[idx]
        vel = frames_vel[idx] if idx < len(frames_vel) else np.zeros_like(pos)
        block_ang = frames_block_ang[idx] if idx < len(frames_block_ang) else np.zeros(len(pos))
        disp = compute_displacement(pos, init_pos)
        angle = frames_angle[idx]
        t = frames_time[idx]

        # Calculate rotational transform to keep gravity pointing down
        tr = matplotlib.transforms.Affine2D().rotate_deg_around(cx, cz, -angle)
        transform = tr + ax_main.transData

        # Apply transformation to rotating static elements
        for elem in rotating_elements:
            elem.set_transform(transform)

        # Update particle patches
        for i in range(n_particles):
            speed = math.hypot(vel[i, 0], vel[i, 1]) if len(vel) else 0.0
            metric = speed if is_dfn_mode else disp[i]
            d_norm = min(metric / max(color_max, 1e-9), 1.0)
            color = cmap_disp(d_norm)

            if is_dfn_mode and draw_local_vertices is not None and draw_local_vertices[i] is not None:
                lv = draw_local_vertices[i]
                ca = math.cos(block_ang[i])
                sa = math.sin(block_ang[i])
                rot = np.array([[ca, -sa], [sa, ca]])
                world = lv @ rot.T + pos[i]
                particle_patches[i].set_xy(world)
            elif PARTICLE_SHAPE == "circle":
                particle_patches[i].center = (pos[i, 0], pos[i, 1])
            else:
                particle_patches[i].set_xy((pos[i, 0] - rect_w / 2,
                                             pos[i, 1] - rect_h / 2))
            particle_patches[i].set_facecolor(color)
            particle_patches[i].set_transform(transform)

        if is_dfn_mode:
            vmag = np.linalg.norm(vel, axis=1)
            if len(vmag) > 0:
                k = min(dfn_quiver_n, len(vmag))
                sel = np.argpartition(vmag, -k)[-k:]

                off = np.zeros((dfn_quiver_n, 2), dtype=float)
                uu = np.zeros(dfn_quiver_n, dtype=float)
                vv = np.zeros(dfn_quiver_n, dtype=float)
                off[:k, :] = pos[sel]
                uu[:k] = vel[sel, 0]
                vv[:k] = vel[sel, 1]

                dfn_quiver.set_offsets(off)
                dfn_quiver.set_UVC(uu, vv)
                dfn_quiver.set_transform(transform)

        # Inset: displacement vs angle
        disp_line.set_data(arr_angle[:idx + 1], arr_pct[:idx + 1])

        # Diagnostics
        fs_line.set_data(arr_time[:idx + 1], arr_fs[:idx + 1])
        coord_line.set_data(arr_time[:idx + 1], arr_coord[:idx + 1])
        ke_line.set_data(arr_time[:idx + 1], arr_ke[:idx + 1])
        fab_line.set_data(arr_time[:idx + 1], arr_fa[:idx + 1])
        touch_line.set_data(arr_time[:idx + 1], arr_touch[:idx + 1] * 100.0)
        mob_small_line.set_data(arr_time[:idx + 1], arr_mob_small[:idx + 1])
        mob_med_line.set_data(arr_time[:idx + 1], arr_mob_medium[:idx + 1])
        mob_large_line.set_data(arr_time[:idx + 1], arr_mob_large[:idx + 1])

        # Title
        pct = arr_pct[idx]
        fs_now = arr_fs[idx]
        status = "STABLE" if pct < 50 and fs_now > 1.0 else "FAILURE"
        title_text.set_text(
            f"t = {t:.2f} s  |  Tilt = {angle:.1f}°  |  "
            f"Moved = {pct:.1f}%  |  FS = {fs_now:.2f}  |  "
            f"Coord# = {arr_coord[idx]:.1f}  [{status}]"
        )
        return particle_patches + rotating_elements + [disp_line, fs_line, coord_line,
                ke_line, fab_line, touch_line, mob_small_line, mob_med_line,
                mob_large_line, dfn_quiver, title_text]

    anim = animation.FuncAnimation(
        fig, _update_frame, frames=n_frames,
        interval=1000 / FPS, blit=False, repeat=False,
    )

    # Save
    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=2500)
        anim.save(VIDEO_FILE, writer=writer)
        print(f"  Saved: {VIDEO_FILE}")
    except Exception as e:
        print(f"  Could not save animation ({e})")
        print("  Showing interactive plot instead...")
        plt.show()

    # =====================================================================
    # 9. SUMMARY REPORT
    # =====================================================================
    print("\n" + "=" * 65)
    print("STABILITY ANALYSIS SUMMARY")
    print("=" * 65)

    # Find failure angle (first time >50% moved, after settling phase)
    failure_idx = None
    for i, pct in enumerate(arr_pct):
        if arr_time[i] >= SETTLE_TIME and pct >= 50.0:
            failure_idx = i
            break

    if failure_idx is not None:
        print(f"  FAILURE at tilt = {arr_angle[failure_idx]:.1f}° "
              f"(t = {arr_time[failure_idx]:.2f} s)")
    else:
        print(f"  NO FAILURE detected (max moved = {max(arr_pct):.1f}%)")

    print(f"\n  --- Geotechnical Diagnostics ---")
    print(f"  Particles              : {n_particles}")
    print(f"  Final % moved          : {arr_pct[-1]:.1f}%")
    print(f"  Min Factor of Safety   : {min(arr_fs):.3f}")
    print(f"  Final FS               : {arr_fs[-1]:.3f}")
    print(f"  Max kinetic energy     : {max(arr_ke):.4e} J")
    print(f"  Final coordination #   : {arr_coord[-1]:.2f}")
    print(f"  Final touching surface : {arr_touch[-1] * 100:.1f}%")
    print(f"  Min touching surface   : {arr_touch.min() * 100:.1f}%")
    if is_dfn_mode and len(arr_mob_small) > 0:
        print(f"  Final class mobility   : small={arr_mob_small[-1]:.1f}%  "
              f"medium={arr_mob_medium[-1]:.1f}%  large={arr_mob_large[-1]:.1f}%")
        print(f"  Peak class mobility    : small={arr_mob_small.max():.1f}%  "
              f"medium={arr_mob_medium.max():.1f}%  large={arr_mob_large.max():.1f}%")
    print(f"  Final fabric anisotropy: {arr_fa[-1]:.3f}")
    print(f"  Final mob. friction    : {arr_mp[-1]:.1f}°")
    print(f"  Final bulk density     : {arr_bd[-1]:.0f} kg/m³")

    print(f"\n  --- Metric Definitions ---")
    print(f"  Factor of Safety (FS): Ratio of resisting to driving moments")
    print(f"    about the slope toe. FS < 1 indicates instability.")
    print(f"  Coordination Number: Average contacts per particle.")
    print(f"    Higher values indicate a more mechanically stable packing.")
    print(f"  Touching-surface fraction: Estimated contacted surface area")
    print(f"    (material in contact with other material), divided by")
    print(f"    total material surface area.")
    print(f"  Fabric Anisotropy: Directional bias of contact normals (0=isotropic,")
    print(f"    1=all contacts aligned). High values with horizontal bias resist")
    print(f"    lateral sliding.")
    print(f"  Mobilised Friction Angle: Angular deviation of the fabric tensor")
    print(f"    principal direction from gravity. Approaches the material friction")
    print(f"    angle at failure.")
    print(f"  Kinetic Energy: Total KE of all particles. A spike indicates")
    print(f"    dynamic failure / rapid mass movement.")

    print("=" * 65)

    # Show as Jupyter display if available
    try:
        from IPython.display import Video
        return Video(VIDEO_FILE)
    except Exception:
        pass

# =====================================================================
# 10. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    run_simulation()
