#!/usr/bin/env python3
"""
HillslopeSim.py — Geomorphic Slope Stability via Discrete-Element Tilting
==========================================================================
A physics-based hillslope stability model that fractures a cohesive slope
into discrete particles (rectangles or spheres), then progressively tilts
the domain (rotates gravity by 90°) to test slope failure.

The default Stream bed set up uses StreamBed.py to simulate submerged
sediment with planar PyChrono grains, a mean-bed outlet barrier and a
separate 10-second flow clock.

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
from contextlib import contextmanager
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
from ParticleSizes import parse_size_input, ProgressEstimate, format_duration
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
SLOPE_HEIGHT = 0.1      # Stream-bed top before the 0.2 geometry scale (m)
SLOPE_LENGTH = 2.0      # Sediment length before the geometry scale (m)
SLOPE_BASE   = 0.0      # Elevation of slope toe (m)

# -------------------------------------------------------------------
# 1.4  FRACTURE / PARTICLE PARAMETERS
# -------------------------------------------------------------------
PARTICLE_SHAPE = "circle" # "circle", "rectangle", or "mixed"
PARTICLE_SIZE  = 0.003       # Representative diameter for the default mixture (m)
                             #   rectangle long edge (X-dimension)
PARTICLE_SIZE_INPUT = ".002;.004" # Equal counts of 2 mm and 4 mm grains
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
FRACTURE_MODE = "grid"      # "grid" (particle packing) or "DFN" (joint-set blocks)
TARGET_POROSITY = 0.55      # Attainable initial porosity for this random size mixture
GRAVITY_PACKED = True       # Close vertical voids before t=0; sliding/rolling remain free
FILL_VOID = False           # With Gravity Packed: add supported particles up to the slope outline
HOTSTART_PACK = False       # Repeat gravity packing, filling and actual solver settling before t=0
HOTSTART_AREA_TOLERANCE = 0.02  # Relative tolerance on settled bulk area (2%, not solid-particle area)
HOTSTART_MAX_CYCLES = 25
HOTSTART_MIN_SETTLE_TIME = 0.5  # Minimum solver time per cycle (s)
HOTSTART_MAX_SETTLE_TIME = 6.0  # Maximum solver time per cycle; unsettled cycles continue without filling
HOTSTART_SPEED_TOLERANCE = 0.003  # Maximum linear/rotational surface speed for rest (m/s)
_HOTSTART_OPTIONS = ("HOTSTART_AREA_TOLERANCE", "HOTSTART_MAX_CYCLES", "HOTSTART_MIN_SETTLE_TIME",
                     "HOTSTART_MAX_SETTLE_TIME", "HOTSTART_SPEED_TOLERANCE")
_HOTSTART_DEFAULTS = {name: globals()[name] for name in _HOTSTART_OPTIONS}

# Mixed-shape initial conditions (PARTICLE_SHAPE = "mixed"). Uses random
# packing instead of DFN fractures; circles become spheres in pychrono.
# All fractions below are by NUMBER, not area, volume, or mass.
CIRCLE_FRACTION = 0.5        # 0 = rectangles only, 1 = circles only; rounded to nearest count
INITIAL_SEED = 42            # Repeatable locations, categories, sizes and orientations
INITIAL_PARTICLE_COUNT = 0   # 0: estimate from TARGET_POROSITY; >0: require this exact count
CIRCLE_DIAMETER = 0.06       # Circle/sphere diameter independent of rectangle a-axis (m)
MIX_SIZE_SPREAD = 0.0        # Uniform fractional size variation +/- this value (0 <= value < 1)
INITIAL_CLEARANCE = 0.00005  # Minimum initial gap, suitable for 2 mm grains (m)
MIX_RECT_ANGLE_DEG = 0.0     # Mean rectangle orientation in the XZ plane (deg from +X)
MIX_RECT_ANGLE_SPREAD = 180.0 # Uniform angular range centered on mean; 180 = all orientations
PACKING_ATTEMPTS = 500      # Placement trials per particle before reducing automatic count

_MIXED_FIELDS = {
    "CIRCLE_FRACTION": ("Circle fraction by count (0-1):", CIRCLE_FRACTION),
    "INITIAL_SEED": ("Initial random seed:", INITIAL_SEED),
    "INITIAL_PARTICLE_COUNT": ("Particle count (0 = automatic):", INITIAL_PARTICLE_COUNT),
    "CIRCLE_DIAMETER": ("Circle diameter (m):", CIRCLE_DIAMETER),
    "MIX_SIZE_SPREAD": ("Size variation +/- fraction:", MIX_SIZE_SPREAD),
    "INITIAL_CLEARANCE": ("Minimum initial gap (m):", INITIAL_CLEARANCE),
    "MIX_RECT_ANGLE_DEG": ("Mean rectangle angle (deg):", MIX_RECT_ANGLE_DEG),
    "MIX_RECT_ANGLE_SPREAD": ("Rectangle angular range (deg):", MIX_RECT_ANGLE_SPREAD),
    "PACKING_ATTEMPTS": ("Placement trials per particle:", PACKING_ATTEMPTS),
}

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
PARTICLE_DENSITY = 2650.0   # kg/m³, typical quartz gravel
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

# Contact physics can be disabled independently without losing material values.
SIMULATE_CONTACT_FRICTION = True
SIMULATE_RESTITUTION = True
SIMULATE_COHESION = True
SIMULATE_BUOYANCY = True
SIMULATE_HYDRAULIC_FORCES = True
SIMULATE_LIFT = True
_PARTICLE_PHYSICS_FIELDS = {
    "SIMULATE_CONTACT_FRICTION": ("Contact friction", True),
    "SIMULATE_RESTITUTION": ("Collision restitution (bounce)", True),
    "SIMULATE_COHESION": ("Cohesion / breakable bonds", True),
    "SIMULATE_BUOYANCY": ("Buoyancy (stream bed)", True),
    "SIMULATE_HYDRAULIC_FORCES": ("Fluid drag / traction (stream bed)", True),
    "SIMULATE_LIFT": ("Additional lift closure (stream bed)", True),
}


def _contact_parameters():
    """Effective contact properties; stored material coefficients stay unchanged."""
    return (FRICTION_COEFF if SIMULATE_CONTACT_FRICTION else 0.0,
            RESTITUTION if SIMULATE_RESTITUTION else 0.0,
            COHESION_FORCE if SIMULATE_COHESION and MATERIAL_TYPE != "granular" else 0.0)

# -------------------------------------------------------------------
# 1.6  TILTING / ROTATION PARAMETERS
# -------------------------------------------------------------------
TOTAL_ROTATION   = 90.0     # Total clockwise rotation of domain (degrees)
ROTATION_RATE    = 2.0      # Degrees per second of simulation time
SETTLE_TIME      = 0.5      # Submerged preparation time before stream flow starts (s)

# Stream-bed runs use an XZ section (unit width for Pymunk; SI 3D grains for Chrono).
# Geometry: H = flat bed top, L = sediment length, base = rigid substrate level.
STREAM_BED_SETUP = True
STREAM_DOMAIN_SCALE = 0.2   # Scale box/grid/bed geometry; particle sizes stay physical
_STREAM_SCALE_ACTIVE = False
_STREAM_GEOMETRY_FIELDS = ("DOMAIN_X", "DOMAIN_Y", "DOMAIN_Z", "CELL_SIZE",
                           "SLOPE_HEIGHT", "SLOPE_LENGTH", "SLOPE_BASE")
RECIRCULATION = False       # Return exported grains to the inlet at unchanged Z/velocity
STREAM_FORCING = "velocity"
BED_STRESS_MEAN = 10.0       # N/m2 (Pa)
BED_STRESS_AMPLITUDE = 5.0   # Pa; harmonic only, <= mean (left-to-right flow)
BED_STRESS_FREQUENCY = 0.1   # Hz
STREAM_DURATION = 10.0      # Flow duration; initialization settling is separate
WATER_DENSITY = 1000.0      # kg/m3; grains are assumed fully submerged
GRAIN_STRESS_FACTOR = 1.0   # Effective grain traction / prescribed bed stress
STREAM_LIFT_RATIO = 0.0     # Lift / horizontal force; optional closure
HYDRAULIC_SOLVER = "simple"  # "simple" log law or "advanced" equilibrium 1DV RANS
FLOW_MEAN_VELOCITY = 1.0    # Depth-mean inlet velocity (m/s), velocity forcing
FLOW_DEPTH = 5.            # Reference depth above initial H; fixed stage H + depth
WATER_VISCOSITY = 1e-6      # Kinematic viscosity (m2/s)
ROUGHNESS_D84_FACTOR = 3.0  # ks / surface D84; z0 = ks / 30 (grain roughness)
FLOW_COLUMNS = 24
FLOW_LEVELS = 96
FLOW_UPDATE_INTERVAL = .05 # Hydraulic bed/roughness refresh (s)
HYDRAULIC_LOAD = "drag"     # Relative-velocity drag for velocity-driven flow
STREAM_DRAG_COEFFICIENT = .8
STREAM_DRAG_INTEGRATION = "implicit"  # Stable drag impulse; "explicit" retains the old force evaluation
STREAM_STARTUP_TIME = 2.   # Smooth velocity/u* ramp in flow seconds; 0 = immediate full forcing
STREAM_SETTLE_DAMPING = 5. # Preparation-only velocity decay rate (1/s); 0 = disabled
STREAM_CONTACT_TRAVEL_FRACTION = .02  # Additional stream contact-resolution travel / grain thickness
SIMULATE_TURBULENCE = False
TURBULENCE_TARGET = "stress"  # "stress" or "velocity" (requires drag loads)
TURBULENCE_INTENSITY = 1.0    # Multiplier on river u*/Reynolds-stress scaling
TURBULENCE_STRESS_CV = .5     # std(tau)/mean(tau) at multiplier 1
TURBULENCE_LENGTH_X = 1.0    # Streamwise correlation length / water depth
TURBULENCE_LENGTH_Z = .25    # Vertical correlation length / water depth
TURBULENCE_TIME_FACTOR = 1.0 # Correlation time = factor * depth / mean speed
TURBULENCE_SEED = 314159
_STREAM_FIELDS = {
    "STREAM_DOMAIN_SCALE": ("Domain / bed scale factor:", STREAM_DOMAIN_SCALE),
    "STREAM_FORCING": ("Hydraulic forcing:", STREAM_FORCING),
    "BED_STRESS_MEAN": ("Reference bed stress (N/m²):", BED_STRESS_MEAN),
    "BED_STRESS_AMPLITUDE": ("Harmonic amplitude (N/m²):", BED_STRESS_AMPLITUDE),
    "BED_STRESS_FREQUENCY": ("Harmonic frequency (Hz):", BED_STRESS_FREQUENCY),
    "STREAM_DURATION": ("Flow duration (s):", STREAM_DURATION),
    "WATER_DENSITY": ("Water density (kg/m³):", WATER_DENSITY),
    "GRAIN_STRESS_FACTOR": ("Grain stress multiplier:", GRAIN_STRESS_FACTOR),
    "STREAM_LIFT_RATIO": ("Lift / horizontal force:", STREAM_LIFT_RATIO),
    "HYDRAULIC_SOLVER": ("Flow solver:", HYDRAULIC_SOLVER),
    "FLOW_MEAN_VELOCITY": ("Mean inlet velocity (m/s):", FLOW_MEAN_VELOCITY),
    "FLOW_DEPTH": ("Depth above initial bed H (m):", FLOW_DEPTH),
    "WATER_VISCOSITY": ("Kinematic viscosity (m2/s):", WATER_VISCOSITY),
    "ROUGHNESS_D84_FACTOR": ("Roughness ks / surface D84:", ROUGHNESS_D84_FACTOR),
    "FLOW_COLUMNS": ("Hydraulic X columns:", FLOW_COLUMNS),
    "FLOW_LEVELS": ("Vertical grid nodes:", FLOW_LEVELS),
    "FLOW_UPDATE_INTERVAL": ("Hydraulic refresh (s):", FLOW_UPDATE_INTERVAL),
    "HYDRAULIC_LOAD": ("Grain force closure:", HYDRAULIC_LOAD),
    "STREAM_DRAG_COEFFICIENT": ("Drag coefficient Cd:", STREAM_DRAG_COEFFICIENT),
    "STREAM_DRAG_INTEGRATION": ("Drag integration:", STREAM_DRAG_INTEGRATION),
    "STREAM_STARTUP_TIME": ("Flow startup ramp (s, 0 = off):", STREAM_STARTUP_TIME),
    "STREAM_SETTLE_DAMPING": ("Settling damping rate (1/s):", STREAM_SETTLE_DAMPING),
    "STREAM_CONTACT_TRAVEL_FRACTION": ("Contact travel / grain thickness:", STREAM_CONTACT_TRAVEL_FRACTION),
    "SIMULATE_TURBULENCE": ("Simulate turbulence", SIMULATE_TURBULENCE),
    "TURBULENCE_TARGET": ("Turbulence forcing field:", TURBULENCE_TARGET),
    "TURBULENCE_INTENSITY": ("Turbulence intensity multiplier:", TURBULENCE_INTENSITY),
    "TURBULENCE_STRESS_CV": ("Bed-stress RMS / mean:", TURBULENCE_STRESS_CV),
    "TURBULENCE_LENGTH_X": ("Correlation length X / depth:", TURBULENCE_LENGTH_X),
    "TURBULENCE_LENGTH_Z": ("Correlation length Z / depth:", TURBULENCE_LENGTH_Z),
    "TURBULENCE_TIME_FACTOR": ("Eddy time multiplier:", TURBULENCE_TIME_FACTOR),
    "TURBULENCE_SEED": ("Turbulence random seed:", TURBULENCE_SEED),
}
_STREAM_CHOICES = {"STREAM_FORCING": ["steady", "harmonic", "velocity"],
                   "HYDRAULIC_SOLVER": ["simple", "advanced"],
                   "HYDRAULIC_LOAD": ["traction", "drag"],
                   "STREAM_DRAG_INTEGRATION": ["implicit", "explicit"],
                   "TURBULENCE_TARGET": ["stress", "velocity"]}


def _parse_control(value, default):
    """Strict JSON/Tk conversion shared by hydraulic and numerical controls."""
    if isinstance(default, bool):
        if str(value).lower() not in ("true", "false", "1", "0"):
            raise ValueError("Expected a checkbox/boolean value.")
        return str(value).lower() in ("true", "1")
    if isinstance(default, int):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError("Grid sizes, iterations and seeds must be integers.")
        return int(number)
    return str(value).strip() if isinstance(default, str) else float(value)

# -------------------------------------------------------------------
# 1.7  SIMULATION CONTROLS
# -------------------------------------------------------------------
DT               = 0.0005   # Maximum physics time-step (s) for small grains
SOLVER_ITERS     = 150      # Constraint solver iterations per step
GRAVITY_MAG      = 9.81     # m/s²
ADAPTIVE_DT = True         # Bound translation + rotational surface travel per substep
DT_MIN = 1e-6              # Fail explicitly if accuracy requires a smaller step
DT_TRAVEL_FRACTION = .1    # Maximum predicted travel / smallest grain thickness
CHRONO_SOLVER = "PSOR"
CHRONO_TOLERANCE = 1e-8
CHRONO_RECOVERY_SPEED = .2 # m/s, penetration stabilization (NSC)
CHRONO_BOUNCE_SPEED = .05  # m/s, no restitution below this speed
CHRONO_ENVELOPE_FACTOR = .01
_SOLVER_FIELDS = {
    "ADAPTIVE_DT": ("Adaptive computational timesteps", ADAPTIVE_DT),
    "DT_MIN": ("Minimum adaptive dt (s):", DT_MIN),
    "DT_TRAVEL_FRACTION": ("Max travel / grain thickness:", DT_TRAVEL_FRACTION),
    "CHRONO_SOLVER": ("Chrono contact solver:", CHRONO_SOLVER),
    "CHRONO_TOLERANCE": ("Chrono solver tolerance:", CHRONO_TOLERANCE),
    "CHRONO_RECOVERY_SPEED": ("Penetration recovery (m/s):", CHRONO_RECOVERY_SPEED),
    "CHRONO_BOUNCE_SPEED": ("Restitution threshold (m/s):", CHRONO_BOUNCE_SPEED),
    "CHRONO_ENVELOPE_FACTOR": ("Collision envelope / min grain:", CHRONO_ENVELOPE_FACTOR),
}

# -------------------------------------------------------------------
# 1.8  OUTPUT
# -------------------------------------------------------------------
FPS              = 30       # Animation frames per second
FRAME_INTERVAL   = 5        # Save a frame every N physics steps
FRAME_INTERVAL_MODE = "seconds"  # Record nearest solver state to 0.1 s targets
FRAME_INTERVAL_SECONDS = .1
ANIMATION_START_TIME = 0.0  # Simulation clock (stream flow clock excludes settling)
ANIMATION_STOP_TIME = None # None / blank: record through simulation end
_RECORDING_FIELDS = {
    "FRAME_INTERVAL_MODE": ("Frame interval units:", FRAME_INTERVAL_MODE),
    "FRAME_INTERVAL_SECONDS": ("Frame interval (seconds):", FRAME_INTERVAL_SECONDS),
    "ANIMATION_START_TIME": ("Animation start time (s):", ANIMATION_START_TIME),
    "ANIMATION_STOP_TIME": ("Animation stop (s, blank = end):", ANIMATION_STOP_TIME),
}
VIDEO_FILE       = "stream_bed_sim.mp4"
DISPLACEMENT_THRESHOLD = 2.0 * PARTICLE_SIZE  # "significant movement" threshold (2 diameters)

# -------------------------------------------------------------------
# 1.9  MATERIAL PRESETS (auto-applied if MATERIAL defaults are used)
# -------------------------------------------------------------------
# Adjust FRICTION_COEFF and COHESION_FORCE based on MATERIAL_TYPE.
# Uncomment to use presets; otherwise manually set above.
if MATERIAL_TYPE == "granular":
    FRICTION_COEFF = 0.80      # φ ≈ 39° (sand: 30–45°, gravel: 35–50°)
    COHESION_FORCE = 0.0       # Non-cohesive; zero bonding
    if STREAM_BED_SETUP:
        FRICTION_COEFF = 0.60
elif MATERIAL_TYPE == "cohesive":
    # FRICTION_COEFF and COHESION_FORCE retain dashboard values (user-set)
    # Default: FRICTION_COEFF = 0.6 (φ ≈ 31°), COHESION_FORCE = 5 N
    pass

# =====================================================================
# 2. DERIVED QUANTITIES
# =====================================================================
def _recompute_derived_quantities():
    """Recompute derived simulation and geometry variables from current globals."""
    global _rotation_duration, SIM_DURATION, N_STEPS
    global _is_quasi2d, _is_3d_full, _rect_b, _rect_c, DISPLACEMENT_THRESHOLD
    global _DFN_BLOCKS_CACHE, _MIXED_PARTICLES_CACHE, _MIXED_PACKING_INFO, FRACTURE_MODE
    global _PACKED_SPECS_CACHE, _GRAVITY_PACKING_INFO, _FILL_VOID_INFO
    global _HOTSTART_INFO
    global GRAVITY_PACKED, FILL_VOID
    global ENGINE, QUASI_2D
    global PARTICLE_SIZE

    from TimeStepping import validate_controls
    validate_controls(sys.modules[__name__])
    from Recording import validate_recording
    validate_recording(sys.modules[__name__])
    distribution = _size_distribution()
    if PARTICLE_SIZE_INPUT:
        PARTICLE_SIZE = distribution.representative
    if distribution.mode != "single" and FRACTURE_MODE.upper() == "DFN":
        FRACTURE_MODE = "grid"
        print("  [INFO] Particle-size distributions use particle packing; switching DFN to grid.")

    if STREAM_BED_SETUP:
        from StreamBed import validate_parameters
        validate_parameters(sys.modules[__name__])
        if ENGINE == "pychrono":
            QUASI_2D = True  # Surface exposure is evaluated in one XZ section.

    _DFN_BLOCKS_CACHE = None
    _MIXED_PARTICLES_CACHE = None
    _MIXED_PACKING_INFO = None
    _PACKED_SPECS_CACHE = None
    _GRAVITY_PACKING_INFO = None
    _FILL_VOID_INFO = None
    _HOTSTART_INFO = None
    if HOTSTART_PACK:
        GRAVITY_PACKED = FILL_VOID = True
        if (not 0 < HOTSTART_AREA_TOLERANCE < 1 or HOTSTART_MAX_CYCLES < 1
                or not isinstance(HOTSTART_MAX_CYCLES, (int, np.integer))
                or not 0 < HOTSTART_MIN_SETTLE_TIME <= HOTSTART_MAX_SETTLE_TIME
                or not math.isfinite(DT) or DT <= 0
                or not math.isfinite(HOTSTART_SPEED_TOLERANCE) or HOTSTART_SPEED_TOLERANCE <= 0):
            raise ValueError("Invalid Hotstart Pack tolerance, cycle count, settling time or speed threshold.")
    if PARTICLE_SHAPE == "mixed":
        _validate_mixed_parameters()
        if FRACTURE_MODE.upper() == "DFN":
            print("  [INFO] Mixed shapes use random packing; switching FRACTURE_MODE to 'grid'.")
            FRACTURE_MODE = "grid"

    _rotation_duration = TOTAL_ROTATION / max(ROTATION_RATE, 1e-6)
    SIM_DURATION = STREAM_DURATION if STREAM_BED_SETUP else SETTLE_TIME + _rotation_duration + 2.0
    N_STEPS = int(math.ceil(SIM_DURATION / DT))
    DISPLACEMENT_THRESHOLD = 2.0 * PARTICLE_SIZE

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


def _wall_inset():
    """Actual Pymunk wall radius, scaled with the stream geometry."""
    if ENGINE != "pymunk":
        return 0.
    return .02 * (STREAM_DOMAIN_SCALE if STREAM_BED_SETUP and _STREAM_SCALE_ACTIVE else 1.)


@contextmanager
def _scaled_stream_geometry():
    """Apply the scale once for a preview/run; retain reference values in controls/configs."""
    global _STREAM_SCALE_ACTIVE
    if not STREAM_BED_SETUP or _STREAM_SCALE_ACTIVE:
        yield
        return
    if not math.isfinite(STREAM_DOMAIN_SCALE) or STREAM_DOMAIN_SCALE <= 0:
        raise ValueError("Domain / bed scale factor must be positive and finite.")
    reference = {name: globals()[name] for name in _STREAM_GEOMETRY_FIELDS}
    _STREAM_SCALE_ACTIVE = True
    try:
        for name, value in reference.items():
            globals()[name] = value * STREAM_DOMAIN_SCALE
        _recompute_derived_quantities()
        yield
    finally:
        globals().update(reference)
        _STREAM_SCALE_ACTIVE = False
        _recompute_derived_quantities()

# =====================================================================
# 3. HILLSLOPE GENERATION + FRACTURE INTO PARTICLES
# =====================================================================

def _slope_surface_z(x: float) -> float:
    """Return the slope surface elevation at horizontal position x."""
    if STREAM_BED_SETUP:
        return SLOPE_HEIGHT if x <= SLOPE_LENGTH else SLOPE_BASE
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
    if STREAM_BED_SETUP:
        from shapely.geometry import box
        # Build inside the actual Pymunk wall faces to avoid compressing an
        # edge-to-edge DFN mosaic when gravity packing adjusts wall overlaps.
        wall = _wall_inset()
        slope_poly = box(wall, SLOPE_BASE, min(SLOPE_LENGTH, DOMAIN_X-wall), SLOPE_HEIGHT)

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

    if GRAVITY_PACKED and block_desc:
        specs = [dict(x=b["centroid"][0], z=b["centroid"][1], shape="rectangle", dfn_source=i,
                      world_local=np.asarray(b["verts"]) - b["centroid"])
                 for i, b in enumerate(block_desc)]
        packed = _gravity_pack_particles(specs)
        if FILL_VOID:
            packed = _fill_packed_voids(packed)
        filled_blocks = []
        for p in packed:
            b = dict(block_desc[p["dfn_source"]])
            b["centroid"] = (p["x"], p["z"])
            b["verts"] = (p["world_local"] + [p["x"], p["z"]]).tolist()
            filled_blocks.append(b)
        block_desc = filled_blocks
    _DFN_BLOCKS_CACHE = block_desc
    return _DFN_BLOCKS_CACHE


def _validate_mixed_parameters():
    values = (CIRCLE_FRACTION, CIRCLE_DIAMETER, MIX_SIZE_SPREAD, INITIAL_CLEARANCE,
              MIX_RECT_ANGLE_DEG, MIX_RECT_ANGLE_SPREAD, TARGET_POROSITY,
              PARTICLE_SIZE, _b_axis_input(), PARTICLE_C_AXIS or _b_axis_input())
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Mixed-shape parameters must be finite.")
    if not 0 <= CIRCLE_FRACTION <= 1:
        raise ValueError("Circle fraction must be between 0 and 1 (by particle count).")
    if min(CIRCLE_DIAMETER, PARTICLE_SIZE, _b_axis_input(),
           PARTICLE_C_AXIS if PARTICLE_C_AXIS is not None else _b_axis_input()) <= 0:
        raise ValueError("Particle dimensions must be positive.")
    if not 0 <= MIX_SIZE_SPREAD < 1 or not 0 <= TARGET_POROSITY < 1:
        raise ValueError("Size spread and target porosity must be in [0, 1).")
    if INITIAL_CLEARANCE < 0 or not 0 <= MIX_RECT_ANGLE_SPREAD <= 180:
        raise ValueError("Clearance must be nonnegative and angular range must be 0 to 180 degrees.")
    for name, value, minimum in (("Initial particle count", INITIAL_PARTICLE_COUNT, 0),
                                  ("Packing attempts", PACKING_ATTEMPTS, 1),
                                  ("Initial seed", INITIAL_SEED, 0)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")


def _b_axis_input():
    return PARTICLE_B_AXIS if PARTICLE_B_AXIS is not None else PARTICLE_SIZE


def _size_distribution():
    return parse_size_input(PARTICLE_SIZE_INPUT or PARTICLE_SIZE)


def _generate_mixed_particles():
    """Seeded, nonintersecting random XZ packing shared by preview and both engines.

    Exact category counts are rounded once per candidate assembly. Automatic
    counts are reduced if random sequential placement jams; explicit counts
    fail with an actionable error instead of silently changing the mixture.
    Rectangle orientations rotate around Y, including for full-3D Chrono.
    """
    global _MIXED_PARTICLES_CACHE, _MIXED_PACKING_INFO
    if _MIXED_PARTICLES_CACHE is not None:
        return _MIXED_PARTICLES_CACHE
    from shapely.geometry import Point, Polygon, box
    from shapely.prepared import prep

    _validate_mixed_parameters()
    distribution = _size_distribution()
    sized = distribution.mode != "single"
    circle_fraction = CIRCLE_FRACTION if PARTICLE_SHAPE == "mixed" else float(PARTICLE_SHAPE == "circle")
    boundary = Polygon([(0, 0), (SLOPE_LENGTH, 0),
                        (SLOPE_LENGTH, SLOPE_BASE), (0, SLOPE_HEIGHT)])
    if STREAM_BED_SETUP:
        boundary = box(0, SLOPE_BASE, SLOPE_LENGTH, SLOPE_HEIGHT)
    boundary = boundary.intersection(box(0, 0, DOMAIN_X, DOMAIN_Z))
    available = boundary.buffer(-INITIAL_CLEARANCE)
    if ENGINE == "pymunk":
        # Pymunk wall segments extend 0.02 m into the nominal domain.
        inset = _wall_inset() + INITIAL_CLEARANCE
        available = available.intersection(box(inset, SLOPE_BASE+INITIAL_CLEARANCE if STREAM_BED_SETUP else inset,
                                               DOMAIN_X - inset, DOMAIN_Z - inset))
    if available.is_empty:
        raise ValueError("Initial clearance leaves no room inside the slope.")
    inside = prep(available)
    if sized:
        estimate_sizes = distribution.sample(300*distribution.class_count, np.random.default_rng(INITIAL_SEED))
        mean_area = float(np.mean(estimate_sizes**2)) * (
            circle_fraction*math.pi/4 + (1-circle_fraction)*_rect_c/PARTICLE_SIZE)
    else:
        mean_area = (CIRCLE_FRACTION * math.pi * CIRCLE_DIAMETER ** 2 / 4
                     + (1 - CIRCLE_FRACTION) * PARTICLE_SIZE * _rect_c)
        mean_area *= 1 + MIX_SIZE_SPREAD ** 2 / 3
    count = INITIAL_PARTICLE_COUNT or max(1, int(boundary.area * (1 - TARGET_POROSITY) / mean_area))
    if sized and distribution.mode == "equal":
        if INITIAL_PARTICLE_COUNT and count % distribution.class_count:
            raise ValueError(f"Particle count must be a multiple of {distribution.class_count} for equal-count sizes.")
        count = max(distribution.class_count, count//distribution.class_count*distribution.class_count)
    rng = np.random.default_rng(INITIAL_SEED)
    xmin, zmin, xmax, zmax = available.bounds
    for _ in range(24):
        nc = int(math.floor(count * circle_fraction + 0.5))
        kinds = np.array(["circle"] * nc + ["rectangle"] * (count - nc))
        rng.shuffle(kinds)
        sizes = distribution.sample(count, rng) if sized else None
        specs = []
        for index, kind in enumerate(kinds):
            scale = sizes[index]/PARTICLE_SIZE if sized else rng.uniform(1 - MIX_SIZE_SPREAD, 1 + MIX_SIZE_SPREAD)
            a = sizes[index] if sized else (CIRCLE_DIAMETER if kind == "circle" else PARTICLE_SIZE) * scale
            c = a if kind == "circle" else _rect_c * scale
            b = a if kind == "circle" else (_rect_c if _is_quasi2d else _rect_b) * scale
            angle = math.radians(MIX_RECT_ANGLE_DEG + rng.uniform(
                -MIX_RECT_ANGLE_SPREAD / 2, MIX_RECT_ANGLE_SPREAD / 2)) if kind == "rectangle" else 0.0
            if sized and PARTICLE_SHAPE == "rectangle":
                angle = rng.uniform(0, math.pi) if RECT_ARRANGEMENT == "mixed" else 0.
            local = None if kind == "circle" else np.array([[-a/2, -c/2], [a/2, -c/2],
                                                                         [a/2, c/2], [-a/2, c/2]])
            radius = a / 2 if kind == "circle" else math.hypot(a, c) / 2
            specs.append(dict(shape=str(kind), size=a, c=c, b=b, angle=angle,
                              local=local, radius=radius))
        # Large particles first avoids filling all usable space with small ones.
        specs.sort(key=lambda p: p["radius"], reverse=True)
        cell = max(p["radius"] for p in specs) * 2 + INITIAL_CLEARANCE
        buckets = {}
        placed = []
        for p in specs:
            if p["local"] is not None:
                ca, sa = math.cos(p["angle"]), math.sin(p["angle"])
                rotated = p["local"] @ np.array([[ca, sa], [-sa, ca]])
            for trial in range(PACKING_ATTEMPTS):
                x, z = rng.uniform(xmin, xmax), rng.uniform(zmin, zmax)
                if p["shape"] == "circle":
                    # Circumscribed polygon conservatively encloses the true circle.
                    geom = Point(x, z).buffer(p["radius"] / math.cos(math.pi / 64), 16)
                else:
                    geom = Polygon(rotated + [x, z])
                if not inside.covers(geom):
                    continue
                ix, iz = math.floor(x / cell), math.floor(z / cell)
                neighbors = (q for dx in (-1, 0, 1) for dz in (-1, 0, 1)
                             for q in buckets.get((ix + dx, iz + dz), []))
                if any(geom.intersects(q) or geom.distance(q) < INITIAL_CLEARANCE for q in neighbors):
                    continue
                p.update(x=x, z=z)
                buckets.setdefault((ix, iz), []).append(geom)
                placed.append(p)
                break
            else:
                break
        if len(placed) == count:
            rng.shuffle(placed)
            area = sum(math.pi * p["size"] ** 2 / 4 if p["shape"] == "circle"
                       else p["size"] * p["c"] for p in placed)
            porosity = 1 - area / boundary.area
            _MIXED_PACKING_INFO = dict(circle_count=nc, rectangle_count=count-nc,
                                      circle_fraction=nc/count, porosity=porosity,
                                      target_porosity=TARGET_POROSITY, seed=INITIAL_SEED)
            print(f"  Particle packing: {nc} circles, {count-nc} rectangles; "
                  f"circle fraction={nc/count:.3f}, initial porosity={porosity:.3f} "
                  f"(target={TARGET_POROSITY:.3f}), seed={INITIAL_SEED}")
            if not INITIAL_PARTICLE_COUNT and porosity > TARGET_POROSITY + 0.03:
                warnings.warn("Random packing could not reach target porosity without overlaps; "
                              "the achieved initial porosity is reported above. Settling may compact it.")
            _MIXED_PARTICLES_CACHE = placed
            return placed
        if INITIAL_PARTICLE_COUNT or count == distribution.class_count:
            raise ValueError(f"Could not place {count} particles without overlaps. "
                             "Reduce count, size or clearance, or increase packing attempts.")
        count = max(distribution.class_count, int(count * .85)//distribution.class_count*distribution.class_count)
    raise ValueError("Unable to generate mixed packing; reduce particle dimensions or clearance.")


def _particle_specs():
    """Per-particle dimensions and geometry; legacy single-shape packing is retained."""
    global _PACKED_SPECS_CACHE
    if GRAVITY_PACKED and _PACKED_SPECS_CACHE is not None:
        return _PACKED_SPECS_CACHE
    if PARTICLE_SHAPE == "mixed" or _size_distribution().mode != "single":
        specs = _generate_mixed_particles()
    else:
        specs = [dict(x=x, z=z, angle=ang, size=size, shape=PARTICLE_SHAPE,
                      c=size if PARTICLE_SHAPE == "circle" else _rect_c,
                      b=_rect_c if _is_quasi2d else _rect_b)
                 for x, z, ang, size in _generate_particle_centres()]
    if GRAVITY_PACKED:
        specs = _gravity_pack_particles(specs)
        if FILL_VOID:
            specs = _fill_packed_voids(specs)
        _PACKED_SPECS_CACHE = specs
    return specs


def _vertical_contact_height(vertices, radius, dx):
    """Top of a convex Minkowski polygon plus a disk at horizontal offset dx.

    Analytic edge/circle intersections keep circular contacts exact rather
    than introducing gaps through polygonal approximations of circles.
    """
    top = -math.inf
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        ex, ez = end - start
        if radius > 0:
            delta = dx - start[0]
            if abs(delta) <= radius:
                top = max(top, start[1] + math.sqrt(max(0., radius**2 - delta**2)))
            if abs(ex) < 1e-14:
                continue
            lo = max(min(start[0], end[0]), dx - radius)
            hi = min(max(start[0], end[0]), dx + radius)
            if lo > hi:
                continue
            slope = ez / ex
            x = np.clip(dx + radius * slope / math.hypot(1., slope), lo, hi)
            top = max(top, start[1] + slope * (x - start[0])
                      + math.sqrt(max(0., radius**2 - (dx - x)**2)))
        elif abs(ex) > 1e-14:
            t = (dx - start[0]) / ex
            if -1e-12 <= t <= 1 + 1e-12:
                top = max(top, start[1] + t * ez)
        elif abs(dx - start[0]) < 1e-12:
            top = max(top, start[1], end[1])
    return top


def _packed_shape_geometry(p):
    """World-oriented local polygon, or circle represented by a point and radius."""
    if p["shape"] == "circle":
        return np.zeros((1, 2)), p["size"] / 2
    if "projection_local" in p:
        return np.asarray(p["projection_local"]), 0.
    if "world_local" in p:
        return np.asarray(p["world_local"], dtype=float), 0.
    a, c, angle = p["size"], p["c"], p["angle"]
    ca, sa = math.cos(angle), math.sin(angle)
    local = np.array([[-a/2, -c/2], [a/2, -c/2], [a/2, c/2], [-a/2, c/2]])
    return local @ np.array([[ca, sa], [-sa, ca]]), 0.


def _vertical_exclusion(local, radius, other_local, other_radius, dx, cache):
    """Open interval of relative Z positions that would overlap a neighboring shape."""
    tolerance = 1e-12
    r = radius + other_radius
    if not (other_local[:, 0].min() - local[:, 0].max() - r + tolerance < dx
            < other_local[:, 0].max() - local[:, 0].min() + r - tolerance):
        return None
    if len(local) == len(other_local) == 1:
        height = math.sqrt(max(0., r*r - dx*dx))
        return -height, height
    key = (local.tobytes(), other_local.tobytes())
    differences = cache.get(key)
    if differences is None:
        from shapely.geometry import MultiPoint
        differences = (other_local[:, None, :] - local[None, :, :]).reshape(-1, 2)
        differences = np.asarray(MultiPoint(differences).convex_hull.exterior.coords[:-1])
        # Bound cache growth when dimensions and orientations vary continuously.
        if len(cache) < 4096:
            cache[key] = differences
    top = _vertical_contact_height(differences, r, dx)
    bottom = -_vertical_contact_height(-differences, r, -dx)
    return bottom, top


def _gravity_pack_particles(specs):
    """Close vertical voids with fixed X/orientation, using exact contact geometry.

    Pairwise vertical exclusion intervals define which particle is above the
    other. The minimum feasible heights form a support network rooted at the
    floor. This is geometric deposition, not a frictional-equilibrium solve:
    particles start at rest, but may immediately slide or roll when released.
    Only invalid initial wall penetrations require horizontal repositioning.
    No motion constraints are installed in the running physics engine.
    """
    global _GRAVITY_PACKING_INFO

    packed = [dict(p) for p in specs]
    if not packed:
        return packed
    wall = _wall_inset()
    tolerance = 1e-10 * max(1., DOMAIN_X, DOMAIN_Z)
    vertices, radii, floors, bounds = [], [], [], []
    for p in packed:
        local, radius = _packed_shape_geometry(p)
        minx, minz = local.min(axis=0) - radius
        maxx, maxz = local.max(axis=0) + radius
        if maxx - minx > DOMAIN_X - 2 * wall:
            raise ValueError("A gravity-packed particle is wider than the domain.")
        p["x"] = float(np.clip(p["x"], wall - minx, DOMAIN_X - wall - maxx))
        vertices.append(local)
        radii.append(radius)
        floors.append((SLOPE_BASE if STREAM_BED_SETUP else wall) - minz)
        bounds.append((minx, maxx, maxz))

    # Each edge (lower, upper, separation) is a vertical nonpenetration constraint.
    edges = []
    contact_cache = {}
    for i in range(len(packed)):
        for j in range(i):
            dx = packed[i]["x"] - packed[j]["x"]
            interval = _vertical_exclusion(vertices[i], radii[i], vertices[j], radii[j], dx, contact_cache)
            if interval is None:
                continue  # Side tangency does not prevent vertical motion.
            bottom, top = interval
            dz = packed[i]["z"] - packed[j]["z"]
            if dz >= top - tolerance:
                edges.append((j, i, top))
            elif dz <= bottom + tolerance:
                edges.append((i, j, -bottom))
            elif (packed[i]["z"], i) >= (packed[j]["z"], j):
                edges.append((j, i, top))  # Repair initially overlapping legacy grids.
            else:
                edges.append((i, j, -bottom))

    heights = np.asarray(floors)
    for _ in range(len(packed)):
        changed = False
        for lower, upper, separation in edges:
            required = heights[lower] + separation
            if required > heights[upper] + tolerance:
                heights[upper] = required
                changed = True
        if not changed:
            break
    else:
        raise ValueError("Gravity packing cannot resolve this interlocked geometry. "
                         "Try another seed, fewer particles or smaller dimensions.")
    if any(z + b[2] > DOMAIN_Z - wall + tolerance for z, b in zip(heights, bounds)):
        raise ValueError("Gravity-packed particles exceed the domain height. "
                         "Increase domain height or reduce particle count/size.")
    for p, z in zip(packed, heights):
        p["z"] = float(z)
    _GRAVITY_PACKING_INFO = dict(particle_count=len(packed),
                                max_vertical_shift=max(abs(p["z"] - q["z"]) for p, q in zip(packed, specs)))
    print(f"  Gravity packed: {len(packed)} particles supported by the floor/contact network; "
          f"maximum vertical shift={_GRAVITY_PACKING_INFO['max_vertical_shift']:.4f} m")
    return packed


def _update_packed_surface(profile, xs, p, local, radius):
    """Update the upper surface at X samples; interior pore space is not solid material."""
    dx = xs - p["x"]
    if radius:
        mask = np.abs(dx) <= radius
        profile[mask] = np.maximum(profile[mask], p["z"] + np.sqrt(np.maximum(0., radius**2 - dx[mask]**2)))
    else:
        for start, end in zip(local, np.roll(local, -1, axis=0)):
            ex, ez = end - start
            if abs(ex) < 1e-14:
                continue
            t = (dx - start[0]) / ex
            mask = (t >= 0.) & (t <= 1.)
            profile[mask] = np.maximum(profile[mask], p["z"] + start[1] + t[mask] * ez)


def _new_fill_particle(rng, packed, circle_count, templates):
    """Use configured dimensions/orientations; DFN additions reuse generated block shapes."""
    if templates and "world_local" in templates[0]:
        return dict(templates[int(rng.integers(len(templates)))])
    kind = PARTICLE_SHAPE
    if kind == "mixed":
        wanted = int(math.floor((len(packed) + 1) * CIRCLE_FRACTION + .5))
        kind = "circle" if circle_count < wanted else "rectangle"
    scale = rng.uniform(1 - MIX_SIZE_SPREAD, 1 + MIX_SIZE_SPREAD) if PARTICLE_SHAPE == "mixed" else 1.
    a = (CIRCLE_DIAMETER if PARTICLE_SHAPE == "mixed" and kind == "circle" else PARTICLE_SIZE) * scale
    distribution = _size_distribution()
    if distribution.mode != "single":
        a = distribution.next_size(rng, packed)
        scale = a/PARTICLE_SIZE
    c = a if kind == "circle" else _rect_c * scale
    b = a if kind == "circle" else (_rect_c if _is_quasi2d else _rect_b) * scale
    angle = 0.
    if kind == "rectangle":
        if PARTICLE_SHAPE == "mixed":
            angle = math.radians(MIX_RECT_ANGLE_DEG + rng.uniform(-MIX_RECT_ANGLE_SPREAD/2, MIX_RECT_ANGLE_SPREAD/2))
        elif RECT_ARRANGEMENT == "mixed":
            angle = rng.uniform(0., math.pi)
    return dict(shape=kind, size=a, c=c, b=b, angle=angle)


def _fill_packed_voids(specs, *, hotstart_batch=None, cycle=0):
    """Incrementally insert and vertically pack particles beneath the target slope.

    Stop at 95% coverage of the accessible target's bulk surface envelope,
    or when the configured sizes no longer fit. This is not a request for
    zero microscopic porosity. Existing particles are retained; each addition
    starts at its lowest nonintersecting supported height. Bounded trials
    prevent impossible particle sizes from hanging initialization.
    """
    global _FILL_VOID_INFO, _MIXED_PACKING_INFO
    packed = [dict(p) for p in specs]
    if not packed:
        raise ValueError("Fill Void needs an initial particle assembly to pack.")
    templates = list(packed)
    wall = _wall_inset()
    floor = SLOPE_BASE if STREAM_BED_SETUP else wall
    left, right = wall, DOMAIN_X - wall if hotstart_batch is not None else min(SLOPE_LENGTH, DOMAIN_X - wall)
    if right <= left:
        raise ValueError("Fill Void has no accessible area inside the slope and domain.")
    xs = left + (np.arange(512) + .5) * (right - left) / 512
    target = np.minimum(np.array([_slope_surface_z(float(x)) for x in xs]), DOMAIN_Z - wall)
    target = np.maximum(target, floor)
    target_area = float(np.sum(target - floor) * (right - left) / len(xs))
    if target_area <= 1e-12:
        raise ValueError("Fill Void has no accessible area below the target slope line.")
    profile = np.full(len(xs), floor)
    seed = DFN_SEED if "world_local" in packed[0] else INITIAL_SEED
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7001]))
    contact_cache = {}
    geometries, buckets = [], {}
    bin_width = max(PARTICLE_SIZE, CIRCLE_DIAMETER if PARTICLE_SHAPE == "mixed" else 0., .01)

    def columns(x, local, radius):
        start = math.floor((x + local[:, 0].min() - radius) / bin_width)
        stop = math.floor((x + local[:, 0].max() + radius) / bin_width)
        return range(start, stop + 1)

    def register(p, local, radius):
        idx = len(geometries)
        geometries.append((local, radius))
        for col in columns(p["x"], local, radius):
            buckets.setdefault(col, []).append(idx)
        _update_packed_surface(profile, xs, p, local, radius)

    for p in packed:
        register(p, *_packed_shape_geometry(p))
    if hotstart_batch is not None:
        # Feed the crest region, allowing the solver to form its own repose
        # angle and spread beyond the original toe. No slope-line clipping.
        feed_end = max(left + bin_width, min(SLOPE_LENGTH * .65, right))
        feed = np.clip((feed_end - xs) / max(feed_end - left, 1e-12), 0., 1.)
        target = (np.minimum(DOMAIN_Z-wall, profile + max(.15*DOMAIN_Z, 3*bin_width)
                             * (xs <= SLOPE_LENGTH)) if STREAM_BED_SETUP else
                  np.minimum(DOMAIN_Z - wall, profile + max(.15 * DOMAIN_Z, 3 * bin_width) * feed))
        target = np.maximum(target, floor)
        target_area = float(np.sum(target-floor) * (right-left) / len(xs))
        rng = np.random.default_rng(np.random.SeedSequence([seed, 8001, cycle]))
    circle_count = sum(p["shape"] == "circle" for p in packed)
    initial_count = len(packed)
    failures, trials = 0, 0
    max_failures, max_additions = 600, hotstart_batch if hotstart_batch is not None else 2000
    gap = np.maximum(target - profile, 0.)
    initial_coverage = 1 - float(gap.sum() / np.sum(target - floor))
    coverage = initial_coverage
    slope_gradient = 0. if STREAM_BED_SETUP else (SLOPE_HEIGHT - SLOPE_BASE) / SLOPE_LENGTH

    while (hotstart_batch is not None or coverage < .95) and failures < max_failures and len(packed) - initial_count < max_additions:
        if gap.sum() <= 1e-12:
            break
        trials += 1
        p = _new_fill_particle(rng, packed, circle_count, templates)
        local, radius = _packed_shape_geometry(p)
        # Favor deficits in the packed surface, while sampling across each bin.
        sample = int(rng.choice(len(xs), p=gap / gap.sum()))
        p["x"] = float(xs[sample] + rng.uniform(-.5, .5) * (right - left) / len(xs))
        if (p["x"] + local[:, 0].min() - radius < wall
                or p["x"] + local[:, 0].max() + radius > DOMAIN_X - wall):
            failures += 1
            continue
        neighbors = sorted({idx for col in columns(p["x"], local, radius) for idx in buckets.get(col, [])})
        intervals = []
        for idx in neighbors:
            q = packed[idx]
            other_local, other_radius = geometries[idx]
            interval = _vertical_exclusion(local, radius, other_local, other_radius, p["x"] - q["x"], contact_cache)
            if interval is not None:
                intervals.append((q["z"] + interval[0], q["z"] + interval[1]))
        z = floor - local[:, 1].min() + radius
        for lo, hi in sorted(intervals):
            if lo + 1e-10 < z < hi - 1e-10:
                z = hi
        p["z"] = float(z)
        if radius:
            inside = (z + radius <= DOMAIN_Z - wall + 1e-10
                      and (hotstart_batch is not None or
                           (p["x"] + radius <= SLOPE_LENGTH + 1e-10
                            and _slope_surface_z(p["x"]) - z >= radius * math.hypot(1., slope_gradient) - 1e-10)))
        else:
            world = local + [p["x"], z]
            inside = (np.all(world[:, 1] <= DOMAIN_Z - wall + 1e-10)
                      and (hotstart_batch is not None or
                           (np.all(world[:, 0] <= SLOPE_LENGTH + 1e-10)
                            and all(vz <= _slope_surface_z(float(vx)) + 1e-10 for vx, vz in world))))
        if not inside:
            failures += 1
            continue
        packed.append(p)
        register(p, local, radius)
        circle_count += p["shape"] == "circle"
        failures = 0
        gap = np.maximum(target - profile, 0.)
        coverage = 1 - float(gap.sum() / np.sum(target - floor))

    distribution = _size_distribution()
    if distribution.mode == "equal":
        incomplete = (len(packed)-initial_count) % distribution.class_count
        if incomplete:
            packed = packed[:-incomplete]
            circle_count = sum(p["shape"] == "circle" for p in packed)
            profile = np.full(len(xs), floor)
            for p in packed:
                _update_packed_surface(profile, xs, p, *_packed_shape_geometry(p))
            gap = np.maximum(target-profile, 0.)
            coverage = 1-float(gap.sum()/np.sum(target-floor))
    added = len(packed) - initial_count
    reason = "target reached" if coverage >= .95 else ("addition limit reached" if added == max_additions else "particle sizes no longer fit")
    _FILL_VOID_INFO = dict(initial_count=initial_count, added_count=added, final_count=len(packed),
                          initial_coverage=initial_coverage, coverage=coverage,
                          target_coverage=.95, target_area=target_area, trials=trials, stop_reason=reason,
                          hotstart_batch=hotstart_batch is not None)
    if _GRAVITY_PACKING_INFO is not None:
        _GRAVITY_PACKING_INFO["particle_count"] = len(packed)
    if PARTICLE_SHAPE == "mixed" and _MIXED_PACKING_INFO is not None:
        area = sum(math.pi * p["size"]**2 / 4 if p["shape"] == "circle" else p["size"] * p["c"] for p in packed)
        slope_area = _target_slope_area() if STREAM_BED_SETUP else .5 * (SLOPE_HEIGHT + SLOPE_BASE) * SLOPE_LENGTH
        _MIXED_PACKING_INFO.update(circle_count=circle_count, rectangle_count=len(packed)-circle_count,
                                   circle_fraction=circle_count/len(packed), porosity=1-area/max(slope_area, 1e-12))
    print(f"  Fill Void: added {added} particles; target slope coverage "
          f"{initial_coverage:.1%} -> {coverage:.1%} ({reason}).")
    if coverage < .95 and hotstart_batch is None and not HOTSTART_PACK:
        warnings.warn(f"Fill Void reached {coverage:.1%} of the target slope envelope: {reason}. "
                      "Smaller particles or a different seed may fill the remaining gaps.")
    return packed


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
    if PARTICLE_SHAPE == "mixed" or _size_distribution().mode != "single":
        return [(p["x"], p["z"], p["angle"], p["size"]) for p in _generate_mixed_particles()]
    if FRACTURE_MODE.upper() == "DFN":
        blocks = _fracture_domain_with_dfn()
        return [
            (b["centroid"][0], b["centroid"][1], b["angle"], b["a"])
            for b in blocks
        ]

    if STREAM_BED_SETUP:
        from StreamBed import flat_particle_centres
        return flat_particle_centres(sys.modules[__name__])

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


_recompute_derived_quantities()

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

AUTO_STABILIZE = False  # Global: whether to automatically raise COHESION_FORCE if initial FS < 1

def _stabilize_initial_slope() -> float:
    """
    Compute the minimum cohesion force (N) for FS ≥ 1 on the initial slope.
    Updates the global COHESION_FORCE if it is insufficient, and returns
    the required value.
    """
    global COHESION_FORCE, AUTO_STABILIZE

    if not SIMULATE_COHESION:
        print("  Cohesion disabled; preserving the configured bond strength.")
        return 0.0
    if MATERIAL_TYPE == "granular":
        COHESION_FORCE = 0.0

    if SLOPE_HEIGHT <= 0 or SLOPE_LENGTH <= 0:
        return COHESION_FORCE

    beta = math.atan2(SLOPE_HEIGHT - SLOPE_BASE, SLOPE_LENGTH)
    phi  = math.atan(_contact_parameters()[0])
    # Bulk unit weight accounting for porosity
    gamma_bulk = PARTICLE_DENSITY * (1.0 - TARGET_POROSITY)   # kg/m³
    unit_weight = gamma_bulk * GRAVITY_MAG                     # N/m³

    # Frictional contribution to FS on an infinite slope
    fs_friction = math.tan(phi) / math.tan(beta) if math.tan(beta) > 1e-12 else 99.0

    if fs_friction >= 1.0:
        # Friction alone is sufficient — zero cohesion needed
        c_prime_min = 0.0
    else:
        # Cohesion required to raise FS to unity
        # Infinite-slope formula: FS = c'/(γ H sinβ cosβ) + tanφ/tanβ
        c_prime_min = (unit_weight * SLOPE_HEIGHT
                       * math.sin(beta) * math.cos(beta)
                       * (1.0 - fs_friction))

    # Contact area: one face of the characteristic particle
    if PARTICLE_SHAPE == "circle":
        contact_area = math.pi * (PARTICLE_SIZE / 2.0) ** 2   # projected end-face
    else:
        contact_area = PARTICLE_SIZE * _rect_c                  # a × c face

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
        if AUTO_STABILIZE:
            print(f"  [AUTO-ADJUST] COHESION_FORCE raised: "
                  f"{COHESION_FORCE:.4f} → {coh_force_min:.4f} N "
                  f"(parameterised value was insufficient to hold slope at FS = 1)")
            COHESION_FORCE = coh_force_min
        else:
            print(f"  [COHESIVE MODE] Preserving user COHESION_FORCE = {COHESION_FORCE:.4f} N "
                  f"(min required for initial FS≥1 is {coh_force_min:.4f} N).")
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

    def __init__(self, particle_specs=None):
        self._spec_override = particle_specs
        import pymunk
        self.pymunk = pymunk
        self.space = pymunk.Space()
        self.space.gravity = (0, -m2px(GRAVITY_MAG))
        self.space.iterations = SOLVER_ITERS
        self.space.damping = 0.95  # slight global damping for stability
        if GRAVITY_PACKED or STREAM_BED_SETUP:
            self.space.collision_slop = m2px(1e-6)

        self.particle_bodies: List[pymunk.Body] = []
        self.particle_shapes = []
        self.particle_kinds = []
        self.particle_b_sizes = []
        self.particle_sizes: List[float] = []
        self.particle_c_sizes: List[float] = []
        self.particle_planform_areas: List[float] = []
        self.particle_draw_local_vertices: List[Optional[np.ndarray]] = []
        self.initial_positions: List[Tuple[float, float]] = []  # in metres
        # Breakable bond springs (DampedSpring); populated by setup_bonds()
        # which is called externally after the settle phase.
        self.bond_springs: List = []
        self._bond_break_distances = {}
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
        thickness = m2px(_wall_inset())
        friction, restitution, _ = _contact_parameters()

        walls = [
            pm.Segment(sb, (0, 0), (W, 0), thickness),           # floor
            pm.Segment(sb, (0, 0), (0, H), thickness),           # left wall
            pm.Segment(sb, (W, 0), (W, H), thickness),           # right wall
            pm.Segment(sb, (0, H), (W, H), thickness),           # ceiling
        ]
        if STREAM_BED_SETUP:
            # Confine sediment while initializing; open the outlet before flow.
            floor_center = m2px(SLOPE_BASE) - thickness
            walls[0] = pm.Segment(sb, (0, floor_center), (W, floor_center), thickness)
        self.walls = walls
        for w in walls:
            w.friction = math.sqrt(friction) if STREAM_BED_SETUP else friction
            w.elasticity = restitution
            w.collision_type = 2  # WALL
        self.space.add(*walls)

    # ── Particles ──────────────────────────────────────────────────
    def _build_particles(self):
        pm = self.pymunk
        friction, restitution, _ = _contact_parameters()
        if self._spec_override is None and FRACTURE_MODE.upper() == "DFN" and PARTICLE_SHAPE == "rectangle":
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
                shape.friction = math.sqrt(friction) if STREAM_BED_SETUP else friction
                shape.elasticity = restitution
                shape.collision_type = 1

                self.space.add(body, shape)
                self.particle_bodies.append(body)
                self.particle_shapes.append(shape)
                self.particle_kinds.append("rectangle")
                self.particle_b_sizes.append(_rect_b)
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

        for p in (self._spec_override if self._spec_override is not None else _particle_specs()):
            cx, cz, ang, size = p["x"], p["z"], p["angle"], p["size"]
            kind, c_size = p["shape"], p["c"]
            self.particle_kinds.append(kind)
            self.particle_b_sizes.append(p["b"])
            body = pm.Body()
            body.position = (m2px(cx), m2px(cz))

            if kind == "circle":
                shape = pm.Circle(body, m2px(size / 2))
                shape.density = PARTICLE_DENSITY / (PX_PER_M ** 2)
            elif "world_local" in p:
                shape = pm.Poly(body, (np.asarray(p["world_local"]) * PX_PER_M).tolist())
                shape.density = PARTICLE_DENSITY / (PX_PER_M ** 2)
            else:
                shape = pm.Poly.create_box(body, size=(m2px(size), m2px(c_size)))
                shape.density = PARTICLE_DENSITY / (PX_PER_M ** 2)
                body.angle = ang   # apply in-plane orientation

            shape.friction = math.sqrt(friction) if STREAM_BED_SETUP else friction
            shape.elasticity = restitution
            shape.collision_type = 1  # PARTICLE

            self.space.add(body, shape)
            self.particle_bodies.append(body)
            self.particle_shapes.append(shape)
            if kind == "circle":
                self.particle_sizes.append(size)
                self.particle_c_sizes.append(size)
                self.particle_planform_areas.append(math.pi * (size / 2.0) ** 2)
                self.particle_draw_local_vertices.append(None)
            else:
                self.particle_sizes.append(size)
                self.particle_c_sizes.append(c_size)
                self.particle_planform_areas.append(p.get("area", size * c_size))
                rect_local = np.array([
                    [-size * 0.5, -c_size * 0.5],
                    [ size * 0.5, -c_size * 0.5],
                    [ size * 0.5,  c_size * 0.5],
                    [-size * 0.5,  c_size * 0.5],
                ], dtype=float)
                self.particle_draw_local_vertices.append(np.asarray(p.get("world_local", rect_local)))
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
        cohesion = _contact_parameters()[2]
        if cohesion <= 0:
            return
        pm = self.pymunk
        pos = self.get_positions()  # metres after settling
        n = len(pos)
        bond_search = PARTICLE_SIZE * 1.05  # only truly touching pairs bond
        break_ext = max((BOND_BREAK_MULT - 1.0) * PARTICLE_SIZE, 1e-4)  # m
        # Spring stiffness: F rises linearly from 0 at contact to
        # COHESION_FORCE at break extension (see unit analysis: k_pymunk = k_SI)
        k = cohesion / break_ext  # N/m
        count = 0
        variable_sizes = PARTICLE_SHAPE == "mixed" or _size_distribution().mode != "single"
        for i in range(n):
            for j in range(i + 1, n):
                if variable_sizes:
                    pair_size = (self.particle_sizes[i] + self.particle_sizes[j]) / 2
                    bond_search = pair_size * 1.05
                    break_ext = max((BOND_BREAK_MULT - 1.0) * pair_size, 1e-4)
                    k = cohesion / break_ext
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
                    if variable_sizes:
                        self._bond_break_distances[spring] = (m2px(pair_size) * BOND_BREAK_MULT) ** 2
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
             (s.b.position.y - s.a.position.y) ** 2) > self._bond_break_distances.get(s, bd_sq)
        ]
        for s in to_remove:
            self.bond_springs.remove(s)
            self._bond_break_distances.pop(s, None)
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
    def step(self, dt=None):
        if ADAPTIVE_DT:
            from TimeStepping import computational_dt
            remaining = DT if dt is None else dt
            loads = [(tuple(body.force), body.torque) for body in self.particle_bodies]
            while remaining > 1e-14:
                substep = computational_dt(sys.modules[__name__], self, remaining)
                for body, (force, torque) in zip(self.particle_bodies, loads):
                    body.force, body.torque = force, torque
                self._break_springs()
                self.space.step(substep)
                remaining = max(0., remaining-substep)
        else:
            self._break_springs()  # remove broken cohesive bonds before stepping
            self.space.step(DT if dt is None else dt)

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

    def __init__(self, particle_specs=None):
        self._spec_override = particle_specs
        import pychrono as chrono
        self.chrono = chrono

        # ── Shrink Bullet collision envelope BEFORE creating any collision shape.
        # Default envelope (0.01 m) is close to particle radius (0.03 m), causing
        # false contacts and tunneling through thin walls.
        collision_size = min(PARTICLE_SIZE, CIRCLE_DIAMETER, _rect_c)
        if self._spec_override:
            collision_size = min(min(p["size"], p["c"]) for p in self._spec_override)
        elif _size_distribution().mode != "single":
            self._spec_override = _particle_specs()
            collision_size = min(min(p["size"], p["c"]) for p in self._spec_override)
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(collision_size*CHRONO_ENVELOPE_FACTOR)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(min(.0005, collision_size*.005))

        self.sys = chrono.ChSystemNSC()
        self.sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY_MAG))
        self.sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

        # Solver settings — increase iterations for tight quasi-2D packing.
        self.sys.SetSolverType(getattr(chrono.ChSolver, "Type_"+CHRONO_SOLVER))
        self.sys.SetMaxPenetrationRecoverySpeed(CHRONO_RECOVERY_SPEED)
        self.sys.SetMinBounceSpeed(CHRONO_BOUNCE_SPEED)
        _solver = self.sys.GetSolver()
        _it_solver = _solver.AsIterative()
        if _it_solver is not None:
            _it_solver.SetMaxIterations(SOLVER_ITERS)
            _it_solver.SetTolerance(CHRONO_TOLERANCE)
        self.sys.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)

        self.particle_bodies = []
        self.particle_kinds = []
        self.particle_b_sizes = []
        self.particle_sizes = []
        self.particle_c_sizes = []
        self.particle_planform_areas = []
        self.particle_draw_local_vertices = []
        self.initial_positions = []

        # Determine Y extent for quasi-2D
        self._y_extent = DOMAIN_Y
        if QUASI_2D:
            self._y_extent = PARTICLE_SIZE * 1.5  # just wide enough for 1 particle
            if PARTICLE_SHAPE == "mixed":
                self._y_extent = max(CIRCLE_DIAMETER, _rect_c) * (1 + MIX_SIZE_SPREAD) + 2 * INITIAL_CLEARANCE
            if _size_distribution().mode != "single":
                if self._spec_override is None:
                    self._spec_override = _particle_specs()
                self._y_extent = max(p["size"] if p["shape"] == "circle" else p["b"]
                                     for p in self._spec_override) + 2*INITIAL_CLEARANCE
            if STREAM_BED_SETUP:
                self._y_extent = DOMAIN_Y
            print(f"  Quasi-2D mode: Y extent = {self._y_extent:.4f} m")

        self._build_walls()
        self._build_particles()
        if STREAM_BED_SETUP:
            from StreamBed import configure_chrono_stream
            configure_chrono_stream(sys.modules[__name__], self)

    def _make_material(self, is_wall=False):
        chrono = self.chrono
        mat = chrono.ChContactMaterialNSC()
        friction, restitution, cohesion = _contact_parameters()
        mat.SetFriction(friction)
        mat.SetRestitution(restitution)
        mat.SetCohesion(cohesion if not is_wall else 0.0)
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
        if STREAM_BED_SETUP:
            walls[0] = ((W/2, 0, SLOPE_BASE-t/2), (W+2*t, Y+2*t, t))
        if QUASI_2D:
            # Front Y-wall — inner face at y = -Y/2
            walls.append(((W / 2, -Y / 2 - t / 2, H / 2),
                          (W + 2 * t, t, H + 2 * t)))
            # Back Y-wall  — inner face at y = +Y/2
            walls.append(((W / 2,  Y / 2 + t / 2, H / 2),
                          (W + 2 * t, t, H + 2 * t)))

        self.wall_bodies = []
        for wall_index, (pos, sz) in enumerate(walls):
            wall_mat = mat
            if STREAM_BED_SETUP and wall_index >= 4:
                # A backup to the planar constraints, without introducing
                # artificial cross-section wall friction into river hydraulics.
                wall_mat = self._make_material(is_wall=True)
                wall_mat.SetFriction(0.)
            body = chrono.ChBodyEasyBox(sz[0], sz[1], sz[2], 1000, True, True, wall_mat)
            body.SetPos(chrono.ChVector3d(pos[0], pos[1], pos[2]))
            body.SetFixed(True)
            self.sys.Add(body)
            self.wall_bodies.append(body)

    def _build_particles(self):
        chrono = self.chrono
        mat = self._make_material(is_wall=False)
        y_pos = 0.0  # centre of Y-extent

        # For quasi-2D the Y thickness is the c-axis so the block fits
        # exactly in the Y-confined channel.  In full 3D use the b-axis.
        y_size = _rect_c if (_is_quasi2d and PARTICLE_SHAPE == "rectangle") else _rect_b

        if self._spec_override is None and FRACTURE_MODE.upper() == "DFN" and PARTICLE_SHAPE == "rectangle":
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
                            cloud.push_back(chrono.ChVector3d(vx - cx, -y_half, vz - cz))
                            cloud.push_back(chrono.ChVector3d(vx - cx,  y_half, vz - cz))
                        body = chrono.ChBodyEasyConvexHull(cloud, PARTICLE_DENSITY, True, True, mat)
                    except Exception:
                        body = None

                # Robust fallback: oriented box from minimum rotated rectangle.
                if body is None:
                    if GRAVITY_PACKED or STREAM_BED_SETUP:
                        raise ValueError("Gravity Packed DFN blocks require Chrono convex-hull support. "
                                         "Use grid packing or the pymunk engine with this Chrono version.")
                    body = chrono.ChBodyEasyBox(
                        a_len, y_size, c_len,
                        PARTICLE_DENSITY, True, True, mat
                    )
                    q = chrono.ChQuaterniond()
                    q.SetFromAngleAxis(-ang, chrono.ChVector3d(0, 1, 0))
                    body.SetRot(q)

                body.SetPos(chrono.ChVector3d(cx, y_pos, cz))
                try:
                    body.SetLinDamping(0.15)
                    body.SetAngDamping(0.05)
                except AttributeError:
                    pass

                self.sys.Add(body)
                self.particle_bodies.append(body)
                self.particle_kinds.append("rectangle")
                self.particle_b_sizes.append(y_size)
                self.particle_sizes.append(a_len)
                self.particle_c_sizes.append(c_len)
                self.particle_planform_areas.append(float(b["area"]))
                if GRAVITY_PACKED or STREAM_BED_SETUP:
                    self.particle_draw_local_vertices.append(np.asarray(verts) - [cx, cz])
                else:
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

        for p in (self._spec_override if self._spec_override is not None else _particle_specs()):
            cx, cz, ang, size = p["x"], p["z"], p["angle"], p["size"]
            kind, c_size = p["shape"], p["c"]
            self.particle_kinds.append(kind)
            self.particle_b_sizes.append(p["b"])
            if kind == "circle":
                r = size / 2.0
                body = chrono.ChBodyEasySphere(r, PARTICLE_DENSITY, True, True, mat)
                # Circles/spheres have no orientation
            elif "world_local" in p:
                if not (hasattr(chrono, "ChBodyEasyConvexHull") and hasattr(chrono, "vector_ChVector3d")):
                    raise ValueError("Hotstart Pack DFN blocks require Chrono convex-hull support.")
                cloud = chrono.vector_ChVector3d()
                vertices3d = p.get("hull_vertices_3d")
                if vertices3d is None:
                    vertices3d = [(vx, y, vz) for vx, vz in p["world_local"] for y in (-p["b"]/2, p["b"]/2)]
                for vx, vy, vz in vertices3d:
                    cloud.push_back(chrono.ChVector3d(vx, vy, vz))
                body = chrono.ChBodyEasyConvexHull(cloud, PARTICLE_DENSITY, True, True, mat)
            else:
                # Box dims: a-axis (X), b/c-axis (Y), c-axis (Z)
                body = chrono.ChBodyEasyBox(
                    size, p["b"], c_size,
                    PARTICLE_DENSITY, True, True, mat
                )
                if (self._spec_override is not None or PARTICLE_SHAPE == "mixed" or RECT_ARRANGEMENT == "mixed") and abs(ang) > 1e-9:
                    # Rotate around Y-axis by ang (in-plane XZ rotation)
                    # ChQuaterniond(axis, angle) uses angle in radians
                    q = chrono.ChQuaterniond()
                    q.SetFromAngleAxis(-ang, chrono.ChVector3d(0, 1, 0))
                    body.SetRot(q)
                elif not GRAVITY_PACKED and PARTICLE_SHAPE != "mixed" and _is_3d_full and RECT_ARRANGEMENT == "mixed":
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

            if "quaternion" in p:
                body.SetRot(chrono.ChQuaterniond(*p["quaternion"]))
            body.SetPos(chrono.ChVector3d(cx, p.get("y", y_pos), cz))
            try:
                body.SetLinDamping(0.15)
                body.SetAngDamping(0.05)
            except AttributeError:
                pass
            self.sys.Add(body)
            self.particle_bodies.append(body)
            self.particle_sizes.append(size)
            self.particle_c_sizes.append(size if kind == "circle" else c_size)
            if kind == "circle":
                self.particle_planform_areas.append(math.pi * (size / 2.0) ** 2)
                self.particle_draw_local_vertices.append(None)
            else:
                self.particle_planform_areas.append(p.get("area", size * c_size))
                self.particle_draw_local_vertices.append(np.asarray(p.get("world_local", [
                    [-size * 0.5, -c_size * 0.5],
                    [ size * 0.5, -c_size * 0.5],
                    [ size * 0.5,  c_size * 0.5],
                    [-size * 0.5,  c_size * 0.5],
                ]), dtype=float))
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

    def step(self, dt=None):
        from TimeStepping import computational_dt
        remaining = DT if dt is None else dt
        while remaining > 1e-14:
            substep = computational_dt(sys.modules[__name__], self, remaining,
                                      active=getattr(self, "stream_active", None))
            self.sys.DoStepDynamics(substep)
            remaining = max(0., remaining-substep)

    def get_positions(self) -> np.ndarray:
        chrono = self.chrono
        pos = np.zeros((len(self.particle_bodies), 2))
        for i, b in enumerate(self.particle_bodies):
            p = b.GetPos()
            pos[i, 0] = p.x
            pos[i, 1] = p.z
        return pos

    def get_positions3d(self):
        return np.array([(body.GetPos().x, body.GetPos().y, body.GetPos().z) for body in self.particle_bodies])

    def check_stream_boundaries(self, active):
        """Never silently render numerical floor/plane escape as sediment export."""
        ids = np.flatnonzero(active)
        if not len(ids):
            return
        positions = self.get_positions3d()
        angles = self.get_angles()
        for i in ids:
            # A polydisperse bed must not test an 8 mm grain's contact
            # tolerance against the smallest tail grain's sub-mm diameter.
            grain_thickness = min(self.particle_sizes[i], self.particle_c_sizes[i])
            tol = max(1e-7, grain_thickness*max(.05, 5*CHRONO_ENVELOPE_FACTOR))
            local = self.particle_draw_local_vertices[i]
            bottom = positions[i, 2]-self.particle_sizes[i]/2
            if local is not None:
                bottom = positions[i, 2]+float((np.asarray(local)[:, 0]*math.sin(angles[i])
                                                +np.asarray(local)[:, 1]*math.cos(angles[i])).min())
            half_width = self.particle_sizes[i]/2 if local is None else self.particle_b_sizes[i]/2
            if not np.isfinite(positions[i]).all() or abs(positions[i, 1])+half_width > DOMAIN_Y/2+tol:
                raise ValueError(f"Chrono lateral constraint failure for grain {i}: Y={positions[i, 1]:.6g} m. "
                                 "Increase solver iterations or reduce maximum DT; this is not a valid export.")
            # Only test grains still overlapping the physical substrate in X.
            if -tol <= positions[i, 0] <= DOMAIN_X+tol and bottom < SLOPE_BASE-tol:
                raise ValueError(f"Chrono floor penetration for grain {i}: bottom Z={bottom:.6g} m. "
                                 "Reduce maximum DT or increase solver iterations; the run was stopped before recording an escaped grain.")

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
                ang[i] = math.atan2(2.0 * (x * z - w * y),
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

class HotstartCancelled(RuntimeError):
    """Raised when the user stops initial-condition preparation."""


def _snapshot_particles(backend):
    """Current solver geometry in metres, preserving each particle's dimensions."""
    positions, angles = backend.get_positions(), backend.get_angles()
    specs = []
    for i, (x, z) in enumerate(positions):
        p = dict(x=float(x), z=float(z), angle=float(angles[i]),
                 shape=backend.particle_kinds[i], size=backend.particle_sizes[i],
                 c=backend.particle_c_sizes[i], b=backend.particle_b_sizes[i],
                 area=backend.particle_planform_areas[i])
        if FRACTURE_MODE.upper() == "DFN":
            local = backend.particle_draw_local_vertices[i]
            ca, sa = math.cos(angles[i]), math.sin(angles[i])
            p["world_local"] = local @ np.array([[ca, sa], [-sa, ca]])
            p["dfn_source"] = i
        if isinstance(backend, PychronoBackend):
            p["y"] = float(backend.particle_bodies[i].GetPos().y)
            if p["shape"] != "circle":
                from shapely.geometry import MultiPoint
                q = backend.particle_bodies[i].GetRot()
                w, qx, qy, qz = (float(q.e0), float(q.e1), float(q.e2), float(q.e3))
                p["quaternion"] = (w, qx, qy, qz)
                rotation = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-w*qz), 2*(qx*qz+w*qy)],
                                     [2*(qx*qy+w*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-w*qx)],
                                     [2*(qx*qz-w*qy), 2*(qy*qz+w*qx), 1-2*(qx*qx+qy*qy)]])
                if FRACTURE_MODE.upper() == "DFN":
                    override = backend._spec_override[i] if backend._spec_override is not None else {}
                    cloud = override.get("hull_vertices_3d")
                    if cloud is None:
                        cloud = [(vx, y, vz) for vx, vz in backend.particle_draw_local_vertices[i]
                                 for y in (-p["b"]/2, p["b"]/2)]
                    p["hull_vertices_3d"] = cloud
                else:
                    cloud = [(x, y, z) for x in (-p["size"]/2, p["size"]/2)
                             for y in (-p["b"]/2, p["b"]/2) for z in (-p["c"]/2, p["c"]/2)]
                projected = (np.asarray(cloud) @ rotation.T)[:, [0, 2]]
                p["projection_local"] = np.asarray(MultiPoint(projected).convex_hull.exterior.coords[:-1])
                if FRACTURE_MODE.upper() == "DFN":
                    p["world_local"] = p["projection_local"]
        specs.append(p)
    return specs


def _target_slope_area():
    """Nominal area under the dashed line, clipped only by the domain."""
    from shapely.geometry import Polygon, box
    if STREAM_BED_SETUP:
        return (SLOPE_HEIGHT - SLOPE_BASE) * SLOPE_LENGTH
    return float(Polygon([(0., 0.), (SLOPE_LENGTH, 0.),
                          (SLOPE_LENGTH, SLOPE_BASE), (0., SLOPE_HEIGHT)])
                 .intersection(box(0., 0., DOMAIN_X, DOMAIN_Z)).area)


def _snapshot_dfn_blocks(specs):
    return [dict(centroid=(p["x"], p["z"]), verts=(p["world_local"] + [p["x"], p["z"]]).tolist(),
                 angle=p["angle"], a=p["size"], c=p["c"], area=p["area"]) for p in specs]


def _settled_bulk_area(specs):
    """Area under the pile's free surface across the FULL domain, including its toe.

    Measured only after motion has subsided; it includes interparticle pores
    and is not clipped to the original target line or slope length.
    """
    wall = _wall_inset()
    floor = SLOPE_BASE if STREAM_BED_SETUP else wall
    width = DOMAIN_X - 2 * wall
    xs = wall + (np.arange(1024) + .5) * width / 1024
    profile = np.full(len(xs), floor)
    for p in specs:
        if not math.isfinite(p["x"]) or not math.isfinite(p["z"]):
            raise ValueError("Hotstart solver produced non-finite particle positions. Reduce the timestep.")
        _update_packed_surface(profile, xs, p, *_packed_shape_geometry(p))
    return float(np.sum(np.maximum(profile - floor, 0.)) * width / len(xs))


def _hotstart_speed(backend):
    """Maximum translational or rotational surface speed, in m/s."""
    velocities = backend.get_velocities()
    speed = float(np.max(np.linalg.norm(velocities, axis=1))) if len(velocities) else 0.
    for body, size, c_size in zip(backend.particle_bodies, backend.particle_sizes, backend.particle_c_sizes):
        if isinstance(backend, PymunkBackend):
            spin = abs(body.angular_velocity)
        else:
            omega = body.GetAngVelLocal()
            spin = math.sqrt(omega.x**2 + omega.y**2 + omega.z**2)
            v = body.GetPosDt()
            speed = max(speed, math.sqrt(v.x**2 + v.y**2 + v.z**2))
        speed = max(speed, spin * math.hypot(size, c_size) / 2)
    return speed


def _hotstart_pack(backend, factory, progress_callback=None):
    """Fill/pack/solve feedback loop; the returned backend retains final contacts and motion.

    Only a solver-settled area within tolerance counts as convergence. Filling
    can extend beyond the dashed line: its integral, not its shape, is the goal.
    Overshooting a discrete-particle target rolls back particles and settles
    again. Cycle limits are reported, never relabeled as successful packing.
    """
    global _HOTSTART_INFO, _MIXED_PACKING_INFO, _GRAVITY_PACKING_INFO
    target = _target_slope_area()
    wall = _wall_inset()
    capacity = (DOMAIN_X - 2 * wall) * (DOMAIN_Z - 2 * wall)
    if target <= 0 or target > capacity:
        raise ValueError("Hotstart target area does not fit inside the domain walls. Enlarge the domain.")
    history, elapsed = [], 0.
    initial_count = backend.num_particles()
    dt = min(DT, .002)
    rest_window = .25
    final_area, converged, settled, reason = 0., False, False, "cycle limit reached"

    def progress(cycle, phase, **extra):
        if progress_callback is not None:
            state = dict(cycle=cycle, max_cycles=HOTSTART_MAX_CYCLES, phase=phase,
                         particle_count=backend.num_particles(), solver_time=elapsed,
                         target_area=target, **extra)
            if progress_callback(state) is False:
                raise HotstartCancelled("Hotstart packing cancelled.")

    for cycle in range(1, HOTSTART_MAX_CYCLES + 1):
        backend.set_rotation_angle(0.)
        from collections import deque
        window_steps = max(2, int(math.ceil(rest_window / dt)))
        recent_speeds, recent_positions = deque(maxlen=window_steps), deque(maxlen=window_steps)
        settled = False
        for step in range(int(math.ceil(HOTSTART_MAX_SETTLE_TIME / dt))):
            backend.step(dt)
            elapsed += dt
            speed = _hotstart_speed(backend)
            recent_speeds.append(speed)
            recent_positions.append(backend.get_positions())
            if step % max(1, int(.2 / dt)) == 0:
                progress(cycle, "settling", max_speed=speed)
            window_speed = math.sqrt(sum(v*v for v in recent_speeds) / len(recent_speeds))
            drift = float(np.max(np.linalg.norm(recent_positions[-1] - recent_positions[0], axis=1)))
            if ((step + 1) * dt >= HOTSTART_MIN_SETTLE_TIME and len(recent_speeds) == window_steps
                    and speed <= HOTSTART_SPEED_TOLERANCE and window_speed <= HOTSTART_SPEED_TOLERANCE
                    and drift <= HOTSTART_SPEED_TOLERANCE * rest_window):
                settled = True
                break
        specs = _snapshot_particles(backend)
        final_area = _settled_bulk_area(specs)
        error = (final_area - target) / target
        history.append(dict(cycle=cycle, area=final_area, relative_error=error,
                            settled=settled, max_speed=speed, particle_count=len(specs)))
        print(f"  Hotstart {cycle}/{HOTSTART_MAX_CYCLES}: area={final_area:.4f}/{target:.4f} m2 "
              f"({error:+.1%}), speed={speed:.5f} m/s, settled={settled}")
        progress(cycle, "measured", area=final_area, relative_error=error, settled=settled, max_speed=speed)
        if settled and abs(error) <= HOTSTART_AREA_TOLERANCE:
            converged, reason = True, "area and motion tolerances reached"
            break
        if cycle == HOTSTART_MAX_CYCLES:
            reason = "cycle limit reached before rest" if not settled else "cycle limit reached before area match"
            break
        if not settled:
            continue  # Never fill based on a moving or airborne surface.

        solid_area = sum(p["area"] for p in specs)
        packing_fraction = min(1., solid_area / max(final_area, 1e-12))
        mean_area = solid_area / max(len(specs), 1)
        count_change = max(1, int(round(abs(target - final_area) * packing_fraction / max(mean_area, 1e-12))))
        count_change = min(count_change, max(8, len(specs) // 4))
        distribution = _size_distribution()
        if distribution.mode == "equal":
            count_change = max(distribution.class_count,
                               count_change//distribution.class_count*distribution.class_count)
        if error > 0:
            # Remove the latest additions first, then let the remaining pile resettle.
            count_change = min(count_change, len(specs) - distribution.class_count)
            if count_change <= 0:
                reason = "particle resolution prevents area match"
                break
            if distribution.mode == "equal":
                count_change = count_change//distribution.class_count*distribution.class_count
                to_remove = {value: count_change//distribution.class_count for value in distribution.values}
                retained = []
                for p in reversed(specs):
                    if to_remove[p["size"]] > 0:
                        to_remove[p["size"]] -= 1
                    else:
                        retained.append(p)
                specs = list(reversed(retained))
            elif PARTICLE_SHAPE == "mixed":
                nc = sum(p["shape"] == "circle" for p in specs)
                remove_circle = nc - int(math.floor((len(specs)-count_change) * CIRCLE_FRACTION + .5))
                to_remove = {"circle": remove_circle, "rectangle": count_change-remove_circle}
                retained = []
                for p in reversed(specs):
                    if to_remove[p["shape"]] > 0:
                        to_remove[p["shape"]] -= 1
                    else:
                        retained.append(p)
                specs = list(reversed(retained))
            else:
                specs = specs[:-count_change]
            progress(cycle, "trimming", removed=count_change, area=final_area)
        else:
            progress(cycle, "packing and filling", area=final_area)
            specs = _gravity_pack_particles(specs)
            before = len(specs)
            specs = _fill_packed_voids(specs, hotstart_batch=count_change, cycle=cycle)
            if len(specs) == before:
                reason = "no additional particles fit inside the domain"
                break
        backend = factory(particle_specs=specs)

    # Preserve the final solver state; resetting velocities would disguise residual motion.
    backend.initial_positions = [tuple(p) for p in backend.get_positions()]
    _HOTSTART_INFO = dict(converged=converged, settled=settled, stop_reason=reason,
                          target_area=target, area=final_area,
                          relative_error=(final_area-target)/target,
                          tolerance=HOTSTART_AREA_TOLERANCE, cycles=len(history),
                          initial_count=initial_count, final_count=backend.num_particles(),
                          solver_time=elapsed, history=history)
    if _GRAVITY_PACKING_INFO is not None:
        _GRAVITY_PACKING_INFO["particle_count"] = backend.num_particles()
    if _MIXED_PACKING_INFO is not None:
        nc = backend.particle_kinds.count("circle")
        n = backend.num_particles()
        _MIXED_PACKING_INFO.update(circle_count=nc, rectangle_count=n-nc,
                                   circle_fraction=nc/max(n, 1),
                                   porosity=1-sum(backend.particle_planform_areas)/max(final_area, 1e-12))
    if not converged:
        warnings.warn(f"Hotstart Pack did not converge: {reason}; settled area "
                      f"{final_area:.4f} / {target:.4f} m2, error={(final_area-target)/target:+.1%}.")
    return backend


def _initialize_backend(initialization_callback=None):
    """Use the selected engine for both initialization and the actual run."""
    global ENGINE
    requested_engine, fallback_reason = ENGINE, None
    factory = PychronoBackend if ENGINE == "pychrono" else PymunkBackend
    try:
        backend = factory()
    except Exception as exc:
        if ENGINE != "pychrono":
            raise
        print(f"  [WARN] pychrono initialization failed ({exc}); using pymunk for initialization and simulation.")
        fallback_reason = str(exc)
        ENGINE = "pymunk"
        _recompute_derived_quantities()
        factory = PymunkBackend
        backend = factory()
    backend = _hotstart_pack(backend, factory, initialization_callback) if HOTSTART_PACK else backend
    backend.requested_engine = requested_engine
    backend.engine_fallback_reason = fallback_reason
    return backend


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
                                      particle_c_sizes: Optional[np.ndarray] = None,
                                      particle_kinds=None, particle_b_sizes=None) -> float:
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

    kinds = np.asarray(particle_kinds if particle_kinds is not None else [shape] * n)
    circles = kinds == "circle"
    a = particle_sizes
    b = np.asarray(particle_b_sizes) if particle_b_sizes is not None else np.full(n, _rect_b)
    c = particle_c_sizes if particle_c_sizes is not None else np.full(n, _rect_c)
    total_areas = np.where(circles, math.pi * a * a, 2.0 * (a * b + b * c + a * c))
    face_areas = np.where(circles, math.pi * (a / 2) ** 2,
                          np.minimum(np.minimum(a * b, b * c), a * c))

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

            if circles[i] and circles[j]:
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
                min_face_i = face_areas[i]
                min_face_j = face_areas[j]
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
    friction, _, cohesion = _contact_parameters()
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
        resist = normal_comp * friction + cohesion
        resisting += resist * arm

    if driving < 1e-12:
        return 99.0
    return resisting / driving


# =====================================================================
# 7. MAIN SIMULATION LOOP
# =====================================================================

def run_simulation(progress_callback=None, save_video=True, return_data=False, initialization_callback=None):
    global PARTICLE_SHAPE

    if STREAM_BED_SETUP and not _STREAM_SCALE_ACTIVE:
        with _scaled_stream_geometry():
            return run_simulation(progress_callback, save_video, return_data, initialization_callback)

    _recompute_derived_quantities()

    if FRACTURE_MODE.upper() == "DFN" and PARTICLE_SHAPE != "rectangle":
        print("  [WARN] FRACTURE_MODE='DFN' requires polygonal blocks; forcing PARTICLE_SHAPE='rectangle'.")
        PARTICLE_SHAPE = "rectangle"

    if STREAM_BED_SETUP:
        from StreamBed import run_stream_simulation
        return run_stream_simulation(sys.modules[__name__], progress_callback, save_video,
                                     return_data, initialization_callback)

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
        if PARTICLE_SHAPE == "mixed":
            _shape_info = (f"circles={CIRCLE_FRACTION:.1%} by count, circle d={CIRCLE_DIAMETER} m, "
                           f"rectangle a={PARTICLE_SIZE} c={_rect_c:.3f} m, seed={INITIAL_SEED}")
    print(f"Particles    : shape={PARTICLE_SHAPE}, {_shape_info}, "
          f"porosity={TARGET_POROSITY}")
    print(f"Gravity packed: {GRAVITY_PACKED}")
    print(f"Fill Void    : {FILL_VOID and GRAVITY_PACKED}")
    print(f"Hotstart Pack: {HOTSTART_PACK}")
    print(f"Rotation     : {TOTAL_ROTATION} deg @ {ROTATION_RATE} deg/s "
          f"(settle={SETTLE_TIME} s)")
    print(f"Sim          : T={SIM_DURATION:.1f} s, dt={DT} s, "
          f"{N_STEPS} steps")
    print(f"Output       : {VIDEO_FILE}")
    print("-" * 65)

    # ── Run the slope stability pre-check; may raise COHESION_FORCE ──
    _stabilize_initial_slope()

    friction, restitution, cohesion = _contact_parameters()
    print(f"Material     : type={MATERIAL_TYPE}, rho={PARTICLE_DENSITY} kg/m3, "
          f"mu={friction}, coh={cohesion:.4f} N, e={restitution}")

    # ── Initialise backend ────────────────────────────────────────
    backend = _initialize_backend(initialization_callback)
    hotstart_initial_specs = _snapshot_particles(backend) if HOTSTART_PACK else None

    n_particles = backend.num_particles()
    init_pos = np.array(backend.initial_positions) if len(backend.initial_positions) > 0 else np.zeros((0, 2))
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
    from Recording import FrameSchedule
    schedule = FrameSchedule(sys.modules[__name__], SIM_DURATION)
    last_progress = wall_t0
    cancelled = False

    for step in range(N_STEPS + 1):
        t = min(step * DT, SIM_DURATION)

        # Determine rotation angle
        if t < SETTLE_TIME:
            current_angle = 0.0
        elif t < SETTLE_TIME + _rotation_duration:
            current_angle = ROTATION_RATE * (t - SETTLE_TIME)
        else:
            current_angle = TOTAL_ROTATION

        backend.set_rotation_angle(current_angle)
        if step > 0:
            backend.step(t-(step-1)*DT)

        # Capture settled positions at end of settling phase, then form bonds
        if settled_positions is None and t >= SETTLE_TIME:
            settled_positions = backend.get_positions().copy()
            if not GRAVITY_PACKED:
                init_pos = settled_positions  # packed runs measure from their supported t=0 geometry
            if hasattr(backend, 'setup_bonds'):
                backend.setup_bonds()

        # Save frame
        should_save = schedule.due(step, t, min((step+1)*DT, SIM_DURATION) if step < N_STEPS else None)
        now = time.time()
        if progress_callback and (should_save or now-last_progress >= .15 or step == N_STEPS):
            last_progress = now
            progress_pos = backend.get_positions()
            progress_disp = compute_displacement(progress_pos, init_pos)
            progress_pct = compute_pct_moved(progress_disp, sizes_static)
            if progress_callback(step, N_STEPS, t, current_angle, progress_pct, float("nan")) is False:
                cancelled = True
                should_save |= not frames_time or schedule.contains(t)
        if should_save:
            pos = backend.get_positions()
            if hasattr(backend, "particle_sizes") and len(getattr(backend, "particle_sizes", [])) == len(pos):
                sizes = np.array(backend.particle_sizes, dtype=float)
            else:
                sizes = np.full(len(pos), PARTICLE_SIZE, dtype=float)

            if hasattr(backend, "particle_c_sizes") and len(getattr(backend, "particle_c_sizes", [])) == len(pos):
                c_sizes = np.array(backend.particle_c_sizes, dtype=float)
            else:
                c_sizes = np.full(len(pos), _rect_c if PARTICLE_SHAPE == "rectangle" else PARTICLE_SIZE, dtype=float)

            disp = compute_displacement(pos, init_pos) if len(pos) > 0 else np.zeros(0)
            pct = compute_pct_moved(disp, sizes) if len(disp) > 0 else 0.0
            coord, normals = backend.get_contact_info()
            vel = backend.get_velocities()
            ke = backend.get_kinetic_energy()
            bd = compute_bulk_density(pos, planform_static)
            fa = compute_fabric_anisotropy(normals)
            mp = compute_mobilised_friction_angle(normals, pos, current_angle)
            fs = estimate_factor_of_safety(pos, current_angle, coord, planform_static)
            touch_frac = compute_touching_surface_fraction(
                pos, sizes, PARTICLE_SHAPE, c_sizes, backend.particle_kinds, backend.particle_b_sizes)
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

        if cancelled:
            print("\n  [INFO] Simulation cancelled by user.")
            break

        # Progress
        if step % max(1, N_STEPS // 10) == 0:
            pos = backend.get_positions()
            disp = compute_displacement(pos, init_pos) if len(pos) > 0 else np.zeros(0)
            pct = compute_pct_moved(disp, sizes_static) if len(disp) > 0 else 0.0
            elapsed = time.time() - wall_t0
            print(f"  t={t:6.2f}s  angle={current_angle:5.1f}°  "
                  f"moved={pct:5.1f}%  KE={backend.get_kinetic_energy():.2e}  "
                  f"[{elapsed:.1f}s wall]")

    print(f"\nSimulation complete ({time.time() - wall_t0:.1f}s wall time).\n")

    # Convert lists to arrays for plotting & analysis
    n_frames = len(frames_time)
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
    is_dfn_mode = (FRACTURE_MODE.upper() == "DFN")
    vel_mag_max = (float(np.percentile(np.linalg.norm(arr_vel.reshape(-1, 2), axis=1), 98))
                   if arr_vel.size > 0 else 1.0)
    vel_mag_max = max(vel_mag_max, 1e-3)

    # =====================================================================
    # 8. VISUALISATION (when save_video is enabled)
    # =====================================================================
    if save_video and n_frames > 0:
        print(f"Rendering {n_frames} frames → {VIDEO_FILE} ...")

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
        cmap_disp = plt.colormaps.get_cmap("viridis" if is_dfn_mode else "YlOrRd")
        max_displacement = max(float(np.max(sizes_static)) * 5.0 if len(sizes_static) else PARTICLE_SIZE * 5, 0.3)

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
            if draw_local_vertices is not None and draw_local_vertices[i] is not None:
                p = MplPolygon(np.zeros((4, 2)), closed=True, lw=0.4, ec="0.2")
            elif backend.particle_kinds[i] == "circle":
                p = MplCircle((0, 0), sizes_static[i] / 2, lw=0.3, ec="0.2")
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

                if draw_local_vertices is not None and draw_local_vertices[i] is not None:
                    lv = draw_local_vertices[i]
                    ca = math.cos(block_ang[i])
                    sa = math.sin(block_ang[i])
                    rot = np.array([[ca, -sa], [sa, ca]])
                    world = lv @ rot.T + pos[i]
                    particle_patches[i].set_xy(world)
                elif backend.particle_kinds[i] == "circle":
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
            if not return_data:
                print("  Showing interactive plot instead...")
                plt.show()
        plt.close(fig)

    # =====================================================================
    # 9. SUMMARY REPORT
    # =====================================================================
    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("STABILITY ANALYSIS SUMMARY")
    summary_lines.append(f"Calculated time: {t:.6g} s; recording window: {arr_time[0]:.6g} to {arr_time[-1]:.6g} s "
                         f"({schedule.mode}, nearest solver state for seconds)")
    summary_lines.append("=" * 65)

    failure_idx = None
    for i, pct in enumerate(arr_pct):
        if arr_time[i] >= SETTLE_TIME and pct >= 50.0:
            failure_idx = i
            break

    if failure_idx is not None:
        summary_lines.append(f"  FAILURE at tilt = {arr_angle[failure_idx]:.1f}° "
                             f"(t = {arr_time[failure_idx]:.2f} s)")
    elif len(arr_pct) > 0:
        summary_lines.append(f"  NO FAILURE detected (max moved = {max(arr_pct):.1f}%)")
    else:
        summary_lines.append("  NO DATA recorded.")

    if len(arr_time) > 0:
        summary_lines.append(f"\n  --- Geotechnical Diagnostics ---")
        summary_lines.append(f"  Particles              : {n_particles}")
        if PARTICLE_SIZE_INPUT:
            summary_lines.append(f"  Size input (m, by count): {PARTICLE_SIZE_INPUT}")
            summary_lines.append(f"  Realized D16/D50/D84    : {np.quantile(sizes_static, [.16, .5, .84])}")
        summary_lines.append(f"  Gravity packed         : {GRAVITY_PACKED}")
        if _HOTSTART_INFO is not None:
            info = _HOTSTART_INFO
            summary_lines.append(f"  Hotstart Pack          : {'converged' if info['converged'] else 'NOT converged'}, "
                                 f"{info['cycles']} cycles; {info['stop_reason']}")
            summary_lines.append(f"  Settled bulk area      : {info['area']:.4f} / {info['target_area']:.4f} m2 "
                                 f"(error={info['relative_error']:+.1%})")
        if _FILL_VOID_INFO is not None:
            info = _FILL_VOID_INFO
            if info.get("hotstart_batch"):
                summary_lines.append(f"  Last hotstart fill     : added {info['added_count']} particles")
            else:
                summary_lines.append(f"  Fill Void              : added {info['added_count']} particles, "
                                     f"slope envelope coverage {info['coverage']:.1%} ({info['stop_reason']})")
        if _MIXED_PACKING_INFO is not None:
            info = _MIXED_PACKING_INFO
            summary_lines.append(f"  Initial mixture        : {info['circle_count']} circles / "
                                 f"{info['rectangle_count']} rectangles ({info['circle_fraction']:.1%} circles by count)")
            summary_lines.append(f"  Initial XZ porosity    : {info['porosity']:.3f} "
                                 f"(target={info['target_porosity']:.3f}, seed={info['seed']})")
        summary_lines.append(f"  Final % moved          : {arr_pct[-1]:.1f}%")
        summary_lines.append(f"  Min Factor of Safety   : {min(arr_fs):.3f}")
        summary_lines.append(f"  Final FS               : {arr_fs[-1]:.3f}")
        summary_lines.append(f"  Max kinetic energy     : {max(arr_ke):.4e} J")
        summary_lines.append(f"  Final coordination #   : {arr_coord[-1]:.2f}")
        summary_lines.append(f"  Final touching surface : {arr_touch[-1] * 100:.1f}%")
        summary_lines.append(f"  Min touching surface   : {arr_touch.min() * 100:.1f}%")
        if is_dfn_mode and len(arr_mob_small) > 0:
            summary_lines.append(f"  Final class mobility   : small={arr_mob_small[-1]:.1f}%  "
                                 f"medium={arr_mob_medium[-1]:.1f}%  large={arr_mob_large[-1]:.1f}%")
            summary_lines.append(f"  Peak class mobility    : small={arr_mob_small.max():.1f}%  "
                                 f"medium={arr_mob_medium.max():.1f}%  large={arr_mob_large.max():.1f}%")
        summary_lines.append(f"  Final fabric anisotropy: {arr_fa[-1]:.3f}")
        summary_lines.append(f"  Final mob. friction    : {arr_mp[-1]:.1f}°")
        summary_lines.append(f"  Final bulk density     : {arr_bd[-1]:.0f} kg/m³")

        summary_lines.append(f"\n  --- Metric Definitions ---")
        summary_lines.append(f"  Factor of Safety (FS): Ratio of resisting to driving moments")
        summary_lines.append(f"    about the slope toe. FS < 1 indicates instability.")
        summary_lines.append(f"  Coordination Number: Average contacts per particle.")
        summary_lines.append(f"    Higher values indicate a more mechanically stable packing.")
        summary_lines.append(f"  Touching-surface fraction: Estimated contacted surface area")
        summary_lines.append(f"    (material in contact with other material), divided by")
        summary_lines.append(f"    total material surface area.")
        summary_lines.append(f"  Fabric Anisotropy: Directional bias of contact normals (0=isotropic,")
        summary_lines.append(f"    1=all contacts aligned). High values with horizontal bias resist")
        summary_lines.append(f"    lateral sliding.")
        summary_lines.append(f"  Mobilised Friction Angle: Angular deviation of the fabric tensor")
        summary_lines.append(f"    principal direction from gravity. Approaches the material friction")
        summary_lines.append(f"    angle at failure.")
        summary_lines.append(f"  Kinetic Energy: Total KE of all particles. A spike indicates")
        summary_lines.append(f"    dynamic failure / rapid mass movement.")

    summary_lines.append("=" * 65)
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    if return_data:
        dfn_blocks = (_snapshot_dfn_blocks(hotstart_initial_specs) if HOTSTART_PACK else
                      _fracture_domain_with_dfn()) if is_dfn_mode else []
        return {
            "n_particles": n_particles,
            "init_pos": init_pos,
            "sizes_static": sizes_static,
            "c_sizes_static": c_sizes_static,
            "planform_static": planform_static,
            "draw_local_vertices": getattr(backend, "particle_draw_local_vertices", None),
            "is_dfn_mode": is_dfn_mode,
            "dfn_blocks": dfn_blocks,
            "particle_shape": PARTICLE_SHAPE,
            "particle_size_input": PARTICLE_SIZE_INPUT or str(PARTICLE_SIZE),
            "particle_kinds": list(backend.particle_kinds),
            "initial_mixture": dict(_MIXED_PACKING_INFO) if _MIXED_PACKING_INFO is not None else None,
            "gravity_packed": GRAVITY_PACKED,
            "fill_void": bool(FILL_VOID and GRAVITY_PACKED),
            "fill_void_info": dict(_FILL_VOID_INFO) if _FILL_VOID_INFO is not None else None,
            "hotstart_pack": HOTSTART_PACK,
            "hotstart_info": dict(_HOTSTART_INFO) if _HOTSTART_INFO is not None else None,
            "hotstart_initial_specs": hotstart_initial_specs,
            "gravity_packing_info": dict(_GRAVITY_PACKING_INFO) if _GRAVITY_PACKING_INFO is not None else None,
            "b_sizes_static": np.array(backend.particle_b_sizes),
            "rect_b": _rect_b,
            "rect_c": _rect_c,
            "domain_x": DOMAIN_X,
            "domain_y": DOMAIN_Y,
            "domain_z": DOMAIN_Z,
            "slope_height": SLOPE_HEIGHT,
            "slope_length": SLOPE_LENGTH,
            "slope_base": SLOPE_BASE,
            "total_rotation": TOTAL_ROTATION,
            "settle_time": SETTLE_TIME,
            "sim_duration": SIM_DURATION,
            "calculated_time": t,
            "cancelled": cancelled,
            "recording_parameters": schedule.metadata(),
            "fps": FPS,
            "frames_pos": frames_pos,
            "frames_vel": frames_vel,
            "arr_vel": arr_vel,
            "vel_mag_max": vel_mag_max,
            "frames_block_ang": frames_block_ang,
            "arr_time": arr_time,
            "arr_angle": arr_angle,
            "arr_pct": arr_pct,
            "arr_coord": arr_coord,
            "arr_bd": arr_bd,
            "arr_ke": arr_ke,
            "arr_fa": arr_fa,
            "arr_mp": arr_mp,
            "arr_fs": arr_fs,
            "arr_touch": arr_touch,
            "arr_mob_small": arr_mob_small,
            "arr_mob_medium": arr_mob_medium,
            "arr_mob_large": arr_mob_large,
            "failure_idx": failure_idx,
            "summary_text": summary_text,
        }

    # Show as Jupyter display if available
    try:
        from IPython.display import Video
        return Video(VIDEO_FILE)
    except Exception:
        pass


# =====================================================================
# 10. INTERACTIVE GUI (Tkinter + Matplotlib Canvas)
# =====================================================================

PRESETS = {
    "1. Default: DFN Fractured Rock Slope": {
        "ENGINE": "pymunk", "QUASI_2D": True,
        "DOMAIN_X": 2.0, "DOMAIN_Y": 0.2, "DOMAIN_Z": 2.0, "CELL_SIZE": 0.05,
        "SLOPE_HEIGHT": 1.0, "SLOPE_LENGTH": 1.5, "SLOPE_BASE": 0.0,
        "FRACTURE_MODE": "DFN", "PARTICLE_SHAPE": "rectangle",
        "PARTICLE_SIZE": 0.06, "PARTICLE_B_AXIS": 0.06, "PARTICLE_C_AXIS": None,
        "RECT_ARRANGEMENT": "uniform", "TARGET_POROSITY": 0.15,
        "DFN_SEED": 42, "DFN_LINE_JITTER": 0.08, "DFN_MIN_BLOCK_AREA": 0.00025,
        "DFN_JOINT_SETS": [
            {"dip_deg": 35.0, "spacing": 0.18, "persistence": 0.90, "persistence_std": 0.20},
            {"dip_deg": 70.0, "spacing": 0.24, "persistence": 0.65, "persistence_std": 0.15}
        ],
        "MATERIAL_TYPE": "granular", "PARTICLE_DENSITY": 2000.0,
        "FRICTION_COEFF": 0.80, "COHESION_FORCE": 0.0, "BOND_BREAK_MULT": 1.5, "RESTITUTION": 0.01,
        "TOTAL_ROTATION": 90.0, "ROTATION_RATE": 2.0, "SETTLE_TIME": 2.0,
        "DT": 0.005, "SOLVER_ITERS": 20, "GRAVITY_MAG": 9.81,
        "AUTO_STABILIZE": False,
    },
    "2. Granular Talus / Scree (Friction Only)": {
        "ENGINE": "pymunk",
        "FRACTURE_MODE": "grid", "PARTICLE_SHAPE": "rectangle",
        "PARTICLE_SIZE": 0.05, "PARTICLE_B_AXIS": 0.05,
        "RECT_ARRANGEMENT": "uniform", "TARGET_POROSITY": 0.20,
        "MATERIAL_TYPE": "granular", "PARTICLE_DENSITY": 1800.0,
        "FRICTION_COEFF": 0.70, "COHESION_FORCE": 0.0, "RESTITUTION": 0.05,
        "SLOPE_HEIGHT": 0.9, "SLOPE_LENGTH": 1.6,
        "TOTAL_ROTATION": 60.0, "ROTATION_RATE": 2.0,
        "AUTO_STABILIZE": False,
    },
    "3. Cohesive Soil / Clay Slope": {
        "ENGINE": "pymunk",
        "FRACTURE_MODE": "grid", "PARTICLE_SHAPE": "circle",
        "PARTICLE_SIZE": 0.05,
        "MATERIAL_TYPE": "cohesive", "PARTICLE_DENSITY": 1900.0,
        "FRICTION_COEFF": 0.50, "COHESION_FORCE": 6.0, "BOND_BREAK_MULT": 1.4,
        "SLOPE_HEIGHT": 1.0, "SLOPE_LENGTH": 1.4,
        "TOTAL_ROTATION": 75.0, "ROTATION_RATE": 2.5,
        "AUTO_STABILIZE": False,
    },
    "4. Steep Bedrock Slope (H=1.5m, L=1.0m)": {
        "ENGINE": "pymunk",
        "SLOPE_HEIGHT": 1.5, "SLOPE_LENGTH": 1.0,
        "FRACTURE_MODE": "DFN", "PARTICLE_SHAPE": "rectangle",
        "DFN_SEED": 101, "DFN_MIN_BLOCK_AREA": 0.0003,
        "MATERIAL_TYPE": "cohesive", "FRICTION_COEFF": 0.75, "COHESION_FORCE": 8.0,
        "TOTAL_ROTATION": 60.0,
        "AUTO_STABILIZE": False,
    },
    "5. Fine-Grained Talus (d=0.03m, Porosity=0.25)": {
        "ENGINE": "pymunk",
        "FRACTURE_MODE": "grid", "PARTICLE_SHAPE": "circle",
        "PARTICLE_SIZE": 0.03, "TARGET_POROSITY": 0.25,
        "MATERIAL_TYPE": "granular", "FRICTION_COEFF": 0.85, "COHESION_FORCE": 0.0,
        "SLOPE_HEIGHT": 0.8, "SLOPE_LENGTH": 1.5,
        "AUTO_STABILIZE": False,
    },
    "6. Bedding-Plane Jointed Slope (Dip=30°)": {
        "ENGINE": "pymunk",
        "FRACTURE_MODE": "DFN", "PARTICLE_SHAPE": "rectangle",
        "DFN_JOINT_SETS": [
            {"dip_deg": 30.0, "spacing": 0.12, "persistence": 1.50, "persistence_std": 0.10},
            {"dip_deg": 105.0, "spacing": 0.30, "persistence": 0.40, "persistence_std": 0.10}
        ],
        "MATERIAL_TYPE": "granular", "FRICTION_COEFF": 0.75, "COHESION_FORCE": 0.0,
        "SLOPE_HEIGHT": 1.1, "SLOPE_LENGTH": 1.4,
        "AUTO_STABILIZE": False,
    }
}

PRESETS["7. Mixed Circles and Rectangles (50:50 by count)"] = {
    "ENGINE": "pymunk", "FRACTURE_MODE": "grid", "PARTICLE_SHAPE": "mixed",
    "PARTICLE_SIZE": 0.06, "PARTICLE_B_AXIS": 0.04, "PARTICLE_C_AXIS": None,
    "TARGET_POROSITY": 0.55,
    **{name: default for name, (_, default) in _MIXED_FIELDS.items()},
}

PRESETS["8. Stream Bed: Flat Gravel, Prescribed Stress"] = {
    "STREAM_BED_SETUP": True, "ENGINE": "pymunk", "FRACTURE_MODE": "grid",
    "DOMAIN_X": 2., "DOMAIN_Z": .8, "SLOPE_HEIGHT": .3, "SLOPE_BASE": 0., "SLOPE_LENGTH": 2.,
    "PARTICLE_SHAPE": "circle", "PARTICLE_SIZE": .06, "TARGET_POROSITY": .3,
    "GRAVITY_PACKED": True, "FILL_VOID": False, "HOTSTART_PACK": False,
    "MATERIAL_TYPE": "granular", "PARTICLE_DENSITY": 2650., "FRICTION_COEFF": .6,
    "COHESION_FORCE": 0., "SETTLE_TIME": 2., "DT": .002, "SOLVER_ITERS": 50,
    "RECIRCULATION": False, "PARTICLE_SIZE_INPUT": "",
    "FRAME_INTERVAL": 25, "VIDEO_FILE": "stream_bed_sim.mp4",
    **{name: default for name, (_, default) in _STREAM_FIELDS.items()},
    **{name: default for name, (_, default) in _SOLVER_FIELDS.items()},
    **{name: default for name, (_, default) in _RECORDING_FIELDS.items()},
    "STREAM_DOMAIN_SCALE": 1., "STREAM_FORCING": "steady", "STREAM_DURATION": 60.,
    "FLOW_DEPTH": .5, "HYDRAULIC_LOAD": "traction", "STREAM_STARTUP_TIME": .5,
    "ADAPTIVE_DT": False, "FRAME_INTERVAL_MODE": "steps",
}
PRESETS["9. Stream Bed: Mean Velocity, RANS and Turbulence"] = {
    **PRESETS["8. Stream Bed: Flat Gravel, Prescribed Stress"],
    "ENGINE": "pychrono", "STREAM_FORCING": "velocity", "HYDRAULIC_SOLVER": "advanced",
    "FLOW_MEAN_VELOCITY": 1., "FLOW_DEPTH": .5, "HYDRAULIC_LOAD": "drag",
    "SIMULATE_TURBULENCE": True, "TURBULENCE_TARGET": "velocity", "ADAPTIVE_DT": True,
    "SOLVER_ITERS": 80,
}

DEFAULT_PRESET_NAME = "0. Default: Scaled Stream Bed, 2-4 mm Gravel"
_DEFAULT_STREAM_KEYS = (
    "STREAM_BED_SETUP", "ENGINE", "QUASI_2D", "DOMAIN_X", "DOMAIN_Y", "DOMAIN_Z",
    "CELL_SIZE", "SLOPE_HEIGHT", "SLOPE_LENGTH", "SLOPE_BASE", "FRACTURE_MODE",
    "PARTICLE_SHAPE", "PARTICLE_SIZE", "PARTICLE_SIZE_INPUT", "PARTICLE_B_AXIS",
    "PARTICLE_C_AXIS", "RECT_ARRANGEMENT", "TARGET_POROSITY", "GRAVITY_PACKED",
    "FILL_VOID", "HOTSTART_PACK", "MATERIAL_TYPE", "PARTICLE_DENSITY",
    "FRICTION_COEFF", "COHESION_FORCE", "BOND_BREAK_MULT", "RESTITUTION",
    "SETTLE_TIME", "DT", "SOLVER_ITERS", "GRAVITY_MAG", "RECIRCULATION",
    "FRAME_INTERVAL", "FPS", "VIDEO_FILE", "AUTO_STABILIZE",
)
PRESETS = {
    DEFAULT_PRESET_NAME: {
        **{name: globals()[name] for name in _DEFAULT_STREAM_KEYS},
        **{name: globals()[name] for name in _MIXED_FIELDS},
        **{name: globals()[name] for name in _HOTSTART_OPTIONS},
        **{name: globals()[name] for name in _STREAM_FIELDS},
        **{name: globals()[name] for name in _SOLVER_FIELDS},
        **{name: globals()[name] for name in _PARTICLE_PHYSICS_FIELDS},
        **{name: globals()[name] for name in _RECORDING_FIELDS},
    },
    **PRESETS,
}


class HillslopeSimGUI:
    """
    Logically structured desktop GUI for operating the HillslopeSim model,
    tuning parameters, visualizing initial geometry and live/scrubbable
    simulation output, and analyzing/saving outputs.
    """

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.NavigationToolbar2Tk = NavigationToolbar2Tk
        self.Figure = Figure

        self.root = root
        self.root.title("HillslopeSim — Hillslope & Stream-Bed Discrete-Element Model")
        self.root.geometry("1550x940")
        self.root.minsize(1200, 750)

        # State
        self.sim_data: Optional[Dict] = None
        self.is_running = False
        self.cancel_requested = False
        self.is_playing = False
        self.current_frame_idx = 0
        self._play_job = None

        self._init_variables()
        self._build_layout()
        self._update_stream_controls()
        self.preview_geometry()

    # ── Tkinter Variables ──────────────────────────────────────────
    def _init_variables(self):
        tk = self.tk
        self.var_stream_bed = tk.BooleanVar(value=STREAM_BED_SETUP)
        self.var_recirculation = tk.BooleanVar(value=RECIRCULATION)
        self.stream_vars = {name: (tk.BooleanVar if isinstance(default, bool) else tk.StringVar)(value=globals()[name])
                            for name, (_, default) in _STREAM_FIELDS.items()}
        self.particle_physics_vars = {name: tk.BooleanVar(value=globals()[name])
                                     for name in _PARTICLE_PHYSICS_FIELDS}
        self.solver_vars = {name: (tk.BooleanVar if isinstance(default, bool) else tk.StringVar)(value=globals()[name])
                            for name, (_, default) in _SOLVER_FIELDS.items()}
        # 1. Engine & Domain
        self.var_engine = tk.StringVar(value=ENGINE)
        self.var_quasi2d = tk.BooleanVar(value=QUASI_2D)
        self.var_domain_x = tk.DoubleVar(value=DOMAIN_X)
        self.var_domain_y = tk.DoubleVar(value=DOMAIN_Y)
        self.var_domain_z = tk.DoubleVar(value=DOMAIN_Z)
        self.var_cell_size = tk.DoubleVar(value=CELL_SIZE)
        self.var_slope_h = tk.DoubleVar(value=SLOPE_HEIGHT)
        self.var_slope_l = tk.DoubleVar(value=SLOPE_LENGTH)
        self.var_slope_base = tk.DoubleVar(value=SLOPE_BASE)

        # 2. Fracture & Particles
        self.var_fracture_mode = tk.StringVar(value=FRACTURE_MODE)
        self.var_particle_shape = tk.StringVar(value=PARTICLE_SHAPE)
        self.var_particle_size = tk.StringVar(value=PARTICLE_SIZE_INPUT or str(PARTICLE_SIZE))
        self.var_particle_b = tk.DoubleVar(value=PARTICLE_B_AXIS if PARTICLE_B_AXIS is not None else PARTICLE_SIZE)
        self.var_particle_c = tk.StringVar(value="" if PARTICLE_C_AXIS is None else str(PARTICLE_C_AXIS))
        self.var_rect_arr = tk.StringVar(value=RECT_ARRANGEMENT)
        self.var_porosity = tk.DoubleVar(value=TARGET_POROSITY)
        self.var_gravity_packed = tk.BooleanVar(value=GRAVITY_PACKED)
        self.var_fill_void = tk.BooleanVar(value=FILL_VOID)
        self.var_hotstart_pack = tk.BooleanVar(value=HOTSTART_PACK)
        self.hotstart_options = {name: globals()[name] for name in _HOTSTART_OPTIONS}

        self.mix_vars = {}
        for name, (_, default) in _MIXED_FIELDS.items():
            var_type = tk.StringVar if isinstance(default, int) else tk.DoubleVar
            self.mix_vars[name] = var_type(value=globals()[name])

        # DFN parameters
        self.var_dfn_seed = tk.IntVar(value=DFN_SEED)
        self.var_dfn_jitter = tk.DoubleVar(value=DFN_LINE_JITTER)
        self.var_dfn_min_area = tk.DoubleVar(value=DFN_MIN_BLOCK_AREA)
        js1 = DFN_JOINT_SETS[0] if len(DFN_JOINT_SETS) > 0 else {"dip_deg": 35.0, "spacing": 0.18, "persistence": 0.90, "persistence_std": 0.20}
        js2 = DFN_JOINT_SETS[1] if len(DFN_JOINT_SETS) > 1 else {"dip_deg": 70.0, "spacing": 0.24, "persistence": 0.65, "persistence_std": 0.15}
        self.var_js1_dip = tk.DoubleVar(value=js1.get("dip_deg", 35.0))
        self.var_js1_spac = tk.DoubleVar(value=js1.get("spacing", 0.18))
        self.var_js1_pers = tk.DoubleVar(value=js1.get("persistence", 0.90))
        self.var_js1_pstd = tk.DoubleVar(value=js1.get("persistence_std", 0.20))

        self.var_js2_enable = tk.BooleanVar(value=len(DFN_JOINT_SETS) > 1)
        self.var_js2_dip = tk.DoubleVar(value=js2.get("dip_deg", 70.0))
        self.var_js2_spac = tk.DoubleVar(value=js2.get("spacing", 0.24))
        self.var_js2_pers = tk.DoubleVar(value=js2.get("persistence", 0.65))
        self.var_js2_pstd = tk.DoubleVar(value=js2.get("persistence_std", 0.15))

        # 3. Material & Physics
        self.var_mat_type = tk.StringVar(value=MATERIAL_TYPE)
        self.var_density = tk.DoubleVar(value=PARTICLE_DENSITY)
        self.var_friction = tk.DoubleVar(value=FRICTION_COEFF)
        self.var_cohesion = tk.DoubleVar(value=COHESION_FORCE)
        self.var_bond_mult = tk.DoubleVar(value=BOND_BREAK_MULT)
        self.var_restitution = tk.DoubleVar(value=RESTITUTION)
        self.var_auto_stabilize = tk.BooleanVar(value=AUTO_STABILIZE)

        # 4. Tilting & Solver
        self.var_total_rot = tk.DoubleVar(value=TOTAL_ROTATION)
        self.var_rot_rate = tk.DoubleVar(value=ROTATION_RATE)
        self.var_settle_time = tk.DoubleVar(value=SETTLE_TIME)
        self.var_dt = tk.DoubleVar(value=DT)
        self.var_solver_iters = tk.IntVar(value=SOLVER_ITERS)
        self.var_gravity = tk.DoubleVar(value=GRAVITY_MAG)
        self.var_fps = tk.IntVar(value=FPS)
        self.var_frame_int = tk.IntVar(value=FRAME_INTERVAL)
        self.recording_vars = {name: tk.StringVar(value="" if globals()[name] is None else globals()[name])
                               for name in _RECORDING_FIELDS}
        self.var_video_file = tk.StringVar(value=VIDEO_FILE)
        self.var_auto_video = tk.BooleanVar(value=False)

        # Analysis & Visualisation controls
        self.var_color_mode = tk.StringVar(value="auto")  # "auto", "displacement", "speed", "size"
        self.var_show_quiver = tk.BooleanVar(value=True)
        self.var_show_traces = tk.BooleanVar(value=True)
        self.var_rotate_view = tk.BooleanVar(value=True)

    # ── Layout Construction ────────────────────────────────────────
    def _build_layout(self):
        tk = self.tk
        ttk = self.ttk

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Save.TButton", font=("Segoe UI", 9, "bold"))

        # Top-level PanedWindow (Left: Controls, Right: Visualisation & Analysis)
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left_frame = ttk.Frame(paned, width=430)
        right_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=0)
        paned.add(right_frame, weight=1)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

    def _on_param_modified(self, _event=None):
        if hasattr(self, "lbl_param_status"):
            self.lbl_param_status.config(
                text="● Unsaved changes — click 'Save / Apply' or press Enter",
                foreground="#e65100"
            )

    def _on_packing_switch(self, source):
        """Interactively regenerate the preview when either packing switch changes."""
        if source == "hotstart" and self.var_hotstart_pack.get():
            self.var_gravity_packed.set(True)
            self.var_fill_void.set(True)
        elif source == "fill" and self.var_fill_void.get():
            self.var_gravity_packed.set(True)
        elif source == "fill" and not self.var_fill_void.get():
            self.var_hotstart_pack.set(False)
        elif source == "gravity" and not self.var_gravity_packed.get():
            self.var_fill_void.set(False)
            self.var_hotstart_pack.set(False)
        self._on_param_modified()
        if not self.is_running:
            self.preview_geometry()

    def _update_stream_controls(self):
        enabled = self.var_stream_bed.get()
        if enabled and self.var_engine.get() == "pychrono":
            self.var_quasi2d.set(True)
        if hasattr(self, "stream_entries"):
            for name, widget in self.stream_entries.items():
                usable = enabled and (name not in ("BED_STRESS_AMPLITUDE", "BED_STRESS_FREQUENCY")
                                      or self.stream_vars["STREAM_FORCING"].get() == "harmonic")
                if name == "BED_STRESS_MEAN" and self.stream_vars["STREAM_FORCING"].get() == "velocity":
                    usable = False
                if name == "FLOW_MEAN_VELOCITY" and self.stream_vars["STREAM_FORCING"].get() != "velocity":
                    usable = False
                if name.startswith("TURBULENCE_") and not self.stream_vars["SIMULATE_TURBULENCE"].get():
                    usable = False
                if self.is_running and name != "SIMULATE_TURBULENCE":
                    usable = False
                widget.configure(state=("readonly" if name in _STREAM_CHOICES else "normal")
                                 if usable else "disabled")
            for widget in self.tilt_entries:
                widget.configure(state="disabled" if enabled else "normal")
            self.engine_entry.configure(state="readonly")
            self.quasi_entry.configure(state="disabled" if enabled else "normal")
            self.recirculation_entry.configure(state="normal" if enabled else "disabled")
            self.cohesion_label.configure(text="Cohesive Bond Strength (N/m):" if enabled and self.var_engine.get() == "pymunk" else "Cohesion Force (N):")
            self.auto_cohesion_entry.configure(state="disabled" if enabled else "normal")
            self.btn_stability.configure(state="disabled" if enabled else "normal")
            self.geometry_frame.configure(text="Stream Bed Geometry (m)" if enabled else "Hillslope Geometry (m)")
            for label, bed_text, slope_text in self.geometry_labels:
                label.configure(text=bed_text if enabled else slope_text)

    def _on_stream_switch(self):
        self._update_stream_controls()
        self._on_param_modified()
        if not self.is_running:
            self.preview_geometry()

    def _on_turbulence_switch(self):
        """The turbulence checkbox may be toggled during an active stream run."""
        if self.stream_vars["TURBULENCE_TARGET"].get() == "velocity":
            self.stream_vars["HYDRAULIC_LOAD"].set("drag")
        if self.is_running and STREAM_BED_SETUP:
            globals()["SIMULATE_TURBULENCE"] = bool(self.stream_vars["SIMULATE_TURBULENCE"].get())
        self._update_stream_controls()
        self._on_param_modified()

    def _update_recording_controls(self, _event=None):
        if not hasattr(self, "recording_entries"):
            return
        seconds = self.recording_vars["FRAME_INTERVAL_MODE"].get() == "seconds"
        self.frame_steps_entry.configure(state="disabled" if seconds else "normal")
        self.recording_entries["FRAME_INTERVAL_SECONDS"].configure(state="normal" if seconds else "disabled")
        self._on_param_modified()

    def _add_row(self, parent, row: int, label_text: str, var, widget_type="entry", options=None, tooltip=None):
        ttk = self.ttk
        tk = self.tk
        lbl = ttk.Label(parent, text=label_text)
        lbl.grid(row=row, column=0, sticky="w", padx=6, pady=3)
        if widget_type == "entry":
            w = ttk.Entry(parent, textvariable=var, width=14)
            w.bind("<Return>", lambda e: self.apply_model_parameters())
            w.bind("<KeyRelease>", self._on_param_modified)
        elif widget_type == "combo":
            w = ttk.Combobox(parent, textvariable=var, values=options or [], state="readonly", width=12)
            w.bind("<<ComboboxSelected>>", self._on_param_modified)
        elif widget_type == "check":
            w = ttk.Checkbutton(parent, variable=var, command=self._on_param_modified)
        elif widget_type == "spin":
            w = ttk.Spinbox(parent, textvariable=var, from_=options[0], to=options[1], increment=options[2], width=12)
            w.bind("<Return>", lambda e: self.apply_model_parameters())
            w.bind("<KeyRelease>", self._on_param_modified)
        w.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        parent.columnconfigure(1, weight=1)
        return w

    def _build_left_panel(self, parent):
        tk = self.tk
        ttk = self.ttk

        # Header
        hdr = ttk.Label(parent, text="Model Parameters & Configuration", font=("Segoe UI", 11, "bold"))
        hdr.pack(anchor="w", padx=6, pady=(2, 4))

        # Parameter Action Toolbar (Save Changes / Load / Export)
        param_bar = ttk.Frame(parent)
        param_bar.pack(fill=tk.X, padx=4, pady=(0, 2))

        self.btn_apply_top = ttk.Button(
            param_bar, text="💾 Save Changes to Model",
            command=self.apply_model_parameters, style="Save.TButton"
        )
        self.btn_apply_top.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_load_top = ttk.Button(
            param_bar, text="📂 Load Parameters", command=self.load_config_json
        )
        self.btn_load_top.pack(side=tk.LEFT, padx=2)

        self.btn_save_top = ttk.Button(
            param_bar, text="💾 Save to File", command=self.save_config_json
        )
        self.btn_save_top.pack(side=tk.LEFT, padx=(2, 0))

        # Preset selection bar
        preset_bar = ttk.Frame(parent)
        preset_bar.pack(fill=tk.X, padx=4, pady=(2, 3))
        ttk.Label(preset_bar, text="Preset:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 3))
        self.cb_preset = ttk.Combobox(
            preset_bar, state="readonly", width=28,
            values=list(PRESETS.keys())
        )
        self.cb_preset.set("Select a preset to load...")
        self.cb_preset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.cb_preset.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # Sync state indicator label
        self.lbl_param_status = ttk.Label(
            parent, text="✔ Model parameters in sync with simulation engine",
            font=("Segoe UI", 8, "italic"), foreground="#2e7d32"
        )
        self.lbl_param_status.pack(anchor="w", padx=6, pady=(0, 4))

        # Parameter Notebook
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ── Tab 1: Domain & Geometry ──
        tab_dom = ttk.Frame(nb, padding=6)
        nb.add(tab_dom, text="Domain")

        ttk.Checkbutton(tab_dom, text="Stream bed set up", variable=self.var_stream_bed,
                        command=self._on_stream_switch).pack(anchor="w", pady=4)

        lf_eng = ttk.LabelFrame(tab_dom, text="Physics Engine", padding=6)
        lf_eng.pack(fill=tk.X, pady=4)
        self.engine_entry = self._add_row(lf_eng, 0, "Backend Engine:", self.var_engine, "combo", ["pymunk", "pychrono"])
        self.quasi_entry = self._add_row(lf_eng, 1, "Quasi-2D (pychrono):", self.var_quasi2d, "check")

        lf_grid = ttk.LabelFrame(tab_dom, text="Domain Bounding Box (m)", padding=6)
        lf_grid.pack(fill=tk.X, pady=4)
        self._add_row(lf_grid, 0, "Domain X (length):", self.var_domain_x)
        self._add_row(lf_grid, 1, "Domain Y (width):", self.var_domain_y)
        self._add_row(lf_grid, 2, "Domain Z (height):", self.var_domain_z)
        self._add_row(lf_grid, 3, "Reference Cell Size:", self.var_cell_size)

        lf_slope = ttk.LabelFrame(tab_dom, text="Hillslope Geometry (m)", padding=6)
        lf_slope.pack(fill=tk.X, pady=4)
        self._add_row(lf_slope, 0, "Crest Height (H):", self.var_slope_h)
        self._add_row(lf_slope, 1, "Horizontal Run (L):", self.var_slope_l)
        self._add_row(lf_slope, 2, "Toe Base Elev:", self.var_slope_base)
        self.geometry_frame = lf_slope
        self.geometry_labels = [(lf_slope.grid_slaves(row=r, column=0)[0], bed, slope)
                                for r, bed, slope in ((0, "Flat Bed Top (H):", "Crest Height (H):"),
                                                     (1, "Sediment Length (L):", "Horizontal Run (L):"),
                                                     (2, "Rigid Bed Base Elev:", "Toe Base Elev:"))]

        tab_stream = self._scrollable_parameter_tab(nb, "Stream Bed")
        ttk.Checkbutton(tab_stream, text="Stream bed set up", variable=self.var_stream_bed,
                        command=self._on_stream_switch).pack(anchor="w", pady=4)
        self.recirculation_entry = ttk.Checkbutton(tab_stream, text="Recirculation",
                                                   variable=self.var_recirculation, command=self._on_param_modified)
        self.recirculation_entry.pack(anchor="w", pady=3)
        ttk.Label(tab_stream, text="Return outgoing grains upstream at the same elevation and velocity.",
                  wraplength=370).pack(fill=tk.X, pady=(0, 4))
        ttk.Label(tab_stream, text="Horizontal flow from left to right over submerged grains.\n"
                  "Uses the shared Domain and Discretization & Packing settings.\n"
                  "H is the flat bed top; base is the rigid substrate; L is bed length.\n"
                  "Choose pymunk (unit-width 2D) or pychrono (planar 3D grains).\n"
                  "The outlet barrier follows the mean bed surface.",
                  wraplength=370).pack(fill=tk.X, pady=6)
        groups = {}
        for title in ("Hydraulic Forcing & Duration", "Flow Profile & Roughness", "Simulated Turbulence"):
            groups[title] = ttk.LabelFrame(tab_stream, text=title, padding=6)
            groups[title].pack(fill=tk.X, pady=4)
        self.stream_entries = {}
        rows = {title: 0 for title in groups}
        for row, (name, (label, _)) in enumerate(_STREAM_FIELDS.items()):
            title = ("Simulated Turbulence" if name.startswith("TURBULENCE_") or name == "SIMULATE_TURBULENCE"
                     else "Flow Profile & Roughness" if name in ("HYDRAULIC_SOLVER", "HYDRAULIC_LOAD", "ROUGHNESS_D84_FACTOR",
                          "FLOW_MEAN_VELOCITY", "FLOW_DEPTH", "WATER_VISCOSITY", "FLOW_COLUMNS", "FLOW_LEVELS",
                          "FLOW_UPDATE_INTERVAL", "STREAM_DRAG_COEFFICIENT", "STREAM_DRAG_INTEGRATION",
                          "STREAM_STARTUP_TIME", "STREAM_SETTLE_DAMPING", "STREAM_CONTACT_TRAVEL_FRACTION") else "Hydraulic Forcing & Duration")
            lf_stream = groups[title]
            row = rows[title]
            rows[title] += 1
            widget = self._add_row(lf_stream, row, label, self.stream_vars[name],
                                   "combo" if name in _STREAM_CHOICES else "check" if name == "SIMULATE_TURBULENCE" else "entry",
                                   _STREAM_CHOICES.get(name))
            self.stream_entries[name] = widget
            if name in _STREAM_CHOICES:
                widget.bind("<<ComboboxSelected>>", lambda e: self._on_turbulence_switch())
            if name == "SIMULATE_TURBULENCE":
                widget.configure(command=self._on_turbulence_switch)
        ttk.Label(tab_stream, text="Scale multiplies the box, reference cell and bed dimensions.\n"
                  "Example: 0.01 gives dimensions 100 times smaller.\n"
                  "Grain sizes stay in metres as entered; axes show the scaled dimensions.",
                  wraplength=370).pack(fill=tk.X, pady=5)
        ttk.Label(tab_stream, text="Harmonic stress = mean + amplitude × sin(2π × frequency × t).\n"
                  "Use amplitude ≤ mean to maintain downstream flow.\n"
                  "Force/width = stress × multiplier × exposed grain height.\n"
                  "Buoyancy, friction and contact moments govern grain motion.\n"
                  "Velocity forcing: log-law or equilibrium 1DV RANS columns.\n"
                  "z0 = ks/30; ks is based on surface sediment D84.\n"
                  "Drag uses fluid minus grain application-point velocity.\n"
                  "Implicit drag and a smooth startup ramp stabilize initial loading.\n"
                  "Settling damping is applied only before flow time zero.\n"
                  "Turbulence has correlated space/time fluctuations scaled by u*.\n"
                  "Both solvers use hydrostatic, locally developed columns;\n"
                  "Neither resolves 2D/3D eddies or free-surface evolution.\n"
                  "Solver Settle Time runs before the flow clock starts.\n"
                  "Stress is referenced to the initial depth.\n"
                  "The stress multiplier and lift ratio require calibration.",
                  wraplength=370).pack(fill=tk.X, pady=8)

        # ── Tab 2: Fracture & Particles ──
        tab_frac = self._scrollable_parameter_tab(nb, "Particles / DFN")

        lf_part = ttk.LabelFrame(tab_frac, text="Discretization & Packing", padding=6)
        lf_part.pack(fill=tk.X, pady=4)
        self._add_row(lf_part, 0, "Fracture Mode:", self.var_fracture_mode, "combo", ["DFN", "grid"])
        self._add_row(lf_part, 1, "Particle Shape:", self.var_particle_shape, "combo", ["rectangle", "circle", "mixed"])
        self._add_row(lf_part, 2, "a-axis / Diameter(s) (m):", self.var_particle_size).configure(width=24)
        self._add_row(lf_part, 3, "b-axis Width (m):", self.var_particle_b)
        self._add_row(lf_part, 4, "c-axis Height (m, opt):", self.var_particle_c)
        self._add_row(lf_part, 5, "Rect Arrangement:", self.var_rect_arr, "combo", ["uniform", "mixed"])
        self._add_row(lf_part, 6, "Target Porosity (0-1):", self.var_porosity)
        packing_row = ttk.Frame(lf_part)
        packing_row.grid(row=7, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(packing_row, text="Gravity Packed", variable=self.var_gravity_packed,
                        command=lambda: self._on_packing_switch("gravity")).pack(side=tk.LEFT)
        ttk.Checkbutton(packing_row, text="Fill Void", variable=self.var_fill_void,
                        command=lambda: self._on_packing_switch("fill")).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Checkbutton(packing_row, text="Hotstart Pack", variable=self.var_hotstart_pack,
                        command=lambda: self._on_packing_switch("hotstart")).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(lf_part, text="One size: .005 (all grains the same size)\n"
                  "D16,D50,D84: .002,.005,.008 (percentiles by count)\n"
                  "Equal counts: .002; .004 (half of each size)\n"
                  "Multiple sizes use grid packing; rectangle axes scale together.",
                  wraplength=370).grid(row=8, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        lf_physics = ttk.LabelFrame(lf_part, text="Particle Physics", padding=6)
        lf_physics.grid(row=9, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        for name, (label, _) in _PARTICLE_PHYSICS_FIELDS.items():
            ttk.Checkbutton(lf_physics, text=label,
                            variable=self.particle_physics_vars[name],
                            command=self._on_param_modified).pack(anchor="w")
        ttk.Label(lf_physics, text="Applies to the next run; coefficients are retained.\n"
                  "Lift also requires a nonzero lift ratio.\n"
                  "Cohesion also requires a cohesive material.",
                  wraplength=340).pack(fill=tk.X, pady=(4, 0))

        tab_mix = ttk.Frame(nb, padding=6)
        nb.add(tab_mix, text="Mixture")
        ttk.Label(tab_mix, text="Select Particle Shape = mixed in Particles / DFN.\n"
                  "Random packing replaces DFN for this mode.\n"
                  "Fraction 0.3 = 30% circles and 70% rectangles by count.\n"
                  "Rectangle sizes use the a/b/c axes on the Particles tab.\n"
                  "Angles are random in XZ; circles have no visible orientation.",
                  wraplength=350).pack(fill=tk.X, pady=8)
        lf_mix = ttk.LabelFrame(tab_mix, text="Mixed initial conditions", padding=6)
        lf_mix.pack(fill=tk.X, pady=4)
        for row, (name, (label, _)) in enumerate(_MIXED_FIELDS.items()):
            self._add_row(lf_mix, row, label, self.mix_vars[name])
        ttk.Label(tab_mix, text="Count 0 estimates the assembly from target porosity.\n"
                  "Dense targets may produce a looser packing; achieved porosity\n"
                  "is reported. An explicit count must fit or raises an error.\n"
                  "Range 180 gives uniform random rectangle orientations;\n"
                  "range 0 fixes the angle. Size variation scales all axes.",
                  wraplength=350).pack(fill=tk.X, pady=8)

        lf_dfn = ttk.LabelFrame(tab_frac, text="DFN Joint Network Settings", padding=6)
        lf_dfn.pack(fill=tk.X, pady=4)
        self._add_row(lf_dfn, 0, "Random Seed:", self.var_dfn_seed)
        self._add_row(lf_dfn, 1, "Line Jitter (frac):", self.var_dfn_jitter)
        self._add_row(lf_dfn, 2, "Min Block Area (m²):", self.var_dfn_min_area)

        lf_js1 = ttk.LabelFrame(tab_frac, text="Joint Set 1", padding=4)
        lf_js1.pack(fill=tk.X, pady=2)
        self._add_row(lf_js1, 0, "Dip Angle (°):", self.var_js1_dip)
        self._add_row(lf_js1, 1, "Mean Spacing (m):", self.var_js1_spac)
        self._add_row(lf_js1, 2, "Persistence (m):", self.var_js1_pers)
        self._add_row(lf_js1, 3, "Persistence Std (m):", self.var_js1_pstd)

        lf_js2 = ttk.LabelFrame(tab_frac, text="Joint Set 2", padding=4)
        lf_js2.pack(fill=tk.X, pady=2)
        self._add_row(lf_js2, 0, "Enable Joint Set 2:", self.var_js2_enable, "check")
        self._add_row(lf_js2, 1, "Dip Angle (°):", self.var_js2_dip)
        self._add_row(lf_js2, 2, "Mean Spacing (m):", self.var_js2_spac)
        self._add_row(lf_js2, 3, "Persistence (m):", self.var_js2_pers)
        self._add_row(lf_js2, 4, "Persistence Std (m):", self.var_js2_pstd)

        # ── Tab 3: Material Properties ──
        tab_mat = ttk.Frame(nb, padding=6)
        nb.add(tab_mat, text="Material")

        lf_mat = ttk.LabelFrame(tab_mat, text="Geotechnical Properties", padding=6)
        lf_mat.pack(fill=tk.X, pady=4)
        cb_mat = self._add_row(lf_mat, 0, "Material Type:", self.var_mat_type, "combo", ["granular", "cohesive"])
        cb_mat.bind("<<ComboboxSelected>>", self._on_material_type_change)
        self._add_row(lf_mat, 1, "Particle Density (kg/m³):", self.var_density)
        self._add_row(lf_mat, 2, "Friction Coeff tan(φ):", self.var_friction)
        self._add_row(lf_mat, 3, "Cohesion Force (N):", self.var_cohesion)
        self.cohesion_label = lf_mat.grid_slaves(row=3, column=0)[0]
        self._add_row(lf_mat, 4, "Bond Break Mult (×d):", self.var_bond_mult)
        self._add_row(lf_mat, 5, "Restitution (0-1):", self.var_restitution)
        self.auto_cohesion_entry = self._add_row(lf_mat, 6, "Auto-raise Cohesion for FS≥1:", self.var_auto_stabilize, "check")

        btn_stab = ttk.Button(tab_mat, text="Compute Min Cohesion for Initial Stability (FS ≥ 1)",
                              command=self._check_initial_stability_gui)
        self.btn_stability = btn_stab
        btn_stab.pack(fill=tk.X, pady=8)

        self.lbl_stab_info = ttk.Label(tab_mat, text="", justify=tk.LEFT, wraplength=380, foreground="#1b5e20")
        self.lbl_stab_info.pack(fill=tk.X, padx=4, pady=4)

        # ── Tab 4: Tilt & Solver ──
        tab_sim = self._scrollable_parameter_tab(nb, "Solver")

        lf_tilt = ttk.LabelFrame(tab_sim, text="Tilting Schedule", padding=6)
        lf_tilt.pack(fill=tk.X, pady=4)
        self.tilt_entries = [self._add_row(lf_tilt, 0, "Total Rotation (°):", self.var_total_rot),
                             self._add_row(lf_tilt, 1, "Rotation Rate (°/s):", self.var_rot_rate)]
        self._add_row(lf_tilt, 2, "Settle Time (s):", self.var_settle_time)

        lf_solv = ttk.LabelFrame(tab_sim, text="Time-Stepping & Physics", padding=6)
        lf_solv.pack(fill=tk.X, pady=4)
        self._add_row(lf_solv, 0, "Maximum / fixed DT (s):", self.var_dt)
        self._add_row(lf_solv, 1, "Solver Iterations:", self.var_solver_iters)
        self._add_row(lf_solv, 2, "Gravity (m/s²):", self.var_gravity)
        for row, (name, (label, default)) in enumerate(_SOLVER_FIELDS.items(), start=3):
            self._add_row(lf_solv, row, label, self.solver_vars[name],
                          "check" if isinstance(default, bool) else "combo" if name == "CHRONO_SOLVER" else "entry",
                          ["PSOR", "APGD", "BARZILAIBORWEIN"] if name == "CHRONO_SOLVER" else None)
        ttk.Label(tab_sim, text="Adaptive DT bounds grain translation and rotational surface travel.\n"
                  "It also resolves flow/turbulence and explicit drag response in stream mode.\n"
                  "DT is the maximum; recording can use steps or nearest second targets.\n"
                  "NSC impacts still require timestep/iteration convergence checks.", wraplength=370).pack(fill=tk.X, pady=5)

        lf_out = ttk.LabelFrame(tab_sim, text="Animation & Recording", padding=6)
        lf_out.pack(fill=tk.X, pady=4)
        self.frame_steps_entry = self._add_row(lf_out, 0, "Frame interval (steps):", self.var_frame_int)
        self.recording_entries = {}
        for row, (name, (label, default)) in enumerate(_RECORDING_FIELDS.items(), start=1):
            self.recording_entries[name] = self._add_row(lf_out, row, label, self.recording_vars[name],
                "combo" if name == "FRAME_INTERVAL_MODE" else "entry",
                ["steps", "seconds"] if name == "FRAME_INTERVAL_MODE" else None)
        self.recording_entries["FRAME_INTERVAL_MODE"].bind("<<ComboboxSelected>>", self._update_recording_controls)
        self._add_row(lf_out, 5, "Animation FPS:", self.var_fps)
        self._add_row(lf_out, 6, "Default Video File:", self.var_video_file)
        self._add_row(lf_out, 7, "Auto-save MP4 on Run:", self.var_auto_video, "check")
        ttk.Label(tab_sim, text="Recording window uses simulation seconds (flow seconds in Stream Bed).\n"
                  "The full calculation still runs. Second targets select the nearest\n"
                  "existing solver state; midpoint ties select the earlier state.\n"
                  "Both interval values are saved; FPS sets playback speed.", wraplength=370).pack(fill=tk.X, pady=5)
        self._update_recording_controls()

        # ── Action Buttons at Bottom of Left Panel ──
        act_frame = ttk.LabelFrame(parent, text="Simulation Control", padding=6)
        act_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=4, before=nb)

        btn_row1 = ttk.Frame(act_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        self.btn_apply_bottom = ttk.Button(
            btn_row1, text="💾 Save Changes", command=self.apply_model_parameters
        )
        self.btn_apply_bottom.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_preview = ttk.Button(btn_row1, text="🔍 Preview", command=self.preview_geometry)
        self.btn_preview.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_run = ttk.Button(btn_row1, text="▶ Run Simulation", style="Run.TButton", command=self.start_simulation)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        btn_row2 = ttk.Frame(act_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        self.btn_stop = ttk.Button(btn_row2, text="⏹ Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.btn_reset = ttk.Button(btn_row2, text="↺ Reset Defaults", command=self.reset_defaults)
        self.btn_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progressbar = ttk.Progressbar(act_frame, variable=self.progress_var, maximum=100.0)
        self.progressbar.pack(fill=tk.X, pady=(6, 2))
        self.lbl_calculation_progress = ttk.Label(act_frame, text="Calculation progress: ready", wraplength=400)
        self.lbl_calculation_progress.pack(anchor="w", pady=(0, 2))

        self.lbl_status = ttk.Label(act_frame, text="Ready — Preview geometry or click Run Simulation.", font=("Segoe UI", 9))
        self.lbl_status.pack(anchor="w")

        for canvas, content in self._parameter_scrolls:
            def bind_wheel(widget, surface=canvas):
                def scroll(event):
                    surface.yview_scroll(-int(event.delta/120), "units")
                    return "break"
                widget.bind("<MouseWheel>", scroll, add="+")
                for child in widget.winfo_children():
                    bind_wheel(child, surface)
            bind_wheel(content)

    def _scrollable_parameter_tab(self, notebook, title):
        """Keep long parameter panels usable above the pinned run controls."""
        outer = self.ttk.Frame(notebook)
        notebook.add(outer, text=title)
        canvas = self.tk.Canvas(outer, highlightthickness=0, width=400, height=300)
        scrollbar = self.ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        canvas.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        content = self.ttk.Frame(canvas, padding=6)
        item = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta/120), "units"))
        if not hasattr(self, "_parameter_scrolls"):
            self._parameter_scrolls = []
        self._parameter_scrolls.append((canvas, content))
        return content

    def _build_right_panel(self, parent):
        tk = self.tk
        ttk = self.ttk

        # Top toolbar for Analysis & Export Tools
        tools_frame = ttk.LabelFrame(parent, text="Analysis, Playback & Export Tools", padding=6)
        tools_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        # Row 1: Playback controls & frame scrubber
        play_row = ttk.Frame(tools_frame)
        play_row.pack(fill=tk.X, pady=2)

        self.btn_play = ttk.Button(play_row, text="⏯ Play / Pause", width=14, command=self.toggle_playback, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_jump_fail = ttk.Button(play_row, text="⚡ Jump to Failure", command=self.jump_to_failure, state=tk.DISABLED)
        self.btn_jump_fail.pack(side=tk.LEFT, padx=4)

        ttk.Label(play_row, text="Frame:").pack(side=tk.LEFT, padx=(8, 2))
        self.scale_frame = ttk.Scale(play_row, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_slider_move)
        self.scale_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.lbl_frame_info = ttk.Label(play_row, text="0 / 0 (t=0.00s, θ=0.0°)", width=28)
        self.lbl_frame_info.pack(side=tk.LEFT, padx=4)

        # Row 2: Visualisation options & Export buttons
        opt_row = ttk.Frame(tools_frame)
        opt_row.pack(fill=tk.X, pady=(4, 0))

        ttk.Label(opt_row, text="Color By:").pack(side=tk.LEFT, padx=(0, 3))
        cb_col = ttk.Combobox(opt_row, textvariable=self.var_color_mode,
                              values=["auto", "displacement", "speed", "size"], state="readonly", width=13)
        cb_col.pack(side=tk.LEFT, padx=(0, 8))
        cb_col.bind("<<ComboboxSelected>>", lambda e: self.redraw_current_frame())

        chk_rot = ttk.Checkbutton(opt_row, text="Rotate View w/ Gravity", variable=self.var_rotate_view,
                                  command=self.redraw_current_frame)
        chk_rot.pack(side=tk.LEFT, padx=6)

        chk_quiv = ttk.Checkbutton(opt_row, text="Velocity Vectors", variable=self.var_show_quiver,
                                   command=self.redraw_current_frame)
        chk_quiv.pack(side=tk.LEFT, padx=6)

        chk_tr = ttk.Checkbutton(opt_row, text="DFN Traces", variable=self.var_show_traces,
                                 command=self.redraw_current_frame)
        chk_tr.pack(side=tk.LEFT, padx=6)

        # Export buttons aligned right
        ttk.Button(opt_row, text="💾 Save Config JSON", command=self.save_config_json).pack(side=tk.RIGHT, padx=2)
        ttk.Button(opt_row, text="📂 Load Config", command=self.load_config_json).pack(side=tk.RIGHT, padx=2)
        self.btn_exp_mp4 = ttk.Button(opt_row, text="🎬 Export MP4", command=self.export_mp4, state=tk.DISABLED)
        self.btn_exp_mp4.pack(side=tk.RIGHT, padx=2)
        self.btn_exp_png = ttk.Button(opt_row, text="🖼 Save Plot PNG", command=self.export_png)
        self.btn_exp_png.pack(side=tk.RIGHT, padx=2)
        self.btn_exp_csv = ttk.Button(opt_row, text="📊 Export CSV", command=self.export_csv, state=tk.DISABLED)
        self.btn_exp_csv.pack(side=tk.RIGHT, padx=2)
        self.btn_exp_txt = ttk.Button(opt_row, text="📄 Save Report", command=self.export_report, state=tk.DISABLED)
        self.btn_exp_txt.pack(side=tk.RIGHT, padx=2)

        # Main Notebook on Right: Tab 1 = Interactive Dashboard Plot, Tab 2 = Deep Analysis & Summary Report
        self.right_nb = ttk.Notebook(parent)
        self.right_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        tab_plot = ttk.Frame(self.right_nb)
        tab_report = ttk.Frame(self.right_nb)
        self.right_nb.add(tab_plot, text="📈 Interactive Simulation & Diagnostics Plot")
        self.right_nb.add(tab_report, text="📋 Model Summary & Particle Inspector")

        # Matplotlib Figure in Tab 1
        self.fig = self.Figure(figsize=(11, 7.2), dpi=100)
        self.canvas = self.FigureCanvasTkAgg(self.fig, master=tab_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(tab_plot)
        toolbar_frame.pack(fill=tk.X)
        self.mpl_toolbar = self.NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.mpl_toolbar.update()

        # Summary & Inspector in Tab 2
        rep_paned = ttk.PanedWindow(tab_report, orient=tk.HORIZONTAL)
        rep_paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        rep_left = ttk.LabelFrame(rep_paned, text="Stability Analysis Summary & Console Report", padding=6)
        rep_right = ttk.LabelFrame(rep_paned, text="Frame Displacement Histogram & Block Size Distribution", padding=6)
        rep_paned.add(rep_left, weight=1)
        rep_paned.add(rep_right, weight=1)

        self.txt_report = self.scrolledtext.ScrolledText(rep_left, width=55, font=("Consolas", 9))
        self.txt_report.pack(fill=tk.BOTH, expand=True)

        self.fig_inspector = self.Figure(figsize=(5.5, 5.5), dpi=100)
        self.canvas_inspector = self.FigureCanvasTkAgg(self.fig_inspector, master=rep_right)
        self.canvas_inspector.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Synchronize GUI Variables to Module Globals ────────────────
    def _apply_gui_to_globals(self) -> bool:
        if self.is_running:
            return False
        global ENGINE, QUASI_2D, DOMAIN_X, DOMAIN_Y, DOMAIN_Z, CELL_SIZE
        global SLOPE_HEIGHT, SLOPE_LENGTH, SLOPE_BASE
        global FRACTURE_MODE, PARTICLE_SHAPE, PARTICLE_SIZE, PARTICLE_B_AXIS, PARTICLE_C_AXIS
        global RECT_ARRANGEMENT, TARGET_POROSITY, GRAVITY_PACKED, FILL_VOID, HOTSTART_PACK
        global DFN_SEED, DFN_LINE_JITTER, DFN_MIN_BLOCK_AREA, DFN_JOINT_SETS
        global MATERIAL_TYPE, PARTICLE_DENSITY, FRICTION_COEFF, COHESION_FORCE, BOND_BREAK_MULT, RESTITUTION
        global TOTAL_ROTATION, ROTATION_RATE, SETTLE_TIME, DT, SOLVER_ITERS, GRAVITY_MAG
        global FPS, FRAME_INTERVAL, VIDEO_FILE, AUTO_STABILIZE
        global STREAM_BED_SETUP, RECIRCULATION, PARTICLE_SIZE_INPUT

        try:
            STREAM_BED_SETUP = bool(self.var_stream_bed.get())
            RECIRCULATION = bool(self.var_recirculation.get())
            if STREAM_BED_SETUP:
                for name, var in self.stream_vars.items():
                    globals()[name] = _parse_control(var.get(), _STREAM_FIELDS[name][1])
            for name, var in self.solver_vars.items():
                globals()[name] = _parse_control(var.get(), _SOLVER_FIELDS[name][1])
            for name, var in self.particle_physics_vars.items():
                globals()[name] = _parse_control(var.get(), _PARTICLE_PHYSICS_FIELDS[name][1])
            ENGINE = self.var_engine.get().strip()
            QUASI_2D = bool(self.var_quasi2d.get())
            DOMAIN_X = max(0.1, float(self.var_domain_x.get()))
            DOMAIN_Y = max(0.01, float(self.var_domain_y.get()))
            DOMAIN_Z = max(0.1, float(self.var_domain_z.get()))
            CELL_SIZE = max(0.005, float(self.var_cell_size.get()))

            SLOPE_HEIGHT = max(0.05, float(self.var_slope_h.get()))
            SLOPE_LENGTH = max(0.05, float(self.var_slope_l.get()))
            SLOPE_BASE = float(self.var_slope_base.get())

            FRACTURE_MODE = self.var_fracture_mode.get().strip()
            PARTICLE_SHAPE = self.var_particle_shape.get().strip()
            size_text = self.var_particle_size.get().strip()
            distribution = parse_size_input(size_text)
            PARTICLE_SIZE = distribution.representative
            PARTICLE_SIZE_INPUT = size_text if distribution.mode != "single" else ""
            if FRACTURE_MODE.upper() == "DFN" and (PARTICLE_SHAPE == "mixed" or distribution.mode != "single"):
                FRACTURE_MODE = "grid"
                self.var_fracture_mode.set("grid")
            elif FRACTURE_MODE.upper() == "DFN" and PARTICLE_SHAPE != "rectangle":
                PARTICLE_SHAPE = "rectangle"
                self.var_particle_shape.set("rectangle")

            PARTICLE_B_AXIS = max(0.005, float(self.var_particle_b.get()))
            c_str = str(self.var_particle_c.get()).strip()
            PARTICLE_C_AXIS = float(c_str) if c_str else None
            RECT_ARRANGEMENT = self.var_rect_arr.get().strip()
            TARGET_POROSITY = min(max(float(self.var_porosity.get()), 0.0), 0.90)
            GRAVITY_PACKED = bool(self.var_gravity_packed.get())
            FILL_VOID = bool(self.var_fill_void.get())
            HOTSTART_PACK = bool(self.var_hotstart_pack.get())
            for name, value in self.hotstart_options.items():
                globals()[name] = value
            if HOTSTART_PACK:
                FILL_VOID = True
                self.var_fill_void.set(True)
            if FILL_VOID:
                GRAVITY_PACKED = True
                self.var_gravity_packed.set(True)

            for name, var in self.mix_vars.items():
                globals()[name] = int(var.get()) if isinstance(_MIXED_FIELDS[name][1], int) else var.get()

            DFN_SEED = int(self.var_dfn_seed.get())
            DFN_LINE_JITTER = max(0.0, float(self.var_dfn_jitter.get()))
            DFN_MIN_BLOCK_AREA = max(1e-6, float(self.var_dfn_min_area.get()))

            jsets = [{
                "dip_deg": float(self.var_js1_dip.get()),
                "spacing": max(0.01, float(self.var_js1_spac.get())),
                "persistence": max(0.01, float(self.var_js1_pers.get())),
                "persistence_std": max(0.0, float(self.var_js1_pstd.get())),
            }]
            if self.var_js2_enable.get():
                jsets.append({
                    "dip_deg": float(self.var_js2_dip.get()),
                    "spacing": max(0.01, float(self.var_js2_spac.get())),
                    "persistence": max(0.01, float(self.var_js2_pers.get())),
                    "persistence_std": max(0.0, float(self.var_js2_pstd.get())),
                })
            DFN_JOINT_SETS = jsets

            MATERIAL_TYPE = self.var_mat_type.get().strip()
            PARTICLE_DENSITY = max(10.0, float(self.var_density.get()))
            FRICTION_COEFF = max(0.0, float(self.var_friction.get()))
            COHESION_FORCE = 0.0 if MATERIAL_TYPE == "granular" else max(0.0, float(self.var_cohesion.get()))
            self.var_cohesion.set(COHESION_FORCE)
            BOND_BREAK_MULT = max(1.01, float(self.var_bond_mult.get()))
            RESTITUTION = min(max(float(self.var_restitution.get()), 0.0), 1.0)
            AUTO_STABILIZE = bool(self.var_auto_stabilize.get()) if hasattr(self, "var_auto_stabilize") else False

            TOTAL_ROTATION = max(0.0, float(self.var_total_rot.get()))
            ROTATION_RATE = max(0.05, float(self.var_rot_rate.get()))
            SETTLE_TIME = max(0.0, float(self.var_settle_time.get()))
            DT = float(self.var_dt.get())
            SOLVER_ITERS = max(5, int(self.var_solver_iters.get()))
            GRAVITY_MAG = max(0.1, float(self.var_gravity.get()))

            FPS = max(1, int(self.var_fps.get()))
            FRAME_INTERVAL = max(1, int(self.var_frame_int.get()))
            for name, var in self.recording_vars.items():
                value = var.get().strip()
                globals()[name] = (value if name == "FRAME_INTERVAL_MODE" else
                                   None if name == "ANIMATION_STOP_TIME" and not value else float(value))
            VIDEO_FILE = self.var_video_file.get().strip() or "hillslope_sim.mp4"

            _recompute_derived_quantities()
            self._update_stream_controls()
            return True
        except Exception as e:
            self.messagebox.showerror("Invalid Parameter Input", f"Please check your parameter values:\n{e}")
            return False

    def apply_model_parameters(self, update_preview: bool = True) -> bool:
        """
        Explicitly saves and commits all GUI parameter fields to the simulation
        module globals, clears geometry caches, recomputes derived quantities,
        and refreshes the initial slope geometry preview.
        """
        ok = self._apply_gui_to_globals()
        if not ok:
            if hasattr(self, "lbl_param_status"):
                self.lbl_param_status.config(
                    text="❌ Error applying parameters — please check input values",
                    foreground="#c62828"
                )
            return False

        if update_preview and not self.is_running:
            self.preview_geometry()

        if hasattr(self, "lbl_param_status"):
            self.lbl_param_status.config(
                text="✔ Model parameters saved & loaded into simulation engine",
                foreground="#2e7d32"
            )
        self.lbl_status.config(text="Model parameters saved and loaded into simulation engine.")
        return True

    def _apply_dict_to_vars(self, cfg: dict):
        """Populate GUI variables from a parameter dictionary."""
        self.var_stream_bed.set(bool(cfg.get("STREAM_BED_SETUP", False)))
        self.var_recirculation.set(bool(cfg.get("RECIRCULATION", False)))
        for name, var in self.stream_vars.items():
            var.set(cfg.get(name, _STREAM_FIELDS[name][1]))
        for name, var in self.solver_vars.items():
            var.set(cfg.get(name, _SOLVER_FIELDS[name][1]))
        for name, var in self.particle_physics_vars.items():
            var.set(_parse_control(cfg.get(name, _PARTICLE_PHYSICS_FIELDS[name][1]), True))
        for name, var in self.recording_vars.items():
            value = cfg.get(name, _RECORDING_FIELDS[name][1])
            var.set("" if value is None else value)
        self._update_recording_controls()
        for name, var in self.mix_vars.items():
            if name in cfg:
                var.set(cfg[name])
        if "ENGINE" in cfg: self.var_engine.set(cfg["ENGINE"])
        if "QUASI_2D" in cfg: self.var_quasi2d.set(bool(cfg["QUASI_2D"]))
        if "DOMAIN_X" in cfg: self.var_domain_x.set(cfg["DOMAIN_X"])
        if "DOMAIN_Y" in cfg: self.var_domain_y.set(cfg["DOMAIN_Y"])
        if "DOMAIN_Z" in cfg: self.var_domain_z.set(cfg["DOMAIN_Z"])
        if "CELL_SIZE" in cfg: self.var_cell_size.set(cfg["CELL_SIZE"])
        if "SLOPE_HEIGHT" in cfg: self.var_slope_h.set(cfg["SLOPE_HEIGHT"])
        if "SLOPE_LENGTH" in cfg: self.var_slope_l.set(cfg["SLOPE_LENGTH"])
        if "SLOPE_BASE" in cfg: self.var_slope_base.set(cfg["SLOPE_BASE"])
        if "FRACTURE_MODE" in cfg: self.var_fracture_mode.set(cfg["FRACTURE_MODE"])
        if "PARTICLE_SHAPE" in cfg: self.var_particle_shape.set(cfg["PARTICLE_SHAPE"])
        if "PARTICLE_SIZE" in cfg: self.var_particle_size.set(cfg["PARTICLE_SIZE"])
        if cfg.get("PARTICLE_SIZE_INPUT"): self.var_particle_size.set(cfg["PARTICLE_SIZE_INPUT"])
        if "PARTICLE_B_AXIS" in cfg and cfg["PARTICLE_B_AXIS"] is not None:
            self.var_particle_b.set(cfg["PARTICLE_B_AXIS"])
        if "PARTICLE_C_AXIS" in cfg:
            self.var_particle_c.set("" if cfg["PARTICLE_C_AXIS"] is None else str(cfg["PARTICLE_C_AXIS"]))
        if "RECT_ARRANGEMENT" in cfg: self.var_rect_arr.set(cfg["RECT_ARRANGEMENT"])
        if "TARGET_POROSITY" in cfg: self.var_porosity.set(cfg["TARGET_POROSITY"])
        if "GRAVITY_PACKED" in cfg: self.var_gravity_packed.set(bool(cfg["GRAVITY_PACKED"]))
        if "FILL_VOID" in cfg: self.var_fill_void.set(bool(cfg["FILL_VOID"]))
        if "HOTSTART_PACK" in cfg: self.var_hotstart_pack.set(bool(cfg["HOTSTART_PACK"]))
        for name in _HOTSTART_OPTIONS:
            if name in cfg:
                self.hotstart_options[name] = cfg[name]
        if "DFN_SEED" in cfg: self.var_dfn_seed.set(cfg["DFN_SEED"])
        if "DFN_LINE_JITTER" in cfg: self.var_dfn_jitter.set(cfg["DFN_LINE_JITTER"])
        if "DFN_MIN_BLOCK_AREA" in cfg: self.var_dfn_min_area.set(cfg["DFN_MIN_BLOCK_AREA"])
        if "DFN_JOINT_SETS" in cfg and isinstance(cfg["DFN_JOINT_SETS"], list) and len(cfg["DFN_JOINT_SETS"]) > 0:
            js1 = cfg["DFN_JOINT_SETS"][0]
            self.var_js1_dip.set(js1.get("dip_deg", 35.0))
            self.var_js1_spac.set(js1.get("spacing", 0.18))
            self.var_js1_pers.set(js1.get("persistence", 0.90))
            self.var_js1_pstd.set(js1.get("persistence_std", 0.20))
            self.var_js2_enable.set(len(cfg["DFN_JOINT_SETS"]) > 1)
            if len(cfg["DFN_JOINT_SETS"]) > 1:
                js2 = cfg["DFN_JOINT_SETS"][1]
                self.var_js2_dip.set(js2.get("dip_deg", 70.0))
                self.var_js2_spac.set(js2.get("spacing", 0.24))
                self.var_js2_pers.set(js2.get("persistence", 0.65))
                self.var_js2_pstd.set(js2.get("persistence_std", 0.15))
        if "MATERIAL_TYPE" in cfg: self.var_mat_type.set(cfg["MATERIAL_TYPE"])
        if "PARTICLE_DENSITY" in cfg: self.var_density.set(cfg["PARTICLE_DENSITY"])
        if "FRICTION_COEFF" in cfg: self.var_friction.set(cfg["FRICTION_COEFF"])
        if "COHESION_FORCE" in cfg: self.var_cohesion.set(cfg["COHESION_FORCE"])
        if "BOND_BREAK_MULT" in cfg: self.var_bond_mult.set(cfg["BOND_BREAK_MULT"])
        if "RESTITUTION" in cfg: self.var_restitution.set(cfg["RESTITUTION"])
        if "TOTAL_ROTATION" in cfg: self.var_total_rot.set(cfg["TOTAL_ROTATION"])
        if "ROTATION_RATE" in cfg: self.var_rot_rate.set(cfg["ROTATION_RATE"])
        if "SETTLE_TIME" in cfg: self.var_settle_time.set(cfg["SETTLE_TIME"])
        if "DT" in cfg: self.var_dt.set(cfg["DT"])
        if "SOLVER_ITERS" in cfg: self.var_solver_iters.set(cfg["SOLVER_ITERS"])
        if "GRAVITY_MAG" in cfg: self.var_gravity.set(cfg["GRAVITY_MAG"])
        if "FPS" in cfg: self.var_fps.set(cfg["FPS"])
        if "FRAME_INTERVAL" in cfg: self.var_frame_int.set(cfg["FRAME_INTERVAL"])
        if "VIDEO_FILE" in cfg: self.var_video_file.set(cfg["VIDEO_FILE"])
        if "AUTO_STABILIZE" in cfg: self.var_auto_stabilize.set(bool(cfg["AUTO_STABILIZE"]))
        self._update_stream_controls()

    def _on_preset_selected(self, _event=None):
        name = self.cb_preset.get()
        if not name or name not in PRESETS:
            return
        self._apply_dict_to_vars(PRESETS[name])
        self.apply_model_parameters(update_preview=True)
        self.lbl_status.config(text=f"Loaded and applied preset: {name}")

    def _on_material_type_change(self, _event=None):
        mtype = self.var_mat_type.get()
        if mtype == "granular":
            self.var_friction.set(0.80)
            self.var_cohesion.set(0.0)
        else:
            self.var_friction.set(0.60)
            if self.var_cohesion.get() == 0.0:
                self.var_cohesion.set(5.0)

    def _check_initial_stability_gui(self):
        if not self._apply_gui_to_globals():
            return
        if STREAM_BED_SETUP:
            self.lbl_stab_info.config(text="Stream-bed grains use submerged weight and contact force/moment balance. "
                                           "The hillslope cohesion pre-check does not apply.")
            return
        new_coh = _stabilize_initial_slope()
        self.var_cohesion.set(round(COHESION_FORCE, 4))
        beta = math.degrees(math.atan2(SLOPE_HEIGHT - SLOPE_BASE, SLOPE_LENGTH))
        phi = math.degrees(math.atan(_contact_parameters()[0]))
        msg = (f"Slope angle β = {beta:.1f}°, Friction angle φ = {phi:.1f}°\n"
               f"Material: {MATERIAL_TYPE.upper()} | Effective cohesion = {new_coh:.4f} N "
               f"(configured: {COHESION_FORCE:.4f} N)")
        self.lbl_stab_info.config(text=msg)

    # ── Preview Initial Slope Geometry ─────────────────────────────
    def preview_geometry(self):
        if self.is_running:
            return
        if not self._apply_gui_to_globals():
            return
        try:
            with _scaled_stream_geometry():
                self._preview_applied_geometry()
        except ValueError as exc:
            self.messagebox.showerror("Initial Geometry Error", str(exc))
            self.lbl_status.config(text=f"Initial geometry failed: {exc}")

    def _preview_applied_geometry(self):
        if GRAVITY_PACKED and FILL_VOID:
            self.lbl_status.config(text="Packing and filling voids below the target surface...")
            self.root.update_idletasks()

        try:
            if HOTSTART_PACK:
                backend = _initialize_backend(self._preview_hotstart_progress)
                specs = _snapshot_particles(backend)
                blocks = _snapshot_dfn_blocks(specs) if FRACTURE_MODE.upper() == "DFN" else None
            else:
                blocks = _fracture_domain_with_dfn() if FRACTURE_MODE.upper() == "DFN" else None
                specs = _particle_specs() if blocks is None else None
        except ValueError as exc:
            self.messagebox.showerror("Initial Packing Error", str(exc))
            self.lbl_status.config(text=f"Initial packing failed: {exc}")
            return

        self.fig.clf()
        ax = self.fig.add_subplot(111)

        outline_x = np.linspace(0, SLOPE_LENGTH, 60)
        outline_z = np.array([_slope_surface_z(x) for x in outline_x])

        if STREAM_BED_SETUP:
            ax.plot([0, DOMAIN_X], [SLOPE_BASE]*2, color="0.25", lw=2.5, label="Rigid substrate")
            ax.plot([0, 0], [SLOPE_BASE, DOMAIN_Z], color="0.25", lw=2.5)
        else:
            ax.plot([0, DOMAIN_X, DOMAIN_X, 0, 0], [0, 0, DOMAIN_Z, DOMAIN_Z, 0], color="0.25", lw=2.5, label="Domain Walls")
        ax.plot(outline_x, outline_z, "k--", lw=1.5, alpha=0.7,
                label="Target Bed Surface" if STREAM_BED_SETUP else "Initial Slope Surface")

        if FRACTURE_MODE.upper() == "DFN":
            areas = np.array([b["area"] for b in blocks]) if blocks else np.zeros(0)
            amax = float(areas.max()) if len(areas) else 1.0
            cmap = plt.colormaps.get_cmap("viridis")
            for b in blocks:
                verts = b["verts"]
                if len(verts) >= 3:
                    norm_a = b["area"] / max(amax, 1e-12)
                    poly = MplPolygon(verts, closed=True, facecolor=cmap(norm_a * 0.85 + 0.1),
                                      edgecolor="0.15", lw=0.6, alpha=0.88)
                    ax.add_patch(poly)
            n_elem = len(blocks)
            mode_str = f"DFN Fracture Network ({n_elem} polygonal blocks, {len(DFN_JOINT_SETS)} joint sets)"
        else:
            cmap = plt.colormaps.get_cmap("YlOrRd")
            for spec in specs:
                cx, cz, ang, sz = spec["x"], spec["z"], spec["angle"], spec["size"]
                c_size = spec["c"]
                color = "#4c9ed9" if PARTICLE_SHAPE == "mixed" and spec["shape"] == "circle" else cmap(0.35)
                if spec["shape"] == "circle":
                    p = MplCircle((cx, cz), sz / 2.0, facecolor=color, edgecolor="0.2", lw=0.4, alpha=0.85)
                elif "projection_local" in spec:
                    p = MplPolygon(spec["projection_local"] + [cx, cz], closed=True,
                                   facecolor=color, edgecolor="0.2", lw=.4, alpha=.85)
                else:
                    p = MplRect((cx - sz / 2.0, cz - c_size / 2.0), sz, c_size,
                                angle=math.degrees(ang), rotation_point="center",
                                facecolor=color, edgecolor="0.2", lw=0.4, alpha=0.85)
                ax.add_patch(p)
            n_elem = len(specs)
            mode_str = f"Particle Packing ({n_elem} {PARTICLE_SHAPE} particles)"

            if PARTICLE_SHAPE == "mixed":
                nc = sum(p["shape"] == "circle" for p in specs)
                mode_str = f"Random packing ({nc} circles, {n_elem - nc} rectangles)"
        if GRAVITY_PACKED:
            mode_str += " — Gravity Packed"
        if _FILL_VOID_INFO is not None:
            mode_str += f" + Fill Void ({_FILL_VOID_INFO['coverage']:.1%} coverage)"
        if _HOTSTART_INFO is not None:
            info = _HOTSTART_INFO
            mode_str = (f"Hotstart Pack: {n_elem} particles, area error {info['relative_error']:+.1%}, "
                        f"{'converged' if info['converged'] else 'NOT converged'}")
        if STREAM_BED_SETUP:
            from StreamBed import draw_water
            draw_water(ax, outline_x, outline_z, SLOPE_BASE, DOMAIN_Z, DOMAIN_X, 0.)
            surface_x = (np.arange(1024)+.5)*DOMAIN_X/1024
            profile = np.full(len(surface_x), SLOPE_BASE)
            preview_specs = specs if specs is not None else [
                dict(x=b["centroid"][0], z=b["centroid"][1], shape="rectangle",
                     world_local=np.asarray(b["verts"])-b["centroid"]) for b in blocks]
            for p in preview_specs:
                _update_packed_surface(profile, surface_x, p, *_packed_shape_geometry(p))
            inside = (surface_x >= _wall_inset()) & (surface_x <= DOMAIN_X-_wall_inset())
            mean_bed = float(profile[inside].mean()) if inside.any() else SLOPE_BASE
            ax.plot([DOMAIN_X, DOMAIN_X], [0, mean_bed], color="#394b59", lw=4, label="Mean-bed outlet barrier")
            ax.text(DOMAIN_X, SLOPE_BASE+.05*DOMAIN_Z,
                    "Recirculation → inlet" if RECIRCULATION else "Open outlet →", ha="right", fontsize=9)
            mode_str = f"Stream Bed (scale {STREAM_DOMAIN_SCALE:g}) — " + mode_str
        ax.set_xlim(-0.08 * DOMAIN_X, DOMAIN_X * 1.08)
        ax.set_ylim(-0.08 * DOMAIN_Z, DOMAIN_Z * 1.08)
        ax.set_aspect("equal")
        ax.set_xlabel("Domain X (m)")
        ax.set_ylabel("Domain Z (m)")
        ax.set_title(f"Initial Geometry Preview — {mode_str}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2, ls=":")

        self.canvas.draw_idle()
        status = f"Previewed initial {'stream bed' if STREAM_BED_SETUP else 'slope'}: {n_elem} elements generated."
        if _FILL_VOID_INFO is not None:
            status += (f" Fill Void added {_FILL_VOID_INFO['added_count']}; "
                       f"slope coverage {_FILL_VOID_INFO['coverage']:.1%} ({_FILL_VOID_INFO['stop_reason']}).")
        if _HOTSTART_INFO is not None:
            info = _HOTSTART_INFO
            status = (f"Hotstart Pack: {info['stop_reason']}. Settled area {info['area']:.4f} / "
                      f"{info['target_area']:.4f} m2 ({info['relative_error']:+.1%}); {n_elem} particles.")
        self.lbl_status.config(text=status)

    def _preview_hotstart_progress(self, state):
        if state["phase"] == "submerged settling":
            self.lbl_status.config(text=f"Stream bed: submerged settling — {state['particle_count']} grains")
            self.root.update_idletasks()
            return True
        self.lbl_status.config(text=f"Hotstart Pack {state['cycle']}/{state['max_cycles']}: "
                                    f"{state['phase']} — {state['particle_count']} particles")
        self.root.update_idletasks()
        return True

    # ── Run & Stop Simulation ──────────────────────────────────────
    def start_simulation(self):
        if self.is_running:
            return
        if not self.apply_model_parameters(update_preview=False):
            return

        self.stop_playback()
        self.is_running = True
        self._update_stream_controls()
        self.cancel_requested = False
        self.btn_run.config(state=self.tk.DISABLED)
        self.btn_preview.config(state=self.tk.DISABLED)
        self.btn_stop.config(state=self.tk.NORMAL)
        self.progress_var.set(0.0)
        self.progressbar.configure(mode="indeterminate" if STREAM_BED_SETUP else "determinate")
        if STREAM_BED_SETUP:
            self.progressbar.start(15)
        self.lbl_calculation_progress.config(text="Preparing grains; flow ETA will appear when calculation starts."
                                              if STREAM_BED_SETUP else "Calculation progress: 0%")
        estimate = ProgressEstimate()
        self.lbl_status.config(text="Initializing and settling submerged sediment..." if STREAM_BED_SETUP else
                                    "Initializing physics backend and stabilizing slope...")
        self.root.update_idletasks()

        def _progress_cb(step, n_steps, t, angle, pct, fs):
            pct_done = 100.0 * step / max(n_steps, 1)
            self.progress_var.set(pct_done)
            self.lbl_status.config(
                text=f"Running: {pct_done:5.1f}%  |  t={t:.2f}s  |  Tilt={angle:.1f}°  |  Moved={pct:.1f}%  |  FS={fs:.2f}"
            )
            if STREAM_BED_SETUP:
                self.progressbar.stop()
                self.progressbar.configure(mode="determinate")
                self.progress_var.set(pct_done)
                elapsed, eta = estimate.update(step, n_steps, time.perf_counter())
                self.lbl_calculation_progress.config(
                    text=f"Flow calculation: {pct_done:.1f}% | Elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}")
                self.lbl_status.config(text=f"Flow: {pct_done:.1f}% | t={t:.2f}s | "
                                            f"{STREAM_FORCING} forcing | mobile={pct:.1f}% | "
                                            f"turbulence {'on' if SIMULATE_TURBULENCE else 'off'}")
            self.root.update()
            return not self.cancel_requested

        def _initialization_cb(state):
            self._preview_hotstart_progress(state)
            self.root.update()
            return not self.cancel_requested

        try:
            res = run_simulation(
                progress_callback=_progress_cb,
                save_video=bool(self.var_auto_video.get()),
                return_data=True,
                initialization_callback=_initialization_cb,
            )
            self.sim_data = res
            n_frames = len(res["arr_time"])
            if n_frames > 0:
                self.scale_frame.config(from_=0, to=n_frames - 1)
                self.current_frame_idx = n_frames - 1
                self.scale_frame.set(self.current_frame_idx)
                self._setup_dashboard_axes()
                self.redraw_current_frame()
                self._update_inspector_tab()

                self.btn_play.config(state=self.tk.NORMAL)
                self.btn_jump_fail.config(state=self.tk.NORMAL)
                self.btn_jump_fail.config(text="Jump to First Mobility" if res.get("stream_bed_setup") else "⚡ Jump to Failure")
                self.btn_exp_mp4.config(state=self.tk.NORMAL)
                self.btn_exp_csv.config(state=self.tk.NORMAL)
                self.btn_exp_txt.config(state=self.tk.NORMAL)

                self.txt_report.delete("1.0", self.tk.END)
                self.txt_report.insert(self.tk.END, res["summary_text"])

                fail_idx = res["failure_idx"]
                if res.get("stream_bed_setup"):
                    self.var_engine.set(res["engine"])
                    state = "Stopped" if res.get("cancelled") else "Done"
                    self.lbl_status.config(text=f"{state}: {res.get('calculated_time', res['arr_time'][-1]):.2f}s of flow; "
                                                f"mobile {res['arr_pct'][-1]:.1f}%; exported {res['arr_exit'][-1]} grains.")
                    if res.get("recirculation"):
                        self.lbl_status.config(text=f"{state}: {res.get('calculated_time', res['arr_time'][-1]):.2f}s of flow; "
                                                    f"mobile {res['arr_pct'][-1]:.1f}%; recirculated {res['arr_recirculated'][-1]} passages.")
                    self.lbl_calculation_progress.config(
                        text=f"Calculation {state.lower()}: elapsed {format_duration(res['calculation_wall_seconds'])}"
                             + (" | ETA 00:00" if not res.get("cancelled") else ""))
                    if res.get("engine_fallback_reason"):
                        self.lbl_status.config(text=self.lbl_status.cget("text") + " PyChrono unavailable; used Pymunk (see report).")
                elif fail_idx is not None:
                    f_ang = res["arr_angle"][fail_idx]
                    f_t = res["arr_time"][fail_idx]
                    self.lbl_status.config(text=f"Done! FAILURE detected at tilt = {f_ang:.1f}° (t = {f_t:.2f}s).")
                else:
                    self.lbl_status.config(text=f"Done! Slope remained STABLE (max moved = {max(res['arr_pct']):.1f}%).")
        except HotstartCancelled:
            self.lbl_status.config(text="Initialization cancelled.")
            self.lbl_calculation_progress.config(text="Preparation cancelled.")
        except Exception as e:
            self.messagebox.showerror("Simulation Error", f"An error occurred during simulation:\n{e}")
            self.lbl_status.config(text=f"Error: {e}")
            self.lbl_calculation_progress.config(text="Calculation stopped because of an error.")
        finally:
            self.progressbar.stop()
            self.progressbar.configure(mode="determinate")
            self.is_running = False
            self._update_stream_controls()
            self.btn_run.config(state=self.tk.NORMAL)
            self.btn_preview.config(state=self.tk.NORMAL)
            self.btn_stop.config(state=self.tk.DISABLED)

    def stop_simulation(self):
        if self.is_running:
            self.cancel_requested = True
            self.lbl_status.config(text="Stopping simulation...")

    def reset_defaults(self):
        if self.is_running:
            return
        self.hotstart_options = dict(_HOTSTART_DEFAULTS)
        self._apply_dict_to_vars(PRESETS[DEFAULT_PRESET_NAME])
        self.cb_preset.set(DEFAULT_PRESET_NAME)
        self.apply_model_parameters(update_preview=True)
        self.lbl_status.config(text="Reset all model parameters to default configuration.")
        if hasattr(self, "lbl_param_status"):
            self.lbl_param_status.config(
                text="✔ Parameters reset to defaults & loaded into model",
                foreground="#2e7d32"
            )

    # ── Interactive Dashboard Plotting & Playback ──────────────────
    def _setup_dashboard_axes(self):
        """Build the multi-panel figure layout inside self.fig for fast frame scrubbing."""
        d = self.sim_data
        if not d:
            return
        if d.get("stream_bed_setup"):
            return  # Stream renderer owns its force/transport diagnostics.

        self.fig.clf()
        self.ax_main = self.fig.add_axes([0.05, 0.15, 0.53, 0.75])
        self.cbar_ax = self.fig.add_axes([0.595, 0.15, 0.015, 0.75])

        self.ax_disp    = self.fig.add_axes([0.66, 0.67, 0.14, 0.22])
        self.ax_fs_plot = self.fig.add_axes([0.83, 0.67, 0.14, 0.22])
        self.ax_coord   = self.fig.add_axes([0.66, 0.41, 0.14, 0.22])
        self.ax_ke_plot = self.fig.add_axes([0.83, 0.41, 0.14, 0.22])
        self.ax_fab     = self.fig.add_axes([0.66, 0.15, 0.14, 0.22])
        self.ax_touch   = self.fig.add_axes([0.83, 0.15, 0.14, 0.22])
        self.ax_mobility = self.fig.add_axes([0.05, 0.03, 0.53, 0.09])

        diag_axes = [self.ax_disp, self.ax_fs_plot, self.ax_coord, self.ax_ke_plot, self.ax_fab, self.ax_touch]
        for ax in diag_axes:
            ax.set_facecolor("white")
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
                sp.set_color("0.35")
            ax.grid(True, alpha=0.18, ls=":", lw=0.8)
            ax.tick_params(labelsize=7, width=0.8, length=3)

        arr_time = d["arr_time"]
        arr_angle = d["arr_angle"]
        arr_pct = d["arr_pct"]
        arr_fs = d["arr_fs"]
        arr_coord = d["arr_coord"]
        arr_ke = d["arr_ke"]
        arr_fa = d["arr_fa"]
        arr_touch = d["arr_touch"] * 100.0
        sim_dur = max(float(arr_time[-1]), 1.0) if len(arr_time) else d["sim_duration"]

        # Plot full curves lightly + active time cursor
        self.ax_disp.set_xlim(0, max(d["total_rotation"] + 5, 10))
        self.ax_disp.set_ylim(0, 105)
        self.ax_disp.set_ylabel("% moved", fontsize=7.5)
        self.ax_disp.set_title("Cumulative Movement vs Tilt", fontsize=8)
        self.ax_disp.axhline(50, color="red", ls="--", lw=0.8, alpha=0.5)
        self.ax_disp.plot(arr_angle, arr_pct, "#0b5fa5", lw=1.0, alpha=0.3)
        self.line_disp, = self.ax_disp.plot([], [], "#0b5fa5", lw=2.0)
        self.pt_disp, = self.ax_disp.plot([], [], "ro", ms=4)

        self.ax_fs_plot.set_xlim(0, sim_dur)
        self.ax_fs_plot.set_ylim(0, max(5.0, float(np.max(arr_fs)) * 1.1) if len(arr_fs) else 5.0)
        self.ax_fs_plot.set_ylabel("FS", fontsize=7.5)
        self.ax_fs_plot.set_title("Factor of Safety", fontsize=8)
        self.ax_fs_plot.axhline(1.0, color="red", ls="--", lw=0.8)
        self.ax_fs_plot.plot(arr_time, arr_fs, "#2e7d32", lw=1.0, alpha=0.3)
        self.line_fs, = self.ax_fs_plot.plot([], [], "#2e7d32", lw=2.0)
        self.pt_fs, = self.ax_fs_plot.plot([], [], "ro", ms=4)

        self.ax_coord.set_xlim(0, sim_dur)
        self.ax_coord.set_ylim(0, max(8.0, float(np.max(arr_coord)) * 1.2) if len(arr_coord) else 8.0)
        self.ax_coord.set_ylabel("Contacts", fontsize=7.5)
        self.ax_coord.set_title("Coordination Number", fontsize=8)
        self.ax_coord.plot(arr_time, arr_coord, "#8e24aa", lw=1.0, alpha=0.3)
        self.line_coord, = self.ax_coord.plot([], [], "#8e24aa", lw=2.0)
        self.pt_coord, = self.ax_coord.plot([], [], "ro", ms=4)

        self.ax_ke_plot.set_xlim(0, sim_dur)
        ke_max = float(np.max(arr_ke)) * 1.2 if len(arr_ke) and np.max(arr_ke) > 0 else 1.0
        self.ax_ke_plot.set_ylim(0, ke_max)
        self.ax_ke_plot.set_ylabel("KE (J)", fontsize=7.5)
        self.ax_ke_plot.set_title("Kinetic Energy", fontsize=8)
        self.ax_ke_plot.plot(arr_time, arr_ke, "#c62828", lw=1.0, alpha=0.3)
        self.line_ke, = self.ax_ke_plot.plot([], [], "#c62828", lw=2.0)
        self.pt_ke, = self.ax_ke_plot.plot([], [], "ro", ms=4)

        self.ax_fab.set_xlim(0, sim_dur)
        self.ax_fab.set_ylim(0, 1.05)
        self.ax_fab.set_xlabel("Time (s)", fontsize=7.5)
        self.ax_fab.set_ylabel("Anisotropy", fontsize=7.5)
        self.ax_fab.set_title("Fabric Anisotropy", fontsize=8)
        self.ax_fab.plot(arr_time, arr_fa, "#ef6c00", lw=1.0, alpha=0.3)
        self.line_fab, = self.ax_fab.plot([], [], "#ef6c00", lw=2.0)
        self.pt_fab, = self.ax_fab.plot([], [], "ro", ms=4)

        self.ax_touch.set_xlim(0, sim_dur)
        self.ax_touch.set_ylim(0, 105)
        self.ax_touch.set_xlabel("Time (s)", fontsize=7.5)
        self.ax_touch.set_ylabel("Contact (%)", fontsize=7.5)
        self.ax_touch.set_title("Touching Surface %", fontsize=8)
        self.ax_touch.plot(arr_time, arr_touch, "#00838f", lw=1.0, alpha=0.3)
        self.line_touch, = self.ax_touch.plot([], [], "#00838f", lw=2.0)
        self.pt_touch, = self.ax_touch.plot([], [], "ro", ms=4)

        # DFN mobility strip
        is_dfn = d["is_dfn_mode"]
        self.ax_mobility.set_xlim(0, sim_dur)
        self.ax_mobility.set_ylim(0, 105)
        self.ax_mobility.set_xlabel("Time (s)", fontsize=7.5)
        self.ax_mobility.set_ylabel("Moved (%)", fontsize=7.5)
        self.ax_mobility.set_title("DFN Block-Size Class Mobility", fontsize=8)
        self.ax_mobility.grid(True, alpha=0.2, ls=":")
        self.ax_mobility.tick_params(labelsize=7)
        self.mob_s_ln, = self.ax_mobility.plot([], [], color="#1f77b4", lw=1.6, label="Small")
        self.mob_m_ln, = self.ax_mobility.plot([], [], color="#ff7f0e", lw=1.6, label="Medium")
        self.mob_l_ln, = self.ax_mobility.plot([], [], color="#2ca02c", lw=1.6, label="Large")
        self.ax_mobility.legend(loc="upper left", ncol=3, fontsize=7, frameon=False)
        self.ax_mobility.set_visible(is_dfn)

        self.fig_title = self.fig.suptitle("", fontsize=11, fontweight="bold")

    def redraw_current_frame(self):
        d = self.sim_data
        if not d or len(d["arr_time"]) == 0:
            return
        if d.get("stream_bed_setup"):
            from StreamBed import draw_stream_frame
            idx = min(max(int(self.current_frame_idx), 0), len(d["arr_time"])-1)
            draw_stream_frame(self.fig, d, idx, self.var_color_mode.get(), self.var_show_quiver.get())
            self.lbl_frame_info.config(text=f"{idx+1} / {len(d['arr_time'])} (flow t={d['arr_time'][idx]:.2f}s)")
            self.canvas.draw_idle()
            return

        idx = min(max(int(self.current_frame_idx), 0), len(d["arr_time"]) - 1)
        n_frames = len(d["arr_time"])
        t = d["arr_time"][idx]
        angle = d["arr_angle"][idx]
        pos = d["frames_pos"][idx]
        vel = d["frames_vel"][idx] if idx < len(d["frames_vel"]) else np.zeros_like(pos)
        block_ang = d["frames_block_ang"][idx] if idx < len(d["frames_block_ang"]) else np.zeros(len(pos))
        init_pos = d["init_pos"]
        disp = compute_displacement(pos, init_pos) if len(pos) > 0 else np.zeros(0)
        sizes = d["sizes_static"]
        is_dfn = d["is_dfn_mode"]

        self.lbl_frame_info.config(text=f"{idx + 1} / {n_frames} (t={t:.2f}s, θ={angle:.1f}°)")

        # Clear & redraw main particle axes
        ax = self.ax_main
        ax.cla()

        dom_x, dom_z = d["domain_x"], d["domain_z"]
        cx, cz = dom_x / 2.0, dom_z / 2.0
        R = math.hypot(dom_x / 2.0, dom_z / 2.0)
        ax.set_xlim(cx - R - 0.1, cx + R + 0.1)
        ax.set_ylim(cz - R - 0.1, cz + R + 0.1)
        ax.set_aspect("equal")
        ax.set_xlabel("Domain X (m)", fontsize=8.5)
        ax.set_ylabel("Domain Z (m)", fontsize=8.5)

        rot_deg = -angle if self.var_rotate_view.get() else 0.0
        tr = matplotlib.transforms.Affine2D().rotate_deg_around(cx, cz, rot_deg)
        transform = tr + ax.transData

        # Outline & walls
        outline_x = np.linspace(0, d["slope_length"], 50)
        outline_z = np.array([d["slope_height"] - (x / max(d["slope_length"], 1e-6)) * (d["slope_height"] - d["slope_base"]) for x in outline_x])
        l1, = ax.plot(outline_x, outline_z, "k--", lw=1, alpha=0.45, label="Initial slope")
        l2, = ax.plot([0, 0], [0, d["slope_height"]], "k--", lw=1, alpha=0.45)
        l3, = ax.plot([0, d["slope_length"]], [0, 0], "k--", lw=1, alpha=0.45)
        w1, = ax.plot([0, dom_x], [0, 0], color="0.3", lw=3)
        w2, = ax.plot([0, 0], [0, dom_z], color="0.3", lw=3)
        w3, = ax.plot([dom_x, dom_x], [0, dom_z], color="0.3", lw=3)
        for el in (l1, l2, l3, w1, w2, w3):
            el.set_transform(transform)

        if is_dfn and self.var_show_traces.get():
            for b in d["dfn_blocks"]:
                verts = b.get("verts", [])
                if len(verts) >= 3:
                    xs = [v[0] for v in verts] + [verts[0][0]]
                    zs = [v[1] for v in verts] + [verts[0][1]]
                    ln, = ax.plot(xs, zs, color="0.45", lw=0.5, alpha=0.4)
                    ln.set_transform(transform)

        # Determine color metric
        cmode = self.var_color_mode.get()
        if cmode == "auto":
            cmode = "speed" if is_dfn else "displacement"

        speeds = np.linalg.norm(vel, axis=1) if len(vel) else np.zeros(len(pos))
        if cmode == "speed":
            metric_arr = speeds
            vmax = d.get("vel_mag_max")
            if vmax is None:
                all_vel = d.get("arr_vel")
                if all_vel is None or not isinstance(all_vel, np.ndarray):
                    all_vel = np.array(d.get("frames_vel", []))
                vmax = float(np.percentile(np.linalg.norm(all_vel.reshape(-1, 2), axis=1), 98)) if (all_vel is not None and getattr(all_vel, "size", 0) > 0) else 1.0
            vmax = max(float(vmax), 1e-3)
            cmap_name = "viridis"
            cbar_label = "Block Speed (m/s)"
        elif cmode == "size":
            metric_arr = sizes
            vmax = float(np.max(sizes)) if len(sizes) else 0.1
            cmap_name = "plasma"
            cbar_label = "Particle a-axis Size (m)"
        else:
            metric_arr = disp
            vmax = max(float(np.max(sizes)) * 5.0 if len(sizes) else 0.3, 0.3)
            cmap_name = "YlOrRd"
            cbar_label = "Displacement (m)"

        cmap = plt.colormaps.get_cmap(cmap_name)
        draw_lv = d["draw_local_vertices"]
        n_p = d["n_particles"]
        rect_w = float(np.mean(sizes)) if len(sizes) else 0.06
        rect_h = d["rect_c"]

        for i in range(n_p):
            val = metric_arr[i] if i < len(metric_arr) else 0.0
            col = cmap(min(val / max(vmax, 1e-9), 1.0))
            if draw_lv is not None and draw_lv[i] is not None:
                lv = draw_lv[i]
                ca, sa = math.cos(block_ang[i]), math.sin(block_ang[i])
                rot = np.array([[ca, -sa], [sa, ca]])
                world = lv @ rot.T + pos[i]
                p = MplPolygon(world, closed=True, facecolor=col, edgecolor="0.2", lw=0.4)
            elif d.get("particle_kinds", [d["particle_shape"]] * n_p)[i] == "circle":
                r_i = (sizes[i] if i < len(sizes) else rect_w) / 2.0
                p = MplCircle((pos[i, 0], pos[i, 1]), r_i, facecolor=col, edgecolor="0.2", lw=0.3)
            else:
                w_i = sizes[i] if i < len(sizes) else rect_w
                p = MplRect((pos[i, 0] - w_i / 2.0, pos[i, 1] - rect_h / 2.0), w_i, rect_h,
                            facecolor=col, edgecolor="0.2", lw=0.3)
            ax.add_patch(p)
            p.set_transform(transform)

        # Velocity vectors
        if self.var_show_quiver.get() and len(speeds) > 0 and np.max(speeds) > 1e-4:
            k = min(25, len(speeds))
            sel = np.argpartition(speeds, -k)[-k:]
            q = ax.quiver(pos[sel, 0], pos[sel, 1], vel[sel, 0], vel[sel, 1],
                          color="#08306b", alpha=0.75, width=0.0035, scale_units="xy", scale=1.0)
            q.set_transform(transform)

        # Gravity indicator arrow
        if self.var_rotate_view.get():
            gx_head, gz_head = cx + R, cz - R + 0.2
            gx_tail, gz_tail = cx + R, cz - R + 0.7
        else:
            rad = math.radians(angle)
            gx_tail, gz_tail = cx + R - 0.2, cz - R + 0.5
            gx_head = gx_tail + 0.45 * math.sin(rad)
            gz_head = gz_tail - 0.45 * math.cos(rad)
        ax.annotate("g", xy=(gx_head, gz_head), xytext=(gx_tail, gz_tail),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2.5),
                    fontsize=11, color="blue", fontweight="bold", ha="center", va="bottom")

        # Update colorbar
        self.cbar_ax.cla()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, cax=self.cbar_ax)
        cbar.set_label(cbar_label, fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5)

        # Update diagnostic time-series lines up to idx
        arr_t = d["arr_time"][:idx + 1]
        self.line_disp.set_data(d["arr_angle"][:idx + 1], d["arr_pct"][:idx + 1])
        self.pt_disp.set_data([d["arr_angle"][idx]], [d["arr_pct"][idx]])

        self.line_fs.set_data(arr_t, d["arr_fs"][:idx + 1])
        self.pt_fs.set_data([t], [d["arr_fs"][idx]])

        self.line_coord.set_data(arr_t, d["arr_coord"][:idx + 1])
        self.pt_coord.set_data([t], [d["arr_coord"][idx]])

        self.line_ke.set_data(arr_t, d["arr_ke"][:idx + 1])
        self.pt_ke.set_data([t], [d["arr_ke"][idx]])

        self.line_fab.set_data(arr_t, d["arr_fa"][:idx + 1])
        self.pt_fab.set_data([t], [d["arr_fa"][idx]])

        self.line_touch.set_data(arr_t, d["arr_touch"][:idx + 1] * 100.0)
        self.pt_touch.set_data([t], [d["arr_touch"][idx] * 100.0])

        if is_dfn:
            self.mob_s_ln.set_data(arr_t, d["arr_mob_small"][:idx + 1])
            self.mob_m_ln.set_data(arr_t, d["arr_mob_medium"][:idx + 1])
            self.mob_l_ln.set_data(arr_t, d["arr_mob_large"][:idx + 1])

        pct_now = d["arr_pct"][idx]
        fs_now = d["arr_fs"][idx]
        status = "STABLE" if pct_now < 50 and fs_now > 1.0 else "FAILURE"
        self.fig_title.set_text(
            f"t = {t:.2f} s  |  Tilt = {angle:.1f}°  |  Moved = {pct_now:.1f}%  |  "
            f"FS = {fs_now:.2f}  |  Coord# = {d['arr_coord'][idx]:.1f}  [{status}]"
        )

        self.canvas.draw_idle()

    def _update_inspector_tab(self):
        """Plot final displacement distribution and particle size distribution in Tab 2."""
        d = self.sim_data
        if not d or len(d["frames_pos"]) == 0:
            return

        self.fig_inspector.clf()
        ax1 = self.fig_inspector.add_subplot(211)
        ax2 = self.fig_inspector.add_subplot(212)

        final_pos = d["frames_pos"][-1]
        disp = (d["frames_displacement"][-1] if d.get("stream_bed_setup") and "frames_displacement" in d else
                compute_displacement(final_pos, d["init_pos"]) if len(final_pos) else np.zeros(0))
        sizes = d["sizes_static"]

        if len(disp) > 0:
            ax1.hist(disp, bins=min(25, max(5, len(disp) // 3)), color="#1f77b4", edgecolor="white", alpha=0.85)
            threshold = .5 if d.get("stream_bed_setup") else 2.
            ax1.axvline(threshold * np.mean(sizes), color="red", ls="--", lw=1.2,
                        label=f"Reference travel ({threshold:g} × mean size)")
            ax1.set_title("Final Particle Displacement Distribution", fontsize=9.5, fontweight="bold")
            ax1.set_xlabel("Total Displacement (m)", fontsize=8.5)
            ax1.set_ylabel("Count", fontsize=8.5)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.2, ls=":")

        if len(sizes) > 0:
            ax2.hist(sizes, bins=min(20, max(5, len(sizes) // 3)), color="#2e7d32", edgecolor="white", alpha=0.85)
            ax2.set_title("Block / Particle Size Distribution (a-axis)", fontsize=9.5, fontweight="bold")
            ax2.set_xlabel("Characteristic Length a (m)", fontsize=8.5)
            ax2.set_ylabel("Count", fontsize=8.5)
            ax2.grid(True, alpha=0.2, ls=":")

        self.fig_inspector.tight_layout()
        self.canvas_inspector.draw_idle()

    def _on_slider_move(self, val):
        if not self.sim_data:
            return
        new_idx = int(round(float(val)))
        if new_idx != self.current_frame_idx:
            self.current_frame_idx = new_idx
            self.redraw_current_frame()

    def toggle_playback(self):
        if not self.sim_data:
            return
        if self.is_playing:
            self.stop_playback()
        else:
            self.is_playing = True
            if self.current_frame_idx >= len(self.sim_data["arr_time"]) - 1:
                self.current_frame_idx = 0
            self._play_step()

    def stop_playback(self):
        self.is_playing = False
        if self._play_job is not None:
            try:
                self.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None

    def _play_step(self):
        if not self.is_playing or not self.sim_data:
            return
        n_frames = len(self.sim_data["arr_time"])
        if self.current_frame_idx < n_frames - 1:
            self.current_frame_idx += 1
            self.scale_frame.set(self.current_frame_idx)
            self.redraw_current_frame()
            delay = max(15, int(1000 / max(self.var_fps.get(), 1)))
            self._play_job = self.root.after(delay, self._play_step)
        else:
            self.stop_playback()

    def jump_to_failure(self):
        if not self.sim_data:
            return
        f_idx = self.sim_data.get("failure_idx")
        if f_idx is not None:
            self.current_frame_idx = f_idx
            self.scale_frame.set(f_idx)
            self.redraw_current_frame()
        else:
            self.messagebox.showinfo("No Failure Threshold Crossed",
                                     "No downstream grain mobility was recorded." if self.sim_data.get("stream_bed_setup") else
                                     "Cumulative particle movement did not exceed 50% during this simulation.")

    # ── Export & Save Tools ────────────────────────────────────────
    def export_csv(self):
        if not self.sim_data:
            return
        path = self.filedialog.asksaveasfilename(
            title="Export Diagnostics Time-Series CSV",
            defaultextension=".csv",
            initialfile="stream_bed_diagnostics.csv" if self.sim_data.get("stream_bed_setup") else "hillslope_diagnostics.csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            import csv
            d = self.sim_data
            if d.get("stream_bed_setup"):
                from StreamBed import export_stream_csv
                export_stream_csv(d, path)
                self.lbl_status.config(text=f"Saved stream-bed diagnostics CSV → {path}")
                return
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame", "time_s", "tilt_deg", "pct_moved", "factor_of_safety",
                    "coordination_num", "kinetic_energy_J", "fabric_anisotropy",
                    "mobilised_friction_deg", "bulk_density_kg_m3", "touching_surface_pct",
                    "dfn_small_moved_pct", "dfn_medium_moved_pct", "dfn_large_moved_pct"
                ])
                for i in range(len(d["arr_time"])):
                    writer.writerow([
                        i,
                        f"{d['arr_time'][i]:.4f}",
                        f"{d['arr_angle'][i]:.3f}",
                        f"{d['arr_pct'][i]:.3f}",
                        f"{d['arr_fs'][i]:.4f}",
                        f"{d['arr_coord'][i]:.4f}",
                        f"{d['arr_ke'][i]:.6e}",
                        f"{d['arr_fa'][i]:.4f}",
                        f"{d['arr_mp'][i]:.3f}",
                        f"{d['arr_bd'][i]:.2f}",
                        f"{d['arr_touch'][i] * 100.0:.3f}",
                        f"{d['arr_mob_small'][i]:.3f}",
                        f"{d['arr_mob_medium'][i]:.3f}",
                        f"{d['arr_mob_large'][i]:.3f}",
                    ])
            self.lbl_status.config(text=f"Saved diagnostics CSV → {path}")
        except Exception as e:
            self.messagebox.showerror("Export Error", f"Could not write CSV:\n{e}")

    def export_png(self):
        path = self.filedialog.asksaveasfilename(
            title="Save Current Plot Snapshot",
            defaultextension=".png",
            initialfile="hillslope_snapshot.png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Vector", "*.svg")],
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200)
            self.lbl_status.config(text=f"Saved plot figure → {path}")
        except Exception as e:
            self.messagebox.showerror("Export Error", f"Could not save figure:\n{e}")

    def export_report(self):
        if not self.sim_data:
            return
        path = self.filedialog.asksaveasfilename(
            title="Save Stability Analysis Summary Report",
            defaultextension=".txt",
            initialfile="stream_bed_summary_report.txt" if self.sim_data.get("stream_bed_setup") else "hillslope_summary_report.txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.sim_data["summary_text"] + "\n")
            self.lbl_status.config(text=f"Saved summary report → {path}")
        except Exception as e:
            self.messagebox.showerror("Export Error", f"Could not save report:\n{e}")

    def export_mp4(self):
        if not self.sim_data:
            return
        path = self.filedialog.asksaveasfilename(
            title="Export Animation Video (MP4)",
            defaultextension=".mp4",
            initialfile=self.var_video_file.get() or "hillslope_sim.mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("GIF Animation", "*.gif")],
        )
        if not path:
            return

        self.lbl_status.config(text=f"Rendering animation to {os.path.basename(path)} (please wait)...")
        self.root.update()
        try:
            d = self.sim_data
            n_frames = len(d["arr_time"])
            saved_idx = self.current_frame_idx

            def _anim_step(i):
                self.current_frame_idx = i
                self.redraw_current_frame()
                if i % 5 == 0:
                    self.lbl_status.config(text=f"Encoding video frame {i + 1}/{n_frames}...")
                    self.root.update_idletasks()
                return []

            anim = animation.FuncAnimation(self.fig, _anim_step, frames=n_frames, interval=1000 / max(FPS, 1), blit=False, repeat=False)
            if path.lower().endswith(".gif"):
                writer = animation.PillowWriter(fps=FPS)
            else:
                writer = animation.FFMpegWriter(fps=FPS, bitrate=2500)
            anim.save(path, writer=writer)
            self.current_frame_idx = saved_idx
            self.redraw_current_frame()
            self.lbl_status.config(text=f"Exported animation → {path}")
        except Exception as e:
            self.messagebox.showerror("Video Export Error", f"Could not export video (check ffmpeg installation):\n{e}")
            self.lbl_status.config(text=f"Video export failed: {e}")

    def save_config_json(self):
        if not self.apply_model_parameters(update_preview=False):
            return
        path = self.filedialog.asksaveasfilename(
            title="Save Parameter Configuration (JSON)",
            defaultextension=".json",
            initialfile="hillslope_config.json",
            filetypes=[("JSON Config", "*.json")],
        )
        if not path:
            return
        import json
        cfg = {
            "ENGINE": ENGINE, "QUASI_2D": QUASI_2D,
            "DOMAIN_X": DOMAIN_X, "DOMAIN_Y": DOMAIN_Y, "DOMAIN_Z": DOMAIN_Z, "CELL_SIZE": CELL_SIZE,
            "SLOPE_HEIGHT": SLOPE_HEIGHT, "SLOPE_LENGTH": SLOPE_LENGTH, "SLOPE_BASE": SLOPE_BASE,
            "FRACTURE_MODE": FRACTURE_MODE, "PARTICLE_SHAPE": PARTICLE_SHAPE,
            "PARTICLE_SIZE": PARTICLE_SIZE, "PARTICLE_B_AXIS": PARTICLE_B_AXIS, "PARTICLE_C_AXIS": PARTICLE_C_AXIS,
            "PARTICLE_SIZE_INPUT": PARTICLE_SIZE_INPUT, "RECIRCULATION": RECIRCULATION,
            "RECT_ARRANGEMENT": RECT_ARRANGEMENT, "TARGET_POROSITY": TARGET_POROSITY,
            "GRAVITY_PACKED": GRAVITY_PACKED,
            "FILL_VOID": FILL_VOID,
            "HOTSTART_PACK": HOTSTART_PACK,
            "DFN_SEED": DFN_SEED, "DFN_LINE_JITTER": DFN_LINE_JITTER, "DFN_MIN_BLOCK_AREA": DFN_MIN_BLOCK_AREA,
            "DFN_JOINT_SETS": DFN_JOINT_SETS,
            "MATERIAL_TYPE": MATERIAL_TYPE, "PARTICLE_DENSITY": PARTICLE_DENSITY,
            "FRICTION_COEFF": FRICTION_COEFF, "COHESION_FORCE": COHESION_FORCE,
            "BOND_BREAK_MULT": BOND_BREAK_MULT, "RESTITUTION": RESTITUTION,
            "TOTAL_ROTATION": TOTAL_ROTATION, "ROTATION_RATE": ROTATION_RATE, "SETTLE_TIME": SETTLE_TIME,
            "DT": DT, "SOLVER_ITERS": SOLVER_ITERS, "GRAVITY_MAG": GRAVITY_MAG,
            "FPS": FPS, "FRAME_INTERVAL": FRAME_INTERVAL, "VIDEO_FILE": VIDEO_FILE,
            "AUTO_STABILIZE": AUTO_STABILIZE,
        }
        cfg.update({name: globals()[name] for name in _MIXED_FIELDS})
        cfg.update({name: globals()[name] for name in _HOTSTART_OPTIONS})
        cfg.update({name: globals()[name] for name in _STREAM_FIELDS})
        cfg.update({name: globals()[name] for name in _SOLVER_FIELDS})
        cfg.update({name: globals()[name] for name in _PARTICLE_PHYSICS_FIELDS})
        cfg.update({name: globals()[name] for name in _RECORDING_FIELDS})
        cfg["STREAM_BED_SETUP"] = STREAM_BED_SETUP
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self.lbl_status.config(text=f"Saved parameter configuration → {path}")
        if hasattr(self, "lbl_param_status"):
            self.lbl_param_status.config(
                text=f"✔ Saved config to {os.path.basename(path)}",
                foreground="#2e7d32"
            )

    def load_config_json(self):
        path = self.filedialog.askopenfilename(
            title="Load Parameter Configuration (JSON)",
            filetypes=[("JSON Config", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._apply_dict_to_vars(cfg)
            self.apply_model_parameters(update_preview=True)
            self.lbl_status.config(text=f"Loaded configuration from {os.path.basename(path)}")
            if hasattr(self, "lbl_param_status"):
                self.lbl_param_status.config(
                    text=f"✔ Loaded & applied config from {os.path.basename(path)}",
                    foreground="#2e7d32"
                )
        except Exception as e:
            self.messagebox.showerror("Load Error", f"Could not load configuration:\n{e}")


def launch_gui():
    """Launch the interactive Tkinter GUI for HillslopeSim."""
    import tkinter as tk
    root = tk.Tk()
    app = HillslopeSimGUI(root)
    root.mainloop()


# =====================================================================
# 11. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    if "--cli" in sys.argv or "--no-gui" in sys.argv:
        run_simulation()
    else:
        launch_gui()
