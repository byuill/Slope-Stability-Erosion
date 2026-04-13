#!/usr/bin/env python3
"""
SlopeErosion.py — 3-D Finite-Difference Physics Model of Slope Stability
=========================================================================
A cube-domain model that resolves pore-water pressure, effective stress,
and slope stability (Factor of Safety) in a cohesive soil column subjected
to multiple erosion processes typical of river banks and marsh edges.

Erosion processes (individually toggleable)
-------------------------------------------
ORIGINAL:
 1. TIDAL WETTING/DRYING — Oscillating water stage drives transient seepage.
 2. WAVE EROSION — Wind-generated waves erode the exposed bank face.
 3. VESSEL WAKE — Impulsive wave groups from passing vessels.
 4. HYDRAULIC SHEAR — Lateral current shear erodes exposed soil.

SLOPE3D-INSPIRED:
 5. MULTILAYER SOIL — Stratified soil column with per-layer properties.
 6. SEISMIC (pseudo-static) — Horizontal & vertical acceleration loading.
 7. TENSION CRACKS — Tension crack formation at slope crest.
 8. SURCHARGE — External vertical load applied to the slope top.

XBEACH-INSPIRED:
 9. WAVE BREAKING — Depth-limited wave transformation (Baldock model).
10. INFRAGRAVITY WAVES — Bound long-wave modulation of waterline.
11. SEDIMENT TRANSPORT — Suspended + bed-load transport (Soulsby-Van Rijn).
12. AVALANCHING — Critical-angle slope collapse & mass redistribution.
13. VEGETATION — Wave attenuation by vegetation drag (Mendez-Losada).
14. GROUNDWATER EXCHANGE — Enhanced infiltration/exfiltration coupling.

All processes may be enabled simultaneously or in any combination.

Run
---
    python SlopeErosion.py
Outputs:
    slope_erosion.mp4  —  animated cross-section (XZ plane at mid-Y)
    slope_erosion.nc   —  NetCDF with full 4-D fields (optional)
"""

from __future__ import annotations

import sys
import subprocess
import math
import warnings
import os
from typing import Tuple

# Force UTF-8 console output on Windows to handle Unicode characters
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =====================================================================
# 0. Auto-install lightweight dependencies
# =====================================================================
_REQUIRED = {"numpy": "numpy", "matplotlib": "matplotlib"}
for _mod, _pkg in _REQUIRED.items():
    try:
        __import__(_mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib import cm

# =====================================================================
# 1. MAIN USER CONTROL BOARD & MODEL DOMAIN SETUP
# =====================================================================
# This section centralizes the top-level parameters that define the overall
# dimensions of the simulation space and toggle the active physical processes.

# -------------------------------------------------------------------
# 1.1. ACTIVE PHYSICS SWITCHES
# -------------------------------------------------------------------
# Toggle individual physical models ON (True) or OFF (False).
# The model architecture allows these to operate simultaneously, leading
# to complex, non-linear hydro-geomorphological interactions.

# --- Core Hydrodynamics & Surface Erosion ---
ENABLE_TIDAL           = True    # Tidal wetting/drying (stage oscillation + transient seepage)
ENABLE_WAVES           = False   # Wind-wave erosion on the exposed bank face
ENABLE_VESSEL_WAKE     = False   # Vessel-wake impulsive wave groups
ENABLE_HYDRAULIC_SHEAR = False   # Hydraulic shear from depth-averaged lateral current

# --- Advanced Geotechnics (Slope3D-Inspired) ---
ENABLE_MULTILAYER      = False   # Multi-layer soil with per-layer geomechanical properties
ENABLE_SEISMIC         = False   # Pseudo-static seismic loading (earthquake acceleration)
ENABLE_TENSION_CRACKS  = False   # Tension crack formation near slope crest (reduces cohesion)
ENABLE_SURCHARGE       = False   # External vertical surcharge load on the slope top

# --- Advanced Morphodynamics & Hydrodynamics (XBeach-Inspired) ---
ENABLE_WAVE_BREAKING   = False   # Depth-limited wave breaking & transformation (Baldock 1998)
ENABLE_INFRAGRAVITY    = False   # Infragravity (bound long-wave) modulation of the waterline
ENABLE_SEDIMENT_TRANSPORT = False  # Suspended & bed-load sediment transport (Soulsby-Van Rijn)
ENABLE_MORPHOLOGICAL_UPDATE = False  # Bed level updating based on sediment transport gradients
ENABLE_AVALANCHING     = False   # Critical-angle slope collapse & morphological redistribution
ENABLE_VEGETATION      = False   # Vegetation wave attenuation via cylinder drag (Mendez-Losada)
ENABLE_GROUNDWATER_EXCHANGE = False  # Enhanced groundwater <-> surface water exchange

# -------------------------------------------------------------------
# 1.2. SIMULATION TIME & EXPORT CONTROLS
# -------------------------------------------------------------------
SIM_DURATION   = 60.0     # Total simulation time (s)
DT             = 0.05     # Time step (s) — bounded explicitly by diffusion CFL condition
FPS            = 20       # Frames per second for animation output

EXPORT_NETCDF  = False
NETCDF_FILE    = "slope_erosion.nc"
VIDEO_FILE     = "slope_erosion.mp4"

# -------------------------------------------------------------------
# 1.3. MODEL DOMAIN SETUP & MACRO GEOMETRY
# -------------------------------------------------------------------
# Bounding box for the 3D simulation space (water, air, and soil).
DOMAIN_LENGTH = 1.0       # X-extent (m) — horizontal cross-shore distance
DOMAIN_WIDTH  = 0.1       # Y-extent (m) — along-shore distance (into the screen)
DOMAIN_HEIGHT = 1.0       # Z-extent (m) — vertical distance
CELL_SIZE     = 0.025     # Computational cell edge length (m) — uniform cubic grid

# Soil Column Footprint
SOIL_X_MIN    = 0.00      # Left limit of the soil mass (m)
SOIL_X_MAX    = 0.60      # Right limit of the soil mass (m)
SOIL_Y_MIN    = 0.00      # Front limit (m)
SOIL_Y_MAX    = 1.00      # Back limit (m)

# Exposed lateral boundary style (x = SOIL_X_MAX):
# 'wall'  -> Vertical bluff from SOIL_BOTTOM up to SOIL_TOP
# 'slope' -> Linear bank descending from SOIL_TOP (at X_MIN) to SOIL_BOTTOM (at X_MAX)
LATERAL_BOUNDARY = 'slope'

# =====================================================================
# 2. SECONDARY CONTROLS (PROCESS-SPECIFIC PARAMETERS)
# =====================================================================
# Detailed parameterization grouped by physical process. Each section
# documents the related scientific models and underlying assumptions.

# -------------------------------------------------------------------
# 2.1. BASE SOIL & FLUID PARAMETERS (Always Active)
# -------------------------------------------------------------------
# Defines the homogeneous cohesive soil matrix and ambient fluid.
# Science/Assumptions:
# - Mohr-Coulomb failure criterion: tau_f = c' + sigma_n' * tan(phi')
# - Terzaghi effective stress: sigma' = sigma_total - pore_pressure
# - Seepage follows Darcy's Law, constrained by Hydraulic Conductivity (K_s).
WATER_DENSITY = 1000.0    # Density of ambient fluid (kg/m^3) [Freshwater=1000, Seawater~1025]
GRAVITY       = 9.81      # Gravitational acceleration (m/s^2)

SOIL_TOP      = 0.80      # Elevation of soil surface above domain floor (m)
SOIL_BOTTOM   = 0.00      # Elevation of soil base (m)

BULK_DENSITY      = 1200.0    # Bulk (wet) density of the soil (kg/m^3) — typical marsh mud: 1050-1400
DRY_DENSITY       = 800.0     # Dry density (kg/m^3)
COHESION          = 3000.0    # Effective cohesion c' (Pa) — soft marsh soils typically 1-8 kPa
FRICTION_ANGLE    = 25.0      # Effective internal friction angle phi' (degrees) — typically 15-30
POROSITY          = 0.60      # Porosity n (0-1) — marsh peat ranges ~0.55-0.80
HYDRAULIC_COND    = 1.0e-5    # Saturated hydraulic conductivity K_s (m/s)
SPECIFIC_STORAGE  = 1.0e-4    # Specific storage S_s (1/m) — controls transient pressure response
SLOPE_ANGLE       = 30.0      # Local bank/slope angle alpha for infinite-slope FS calculation (degrees)

# -------------------------------------------------------------------
# 2.2. MULTILAYER SOIL STRATIGRAPHY (ENABLE_MULTILAYER)
# -------------------------------------------------------------------
# Science/Assumptions: Natural soils are highly stratified. This overrides
# the homogeneous properties above with a vertically stacked property field.
# Assumptions: Layers are perfectly horizontal, continuous, and ordered top-to-bottom.
SOIL_LAYERS = [
    {   # Layer 0: topsoil / organic root mat (highly cohesive, permeable)
        'z_top': 0.80,   'z_bottom': 0.50,
        'bulk_density': 1100.0,  'dry_density': 700.0,
        'cohesion': 4000.0,      'friction_angle': 20.0,
        'porosity': 0.65,        'hydraulic_cond': 5.0e-5,
        'specific_storage': 2.0e-4,
    },
    {   # Layer 1: intermediate silty clay
        'z_top': 0.50,   'z_bottom': 0.25,
        'bulk_density': 1300.0,  'dry_density': 850.0,
        'cohesion': 2500.0,      'friction_angle': 28.0,
        'porosity': 0.55,        'hydraulic_cond': 5.0e-6,
        'specific_storage': 8.0e-5,
    },
    {   # Layer 2: stiff basal clay (dense, impermeable, highly cohesive)
        'z_top': 0.25,   'z_bottom': 0.00,
        'bulk_density': 1500.0,  'dry_density': 1000.0,
        'cohesion': 6000.0,      'friction_angle': 30.0,
        'porosity': 0.45,        'hydraulic_cond': 1.0e-6,
        'specific_storage': 5.0e-5,
    },
]

# -------------------------------------------------------------------
# 2.3. TIDAL WETTING/DRYING & SEEPAGE (ENABLE_TIDAL)
# -------------------------------------------------------------------
# Science/Assumptions: A sinusoidal oscillating surface water boundary condition
# creates transient hydraulic gradients driving Darcian seepage into/out of the bank.
STAGE_MEAN      = 0.50    # Mean water surface elevation (m above domain floor)
STAGE_AMPLITUDE = 0.20    # Amplitude of stage oscillation (m)
STAGE_FREQUENCY = 0.05    # Oscillation frequency (Hz) — 0.05 Hz = 20s period (scaled for simulation)

# -------------------------------------------------------------------
# 2.4. HYDRODYNAMIC EROSION (ENABLE_WAVES, ENABLE_VESSEL_WAKE, ENABLE_HYDRAULIC_SHEAR)
# -------------------------------------------------------------------
# Science/Assumptions:
# - Waves: Energy flux P = (rho*g^2*Hs^2*Tp)/(64*pi). Volumetric erosion scales with P.
# - Hydraulic Shear: Excess shear model (Partheniades), E = M*(tau - tau_c).
#
# --- Wind Waves ---
WAVE_HEIGHT     = 0.05    # Significant wave height H_s (m)
WAVE_PERIOD     = 2.0     # Peak wave period T_p (s)
WAVE_EROSION_CE = 5e-5    # Erosion efficiency C_e (m^3/J) — volume eroded per unit wave energy
WAVE_NOTCH_HALFWIDTH = 0.05  # Vertical half-width of the wave impact zone (m)

# --- Vessel Wakes ---
VESSEL_WAKE_HEIGHT       = 0.10   # Wake wave height at the bank (m)
VESSEL_WAKE_PERIOD       = 1.5    # Dominant wake wave period (s)
VESSEL_PASSAGE_INTERVAL  = 20.0   # Time between vessel passages (s)
VESSEL_WAKE_DURATION     = 5.0    # Duration of wake wave train per event (s)
VESSEL_WAKE_CE           = 8e-5   # Wake erosion efficiency (m^3/J)

# --- Hydraulic Current Shear ---
CURRENT_VELOCITY  = 0.5   # Depth-averaged lateral current speed (m/s)
EROSION_RATE_COEF = 1e-4  # Erosion coefficient M (kg/m^2/s per Pa of excess shear)
CRITICAL_SHEAR    = 0.5   # Critical shear stress tau_c (Pa) for cohesive detachment

# -------------------------------------------------------------------
# 2.5. GEOTECHNICAL FORCING (ENABLE_SEISMIC, ENABLE_TENSION_CRACKS, ENABLE_SURCHARGE)
# -------------------------------------------------------------------
# Science/Assumptions:
# - Seismic: Pseudo-static inertial body forces parameterized by seismic coefficients (k_h, k_v).
# - Tension Cracks: Tensile failure at the crest computed via Rankine active earth pressure.
#   Overrides cohesion in the crack depth z_c = 2c' / (gamma * sqrt(K_a)).
SEISMIC_KH = 0.10    # Horizontal seismic coefficient (fraction of g)
SEISMIC_KV = 0.05    # Vertical seismic coefficient (fraction of g)

TENSION_CRACK_WATER_FILLED = True   # If True, hydrostatic pressure in the crack drives slope failure
TENSION_CRACK_DEPTH_OVERRIDE = None  # Hardcoded depth (m), or None to auto-compute analytically

SURCHARGE_LOAD   = 2000.0   # Vertical boundary stress (Pa) simulating infrastructure/vehicles (~2 kPa)
SURCHARGE_X_MIN  = 0.00     # Surcharge application start (m)
SURCHARGE_X_MAX  = 0.20     # Surcharge application end (m)

# -------------------------------------------------------------------
# 2.6. ADVANCED HYDRODYNAMICS (ENABLE_WAVE_BREAKING, ENABLE_INFRAGRAVITY)
# -------------------------------------------------------------------
# Science/Assumptions:
# - Wave Breaking: Baldock (1998) model limits max wave height in shallow water (H_max = gamma_b * h).
# - Infragravity: Long waves bound to wave groups induce low-frequency stage setup/setdown.
BREAKING_GAMMA    = 0.78   # Breaker index gamma_b (dimensionless)
NEARSHORE_SLOPE   = 0.02   # Local bed slope used for wave setup estimations

IG_PERIOD_FACTOR    = 5.0   # Long-wave period scaling (factor * short-wave T_p)
IG_AMPLITUDE_FACTOR = 0.15  # Long-wave amplitude scaling (factor * short-wave H_s)

# -------------------------------------------------------------------
# 2.7. MORPHODYNAMICS (ENABLE_SEDIMENT_TRANSPORT, ENABLE_AVALANCHING)
# -------------------------------------------------------------------
# Science/Assumptions:
# - Transport: Soulsby-Van Rijn total load formulation for currents.
# - Avalanching: Sand-slide cellular automata. Bed slopes exceeding angle-of-repose collapse instantaneously.
GRAIN_SIZE_D50    = 0.0001   # Median grain diameter d50 (m) — e.g., 0.1 mm = fine sand
GRAIN_SIZE_D90    = 0.0003   # d90 grain diameter (m)
SEDIMENT_DENSITY  = 2650.0   # Quartz sand grain density (kg/m^3)
TRANSPORT_CALIB   = 1.0      # Transport rate calibration multiplier
MORFAC            = 1.0      # Morphological Acceleration Factor (XBeach MORFAC) — scales bed updates

CRITICAL_SLOPE_DRY = 45.0    # Angle of repose above waterline (degrees)
CRITICAL_SLOPE_WET = 30.0    # Angle of repose below waterline (degrees)

# -------------------------------------------------------------------
# 2.8. ECO-HYDRAULICS & GROUNDWATER (ENABLE_VEGETATION, ENABLE_GROUNDWATER_EXCHANGE)
# -------------------------------------------------------------------
# Science/Assumptions:
# - Vegetation: Wave dissipation formulated via drag on rigid cylinders (Mendez & Losada 2004).
# - GW Exchange: Elevated inland head generates strong exfiltration gradients, reducing effective
#   stresses at the soil face and subsequently lowering the critical shear stress for detachment.
VEG_STEM_DIAMETER = 0.006    # Cylinder/stem diameter b_v (m)
VEG_STEM_DENSITY  = 400.0    # Stem density N_v (stems/m^2)
VEG_HEIGHT        = 0.30     # Plant height h_v (m)
VEG_DRAG_COEFF    = 1.0      # Bulk drag coefficient C_D
VEG_X_MIN         = 0.50     # Vegetation patch start X (m)
VEG_X_MAX         = 0.60     # Vegetation patch end X (m)

GW_HEAD_OFFSET      = 0.10   # Inland groundwater head above ambient stage (m)
GW_EXCHANGE_COEFF   = 1.0e-4 # Boundary exchange coefficient (m/s)
GW_EXFILTRATION_WEAKENING = 0.3  # Factor by which exfiltration reduces tau_c (0.0 - 1.0)

# =====================================================================
# 4. DERIVED QUANTITIES & GRID SETUP
# =====================================================================
# Number of cells in each direction
NX = max(1, int(round(DOMAIN_LENGTH / CELL_SIZE)))
NY = max(1, int(round(DOMAIN_WIDTH  / CELL_SIZE)))
NZ = max(1, int(round(DOMAIN_HEIGHT / CELL_SIZE)))

dx = DOMAIN_LENGTH / NX
dy = DOMAIN_WIDTH  / NY
dz = DOMAIN_HEIGHT / NZ

N_STEPS   = int(round(SIM_DURATION / DT))
FRAME_INT = max(1, int(round(1.0 / (FPS * DT))))  # steps between saved frames

# Cell-centre coordinates
x_cc = np.linspace(dx / 2, DOMAIN_LENGTH - dx / 2, NX)
y_cc = np.linspace(dy / 2, DOMAIN_WIDTH  - dy / 2, NY)
z_cc = np.linspace(dz / 2, DOMAIN_HEIGHT - dz / 2, NZ)

# Meshgrid for 3-D fields (index order: ix, iy, iz → shape NX×NY×NZ)
XX, YY, ZZ = np.meshgrid(x_cc, y_cc, z_cc, indexing='ij')

# Diffusivity for pressure equation:  D = K_s / S_s
DIFFUSIVITY = HYDRAULIC_COND / SPECIFIC_STORAGE

# CFL check  (3-D explicit FD stability: D·Δt·(1/dx²+1/dy²+1/dz²) ≤ 0.5)
_max_D = DIFFUSIVITY
if ENABLE_MULTILAYER:
    for lyr in SOIL_LAYERS:
        _lyr_D = lyr['hydraulic_cond'] / lyr['specific_storage']
        _max_D = max(_max_D, _lyr_D)

_cfl_1d  = _max_D * DT / min(dx, dy, dz)**2
_cfl_3d  = _cfl_1d * 3
_DT_STABLE = 0.45 * min(dx, dy, dz)**2 / (_max_D * 3)
_N_SUBSTEPS = max(1, math.ceil(DT / _DT_STABLE))
_dt_sub     = DT / _N_SUBSTEPS
if _cfl_3d > 0.5:
    warnings.warn(
        f"3-D CFL = {_cfl_3d:.3f} > 0.5 — using {_N_SUBSTEPS} diffusion substep(s) "
        f"(sub-Δt = {_dt_sub:.4f} s) for stability."
    )

_active_processes = []
if ENABLE_TIDAL:           _active_processes.append("Tidal wetting/drying")
if ENABLE_WAVES:           _active_processes.append("Wave erosion")
if ENABLE_VESSEL_WAKE:     _active_processes.append("Vessel wake")
if ENABLE_HYDRAULIC_SHEAR: _active_processes.append("Hydraulic shear")
if ENABLE_MULTILAYER:      _active_processes.append("Multi-layer soil")
if ENABLE_SEISMIC:         _active_processes.append("Seismic (pseudo-static)")
if ENABLE_TENSION_CRACKS:  _active_processes.append("Tension cracks")
if ENABLE_SURCHARGE:       _active_processes.append("Surcharge loading")
if ENABLE_WAVE_BREAKING:   _active_processes.append("Wave breaking")
if ENABLE_INFRAGRAVITY:    _active_processes.append("Infragravity waves")
if ENABLE_SEDIMENT_TRANSPORT: _active_processes.append("Sediment transport")
if ENABLE_AVALANCHING:     _active_processes.append("Avalanching")
if ENABLE_VEGETATION:      _active_processes.append("Vegetation damping")
if ENABLE_GROUNDWATER_EXCHANGE: _active_processes.append("Groundwater exchange")
if not _active_processes:
    _active_processes.append("(none — passive observation only)")

print("=" * 60)
print("SlopeErosion — 3-D Slope Stability Physics Model")
print("=" * 60)
print(f"Active : {', '.join(_active_processes)}")
print(f"Domain : {DOMAIN_LENGTH}x{DOMAIN_WIDTH}x{DOMAIN_HEIGHT} m")
print(f"Grid   : {NX}x{NY}x{NZ}  ({NX*NY*NZ:,} cells)  d={CELL_SIZE} m")
print(f"Soil   : x in [{SOIL_X_MIN:.2f}, {SOIL_X_MAX:.2f}]  y in [{SOIL_Y_MIN:.2f}, {SOIL_Y_MAX:.2f}]  "
      f"z in [{SOIL_BOTTOM:.2f}, {SOIL_TOP:.2f}] m   boundary={LATERAL_BOUNDARY}")
print(f"       : rho_bulk={BULK_DENSITY} kg/m3   c'={COHESION} Pa   phi'={FRICTION_ANGLE} deg")
print(f"Stage  : {STAGE_MEAN:.2f} +/- {STAGE_AMPLITUDE:.2f} m   f={STAGE_FREQUENCY} Hz")
print(f"Sim    : T={SIM_DURATION} s   dt={DT} s   3-D CFL={_cfl_3d:.4f}  "
      f"({_N_SUBSTEPS} diffusion substep(s))")
print(f"Output : {VIDEO_FILE}" + (f"  +  {NETCDF_FILE}" if EXPORT_NETCDF else ""))
print("=" * 60)

# =====================================================================
# 5. INITIAL FIELDS & GEOMETRY
# =====================================================================

# ---------------------------------------------------------------------------
# 3a. Build the initial soil mask from the footprint + lateral boundary.
# ---------------------------------------------------------------------------
CELL_AIR   = 0
CELL_WATER = 1
CELL_SOIL  = 2
CELL_FAIL  = 3

def _build_soil_surface_z(x_vals):
    """Return the soil surface elevation at each x position."""
    surf = np.full_like(x_vals, SOIL_BOTTOM, dtype=float)
    in_x = (x_vals >= SOIL_X_MIN) & (x_vals <= SOIL_X_MAX)
    if LATERAL_BOUNDARY == 'wall':
        surf[in_x] = SOIL_TOP
    else:  # 'slope'
        width = SOIL_X_MAX - SOIL_X_MIN
        if width <= 0:
            surf[in_x] = SOIL_TOP
        else:
            frac = (x_vals[in_x] - SOIL_X_MIN) / width
            surf[in_x] = np.clip(SOIL_TOP - frac * (SOIL_TOP - SOIL_BOTTOM),
                                  SOIL_BOTTOM, SOIL_TOP)
    return surf

# Soil surface elevation at every (ix, iy) column
soil_surf_xz = _build_soil_surface_z(x_cc)
# Broadcast to 3-D: shape (NX, NY, NZ)
in_x_footprint = (XX >= SOIL_X_MIN) & (XX <= SOIL_X_MAX)
in_y_footprint = (YY >= SOIL_Y_MIN) & (YY <= SOIL_Y_MAX)
soil_surf_3d = np.zeros((NX, NY, NZ))
for ix in range(NX):
    soil_surf_3d[ix, :, :] = soil_surf_xz[ix]

is_soil_init = (
    in_x_footprint & in_y_footprint
    & (ZZ >= SOIL_BOTTOM)
    & (ZZ <  soil_surf_3d)
)

# `is_soil` is the *dynamic* mask that can shrink as cells mobilise.
is_soil = is_soil_init.copy()

# ---------------------------------------------------------------------------
# 3b. Multi-layer soil property fields (Slope3D)
#     When ENABLE_MULTILAYER is True, per-cell arrays override homogeneous values.
# ---------------------------------------------------------------------------
cell_cohesion      = np.full((NX, NY, NZ), COHESION)
cell_friction_deg  = np.full((NX, NY, NZ), FRICTION_ANGLE)
cell_bulk_density  = np.full((NX, NY, NZ), BULK_DENSITY)
cell_dry_density   = np.full((NX, NY, NZ), DRY_DENSITY)
cell_porosity      = np.full((NX, NY, NZ), POROSITY)
cell_K_s           = np.full((NX, NY, NZ), HYDRAULIC_COND)
cell_S_s           = np.full((NX, NY, NZ), SPECIFIC_STORAGE)
cell_layer_id      = np.full((NX, NY, NZ), 0, dtype=np.int8)

if ENABLE_MULTILAYER:
    for lid, lyr in enumerate(SOIL_LAYERS):
        mask = is_soil & (ZZ >= lyr['z_bottom']) & (ZZ < lyr['z_top'])
        cell_cohesion[mask]     = lyr['cohesion']
        cell_friction_deg[mask] = lyr['friction_angle']
        cell_bulk_density[mask] = lyr['bulk_density']
        cell_dry_density[mask]  = lyr['dry_density']
        cell_porosity[mask]     = lyr['porosity']
        cell_K_s[mask]          = lyr['hydraulic_cond']
        cell_S_s[mask]          = lyr['specific_storage']
        cell_layer_id[mask]     = lid
    print(f"  Multi-layer: {len(SOIL_LAYERS)} layers assigned to {int(is_soil.sum())} soil cells")

# Diffusivity field (per-cell if multilayer)
cell_diffusivity = cell_K_s / cell_S_s

# cell_type initialised from geometry + initial stage
h0 = STAGE_MEAN
cell_type = np.where(is_soil, CELL_SOIL,
            np.where(ZZ <= h0, CELL_WATER, CELL_AIR)).astype(np.int8)

def _open_domain(ct):
    return ct != CELL_SOIL

# ---------------------------------------------------------------------------
# 3c. Pore-water pressure (Pa) — hydrostatic initialisation inside soil.
# ---------------------------------------------------------------------------
pore_pressure = np.where(
    (is_soil) & (ZZ <= h0),
    WATER_DENSITY * GRAVITY * (h0 - ZZ),
    np.where(is_soil,
             -WATER_DENSITY * GRAVITY * (ZZ - h0) * 0.1,
             0.0)
)

# ---------------------------------------------------------------------------
# 3d. Total vertical stress — integrate overburden downward within each column.
# ---------------------------------------------------------------------------
def _compute_total_stress(soil_mask):
    """Recompute total vertical stress from the current soil mask."""
    ts = np.zeros((NX, NY, NZ))
    for iz in range(NZ - 1, -1, -1):
        above = 0.0 if iz == NZ - 1 else ts[:, :, iz + 1]
        ts[:, :, iz] = above + np.where(soil_mask[:, :, iz],
                                         cell_bulk_density[:, :, iz] * GRAVITY * dz, 0.0)
    return ts

def _compute_total_stress_with_surcharge(soil_mask):
    """Total stress including surcharge loading on the slope top."""
    ts = _compute_total_stress(soil_mask)
    if ENABLE_SURCHARGE:
        surcharge_mask = (
            (XX >= SURCHARGE_X_MIN) & (XX <= SURCHARGE_X_MAX)
            & soil_mask
        )
        # Apply surcharge as additional stress to all soil cells below the loaded area
        for ix in range(NX):
            if x_cc[ix] < SURCHARGE_X_MIN or x_cc[ix] > SURCHARGE_X_MAX:
                continue
            for iy in range(NY):
                col = soil_mask[ix, iy, :]
                if np.any(col):
                    ts[ix, iy, col] += SURCHARGE_LOAD
    return ts

total_stress = _compute_total_stress_with_surcharge(is_soil) if ENABLE_SURCHARGE else _compute_total_stress(is_soil)

# ---------------------------------------------------------------------------
# 3e. Utility: update soil surface elevation after erosion/failure.
# ---------------------------------------------------------------------------
def _update_soil_surface(soil_mask):
    """Recompute soil_surf_3d from current soil mask."""
    new_surf = np.full((NX, NY, NZ), SOIL_BOTTOM)
    for ix in range(NX):
        for iy in range(NY):
            col = soil_mask[ix, iy, :]
            if np.any(col):
                top_iz = np.max(np.where(col)[0])
                new_surf[ix, iy, :] = z_cc[top_iz] + dz / 2
    return new_surf

# ---------------------------------------------------------------------------
# 3f. Sediment transport fields (XBeach)
# ---------------------------------------------------------------------------
if ENABLE_SEDIMENT_TRANSPORT:
    # Suspended sediment concentration per cell (kg/m³)
    sediment_conc = np.zeros((NX, NY, NZ))
    # Settling velocity (Soulsby 1997)
    _nu = 1.0e-6  # kinematic viscosity of water (m²/s)
    _Dstar = GRAIN_SIZE_D50 * ((SEDIMENT_DENSITY / WATER_DENSITY - 1.0) * GRAVITY / _nu**2) ** (1.0/3.0)
    SETTLING_VELOCITY = (_nu / GRAIN_SIZE_D50) * (
        np.sqrt(10.36**2 + 1.049 * _Dstar**3) - 10.36
    )
    print(f"  Sediment: d50={GRAIN_SIZE_D50*1e3:.2f} mm  w_s={SETTLING_VELOCITY:.4f} m/s  D*={_Dstar:.1f}")

# ---------------------------------------------------------------------------
# 3g. Vegetation field (XBeach)
# ---------------------------------------------------------------------------
if ENABLE_VEGETATION:
    veg_mask = (
        (XX >= VEG_X_MIN) & (XX <= VEG_X_MAX)
        & (ZZ <= VEG_HEIGHT)
    )
    print(f"  Vegetation: zone x=[{VEG_X_MIN:.2f},{VEG_X_MAX:.2f}]  "
          f"hv={VEG_HEIGHT:.2f} m  Nv={VEG_STEM_DENSITY:.0f} stems/m2")

# ---------------------------------------------------------------------------
# 3h. Tension crack depth (Slope3D)
# ---------------------------------------------------------------------------
if ENABLE_TENSION_CRACKS:
    if TENSION_CRACK_DEPTH_OVERRIDE is not None:
        tension_crack_depth = TENSION_CRACK_DEPTH_OVERRIDE
    else:
        # Analytical: z_c = 2c' / (gamma * sqrt(Ka))
        _Ka = math.tan(math.radians(45.0 - FRICTION_ANGLE / 2.0))**2
        tension_crack_depth = min(
            2.0 * COHESION / (BULK_DENSITY * GRAVITY * math.sqrt(_Ka)),
            SOIL_TOP - SOIL_BOTTOM
        )
    print(f"  Tension crack: depth = {tension_crack_depth:.3f} m"
          f"{'  (water-filled)' if TENSION_CRACK_WATER_FILLED else ''}")

# =====================================================================
# 4. PHYSICS FUNCTIONS
# =====================================================================
phi_rad   = math.radians(FRICTION_ANGLE)
alpha_rad = math.radians(SLOPE_ANGLE)

def surface_stage(t):
    """Surface water stage h_w(t) in metres above domain floor."""
    stage = STAGE_MEAN + STAGE_AMPLITUDE * math.sin(2.0 * math.pi * STAGE_FREQUENCY * t)
    # Infragravity wave modulation (XBeach): adds a long-wave oscillation
    if ENABLE_INFRAGRAVITY:
        ig_period = IG_PERIOD_FACTOR * WAVE_PERIOD
        ig_amp    = IG_AMPLITUDE_FACTOR * WAVE_HEIGHT
        stage += ig_amp * math.sin(2.0 * math.pi * t / ig_period)
    return stage


def effective_wave_height_at_bank(stage):
    """
    Compute effective wave height at the bank face, accounting for
    depth-limited breaking (XBeach Baldock model) and vegetation damping.
    Returns the effective Hs that drives erosion.
    """
    H_eff = WAVE_HEIGHT

    # Wave breaking: depth-limited (XBeach)
    if ENABLE_WAVE_BREAKING:
        # Water depth at the bank toe (approximate as stage above bed)
        h_bank = max(stage - SOIL_BOTTOM, 0.01)
        H_max = BREAKING_GAMMA * h_bank
        if H_eff > H_max:
            # Baldock (1998): fraction of breaking waves
            Q_b = np.exp(-(H_max / max(H_eff, 0.01))**2)
            # Dissipated fraction reduces Hs
            H_eff = H_max * (1.0 - Q_b) + H_eff * Q_b * 0.5
            H_eff = min(H_eff, H_max)

    # Vegetation wave attenuation (XBeach Mendez-Losada 2004)
    if ENABLE_VEGETATION:
        # Exponential decay over vegetation field width
        veg_width = max(VEG_X_MAX - VEG_X_MIN, 0.01)
        h_veg = max(stage - SOIL_BOTTOM, 0.01)
        alpha_v = min(VEG_HEIGHT / h_veg, 1.0)  # submergence ratio
        # Damping coefficient (Dalrymple et al. / Mendez & Losada)
        k_wave = 2.0 * math.pi / max(WAVE_PERIOD * math.sqrt(GRAVITY * h_veg), 0.1)
        beta_v = (VEG_DRAG_COEFF * VEG_STEM_DIAMETER * VEG_STEM_DENSITY *
                  alpha_v * k_wave * H_eff) / (3.0 * math.pi * math.sinh(k_wave * h_veg)**3)
        damping = max(1.0 - beta_v * veg_width, 0.1)
        H_eff *= damping

    return max(H_eff, 0.0)


def diffuse_pressure(u, soil_mask, stage, dt):
    """
    Advance pore-pressure field by one time step using explicit finite
    differences for the 3-D diffusion equation inside the soil mask.
    Uses per-cell diffusivity when multilayer is enabled.
    """
    D_field = cell_diffusivity if ENABLE_MULTILAYER else DIFFUSIVITY
    n  = _N_SUBSTEPS
    ds = dt / n

    def _nbr(arr, axis, shift):
        nbr = np.roll(arr, shift, axis=axis)
        if shift == 1:
            idx = [slice(None)] * 3; idx[axis] = 0
            nbr[tuple(idx)] = arr[tuple(idx)]
        else:
            idx = [slice(None)] * 3; idx[axis] = -1
            nbr[tuple(idx)] = arr[tuple(idx)]
        return nbr

    open_mask    = ~soil_mask
    face_exposed = (
        np.roll(open_mask, 1,  axis=0) | np.roll(open_mask, -1, axis=0) |
        np.roll(open_mask, 1,  axis=2) | np.roll(open_mask, -1, axis=2)
    ) & soil_mask

    u_bc = np.array(
        [WATER_DENSITY * GRAVITY * max(stage - z_cc[iz], 0.0)
         if stage >= z_cc[iz] else 0.0
         for iz in range(NZ)]
    )

    # Groundwater exchange (XBeach): modify inland boundary condition
    if ENABLE_GROUNDWATER_EXCHANGE:
        gw_head = stage + GW_HEAD_OFFSET
        u_bc_gw = np.array(
            [WATER_DENSITY * GRAVITY * max(gw_head - z_cc[iz], 0.0)
             for iz in range(NZ)]
        )

    u_cur = u.copy()
    for _ in range(n):
        lap = ((_nbr(u_cur, 0, 1) - 2*u_cur + _nbr(u_cur, 0, -1)) / dx**2
             + (_nbr(u_cur, 1, 1) - 2*u_cur + _nbr(u_cur, 1, -1)) / dy**2
             + (_nbr(u_cur, 2, 1) - 2*u_cur + _nbr(u_cur, 2, -1)) / dz**2)

        if isinstance(D_field, np.ndarray):
            u_new = u_cur + D_field * ds * lap
        else:
            u_new = u_cur + D_field * ds * lap

        # Apply prescribed BC at exposed soil faces
        for iz in range(NZ):
            if stage >= z_cc[iz]:
                u_new[:, :, iz] = np.where(face_exposed[:, :, iz], u_bc[iz], u_new[:, :, iz])
            else:
                face_mask = face_exposed[:, :, iz] & (u_new[:, :, iz] > 0.0)
                u_new[:, :, iz] = np.where(face_mask, 0.0, u_new[:, :, iz])

        # Groundwater exchange: apply elevated head at inland boundary (x=0)
        if ENABLE_GROUNDWATER_EXCHANGE:
            for iz in range(NZ):
                inland_soil = soil_mask[0, :, iz]
                if np.any(inland_soil):
                    u_new[0, inland_soil, iz] = u_bc_gw[iz]

        u_new[~soil_mask] = 0.0
        u_new = np.nan_to_num(u_new, nan=0.0, posinf=0.0, neginf=0.0)
        u_cur = u_new

    return u_cur


def compute_effective_stress(total_stress, pore_pressure):
    """Terzaghi effective stress: sigma' = sigma_total - u."""
    return total_stress - pore_pressure


def compute_factor_of_safety(sigma_eff, soil_mask):
    """
    Infinite-slope Factor of Safety at every soil cell.
    Enhanced with:
      - Per-cell properties when ENABLE_MULTILAYER is True
      - Pseudo-static seismic forces when ENABLE_SEISMIC is True
      - Tension crack effects when ENABLE_TENSION_CRACKS is True
      - Surcharge effects (via modified total stress, already applied)
    """
    # Per-cell or global properties
    if ENABLE_MULTILAYER:
        c_arr      = cell_cohesion
        phi_arr    = np.radians(cell_friction_deg)
        rho_arr    = cell_bulk_density
    else:
        c_arr      = COHESION
        phi_arr    = phi_rad
        rho_arr    = BULK_DENSITY

    cos_a   = math.cos(alpha_rad)
    sin_a   = math.sin(alpha_rad)

    # Overburden depth above each cell
    overburden_z = np.maximum(soil_surf_3d - ZZ, 0.0)

    # Normal effective stress on slope-parallel plane
    sigma_n = np.maximum(sigma_eff, 0.0) * cos_a**2

    if isinstance(phi_arr, np.ndarray):
        tan_phi = np.tan(phi_arr)
    else:
        tan_phi = math.tan(phi_arr)

    tau_f = c_arr + sigma_n * tan_phi

    # Driving shear from overburden along slope
    tau_d = rho_arr * GRAVITY * overburden_z * sin_a * cos_a

    # Seismic pseudo-static loading (Slope3D)
    if ENABLE_SEISMIC:
        W = rho_arr * GRAVITY * overburden_z  # weight per unit area
        # Horizontal seismic force component along slope
        tau_seismic_h = SEISMIC_KH * W * cos_a
        # Vertical seismic force reduces/increases normal stress
        delta_sigma_v = SEISMIC_KV * W * cos_a**2
        tau_d = tau_d + tau_seismic_h
        # Reduce resisting force due to vertical acceleration
        tau_f = tau_f - delta_sigma_v * tan_phi

    # Tension crack effects (Slope3D)
    if ENABLE_TENSION_CRACKS:
        # Cells within the tension crack zone have reduced cohesion contribution
        crack_zone = (soil_surf_3d - ZZ <= tension_crack_depth) & soil_mask
        # No cohesion in the cracked zone (crack breaks the soil)
        if isinstance(c_arr, np.ndarray):
            tau_f = np.where(crack_zone, sigma_n * tan_phi, tau_f)
        else:
            tau_f = np.where(crack_zone, sigma_n * tan_phi, tau_f)

        # Water-filled tension crack adds hydrostatic driving force
        if TENSION_CRACK_WATER_FILLED:
            crack_water_force = (WATER_DENSITY * GRAVITY *
                                 np.minimum(tension_crack_depth, np.maximum(soil_surf_3d - ZZ, 0.0))**2
                                 / (2.0 * max(tension_crack_depth, 0.01)))
            tau_d = np.where(crack_zone, tau_d + crack_water_force * sin_a, tau_d)

    tau_d = np.maximum(tau_d, 1e-12)
    tau_f = np.maximum(tau_f, 0.0)

    fs = np.where(soil_mask, tau_f / tau_d, 999.0)
    return fs


def compute_saturation(pore_pressure, soil_mask):
    """Degree of saturation: 1 (fully saturated) to 0 (dry), inside soil only."""
    ref = max(WATER_DENSITY * GRAVITY * SOIL_TOP * 0.5, 1.0)
    sat = np.where(pore_pressure >= 0, 1.0,
                   np.clip(1.0 + pore_pressure / ref, 0.0, 1.0))
    return sat * soil_mask


def mobilise_failed_cells(cell_type_arr, soil_mask, fs, stage):
    """
    Mark soil cells where FS < 1 as CELL_FAIL (mobilised).
    Returns updated (cell_type_arr, soil_mask) and the count of newly failed cells.
    """
    newly_failed = soil_mask & (fs < 1.0)
    n_failed = int(np.sum(newly_failed))
    if n_failed == 0:
        return cell_type_arr, soil_mask, 0

    new_ct   = cell_type_arr.copy()
    new_soil = soil_mask.copy()

    new_ct[newly_failed]   = np.where(ZZ[newly_failed] <= stage, CELL_WATER, CELL_AIR)
    new_soil[newly_failed] = False
    return new_ct, new_soil, n_failed


# =====================================================================
# 5.  EROSION PROCESS MODELS
# =====================================================================

# ---- 5a. Hydraulic Shear Erosion ----
def compute_bed_shear(current_velocity):
    """Bed shear stress: tau_b = rho_w C_d U^2."""
    C_d = 0.003
    return WATER_DENSITY * C_d * current_velocity**2


def apply_hydraulic_shear_erosion(soil_mask, cell_type_arr, soil_surf_3d_arr,
                                  u_field, current_velocity, stage, dt):
    """
    Excess-shear erosion: E = M max(tau_b - tau_c, 0).
    When ENABLE_GROUNDWATER_EXCHANGE is active, exfiltration reduces tau_c.
    """
    tau_b = compute_bed_shear(current_velocity)

    # Effective critical shear — reduced by exfiltration (XBeach)
    tau_c_eff = CRITICAL_SHEAR
    if ENABLE_GROUNDWATER_EXCHANGE:
        # Identify cells with positive pore pressure gradient toward face (exfiltration)
        open_mask = ~soil_mask
        face_exposed_tmp = soil_mask & (
            np.roll(open_mask, 1, axis=0) | np.roll(open_mask, -1, axis=0)
        )
        exfilt_mask = face_exposed_tmp & (u_field > 0)
        # Where exfiltrating, reduce critical shear
        tau_c_field = np.where(exfilt_mask,
                               CRITICAL_SHEAR * (1.0 - GW_EXFILTRATION_WEAKENING),
                               CRITICAL_SHEAR)
    else:
        tau_c_field = CRITICAL_SHEAR

    if isinstance(tau_c_field, np.ndarray):
        min_tau_c = tau_c_field[soil_mask].min() if np.any(soil_mask) else CRITICAL_SHEAR
    else:
        min_tau_c = tau_c_field

    if tau_b <= min_tau_c and not isinstance(tau_c_field, np.ndarray):
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    E_rate = EROSION_RATE_COEF * np.maximum(tau_b - tau_c_field, 0.0)
    if isinstance(E_rate, np.ndarray):
        dz_erode = E_rate * dt / cell_bulk_density
    else:
        dz_erode = E_rate * dt / BULK_DENSITY

    open_mask = ~soil_mask
    exposed = soil_mask.copy()
    exposed &= (
        np.roll(open_mask, 1, axis=0) | np.roll(open_mask, -1, axis=0)
    )
    exposed &= (ZZ <= stage)

    if not np.any(exposed):
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    if isinstance(dz_erode, np.ndarray):
        frac_removed = dz_erode / dx
    else:
        frac_removed = dz_erode / dx
    remove_mask = exposed & (np.random.random(exposed.shape) < frac_removed)

    n_eroded = int(np.sum(remove_mask))
    if n_eroded == 0:
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    new_soil = soil_mask.copy()
    new_ct   = cell_type_arr.copy()

    new_soil[remove_mask] = False
    new_ct[remove_mask] = np.where(ZZ[remove_mask] <= stage, CELL_WATER, CELL_AIR)
    u_field[remove_mask] = 0.0

    new_surf = _update_soil_surface(new_soil)

    return new_soil, new_ct, new_surf, u_field, n_eroded


# ---- 5b. Wave Erosion ----
def wave_energy_flux(H_s, T_p):
    """Deep-water wave energy flux: P = (rho g^2 Hs^2 Tp) / (64 pi)."""
    return (WATER_DENSITY * GRAVITY**2 * H_s**2 * T_p) / (64.0 * math.pi)


def apply_wave_erosion(soil_mask, cell_type_arr, soil_surf_3d_arr,
                       u_field, stage, dt):
    """
    Wave-notch erosion model using effective wave height (accounts for
    depth-limited breaking and vegetation damping when enabled).
    """
    H_eff = effective_wave_height_at_bank(stage)
    if H_eff <= 0:
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    P = wave_energy_flux(H_eff, WAVE_PERIOD)
    vol_rate = WAVE_EROSION_CE * P

    notch_height = 2.0 * WAVE_NOTCH_HALFWIDTH
    if notch_height < dz:
        notch_height = dz
    dz_erode = vol_rate * dt / notch_height

    open_mask = ~soil_mask
    exposed = soil_mask.copy()
    exposed &= (
        np.roll(open_mask, 1, axis=0) | np.roll(open_mask, -1, axis=0)
    )
    wave_band = (
        (ZZ >= stage - WAVE_NOTCH_HALFWIDTH) &
        (ZZ <= stage + WAVE_NOTCH_HALFWIDTH)
    )
    exposed &= wave_band

    if not np.any(exposed):
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    frac_removed = dz_erode / dx
    remove_mask = exposed & (np.random.random(exposed.shape) < frac_removed)

    n_eroded = int(np.sum(remove_mask))
    if n_eroded == 0:
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    new_soil = soil_mask.copy()
    new_ct   = cell_type_arr.copy()

    new_soil[remove_mask] = False
    new_ct[remove_mask] = np.where(ZZ[remove_mask] <= stage, CELL_WATER, CELL_AIR)
    u_field[remove_mask] = 0.0

    new_surf = _update_soil_surface(new_soil)
    return new_soil, new_ct, new_surf, u_field, n_eroded


# ---- 5c. Vessel Wake Erosion ----
def vessel_wake_active(t: float) -> bool:
    """Return True if a vessel-wake event is active at time t."""
    if VESSEL_PASSAGE_INTERVAL <= 0:
        return False
    phase = t % VESSEL_PASSAGE_INTERVAL
    return phase < VESSEL_WAKE_DURATION


def apply_vessel_wake_erosion(soil_mask: np.ndarray, cell_type_arr: np.ndarray,
                              soil_surf_3d_arr: np.ndarray, u_field: np.ndarray,
                              stage: float, t: float, dt: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Vessel-wake erosion with cyclic pore-pressure perturbation.
    Uses effective wave height when wave breaking/vegetation are enabled.
    """
    if not vessel_wake_active(t):
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    # Use effective height if breaking/vegetation enabled, else raw wake height
    H_wake = VESSEL_WAKE_HEIGHT
    if ENABLE_WAVE_BREAKING:
        h_bank = max(stage - SOIL_BOTTOM, 0.01)
        H_max = BREAKING_GAMMA * h_bank
        H_wake = min(H_wake, H_max)
    if ENABLE_VEGETATION:
        veg_width = max(VEG_X_MAX - VEG_X_MIN, 0.01)
        h_veg = max(stage - SOIL_BOTTOM, 0.01)
        alpha_v = min(VEG_HEIGHT / h_veg, 1.0)
        k_wave = 2.0 * math.pi / max(VESSEL_WAKE_PERIOD * math.sqrt(GRAVITY * h_veg), 0.1)
        beta_v = (VEG_DRAG_COEFF * VEG_STEM_DIAMETER * VEG_STEM_DENSITY *
                  alpha_v * k_wave * H_wake) / (3.0 * math.pi * max(math.sinh(k_wave * h_veg)**3, 1e-12))
        damping = max(1.0 - beta_v * veg_width, 0.1)
        H_wake *= damping

    P = wave_energy_flux(H_wake, VESSEL_WAKE_PERIOD)
    vol_rate = VESSEL_WAKE_CE * P

    notch_height = 2.0 * WAVE_NOTCH_HALFWIDTH * 1.5
    if notch_height < dz:
        notch_height = dz
    dz_erode = vol_rate * dt / notch_height

    # Cyclic pore-pressure spike from wake-wave loading
    du_wake = 0.5 * WATER_DENSITY * GRAVITY * H_wake
    open_mask = ~soil_mask
    face_exposed = soil_mask & (
        np.roll(open_mask, 1, axis=0) | np.roll(open_mask, -1, axis=0)
    )
    wake_pp_zone = face_exposed.copy()
    for _ in range(2):
        wake_pp_zone |= np.roll(wake_pp_zone, -1, axis=0) & soil_mask
    wake_zone_submerged = wake_pp_zone & (ZZ <= stage + H_wake / 2)
    u_field[wake_zone_submerged] += du_wake * dt / VESSEL_WAKE_PERIOD

    # Erosion of exposed cells in the wake impact band
    exposed = face_exposed.copy()
    hw = WAVE_NOTCH_HALFWIDTH * 1.5
    wake_band = (
        (ZZ >= stage - hw) &
        (ZZ <= stage + hw)
    )
    exposed &= wake_band

    if not np.any(exposed):
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    frac_removed = dz_erode / dx
    remove_mask = exposed & (np.random.random(exposed.shape) < frac_removed)

    n_eroded = int(np.sum(remove_mask))
    if n_eroded == 0:
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    new_soil = soil_mask.copy()
    new_ct   = cell_type_arr.copy()

    new_soil[remove_mask] = False
    new_ct[remove_mask] = np.where(ZZ[remove_mask] <= stage, CELL_WATER, CELL_AIR)
    u_field[remove_mask] = 0.0

    new_surf = _update_soil_surface(new_soil)
    return new_soil, new_ct, new_surf, u_field, n_eroded


# ---- 5d. Sediment Transport (XBeach — Soulsby-Van Rijn) ----
def compute_sediment_transport(soil_mask, cell_type_arr, u_field, stage, dt):
    """
    Soulsby-Van Rijn total load transport in the water column.
    Computes suspended + bed-load transport, advects sediment, and deposits
    material where conditions allow.

    Returns (sediment_conc, n_deposited, n_entrained) — modifies sediment_conc in place.
    """
    global sediment_conc

    # Only compute in water cells
    water_mask = (cell_type_arr == CELL_WATER) & (ZZ <= stage)
    if not np.any(water_mask):
        return sediment_conc, 0, 0

    # Approximate depth-averaged current from stage and a simple log-profile
    h_local = np.maximum(stage - ZZ, 0.01)
    # Simplified Soulsby-Van Rijn: total transport q_t
    # q_t = A_sb * U * [max(|U| - U_cr, 0)]  for bed load
    # q_t += A_ss * U * [max(|U| - U_cr, 0)]  for suspended load

    # Critical velocity (Soulsby 1997)
    d50 = GRAIN_SIZE_D50
    d90 = GRAIN_SIZE_D90
    s = SEDIMENT_DENSITY / WATER_DENSITY
    h_mean = max(stage - SOIL_BOTTOM, 0.01)

    U_cr_d50 = 0.19 * d50**0.1 * np.log10(4.0 * h_mean / d90) if d50 >= 0.0005 else \
               8.5 * d50**0.6 * np.log10(4.0 * h_mean / d90)

    # Bed-load and suspended-load coefficients
    A_sb = 0.005 * h_mean * (d50 / h_mean)**1.2 / ((s - 1.0) * GRAVITY * d50)**0.75
    A_ss = 0.012 * d50 * _Dstar**(-0.6) / ((s - 1.0) * GRAVITY * d50)**1.2 if ENABLE_SEDIMENT_TRANSPORT else 0.0

    # Use current velocity as the flow driver
    U_mag = CURRENT_VELOCITY if ENABLE_HYDRAULIC_SHEAR else 0.1
    excess = max(U_mag - U_cr_d50, 0.0)

    q_total = TRANSPORT_CALIB * (A_sb + A_ss) * U_mag * excess**2.4  # m²/s per unit width

    # Entrainment at exposed soil faces (cells eroded become sediment source)
    open_mask = ~soil_mask
    exposed_face = soil_mask & (
        np.roll(open_mask, 1, axis=0) | np.roll(open_mask, -1, axis=0)
    ) & (ZZ <= stage)

    n_entrained = 0
    if np.any(exposed_face) and q_total > 0:
        entrain_rate = q_total / max(h_mean, 0.01) * dt  # kg/m³ entrained
        sediment_conc[water_mask] += entrain_rate * 0.1   # distribute in water column
        n_entrained = int(np.sum(exposed_face))

    # Settling / deposition
    n_deposited = 0
    if np.any(water_mask) and SETTLING_VELOCITY > 0:
        # Reduce concentration by settling
        settle_frac = min(SETTLING_VELOCITY * dt / max(h_mean, dz), 1.0)
        deposited_mass = sediment_conc * settle_frac
        sediment_conc *= (1.0 - settle_frac)

        # Deposition on bed surface (bottom water cells adjacent to soil)
        bed_cells = water_mask & (
            np.roll(soil_mask, 1, axis=2) | (ZZ <= dz)
        )
        # Note: actual bed accretion handled in morphological update

    # Apply MORFAC
    if MORFAC != 1.0:
        sediment_conc *= 1.0  # concentration stays real; MORFAC applied to bed changes

    sediment_conc = np.maximum(sediment_conc, 0.0)
    sediment_conc[~water_mask] = 0.0

    return sediment_conc, n_deposited, n_entrained


# ---- 5e. Morphological Update (XBeach) ----
def apply_morphological_update(soil_mask, cell_type_arr, soil_surf_3d_arr,
                               u_field, stage, dt):
    """
    Update bed level based on sediment transport divergence.
    Continuity equation: dz_bed/dt = -1/(1-p) * dqs/dx, accelerated by MORFAC.
    Deposition can convert water cells back to soil if enough sediment accumulates.

    Returns (soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, n_accreted).
    """
    global sediment_conc

    if not np.any(sediment_conc > 0):
        return soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, 0

    n_accreted = 0
    new_soil = soil_mask.copy()
    new_ct   = cell_type_arr.copy()

    # Identify bottom water cells (just above soil or at domain base)
    for ix in range(NX):
        for iy in range(NY):
            col_soil = new_soil[ix, iy, :]
            if not np.any(col_soil):
                base_iz = 0
            else:
                base_iz = np.max(np.where(col_soil)[0]) + 1

            if base_iz >= NZ:
                continue
            if ZZ[ix, iy, base_iz] > stage:
                continue

            # Deposition from concentration at this cell
            conc = sediment_conc[ix, iy, base_iz]
            if conc <= 0:
                continue

            # Deposition rate: dz = w_s * C * dt * MORFAC / rho_s / (1-p)
            dz_dep = (SETTLING_VELOCITY * conc * dt * MORFAC /
                      (SEDIMENT_DENSITY * (1.0 - POROSITY)))
            frac = dz_dep / dz  # fraction of a cell

            if np.random.random() < frac:
                # Accrete: convert water cell to soil
                new_soil[ix, iy, base_iz] = True
                new_ct[ix, iy, base_iz] = CELL_SOIL
                n_accreted += 1
                sediment_conc[ix, iy, base_iz] = 0.0

    if n_accreted > 0:
        new_surf = _update_soil_surface(new_soil)
    else:
        new_surf = soil_surf_3d_arr

    return new_soil, new_ct, new_surf, u_field, n_accreted


# ---- 5f. Avalanching (XBeach) ----
def apply_avalanching(soil_mask, cell_type_arr, soil_surf_3d_arr,
                      u_field, stage):
    """
    XBeach-style avalanching: if local slope between adjacent columns
    exceeds the critical angle, material redistributes from higher to
    lower column. Uses CRITICAL_SLOPE_DRY above waterline and
    CRITICAL_SLOPE_WET below.

    Returns (soil_mask, cell_type_arr, soil_surf_3d_arr, u_field, n_avalanched).
    """
    new_soil = soil_mask.copy()
    new_ct   = cell_type_arr.copy()
    n_avalanched = 0

    # Work on 2-D column surface — use mid-Y for simplicity, apply to all Y
    for iy in range(NY):
        for ix in range(NX - 1):
            # Surface height in each column
            col_L = new_soil[ix, iy, :]
            col_R = new_soil[ix + 1, iy, :]
            z_L = z_cc[np.max(np.where(col_L)[0])] + dz/2 if np.any(col_L) else SOIL_BOTTOM
            z_R = z_cc[np.max(np.where(col_R)[0])] + dz/2 if np.any(col_R) else SOIL_BOTTOM

            dz_col = z_L - z_R
            if abs(dz_col) < dz:
                continue

            local_slope_deg = math.degrees(math.atan2(abs(dz_col), dx))
            mean_z = (z_L + z_R) / 2.0
            crit = CRITICAL_SLOPE_WET if mean_z <= stage else CRITICAL_SLOPE_DRY

            if local_slope_deg <= crit:
                continue

            # Move one cell from higher column to lower
            if dz_col > 0:  # L is higher → move top of L down, add to top of R
                src_ix, dst_ix = ix, ix + 1
            else:
                src_ix, dst_ix = ix + 1, ix

            src_col = new_soil[src_ix, iy, :]
            if not np.any(src_col):
                continue
            top_iz_src = np.max(np.where(src_col)[0])

            # Remove from source
            new_soil[src_ix, iy, top_iz_src] = False
            new_ct[src_ix, iy, top_iz_src] = (CELL_WATER if z_cc[top_iz_src] <= stage
                                               else CELL_AIR)
            u_field[src_ix, iy, top_iz_src] = 0.0

            # Add to destination
            dst_col = new_soil[dst_ix, iy, :]
            if np.any(dst_col):
                top_iz_dst = np.max(np.where(dst_col)[0]) + 1
            else:
                top_iz_dst = 0

            if top_iz_dst < NZ:
                new_soil[dst_ix, iy, top_iz_dst] = True
                new_ct[dst_ix, iy, top_iz_dst] = CELL_SOIL
                n_avalanched += 1

    if n_avalanched > 0:
        new_surf = _update_soil_surface(new_soil)
    else:
        new_surf = soil_surf_3d_arr

    return new_soil, new_ct, new_surf, u_field, n_avalanched


# =====================================================================
# 6.  TIME-STEPPING LOOP
# =====================================================================
print("\nRunning simulation ...")

iy_mid = NY // 2

# Storage for animation frames (mid-Y cross-section, shape NX×NZ)
frames_pp      = []   # pore pressure
frames_fs      = []   # factor of safety
frames_sat     = []   # saturation
frames_ct      = []   # cell_type (0=air,1=water,2=soil,3=failed)
frames_time    = []
frames_stage   = []
frames_nfail   = []   # cumulative mobilised cell count (FS-based failure)
frames_nerode  = []   # cumulative eroded cell count (all surface erosion)

if EXPORT_NETCDF:
    nc_pp_all  = []; nc_fs_all  = []; nc_sat_all  = []
    nc_ct_all  = []; nc_time_all = []

u          = pore_pressure.copy()
cum_failed = 0
cum_eroded = 0
cum_avalanched = 0
cum_accreted = 0

for step in range(N_STEPS + 1):
    t     = step * DT
    # Stage: if tidal is enabled, oscillate; otherwise hold at mean
    if ENABLE_TIDAL:
        stage = surface_stage(t)
    else:
        stage = STAGE_MEAN

    # --- Pore-pressure diffusion (always active — seepage physics) ---
    u = diffuse_pressure(u, is_soil, stage, DT)

    # --- Effective stress & stability ---
    if ENABLE_SURCHARGE:
        total_stress = _compute_total_stress_with_surcharge(is_soil)
    else:
        total_stress = _compute_total_stress(is_soil)

    sigma_eff    = compute_effective_stress(total_stress, u)
    fs           = compute_factor_of_safety(sigma_eff, is_soil)
    sat          = compute_saturation(u, is_soil)

    # --- Mobilise failed soil cells (FS < 1) into open domain ---
    cell_type, is_soil, n_new = mobilise_failed_cells(cell_type, is_soil, fs, stage)
    cum_failed += n_new
    if n_new > 0:
        u[~is_soil] = 0.0
        soil_surf_3d = _update_soil_surface(is_soil)

    # --- Surface erosion processes ---
    n_eroded_step = 0

    # 5a. Hydraulic shear erosion
    if ENABLE_HYDRAULIC_SHEAR and CURRENT_VELOCITY > 0:
        is_soil, cell_type, soil_surf_3d, u, n_e = apply_hydraulic_shear_erosion(
            is_soil, cell_type, soil_surf_3d, u, CURRENT_VELOCITY, stage, DT)
        n_eroded_step += n_e

    # 5b. Wave erosion
    if ENABLE_WAVES and WAVE_HEIGHT > 0:
        is_soil, cell_type, soil_surf_3d, u, n_e = apply_wave_erosion(
            is_soil, cell_type, soil_surf_3d, u, stage, DT)
        n_eroded_step += n_e

    # 5c. Vessel wake erosion
    if ENABLE_VESSEL_WAKE and VESSEL_WAKE_HEIGHT > 0:
        is_soil, cell_type, soil_surf_3d, u, n_e = apply_vessel_wake_erosion(
            is_soil, cell_type, soil_surf_3d, u, stage, t, DT)
        n_eroded_step += n_e

    cum_eroded += n_eroded_step

    # 5d. Sediment transport (XBeach)
    if ENABLE_SEDIMENT_TRANSPORT:
        sediment_conc, n_dep, n_ent = compute_sediment_transport(
            is_soil, cell_type, u, stage, DT)

    # 5e. Morphological update (XBeach)
    if ENABLE_SEDIMENT_TRANSPORT and ENABLE_MORPHOLOGICAL_UPDATE:
        is_soil, cell_type, soil_surf_3d, u, n_acc = apply_morphological_update(
            is_soil, cell_type, soil_surf_3d, u, stage, DT)
        cum_accreted += n_acc

    # 5f. Avalanching (XBeach)
    if ENABLE_AVALANCHING:
        is_soil, cell_type, soil_surf_3d, u, n_aval = apply_avalanching(
            is_soil, cell_type, soil_surf_3d, u, stage)
        cum_avalanched += n_aval

    # --- Update cell_type for stage-dependent air/water transitions ---
    open_mask = ~is_soil
    cell_type[open_mask] = np.where(ZZ[open_mask] <= stage, CELL_WATER, CELL_AIR)

    # --- Save frame data ---
    if step % FRAME_INT == 0:
        frames_pp.append(u[:, iy_mid, :].copy())
        frames_fs.append(fs[:, iy_mid, :].copy())
        frames_sat.append(sat[:, iy_mid, :].copy())
        frames_ct.append(cell_type[:, iy_mid, :].copy())
        frames_time.append(t)
        frames_stage.append(stage)
        frames_nfail.append(cum_failed)
        frames_nerode.append(cum_eroded)

        if EXPORT_NETCDF:
            nc_pp_all.append(u.copy())
            nc_fs_all.append(fs.copy())
            nc_sat_all.append(sat.copy())
            nc_ct_all.append(cell_type.copy())
            nc_time_all.append(t)

    if step % max(1, N_STEPS // 10) == 0:
        fs_soil = fs[is_soil]
        min_fs   = fs_soil.min() if fs_soil.size > 0 else 999.0
        pct_fail = 100.0 * np.sum(fs_soil < 1.0) / max(fs_soil.size, 1)
        extras = ""
        if ENABLE_AVALANCHING:
            extras += f"  aval: {cum_avalanched}"
        if ENABLE_SEDIMENT_TRANSPORT:
            extras += f"  accr: {cum_accreted}"
        print(f"  t={t:7.2f}s  stage={stage:.3f}m  min(FS)={min_fs:.3f}  "
              f"cells FS<1: {pct_fail:.1f}%  mobilised: {cum_failed}  eroded: {cum_eroded}{extras}")

print("Simulation complete.\n")

# =====================================================================
# 7.  VISUALISATION — Animated cross-section
# =====================================================================
n_frames = len(frames_time)
print(f"Rendering {n_frames} frames → {VIDEO_FILE} ...")

# Precompute time-series arrays before building the figure
# Use the soil mask at each saved frame for FS statistics
# (we saved frames_ct so we can reconstruct the soil mask per frame)
is_soil_mid0 = is_soil_init[:, iy_mid, :]   # initial mid-Y soil mask for reference

ts_time_arr  = np.array(frames_time)
ts_stage_arr = np.array(frames_stage)
ts_nfail_arr = np.array(frames_nfail)
ts_nerode_arr = np.array(frames_nerode)

# For FS time series use the dynamic soil region in each frame's cell_type
def _soil_mid(ct_slice):
    """Boolean mask: soil cells in mid-Y cross-section at a given frame."""
    return ct_slice == CELL_SOIL

ts_minfs_arr = np.array(
    [f[_soil_mid(frames_ct[i])].min()
     if np.any(_soil_mid(frames_ct[i])) else 999.0
     for i, f in enumerate(frames_fs)]
)
ts_pctf_arr = np.array(
    [100.0 * np.sum(f[_soil_mid(frames_ct[i])] < 1.0) /
     max(np.sum(_soil_mid(frames_ct[i])), 1)
     for i, f in enumerate(frames_fs)]
)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
ax_dom = axes[0, 0]   # domain overview (cell types)
ax_pp  = axes[0, 1]   # pore-pressure
ax_fs  = axes[0, 2]   # factor of safety
ax_sat = axes[1, 0]   # saturation
ax_ts  = axes[1, 1]   # time-series (stage & FS)
ax_mob = axes[1, 2]   # mobilised cell count

extent = [0, DOMAIN_LENGTH, 0, DOMAIN_HEIGHT]

# --- Domain overview: discrete colour map for cell types ---
# 0=air (white), 1=water (steelblue), 2=soil (saddlebrown), 3=failed (orange)
from matplotlib.colors import ListedColormap, BoundaryNorm
_ct_cmap = ListedColormap(['white', 'steelblue', 'saddlebrown', 'orange'])
_ct_norm = BoundaryNorm([0, 1, 2, 3, 4], _ct_cmap.N)

im_dom = ax_dom.imshow(frames_ct[0].T, origin='lower', extent=extent,
                        aspect='auto', cmap=_ct_cmap, norm=_ct_norm,
                        interpolation='nearest')
im_pp  = ax_pp.imshow(frames_pp[0].T,  origin='lower', extent=extent,
                       aspect='auto', cmap='coolwarm', vmin=-3000, vmax=8000)
im_fs  = ax_fs.imshow(frames_fs[0].T,  origin='lower', extent=extent,
                       aspect='auto', cmap='RdYlGn', vmin=0, vmax=3)
im_sat = ax_sat.imshow(frames_sat[0].T, origin='lower', extent=extent,
                        aspect='auto', cmap='Blues', vmin=0, vmax=1)

ax_dom.set_title('Domain (air/water/soil/failed)')
ax_pp.set_title('Pore-Water Pressure (Pa)')
ax_fs.set_title('Factor of Safety')
ax_sat.set_title('Saturation')
for a in [ax_dom, ax_pp, ax_fs, ax_sat]:
    a.set_xlabel('X (m)'); a.set_ylabel('Z (m)')

fig.colorbar(im_pp,  ax=ax_pp,  shrink=0.8, label='u (Pa)')
fig.colorbar(im_fs,  ax=ax_fs,  shrink=0.8, label='FS')
fig.colorbar(im_sat, ax=ax_sat, shrink=0.8, label='S')

# Discrete legend patch for domain overview
from matplotlib.patches import Patch
_legend_patches = [
    Patch(facecolor='white',       edgecolor='grey',  label='Air'),
    Patch(facecolor='steelblue',                       label='Water'),
    Patch(facecolor='saddlebrown',                     label='Soil'),
    Patch(facecolor='orange',                          label='Failed/Mobilised'),
]
ax_dom.legend(handles=_legend_patches, loc='upper right', fontsize=7)

# Stage lines on pp, fs, sat panels
stage_lines = []
for a in [ax_pp, ax_fs, ax_sat, ax_dom]:
    ln, = a.plot([], [], 'c-', lw=1.5, label='Water stage')
    stage_lines.append(ln)

# Initial soil boundary outline on domain panel (for reference)
_soil_surf_mid = soil_surf_3d[:, iy_mid, 0]   # surface elevation per x (same for all iz)
_x_outline = np.concatenate([[SOIL_X_MIN], x_cc[(x_cc >= SOIL_X_MIN) & (x_cc <= SOIL_X_MAX)], [SOIL_X_MAX]])
_z_outline = np.concatenate([[SOIL_BOTTOM], _build_soil_surface_z(
    x_cc[(x_cc >= SOIL_X_MIN) & (x_cc <= SOIL_X_MAX)]), [SOIL_BOTTOM]])
ax_dom.plot(_x_outline, _z_outline, 'k--', lw=1, alpha=0.5, label='Init soil surface')

# --- Time-series subplot (stage + FS) ---
ts_stage_ln, = ax_ts.plot([], [], 'c-',  lw=2, label='Stage (m)')
ts_minfs_ln, = ax_ts.plot([], [], 'r-',  lw=2, label='min(FS)')
ax_ts.set_xlim(0, SIM_DURATION)
ax_ts.set_ylim(0, max(3.5, STAGE_MEAN + STAGE_AMPLITUDE + 0.5))
ax_ts.set_xlabel('Time (s)'); ax_ts.set_ylabel('Value')
ax_ts.legend(loc='upper right', fontsize=8); ax_ts.grid(True, alpha=0.3)
ax_ts.set_title('Stage & min(FS)')

# --- Mobilised / eroded soil panel ---
mob_ln, = ax_mob.plot([], [], 'darkorange', lw=2, label='Mobilised (FS<1)')
ero_ln, = ax_mob.plot([], [], 'red', lw=2, label='Eroded (surface)')
ax_mob.set_xlim(0, SIM_DURATION)
max_mob = max(ts_nfail_arr.max(), ts_nerode_arr.max(), 1)
ax_mob.set_ylim(0, max_mob * 1.15)
ax_mob.set_xlabel('Time (s)'); ax_mob.set_ylabel('Cumulative cells')
ax_mob.legend(fontsize=8); ax_mob.grid(True, alpha=0.3)
ax_mob.set_title('Soil Removal')

title_text = fig.suptitle('', fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])


def init_anim():
    im_dom.set_data(frames_ct[0].T)
    im_pp.set_data(frames_pp[0].T)
    im_fs.set_data(frames_fs[0].T)
    im_sat.set_data(frames_sat[0].T)
    for ln in stage_lines:
        ln.set_data([], [])
    ts_stage_ln.set_data([], [])
    ts_minfs_ln.set_data([], [])
    mob_ln.set_data([], [])
    ero_ln.set_data([], [])
    title_text.set_text('')
    return ([im_dom, im_pp, im_fs, im_sat,
             ts_stage_ln, ts_minfs_ln, mob_ln, ero_ln, title_text] + stage_lines)


def update_anim(i):
    t     = frames_time[i]
    stage = frames_stage[i]

    im_dom.set_data(frames_ct[i].T)
    im_pp.set_data(frames_pp[i].T)
    im_fs.set_data(frames_fs[i].T)
    im_sat.set_data(frames_sat[i].T)

    for ln in stage_lines:
        ln.set_data([0, DOMAIN_LENGTH], [stage, stage])

    ts_stage_ln.set_data(ts_time_arr[:i+1], ts_stage_arr[:i+1])
    ts_minfs_ln.set_data(ts_time_arr[:i+1], ts_minfs_arr[:i+1])
    mob_ln.set_data(ts_time_arr[:i+1], ts_nfail_arr[:i+1])
    ero_ln.set_data(ts_time_arr[:i+1], ts_nerode_arr[:i+1])

    min_fs_now = ts_minfs_arr[i]
    pct_now    = ts_pctf_arr[i]
    mob_now    = ts_nfail_arr[i]
    ero_now    = ts_nerode_arr[i]
    status = "STABLE" if min_fs_now >= 1.0 else "FAILURE"
    title_text.set_text(
        f't = {t:.2f} s  |  Stage = {stage:.3f} m  |  '
        f'min(FS) = {min_fs_now:.2f}  |  '
        f'Failed = {pct_now:.1f}%  |  Mobilised = {mob_now}  |  Eroded = {ero_now}  [{status}]'
    )
    return ([im_dom, im_pp, im_fs, im_sat,
             ts_stage_ln, ts_minfs_ln, mob_ln, ero_ln, title_text] + stage_lines)


anim = animation.FuncAnimation(fig, update_anim, init_func=init_anim,
                                frames=n_frames, interval=1000 / FPS, blit=False,
                                repeat=False)

try:
    writer = animation.FFMpegWriter(fps=FPS, metadata={'title': 'Slope Erosion Model'},
                                     bitrate=2000)
    anim.save(VIDEO_FILE, writer=writer)
    print(f"Saved animation → {VIDEO_FILE}")
except Exception as e:
    print(f"[WARNING] Could not save MP4 ({e}). Skipping video export.")

plt.close(fig)

# =====================================================================
# 8.  NetCDF EXPORT  (optional)
# =====================================================================
if EXPORT_NETCDF and nc_time_all:
    try:
        import netCDF4 as nc4
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "netCDF4"])
        import netCDF4 as nc4

    print(f"Writing NetCDF → {NETCDF_FILE} ...")
    with nc4.Dataset(NETCDF_FILE, 'w', format='NETCDF4') as ds:
        ds.title = "SlopeErosion 3-D model output"
        ds.createDimension('time', len(nc_time_all))
        ds.createDimension('x', NX)
        ds.createDimension('y', NY)
        ds.createDimension('z', NZ)

        vt = ds.createVariable('time', 'f4', ('time',))
        vt.units = 'seconds'
        vt[:] = np.array(nc_time_all, dtype=np.float32)

        vx = ds.createVariable('x', 'f4', ('x',))
        vx.units = 'meters'; vx[:] = x_cc.astype(np.float32)
        vy = ds.createVariable('y', 'f4', ('y',))
        vy.units = 'meters'; vy[:] = y_cc.astype(np.float32)
        vz = ds.createVariable('z', 'f4', ('z',))
        vz.units = 'meters'; vz[:] = z_cc.astype(np.float32)

        vpp = ds.createVariable('pore_pressure', 'f4', ('time', 'x', 'y', 'z'),
                                 zlib=True, complevel=4)
        vpp.units = 'Pa'; vpp.long_name = 'Pore-water pressure'

        vfs = ds.createVariable('factor_of_safety', 'f4', ('time', 'x', 'y', 'z'),
                                 zlib=True, complevel=4)
        vfs.long_name = 'Infinite-slope Factor of Safety'

        vsat = ds.createVariable('saturation', 'f4', ('time', 'x', 'y', 'z'),
                                  zlib=True, complevel=4)
        vsat.long_name = 'Degree of saturation (0-1)'

        vct = ds.createVariable('cell_type', 'i1', ('time', 'x', 'y', 'z'),
                                 zlib=True, complevel=4)
        vct.long_name = '0=air 1=water 2=soil 3=failed/mobilised'

        for i in range(len(nc_time_all)):
            vpp[i, :, :, :]  = nc_pp_all[i].astype(np.float32)
            vfs[i, :, :, :]  = nc_fs_all[i].astype(np.float32)
            vsat[i, :, :, :] = nc_sat_all[i].astype(np.float32)
            vct[i, :, :, :]  = nc_ct_all[i].astype(np.int8)

    print(f"Saved NetCDF: {NETCDF_FILE}")

# =====================================================================
# 9.  SUMMARY
# =====================================================================
_per_frame_min = []
for i, f in enumerate(frames_fs):
    mask = _soil_mid(frames_ct[i])
    if np.any(mask):
        v = f[mask].min()
        if np.isfinite(v):
            _per_frame_min.append(v)
global_min_fs = float(np.min(_per_frame_min)) if _per_frame_min else 999.0
max_pct_fail  = ts_pctf_arr.max()

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  Active processes: {', '.join(_active_processes)}")
print(f"  Soil footprint  : x in [{SOIL_X_MIN:.2f}, {SOIL_X_MAX:.2f}]  "
      f"y in [{SOIL_Y_MIN:.2f}, {SOIL_Y_MAX:.2f}]  boundary={LATERAL_BOUNDARY}")
print(f"  Minimum FS      : {global_min_fs:.3f}")
print(f"  Max %  FS<1     : {max_pct_fail:.1f}%")
print(f"  Mobilised cells : {cum_failed}  (approx {cum_failed * CELL_SIZE**3 * 1e6:.1f} cm3)")
print(f"  Eroded cells    : {cum_eroded}  (approx {cum_eroded * CELL_SIZE**3 * 1e6:.1f} cm3)")
if ENABLE_AVALANCHING:
    print(f"  Avalanched cells: {cum_avalanched}")
if ENABLE_SEDIMENT_TRANSPORT:
    print(f"  Accreted cells  : {cum_accreted}")
print(f"  Total removed   : {cum_failed + cum_eroded}")
print(f"  Stage range     : [{STAGE_MEAN - STAGE_AMPLITUDE:.2f}, "
      f"{STAGE_MEAN + STAGE_AMPLITUDE:.2f}] m")
if ENABLE_SEISMIC:
    print(f"  Seismic loading : kh={SEISMIC_KH}  kv={SEISMIC_KV}")
if ENABLE_TENSION_CRACKS:
    print(f"  Tension crack   : depth={tension_crack_depth:.3f} m"
          f"{'  (water-filled)' if TENSION_CRACK_WATER_FILLED else ''}")
if ENABLE_SURCHARGE:
    print(f"  Surcharge       : {SURCHARGE_LOAD:.0f} Pa over x=[{SURCHARGE_X_MIN:.2f},{SURCHARGE_X_MAX:.2f}]")
if ENABLE_MULTILAYER:
    print(f"  Soil layers     : {len(SOIL_LAYERS)}")
if ENABLE_WAVE_BREAKING:
    print(f"  Wave breaking   : gamma_b={BREAKING_GAMMA}")
if ENABLE_VEGETATION:
    print(f"  Vegetation      : hv={VEG_HEIGHT:.2f} m  Nv={VEG_STEM_DENSITY:.0f} stems/m2")
if global_min_fs < 1.0:
    print("  -> SLOPE FAILURE predicted.  Mobilised material entered open domain.")
else:
    print("  -> Soil column remains STABLE under these conditions.")
if cum_eroded > 0:
    erosion_sources = []
    if ENABLE_WAVES:           erosion_sources.append("waves")
    if ENABLE_VESSEL_WAKE:     erosion_sources.append("vessel wake")
    if ENABLE_HYDRAULIC_SHEAR: erosion_sources.append("hydraulic shear")
    if erosion_sources:
        print("  -> Surface erosion active.  Bank face retreated due to "
              + ", ".join(erosion_sources) + ".")
print("=" * 60)

# Try to display in Jupyter / IPython if available
try:
    from IPython.display import Video, display
    display(Video(VIDEO_FILE, embed=True, html_attributes="controls loop autoplay"))
except Exception:
    pass
