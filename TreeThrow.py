#!/usr/bin/env python3
"""
TreeThrow.py — Tree-Throw Hillslope Landscape Evolution Model
=============================================================
A 2-D (plan-view) finite-volume landscape evolution model of a single
hillslope (ridge -> channel) in which soil is moved by two processes:

1. BACKGROUND HILLSLOPE DIFFUSION
   Continuous creep, rain-splash and bioturbation represented as a
   gradient-driven flux,  q = -D_eff * grad(z),  with

       linear     :  D_eff = D
       nonlinear  :  D_eff = D / (1 - (|grad z| / Sc)^2)     (Roering et al. 1999)

   and, when the soil model is on, an availability term
   D_eff *= 1 - exp(-h / h_star) so a bare-bedrock surface cannot creep.

2. TREE THROW (stochastic, individual-based)
   Every tree in the stand is tracked individually.  Each tree has a
   position, an age, a lifespan drawn from the species distribution and a
   mature root-plate (root wad) radius and depth.  The root plate grows with
   age toward its mature size.  When a tree reaches the end of its life it
   dies; a species-specific fraction of deaths are by UPROOTING (the rest snap
   or die standing and move no soil).  An uprooting event:

     * chooses a fall azimuth from a von Mises distribution whose mean and
       concentration combine a downslope bias (proportional to local gradient)
       and an optional prevailing wind,
     * excavates a paraboloid PIT of radius R and depth H at the stump
       (limited by soil thickness; a fraction of any root penetration below
       the soil-bedrock interface plucks bedrock -> biotic soil production),
     * rotates the root plate about a hinge at the pit edge, so the root-wad
       soil centroid moves ~ R + H/3 in the fall direction, and deposits it as
       an elliptical MOUND beyond the hinge.  A fraction of the soil falls
       back into the pit as the wad decays.

   The fallen tree is replaced (after an optional regeneration delay) by a
   seedling at a random location, so stand density is held at the species
   value.  Pit-mound microtopography is then smoothed by diffusion.

Boundary conditions:  x = 0 is the ridge/divide (no flux; tree-throw soil
thrown over the divide is mirrored back, i.e. a symmetric divide); x = L is a
channel whose elevation is fixed (or lowers at a set rate) and which removes
any sediment delivered to it.  The lateral (y) boundaries are periodic.

Three forest types are pre-parameterised (see SPECIES):  'pine',
'deciduous' and 'aspen'.  The values are representative, literature-informed
estimates intended as starting points; they should be calibrated against
field data (stand inventories, pit-mound surveys) for any specific site.

Key references for the process formulation
------------------------------------------
* Gabet, E.J., Reichman, O.J. & Seabloom, E.W. (2003) The effects of
  bioturbation on soil processes and sediment transport. Annu. Rev. Earth
  Planet. Sci.
* Gabet, E.J. & Mudd, S.M. (2010) Bedrock erosion by root fracture and tree
  throw: a coupled biogeomorphic model. J. Geophys. Res.
* Norman, S.A., Schaetzl, R.J. & Small, T.W. (1995) Effects of slope angle
  on mass movement by tree uprooting. Geomorphology.
* Roering, J.J., Kirchner, J.W. & Dietrich, W.E. (1999) Evidence for
  nonlinear, diffusive sediment transport on hillslopes. Water Resour. Res.
* Heimsath, A.M. et al. (1997) The soil production function. Nature.
* Constantine, J.A., Schelhaas, M.-J., Gabet, E., Mudd, S.M. (2012) Limits
  of windthrow-driven hillslope sediment flux due to varying storm frequency
  and intensity. Geomorphology.

Run
---
    python TreeThrow.py --species pine --length 60 --width 30 --slope 0.4
    python TreeThrow.py --species aspen --slope-deg 20 --years 3000
    python TreeThrow.py --compare --length 50 --width 20 --slope 0.35
    python TreeThrow.py --list-species

Outputs (per species) in --output (default ./tree_throw_output/<species>/):
    hillslope_evolution.png   maps, profiles, flux partitioning, roughness
    events.csv                every uprooting event (time, position, size ...)
    summary.json              run configuration and derived diagnostics
    final_state.npz           final elevation, soil thickness and bedrock grids
With --compare a species comparison figure and table are written as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ======================================================================
# 1. Species parameter sets
# ======================================================================


@dataclass(frozen=True)
class TreeSpecies:
    """Tree-throw parameters for one forest type.

    Sizes refer to a mature tree; individual trees draw their own mature size
    from a lognormal distribution with the given coefficient of variation, and
    reach it following  f(age) = 1 - exp(-3 age / maturity_age).
    """

    key: str
    name: str
    description: str
    lifespan_mean: float          # yr, mean tree longevity
    lifespan_sd: float            # yr
    density: float                # stems / ha in a mature stand
    uproot_fraction: float        # fraction of mortality that is by uprooting
    root_plate_diameter: float    # m, mature root-wad diameter
    root_plate_diameter_cv: float
    root_plate_depth: float       # m, mature root-wad (pit) depth
    root_plate_depth_cv: float
    maturity_age: float           # yr, age at ~95 % of mature root-plate size
    dbh: float                    # cm, mature diameter at breast height (informational)
    fall_slope_bias: float = 2.0  # von Mises concentration per unit gradient (downslope lean)
    regeneration_delay: float = 5.0  # yr, mean delay before a gap is re-stocked

    # ---- derived quantities ------------------------------------------------
    @property
    def uproot_rate_per_ha(self) -> float:
        """Steady-state uprooting events per hectare per year."""
        return self.density * self.uproot_fraction / (self.lifespan_mean + self.regeneration_delay)

    @property
    def mature_pit_volume(self) -> float:
        """Volume (m^3) of a paraboloid pit of mature size."""
        r = 0.5 * self.root_plate_diameter
        return 0.5 * math.pi * r * r * self.root_plate_depth

    def validate(self) -> None:
        checks = {
            "lifespan_mean": self.lifespan_mean > 0,
            "lifespan_sd": self.lifespan_sd >= 0,
            "density": self.density > 0,
            "uproot_fraction": 0.0 <= self.uproot_fraction <= 1.0,
            "root_plate_diameter": self.root_plate_diameter > 0,
            "root_plate_depth": self.root_plate_depth > 0,
            "root_plate_diameter_cv": self.root_plate_diameter_cv >= 0,
            "root_plate_depth_cv": self.root_plate_depth_cv >= 0,
            "maturity_age": self.maturity_age > 0,
            "fall_slope_bias": self.fall_slope_bias >= 0,
            "regeneration_delay": self.regeneration_delay >= 0,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"Invalid species parameters for '{self.key}': {', '.join(bad)}")


# Representative parameter sets.  Values are synthesised from the general
# forestry / tree-throw literature (stand inventories, windthrow surveys and
# pit-mound studies) and are deliberately round numbers: treat them as
# plausible defaults, not site measurements.
SPECIES: Dict[str, TreeSpecies] = {
    "pine": TreeSpecies(
        key="pine",
        name="Pine forest (Pinus spp., e.g. ponderosa / lodgepole)",
        description=("Moderately long-lived conifers at intermediate density. "
                     "Tap and sinker roots give a moderately wide but relatively "
                     "deep root plate; a moderate share of deaths are uprootings, "
                     "the rest being snags (beetle kill, fire) and stem breakage."),
        lifespan_mean=200.0, lifespan_sd=60.0,
        density=350.0,
        uproot_fraction=0.35,
        root_plate_diameter=2.4, root_plate_diameter_cv=0.25,
        root_plate_depth=1.0, root_plate_depth_cv=0.30,
        maturity_age=80.0,
        dbh=40.0,
        regeneration_delay=10.0,
    ),
    "deciduous": TreeSpecies(
        key="deciduous",
        name="Deciduous hardwood forest (Acer / Fagus / Quercus)",
        description=("Long-lived, large-crowned broadleaf trees at low density. "
                     "Broad, plate-like lateral root systems produce the largest "
                     "root wads; uprooting is a dominant mortality mode, giving "
                     "classic pit-and-mound microtopography."),
        lifespan_mean=250.0, lifespan_sd=75.0,
        density=220.0,
        uproot_fraction=0.50,
        root_plate_diameter=3.2, root_plate_diameter_cv=0.30,
        root_plate_depth=0.75, root_plate_depth_cv=0.30,
        maturity_age=100.0,
        dbh=50.0,
        regeneration_delay=5.0,
    ),
    "aspen": TreeSpecies(
        key="aspen",
        name="Aspen forest (Populus tremuloides)",
        description=("Short-lived, clonal stems at high density. Shallow lateral "
                     "roots give small, shallow root plates; heart rot makes stem "
                     "snapping common, so only a modest share of deaths uproot. "
                     "Suckering re-stocks gaps quickly."),
        lifespan_mean=90.0, lifespan_sd=25.0,
        density=1100.0,
        uproot_fraction=0.30,
        root_plate_diameter=1.5, root_plate_diameter_cv=0.25,
        root_plate_depth=0.5, root_plate_depth_cv=0.30,
        maturity_age=40.0,
        dbh=22.0,
        regeneration_delay=2.0,
    ),
}


def load_species_file(path: str) -> TreeSpecies:
    """Load a custom species from JSON.  Missing fields fall back to 'pine'."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    base = asdict(SPECIES["pine"])
    base.update(data)
    base.setdefault("key", Path(path).stem)
    sp = TreeSpecies(**base)
    sp.validate()
    return sp


# ======================================================================
# 2. Hillslope / numerical configuration
# ======================================================================


@dataclass
class HillslopeConfig:
    length: float = 50.0               # m, ridge-to-channel horizontal length
    width: float = 20.0                # m, across-slope width (periodic)
    slope: float = 0.35                # m/m, initial planar gradient
    dx: float = 0.25                   # m, grid spacing
    years: float = 5000.0              # yr, run duration
    dt: float = 1.0                    # yr, tree-population time step

    # Background diffusion
    diffusivity: float = 1.0e-3        # m^2/yr, background (non tree-throw) D
    transport_law: str = "linear"      # 'linear' or 'nonlinear'
    critical_slope: float = 1.25       # Sc for nonlinear law

    # Soil / bedrock
    soil_model: bool = True            # track soil thickness over bedrock
    initial_soil_depth: float = 0.8    # m
    soil_production_max: float = 1.0e-4   # m/yr bedrock lowering at h = 0
    soil_production_scale: float = 0.5    # m, h0 in P = P0 exp(-h/h0)
    transport_depth_scale: float = 0.2    # m, h* in D(h) = D (1 - exp(-h/h*))
    rock_to_soil_dilation: float = 2.0    # rho_rock / rho_soil
    bedrock_pluck_fraction: float = 0.3   # fraction of root penetration below soil that plucks rock

    # Tree throw
    tree_throw: bool = True
    infill_fraction: float = 0.2       # fraction of root-wad soil falling back into pit
    wind_direction_deg: float = 0.0    # azimuth of prevailing fall direction (0 = downslope, 90 = +y)
    wind_concentration: float = 0.0    # von Mises kappa of wind control (0 = no wind bias)
    stand_structure: str = "all-aged"  # 'all-aged' (steady state) or 'even-aged' (post-disturbance)

    # Base level
    baselevel_lowering_rate: float = 0.0  # m/yr channel incision

    # Output
    n_snapshots: int = 6
    record_interval: float = 10.0      # yr between time-series records
    seed: Optional[int] = 42

    def validate(self) -> None:
        if self.length <= 0 or self.width <= 0:
            raise ValueError("Hillslope length and width must be positive.")
        if self.slope < 0:
            raise ValueError("Slope must be non-negative.")
        if self.dx <= 0 or self.dx > min(self.length, self.width) / 4:
            raise ValueError("dx must be positive and much smaller than the hillslope dimensions.")
        if self.transport_law not in ("linear", "nonlinear"):
            raise ValueError("transport_law must be 'linear' or 'nonlinear'.")
        if self.transport_law == "nonlinear" and self.slope >= self.critical_slope:
            raise ValueError("Initial slope must be below the critical slope for nonlinear transport.")
        if self.stand_structure not in ("all-aged", "even-aged"):
            raise ValueError("stand_structure must be 'all-aged' or 'even-aged'.")
        if not 0 <= self.infill_fraction < 1:
            raise ValueError("infill_fraction must be in [0, 1).")
        if self.years <= 0 or self.dt <= 0:
            raise ValueError("years and dt must be positive.")


# ======================================================================
# 3. Analytic expectations
# ======================================================================


def _bessel_ratio(kappa: float) -> float:
    """Mean resultant length I1(k)/I0(k) of a von Mises distribution."""
    if kappa <= 0:
        return 0.0
    try:
        from scipy.special import i0e, i1e  # exponentially scaled, overflow-safe
        return float(i1e(kappa) / i0e(kappa))
    except ImportError:  # series / asymptotic fallback
        if kappa < 3:
            t = np.arange(0, 40)
            lg = np.array([math.lgamma(k + 1) for k in t])
            i0 = np.sum(np.exp(2 * t * math.log(kappa / 2) - 2 * lg))
            i1 = np.sum(np.exp((2 * t + 1) * math.log(kappa / 2) - lg - np.array([math.lgamma(k + 2) for k in t])))
            return float(i1 / i0)
        return 1 - 1 / (2 * kappa) - 1 / (8 * kappa ** 2)


def growth_factor(age, maturity_age: float):
    """Fraction of mature root-plate size reached at a given age."""
    age = np.maximum(age, 0.0)
    return 1.0 - np.exp(-3.0 * age / maturity_age)


def lognormal_factor(rng: np.random.Generator, cv: float, size, z=None):
    """Unit-mean lognormal multiplier; z (standard normal) may be supplied for correlation."""
    if cv <= 0:
        return np.ones(size)
    sigma = math.sqrt(math.log(1 + cv * cv))
    if z is None:
        z = rng.standard_normal(size)
    return np.exp(sigma * z - 0.5 * sigma * sigma)


def draw_lifespans(rng: np.random.Generator, sp: TreeSpecies, n: int) -> np.ndarray:
    life = rng.normal(sp.lifespan_mean, sp.lifespan_sd, n)
    return np.maximum(life, 0.25 * sp.lifespan_mean)


def draw_mature_sizes(rng: np.random.Generator, sp: TreeSpecies, n: int):
    """Mature root-plate radius and depth (m); size factors are correlated (r = 0.6)."""
    z1 = rng.standard_normal(n)
    z2 = 0.6 * z1 + 0.8 * rng.standard_normal(n)
    radius = 0.5 * sp.root_plate_diameter * lognormal_factor(rng, sp.root_plate_diameter_cv, n, z1)
    depth = sp.root_plate_depth * lognormal_factor(rng, sp.root_plate_depth_cv, n, z2)
    return radius, depth


def expected_tree_throw(sp: TreeSpecies, slope: float, infill_fraction: float = 0.2,
                        n_samples: int = 200_000, seed: int = 0) -> Dict[str, float]:
    """Monte-Carlo expectation of tree-throw transport on an infinite planar slope.

    Assumes unlimited soil depth and no wind bias.  Returns the event rate,
    mean event volume, mean displacement, mean downslope flux q (m^2/yr) and
    the equivalent linear diffusivity D_tt = q / S.
    """
    rng = np.random.default_rng(seed)
    life = draw_lifespans(rng, sp, n_samples)
    r_m, h_m = draw_mature_sizes(rng, sp, n_samples)
    f = growth_factor(life, sp.maturity_age)
    r, h = r_m * f, h_m * f
    vol = 0.5 * np.pi * r * r * h
    disp = (1 - infill_fraction) * (r + h / 3.0)
    kappa = sp.fall_slope_bias * slope
    mean_cos = _bessel_ratio(kappa)
    rate = sp.uproot_rate_per_ha / 1.0e4           # events / m^2 / yr
    q = rate * float(np.mean(vol * disp)) * mean_cos
    return {
        "events_per_ha_yr": sp.uproot_rate_per_ha,
        "mean_event_volume_m3": float(np.mean(vol)),
        "mean_displacement_m": float(np.sum(vol * disp) / np.sum(vol)),
        "mean_cos_fall": mean_cos,
        "flux_m2_per_yr": q,
        "diffusivity_m2_per_yr": q / slope if slope > 0 else float("nan"),
        "surface_turnover_yr": 1.0 / (rate * float(np.mean(np.pi * r * r))),
    }


# ======================================================================
# 4. The model
# ======================================================================


@dataclass
class RunRecord:
    time: List[float] = field(default_factory=list)
    events: List[int] = field(default_factory=list)
    export_tt: List[float] = field(default_factory=list)         # cumulative m^3 delivered to channel
    export_diff: List[float] = field(default_factory=list)
    moment_tt: List[float] = field(default_factory=list)         # cumulative sum(V * dx_downslope) m^4
    moment_diff: List[float] = field(default_factory=list)
    roughness: List[float] = field(default_factory=list)         # m, RMS detrended elevation
    mean_soil: List[float] = field(default_factory=list)
    relief: List[float] = field(default_factory=list)
    snapshot_times: List[float] = field(default_factory=list)
    snapshot_profiles: List[np.ndarray] = field(default_factory=list)
    snapshot_soil: List[np.ndarray] = field(default_factory=list)


class TreeThrowModel:
    """Hillslope evolution by background diffusion plus individual-tree tree throw."""

    def __init__(self, species: TreeSpecies, config: HillslopeConfig):
        species.validate()
        config.validate()
        self.sp = species
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        dx = config.dx
        self.nx = int(round(config.length / dx)) + 1    # last row is the channel node
        self.ny = max(int(round(config.width / dx)), 4)
        self.x = (np.arange(self.nx) + 0.5) * dx        # cell-centre distance from ridge
        self.y = (np.arange(self.ny) + 0.5) * dx
        self.L_active = (self.nx - 1) * dx              # hillslope (non-channel) length
        self.W = self.ny * dx
        self.cell_area = dx * dx

        # Elevation: planar slope, channel (row nx-1) at z = 0.
        z0 = config.slope * (self.x[-1] - self.x)
        z0 = np.repeat(z0[:, None], self.ny, axis=1)
        self.h = np.full((self.nx, self.ny), config.initial_soil_depth if config.soil_model else 0.0)
        self.zb = z0 - self.h
        self.z_initial = z0.copy()
        self.channel_z0 = 0.0
        self.time = 0.0

        # Bookkeeping
        self.export_tt = 0.0
        self.export_diff = 0.0
        self.moment_tt = 0.0
        self.moment_diff = 0.0
        self.rock_to_soil_tt = 0.0        # m^3 soil produced by root plucking
        self.rock_to_soil_sp = 0.0        # m^3 soil produced by the soil-production function
        self.clip_error = 0.0             # m^3 soil volume lost to negative-depth clipping
        self.n_events = 0
        self.n_deaths = 0
        self.event_log: List[tuple] = []
        self.record = RunRecord()

        self._init_stand()

    # ---------------------------------------------------------------- helpers
    @property
    def z(self) -> np.ndarray:
        return self.zb + self.h

    def _init_stand(self) -> None:
        sp, cfg = self.sp, self.cfg
        area_ha = self.L_active * self.W / 1.0e4
        n = max(int(self.rng.poisson(sp.density * area_ha)), 1)
        self.tx = self.rng.uniform(0, self.L_active, n)
        self.ty = self.rng.uniform(0, self.W, n)
        self.life = draw_lifespans(self.rng, sp, n)
        self.R_mature, self.H_mature = draw_mature_sizes(self.rng, sp, n)
        if cfg.stand_structure == "all-aged":
            # Ages for a stationary renewal process (including gaps awaiting regeneration).
            cycle = self.life + sp.regeneration_delay
            self.age = self.rng.uniform(0, 1, n) * cycle - sp.regeneration_delay
        else:
            self.age = np.zeros(n)

    def _replace_trees(self, idx: np.ndarray) -> None:
        sp, n = self.sp, idx.size
        self.tx[idx] = self.rng.uniform(0, self.L_active, n)
        self.ty[idx] = self.rng.uniform(0, self.W, n)
        self.life[idx] = draw_lifespans(self.rng, sp, n)
        r, h = draw_mature_sizes(self.rng, sp, n)
        self.R_mature[idx], self.H_mature[idx] = r, h
        delay = self.rng.exponential(sp.regeneration_delay, n) if sp.regeneration_delay > 0 else np.zeros(n)
        self.age[idx] = -delay

    def _local_gradient(self, x0: float, y0: float, baseline: float):
        """Downslope-pointing gradient vector (-dz/dx, -dz/dy) at a point, measured
        over +/- `baseline` metres so that it reflects the slope a tree 'feels'
        rather than pit-mound microtopography."""
        z, dx = self.z, self.cfg.dx
        k = max(int(round(baseline / dx)), 1)
        i = min(max(int(x0 / dx), 0), self.nx - 2)
        j = int(y0 / dx) % self.ny
        ip, im = min(i + k, self.nx - 1), max(i - k, 0)
        gx = -(z[ip, j] - z[im, j]) / ((ip - im) * dx)
        gy = -(z[i, (j + k) % self.ny] - z[i, (j - k) % self.ny]) / (2 * k * dx)
        return gx, gy

    def _fall_direction(self, x0: float, y0: float, R: float = 1.0) -> float:
        gx, gy = self._local_gradient(x0, y0, max(2.0 * R, 1.0))
        mvec = self.sp.fall_slope_bias * np.array([gx, gy])
        if self.cfg.wind_concentration > 0:
            a = math.radians(self.cfg.wind_direction_deg)
            mvec = mvec + self.cfg.wind_concentration * np.array([math.cos(a), math.sin(a)])
        kappa = float(np.hypot(*mvec))
        if kappa < 1e-9:
            return float(self.rng.uniform(-math.pi, math.pi))
        return float(self.rng.vonmises(math.atan2(mvec[1], mvec[0]), kappa))

    def _window(self, xc: float, yc: float, reach: float):
        """Unwrapped cell indices and cell-centre offsets covering a disc of radius `reach`."""
        dx = self.cfg.dx
        i = np.arange(int(math.floor((xc - reach) / dx)), int(math.floor((xc + reach) / dx)) + 1)
        j = np.arange(int(math.floor((yc - reach) / dx)), int(math.floor((yc + reach) / dx)) + 1)
        I, J = np.meshgrid(i, j, indexing="ij")
        return I, J, (I + 0.5) * dx - xc, (J + 0.5) * dx - yc

    # ------------------------------------------------------------ tree throw
    def uproot(self, x0: float, y0: float, R: float, H: float, phi: float) -> Dict[str, float]:
        """Apply one uprooting event at (x0, y0) with root plate radius R, depth H and
        fall azimuth phi (radians, 0 = +x = downslope).  Returns event diagnostics."""
        cfg, dx, ca = self.cfg, self.cfg.dx, self.cell_area
        R = max(R, 0.5 * dx)
        H = max(H, 1e-3)

        # ---- pit -----------------------------------------------------------
        I, J, ox, oy = self._window(x0, y0, R + dx)
        shape = np.clip(1.0 - (ox * ox + oy * oy) / (R * R), 0.0, None)
        if shape.sum() <= 0:                            # sub-grid plate: use host cell
            shape = ((I == int(x0 / dx)) & (J == int(y0 / dx))).astype(float)
        # Scale so the discrete pit volume equals the analytic paraboloid volume,
        # then drop any part of the pit beyond the ridge or in the channel cell.
        depth = shape * (0.5 * math.pi * R * R * H) / (shape.sum() * ca)
        depth *= (I >= 0) & (I < self.nx - 1)
        Ii, Jj = np.clip(I, 0, self.nx - 1), J % self.ny
        m = depth > 0
        Ii, Jj, depth = Ii[m], Jj[m], depth[m]

        if cfg.soil_model:
            h_here = self.h[Ii, Jj]
            soil_cut = np.minimum(depth, h_here)
            rock_cut = cfg.bedrock_pluck_fraction * np.maximum(depth - h_here, 0.0)
        else:
            soil_cut, rock_cut = depth, np.zeros_like(depth)
        np.subtract.at(self.h, (Ii, Jj), soil_cut)
        np.subtract.at(self.zb, (Ii, Jj), rock_cut)
        rock_soil = rock_cut.sum() * ca * cfg.rock_to_soil_dilation
        self.rock_to_soil_tt += rock_soil
        volume = soil_cut.sum() * ca + rock_soil         # soil-equivalent volume in root wad
        if volume <= 0:
            return {"volume": 0.0, "moment": 0.0, "export": 0.0, "R": R, "H": H}
        pit_x = np.sum((Ii + 0.5) * dx * (soil_cut + rock_cut * cfg.rock_to_soil_dilation)) * ca / volume

        # ---- infill (soil falling back into the pit) ----------------------
        w_in = soil_cut + rock_cut
        infill = cfg.infill_fraction * volume
        np.add.at(self.h, (Ii, Jj), infill * w_in / (w_in.sum() * ca))
        mound_vol = volume - infill

        # ---- mound: root plate rotated about a hinge at the pit edge -------
        d = R + H / 3.0                                   # centroid travel during rotation
        cphi, sphi = math.cos(phi), math.sin(phi)
        xm, ym = x0 + d * cphi, y0 + d * sphi
        a = max(0.75 * R, dx)                             # semi-axis along fall line
        b = max(R, dx)                                    # semi-axis across fall line
        I, J, ox, oy = self._window(xm, ym, max(a, b) + dx)
        u = ox * cphi + oy * sphi
        v = -ox * sphi + oy * cphi
        mshape = np.clip(1.0 - (u / a) ** 2 - (v / b) ** 2, 0.0, None)
        if mshape.sum() <= 0:
            mshape = np.zeros_like(u)
            mshape.flat[np.argmin(u * u + v * v)] = 1.0
        dep = mshape * mound_vol / (mshape.sum() * ca)    # thickness (m)
        dep_x = (I + 0.5) * dx                            # landing position (unmirrored)
        mound_x = np.sum(dep * dep_x) * ca / mound_vol

        out = I >= self.nx - 1                            # channel row or beyond -> exported
        exported = dep[out].sum() * ca
        keep = (~out) & (dep > 0)
        Ik = I[keep]
        Ik = np.where(Ik < 0, -1 - Ik, Ik)                # mirror across symmetric divide
        Ik = np.minimum(Ik, self.nx - 2)
        np.add.at(self.h, (Ik, J[keep] % self.ny), dep[keep])
        self.export_tt += exported

        # Downslope transport moment.  Mirrored soil really moved down the other
        # side of the divide, so the moment uses the unmirrored landing position.
        moment = mound_vol * (mound_x - pit_x)
        self.moment_tt += moment
        return {"volume": volume, "moment": moment, "export": exported, "R": R, "H": H}

    def _tree_step(self, dt: float) -> int:
        self.age += dt
        dead = np.flatnonzero(self.age >= self.life)
        if dead.size == 0:
            return 0
        self.n_deaths += dead.size
        throw = dead[self.rng.random(dead.size) < self.sp.uproot_fraction]
        for k in throw:
            f = growth_factor(self.life[k], self.sp.maturity_age)
            R, H = self.R_mature[k] * f, self.H_mature[k] * f
            phi = self._fall_direction(self.tx[k], self.ty[k], R)
            ev = self.uproot(self.tx[k], self.ty[k], R, H, phi)
            self.event_log.append((self.time, self.tx[k], self.ty[k], ev["R"], ev["H"],
                                   math.degrees(phi), ev["volume"], ev["export"]))
        self.n_events += throw.size
        self._replace_trees(dead)
        return int(throw.size)

    # ---------------------------------------------------------- diffusion
    def _diffusion_fluxes(self, z: np.ndarray, h: np.ndarray):
        """Return face-normal fluxes qx (nx-1, ny) between rows and qy (nx, ny) to +y."""
        cfg, dx = self.cfg, self.cfg.dx
        Sx = (z[1:] - z[:-1]) / dx                         # faces between rows i, i+1
        Sy = (np.roll(z, -1, axis=1) - z) / dx             # faces between cols j, j+1
        Dx = np.full_like(Sx, cfg.diffusivity)
        Dy = np.full_like(Sy, cfg.diffusivity)
        if cfg.transport_law == "nonlinear":
            gyc = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * dx)
            gxc = np.gradient(z, dx, axis=0)
            Sx_mag2 = Sx ** 2 + (0.5 * (gyc[1:] + gyc[:-1])) ** 2
            Sy_mag2 = Sy ** 2 + (0.5 * (gxc + np.roll(gxc, -1, axis=1))) ** 2
            Dx /= np.maximum(1 - Sx_mag2 / cfg.critical_slope ** 2, 0.05)
            Dy /= np.maximum(1 - Sy_mag2 / cfg.critical_slope ** 2, 0.05)
        if cfg.soil_model:
            hs = cfg.transport_depth_scale
            Dx *= 1 - np.exp(-0.5 * (h[1:] + h[:-1]) / hs)
            Dy *= 1 - np.exp(-0.5 * (h + np.roll(h, -1, axis=1)) / hs)
        return -Dx * Sx, -Dy * Sy, max(Dx.max(), Dy.max())

    def _diffusion_step(self, dt: float) -> None:
        cfg, dx = self.cfg, self.cfg.dx
        if cfg.diffusivity <= 0:
            return
        t_left = dt
        while t_left > 1e-12:
            qx, qy, dmax = self._diffusion_fluxes(self.z, self.h)
            sub = min(t_left, 0.2 * dx * dx / max(dmax, 1e-30))
            div = np.zeros((self.nx, self.ny))
            div[:-1] += qx / dx                            # outflow through lower face
            div[1:] -= qx / dx                             # inflow through upper face
            div += (qy - np.roll(qy, 1, axis=1)) / dx
            div[-1] = 0.0                                  # channel node is fixed
            self.export_diff += float(qx[-1].sum()) * dx * sub
            self.moment_diff += float(qx.sum()) * self.cell_area * sub
            self.h -= div * sub
            t_left -= sub
        neg = self.h < 0
        if cfg.soil_model and neg.any():
            self.clip_error += float(-self.h[neg].sum()) * self.cell_area
            self.h[neg] = 0.0

    def _soil_production(self, dt: float) -> None:
        cfg = self.cfg
        if not cfg.soil_model or cfg.soil_production_max <= 0:
            return
        P = cfg.soil_production_max * np.exp(-self.h[:-1] / cfg.soil_production_scale) * dt
        self.zb[:-1] -= P
        self.h[:-1] += P * cfg.rock_to_soil_dilation
        self.rock_to_soil_sp += float(P.sum()) * self.cell_area * cfg.rock_to_soil_dilation

    def _baselevel(self) -> None:
        zc = self.channel_z0 - self.cfg.baselevel_lowering_rate * self.time
        self.zb[-1] = zc - self.h[-1]

    # ------------------------------------------------------------- driver
    def roughness(self) -> float:
        """RMS deviation from the across-slope mean profile (pit-mound relief)."""
        z = self.z[:-1]
        return float(np.sqrt(np.mean((z - z.mean(axis=1, keepdims=True)) ** 2)))

    def step(self, dt: Optional[float] = None) -> int:
        dt = self.cfg.dt if dt is None else dt
        self.time += dt
        n = self._tree_step(dt) if self.cfg.tree_throw else 0
        self._soil_production(dt)
        self._diffusion_step(dt)
        self._baselevel()
        return n

    def _record(self, events: int) -> None:
        r = self.record
        r.time.append(self.time)
        r.events.append(events)
        r.export_tt.append(self.export_tt)
        r.export_diff.append(self.export_diff)
        r.moment_tt.append(self.moment_tt)
        r.moment_diff.append(self.moment_diff)
        r.roughness.append(self.roughness())
        r.mean_soil.append(float(self.h[:-1].mean()))
        prof = self.z.mean(axis=1)
        r.relief.append(float(prof[0] - prof[-1]))

    def _snapshot(self) -> None:
        self.record.snapshot_times.append(self.time)
        self.record.snapshot_profiles.append(self.z.mean(axis=1))
        self.record.snapshot_soil.append(self.h.mean(axis=1))

    def run(self, years: Optional[float] = None, progress: bool = True) -> RunRecord:
        years = self.cfg.years if years is None else years
        nsteps = int(math.ceil(years / self.cfg.dt))
        snap_every = max(nsteps // max(self.cfg.n_snapshots - 1, 1), 1)
        rec_every = max(int(round(self.cfg.record_interval / self.cfg.dt)), 1)
        self._record(0)
        self._snapshot()
        events = 0
        for k in range(1, nsteps + 1):
            events += self.step()
            if k % rec_every == 0 or k == nsteps:
                self._record(events)
                events = 0
            if k % snap_every == 0 or k == nsteps:
                if not self.record.snapshot_times or self.record.snapshot_times[-1] != self.time:
                    self._snapshot()
            if progress and (k % max(nsteps // 20, 1) == 0 or k == nsteps):
                print(f"\r  [{self.sp.key:>9}] {self.time:8.0f} / {years:.0f} yr  "
                      f"events={self.n_events:6d}", end="", flush=True)
        if progress:
            print()
        return self.record

    # ---------------------------------------------------------- diagnostics
    def summary(self) -> Dict:
        cfg, T = self.cfg, max(self.time, 1e-12)
        area = self.L_active * self.W
        q_tt = self.moment_tt / (area * T)       # mean downslope flux per unit width (m^2/yr)
        q_d = self.moment_diff / (area * T)
        s_mean = (self.z_initial[0, 0] - self.z_initial[-1, 0]) / (self.x[-1] - self.x[0])
        expected = expected_tree_throw(self.sp, cfg.slope, cfg.infill_fraction)
        prof0, prof1 = self.z_initial.mean(axis=1), self.z.mean(axis=1)
        return {
            "species": asdict(self.sp),
            "config": asdict(cfg),
            "grid": {"nx": self.nx, "ny": self.ny, "dx": cfg.dx,
                     "hillslope_length_m": self.L_active, "width_m": self.W},
            "years_simulated": self.time,
            "trees_in_stand": int(self.tx.size),
            "tree_deaths": self.n_deaths,
            "uprooting_events": self.n_events,
            "uprooting_rate_per_ha_yr": self.n_events / (area / 1e4) / T,
            "mean_flux_tree_throw_m2_yr": q_tt,
            "mean_flux_diffusion_m2_yr": q_d,
            "tree_throw_share_of_flux": q_tt / (q_tt + q_d) if (q_tt + q_d) > 0 else float("nan"),
            "effective_D_tree_throw_m2_yr": q_tt / s_mean if s_mean > 0 else float("nan"),
            "expected_D_tree_throw_m2_yr": expected["diffusivity_m2_per_yr"],
            "background_D_m2_yr": cfg.diffusivity,
            "sediment_yield_tree_throw_m3": self.export_tt,
            "sediment_yield_diffusion_m3": self.export_diff,
            "sediment_yield_m3_per_m_width_yr": (self.export_tt + self.export_diff) / (self.W * T),
            "soil_produced_by_root_plucking_m3": self.rock_to_soil_tt,
            "soil_produced_by_weathering_m3": self.rock_to_soil_sp,
            "final_mean_soil_depth_m": float(self.h[:-1].mean()),
            "final_roughness_m": self.roughness(),
            "relief_change_m": float((prof1[0] - prof1[-1]) - (prof0[0] - prof0[-1])),
            "max_ridge_lowering_m": float(prof0[0] - prof1[0]),
            "negative_soil_clip_m3": self.clip_error,
            "expected": expected,
        }

    def save(self, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "events.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_yr", "x_m", "y_m", "root_plate_radius_m", "root_plate_depth_m",
                        "fall_azimuth_deg", "volume_m3", "exported_m3"])
            w.writerows([[f"{v:.6g}" for v in row] for row in self.event_log])
        with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, indent=2)
        np.savez_compressed(outdir / "final_state.npz", x=self.x, y=self.y, z=self.z,
                            soil=self.h, bedrock=self.zb, z_initial=self.z_initial)


# ======================================================================
# 5. Plotting
# ======================================================================


def _import_pyplot():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt
    return plt


def hillshade(z: np.ndarray, dx: float, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    gx, gy = np.gradient(z, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    az, alt = math.radians(azimuth), math.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def plot_run(model: TreeThrowModel, path: Path) -> None:
    plt = _import_pyplot()
    rec, dx = model.record, model.cfg.dx
    t = np.array(rec.time)
    ext = [0, model.W, model.x[-1] + 0.5 * dx, 0]  # y across, x downslope (ridge at top)

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)
    fig.suptitle(f"Tree-throw hillslope evolution — {model.sp.name}\n"
                 f"L = {model.L_active:.0f} m, W = {model.W:.0f} m, S0 = {model.cfg.slope:.2f}, "
                 f"{model.time:.0f} yr, {model.n_events} uprootings", fontsize=12)

    ax = fig.add_subplot(gs[0, 0])
    detr = model.z - model.z.mean(axis=1, keepdims=True)
    ax.imshow(hillshade(model.z, dx), cmap="gray", extent=ext, aspect="equal")
    lim = max(np.percentile(np.abs(detr[:-1]), 99), 1e-3)
    im = ax.imshow(detr, cmap="BrBG", vmin=-lim, vmax=lim, alpha=0.55, extent=ext, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="pit–mound relief (m)")
    ax.set(title="Final surface (hillshade + detrended z)", xlabel="across-slope y (m)",
           ylabel="distance from ridge x (m)")

    ax = fig.add_subplot(gs[0, 1])
    dz = model.z - model.z_initial
    lim = max(np.percentile(np.abs(dz[:-1]), 99), 1e-3)
    im = ax.imshow(dz, cmap="RdBu", vmin=-lim, vmax=lim, extent=ext, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δz (m)")
    ax.set(title="Elevation change", xlabel="y (m)", ylabel="x (m)")

    ax = fig.add_subplot(gs[0, 2])
    if model.cfg.soil_model:
        im = ax.imshow(model.h, cmap="YlOrBr", extent=ext, aspect="equal", vmin=0)
        fig.colorbar(im, ax=ax, shrink=0.8, label="soil thickness (m)")
        ax.set(title="Soil thickness", xlabel="y (m)", ylabel="x (m)")
    else:
        ax.axis("off")

    ev = np.array(model.event_log) if model.event_log else np.zeros((0, 8))
    if len(ev):
        axp = fig.add_subplot(gs[0, 3], projection="polar")
        axp.hist(np.radians(ev[:, 5]), bins=24, weights=ev[:, 6], color="tab:green", alpha=0.75)
        axp.set_theta_zero_location("S")  # downslope (+x) plotted toward the bottom
        axp.set_title("Fall azimuths (volume-weighted)\n0° = downslope", fontsize=10)
    else:
        ax = fig.add_subplot(gs[0, 3])
        ax.text(0.5, 0.5, "no uprooting events", ha="center")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.get_cmap("viridis")
    nsnap = len(rec.snapshot_times)
    for k, (ts, prof) in enumerate(zip(rec.snapshot_times, rec.snapshot_profiles)):
        ax.plot(model.x, prof, color=cmap(k / max(nsnap - 1, 1)), label=f"{ts:.0f} yr")
    ax.set(title="Mean longitudinal profile", xlabel="distance from ridge (m)", ylabel="elevation (m)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    for k, (ts, prof) in enumerate(zip(rec.snapshot_times, rec.snapshot_profiles)):
        if k:
            ax.plot(model.x, prof - rec.snapshot_profiles[0], color=cmap(k / max(nsnap - 1, 1)))
    ax.axhline(0, color="k", lw=0.5)
    ax.set(title="Profile change since t = 0", xlabel="distance from ridge (m)", ylabel="Δz (m)")

    ax = fig.add_subplot(gs[1, 2])
    area_T = model.L_active * model.W
    with np.errstate(invalid="ignore", divide="ignore"):
        q_tt = np.array(rec.moment_tt) / (area_T * t)
        q_d = np.array(rec.moment_diff) / (area_T * t)
    ax.plot(t[1:], q_tt[1:] * 1e4, label="tree throw", color="tab:green")
    ax.plot(t[1:], q_d[1:] * 1e4, label="background diffusion", color="tab:brown")
    ax.plot(t[1:], (q_tt + q_d)[1:] * 1e4, label="total", color="k", lw=1)
    ax.set(title="Running mean downslope flux", xlabel="time (yr)", ylabel="q (cm² / yr)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(t, np.array(rec.roughness) * 100, color="tab:green", label="pit–mound roughness")
    ax.set(xlabel="time (yr)", ylabel="RMS roughness (cm)", title="Microtopography & soil")
    if model.cfg.soil_model:
        ax2 = ax.twinx()
        ax2.plot(t, rec.mean_soil, color="tab:orange", label="mean soil depth")
        ax2.set_ylabel("mean soil depth (m)", color="tab:orange")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(models: List[TreeThrowModel], path: Path) -> None:
    plt = _import_pyplot()
    colors = {"pine": "tab:green", "deciduous": "tab:orange", "aspen": "tab:olive"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Tree-throw landscape evolution: species comparison", fontsize=13)

    for m in models:
        c = colors.get(m.sp.key, None)
        t = np.array(m.record.time)
        area_T = m.L_active * m.W
        with np.errstate(invalid="ignore", divide="ignore"):
            q_tt = np.array(m.record.moment_tt) / (area_T * t)
        axes[0, 0].plot(m.x, m.z.mean(axis=1) - m.z_initial.mean(axis=1), color=c, label=m.sp.key)
        axes[0, 1].plot(t[1:], q_tt[1:] * 1e4, color=c, label=m.sp.key)
        axes[0, 2].plot(t, np.array(m.record.roughness) * 100, color=c, label=m.sp.key)
        y_tt = np.array(m.record.export_tt) / m.W
        y_d = np.array(m.record.export_diff) / m.W
        axes[1, 0].plot(t, y_tt + y_d, color=c, label=f"{m.sp.key} total")
        axes[1, 0].plot(t, y_tt, color=c, ls="--", lw=1)
        if m.cfg.soil_model:
            axes[1, 1].plot(t, m.record.mean_soil, color=c, label=m.sp.key)

    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set(title="Change in mean profile", xlabel="distance from ridge (m)", ylabel="Δz (m)")
    axes[0, 1].set(title="Tree-throw downslope flux (running mean)", xlabel="time (yr)", ylabel="q (cm²/yr)")
    axes[0, 2].set(title="Pit–mound roughness", xlabel="time (yr)", ylabel="RMS (cm)")
    axes[1, 0].set(title="Cumulative sediment yield to channel\n(dashed = tree-throw part)",
                   xlabel="time (yr)", ylabel="m³ per m width")
    axes[1, 1].set(title="Mean soil thickness", xlabel="time (yr)", ylabel="m")
    for ax in axes.flat[:5]:
        ax.legend(fontsize=8)

    ax = axes[1, 2]
    keys = [m.sp.key for m in models]
    sims = [m.summary() for m in models]
    xs = np.arange(len(keys))
    ax.bar(xs - 0.2, [s["effective_D_tree_throw_m2_yr"] * 1e4 for s in sims], 0.4,
           label="simulated D_tt", color=[colors.get(k) for k in keys])
    ax.bar(xs + 0.2, [s["expected_D_tree_throw_m2_yr"] * 1e4 for s in sims], 0.4,
           label="analytic D_tt", color="lightgray", edgecolor="k")
    ax.axhline(models[0].cfg.diffusivity * 1e4, color="tab:brown", ls=":", label="background D")
    ax.set_xticks(xs, keys)
    ax.set(title="Tree-throw diffusivity", ylabel="D (cm²/yr)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 6. Command line interface
# ======================================================================


def species_table(slope: float, infill_fraction: float = 0.2, species=None) -> str:
    species = species or list(SPECIES.values())
    rows = [("Parameter", *[s.key for s in species])]

    def row(label, fn, fmt):
        rows.append((label, *[fmt.format(fn(s)) for s in species]))

    exp = {s.key: expected_tree_throw(s, slope, infill_fraction) for s in species}
    row("Mean lifespan (yr)", lambda s: f"{s.lifespan_mean:.0f} ± {s.lifespan_sd:.0f}", "{}")
    row("Stand density (stems/ha)", lambda s: s.density, "{:.0f}")
    row("Uprooting share of mortality", lambda s: s.uproot_fraction, "{:.2f}")
    row("Mature root-plate diameter (m)", lambda s: s.root_plate_diameter, "{:.2f}")
    row("Mature root-plate depth (m)", lambda s: s.root_plate_depth, "{:.2f}")
    row("Mature DBH (cm)", lambda s: s.dbh, "{:.0f}")
    row("Uprootings (/ha/yr)", lambda s: s.uproot_rate_per_ha, "{:.2f}")
    row("Mean event volume (m³)", lambda s: exp[s.key]["mean_event_volume_m3"], "{:.2f}")
    row("Mean displacement (m)", lambda s: exp[s.key]["mean_displacement_m"], "{:.2f}")
    row(f"Expected D_tt at S={slope:.2f} (m²/yr)", lambda s: exp[s.key]["diffusivity_m2_per_yr"], "{:.2e}")
    row("Surface turnover time (yr)", lambda s: exp[s.key]["surface_turnover_yr"], "{:.0f}")
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "-" * len(lines[0]))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tree-throw hillslope landscape evolution model.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("forest")
    g.add_argument("--species", default="pine", choices=sorted(SPECIES),
                   help="forest type to simulate")
    g.add_argument("--species-file", help="JSON file defining a custom species (overrides --species)")
    g.add_argument("--compare", action="store_true", help="run all three built-in species")
    g.add_argument("--list-species", action="store_true", help="print species parameters and exit")
    g.add_argument("--stand", dest="stand_structure", default="all-aged", choices=["all-aged", "even-aged"])
    g.add_argument("--density", type=float, help="override stand density (stems/ha)")
    g.add_argument("--lifespan", type=float, help="override mean lifespan (yr)")
    g.add_argument("--root-diameter", type=float, help="override mature root-plate diameter (m)")
    g.add_argument("--root-depth", type=float, help="override mature root-plate depth (m)")
    g.add_argument("--uproot-fraction", type=float, help="override share of deaths by uprooting")

    g = p.add_argument_group("hillslope")
    g.add_argument("--length", type=float, default=50.0, help="hillslope length, ridge to channel (m)")
    g.add_argument("--width", type=float, default=20.0, help="hillslope width (m)")
    s = g.add_mutually_exclusive_group()
    s.add_argument("--slope", type=float, help="initial gradient (m/m) [default 0.35]")
    s.add_argument("--slope-deg", type=float, help="initial slope angle (degrees)")
    g.add_argument("--dx", type=float, default=0.25, help="grid spacing (m)")
    g.add_argument("--years", type=float, default=5000.0, help="simulation length (yr)")

    g = p.add_argument_group("processes")
    g.add_argument("--diffusivity", type=float, default=1.0e-3, help="background diffusivity (m²/yr)")
    g.add_argument("--transport", dest="transport_law", default="linear", choices=["linear", "nonlinear"])
    g.add_argument("--critical-slope", type=float, default=1.25)
    g.add_argument("--no-soil", action="store_true", help="disable the soil/bedrock model")
    g.add_argument("--soil-depth", type=float, default=0.8, help="initial soil thickness (m)")
    g.add_argument("--no-tree-throw", action="store_true", help="diffusion-only control run")
    g.add_argument("--infill", type=float, default=0.2, help="fraction of root-wad soil falling back into pit")
    g.add_argument("--wind-dir", type=float, default=0.0, help="prevailing fall azimuth (deg, 0 = downslope)")
    g.add_argument("--wind-kappa", type=float, default=0.0, help="wind concentration (0 = none)")
    g.add_argument("--baselevel-rate", type=float, default=0.0, help="channel lowering rate (m/yr)")

    g = p.add_argument_group("output")
    g.add_argument("--output", default="tree_throw_output", help="output directory")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--quiet", action="store_true")
    return p


def config_from_args(a) -> HillslopeConfig:
    if a.slope_deg is not None:
        slope = math.tan(math.radians(a.slope_deg))
    else:
        slope = 0.35 if a.slope is None else a.slope
    return HillslopeConfig(
        length=a.length, width=a.width, slope=slope, dx=a.dx, years=a.years,
        diffusivity=a.diffusivity, transport_law=a.transport_law, critical_slope=a.critical_slope,
        soil_model=not a.no_soil, initial_soil_depth=a.soil_depth,
        tree_throw=not a.no_tree_throw, infill_fraction=a.infill,
        wind_direction_deg=a.wind_dir, wind_concentration=a.wind_kappa,
        stand_structure=a.stand_structure, baselevel_lowering_rate=a.baselevel_rate, seed=a.seed,
    )


def apply_overrides(sp: TreeSpecies, a) -> TreeSpecies:
    changes = {k: v for k, v in {
        "density": a.density, "lifespan_mean": a.lifespan,
        "root_plate_diameter": a.root_diameter, "root_plate_depth": a.root_depth,
        "uproot_fraction": a.uproot_fraction}.items() if v is not None}
    sp = replace(sp, **changes) if changes else sp
    sp.validate()
    return sp


def run_species(sp: TreeSpecies, cfg: HillslopeConfig, outdir: Path, plots: bool, quiet: bool) -> TreeThrowModel:
    model = TreeThrowModel(sp, cfg)
    if not quiet:
        print(f"Running {sp.name}: grid {model.nx} x {model.ny}, {model.tx.size} trees, {cfg.years:.0f} yr")
    model.run(progress=not quiet)
    model.save(outdir)
    if plots:
        plot_run(model, outdir / "hillslope_evolution.png")
    s = model.summary()
    if not quiet:
        print(f"    uprootings: {s['uprooting_events']}  ({s['uprooting_rate_per_ha_yr']:.2f} /ha/yr)")
        print(f"    mean flux  : tree throw {s['mean_flux_tree_throw_m2_yr']:.2e} m²/yr, "
              f"diffusion {s['mean_flux_diffusion_m2_yr']:.2e} m²/yr "
              f"(tree-throw share {100 * s['tree_throw_share_of_flux']:.0f} %)")
        print(f"    D_tt       : simulated {s['effective_D_tree_throw_m2_yr']:.2e}, "
              f"analytic {s['expected_D_tree_throw_m2_yr']:.2e} m²/yr")
        print(f"    yield      : {s['sediment_yield_m3_per_m_width_yr'] * 1e4:.2f} cm³ per cm width per yr")
        print(f"    outputs in : {outdir}")
    return model


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    if a.list_species:
        print(species_table(cfg.slope, cfg.infill_fraction))
        for sp in SPECIES.values():
            print(f"\n{sp.key}: {sp.name}\n  {sp.description}")
        return 0

    out = Path(a.output)
    if a.compare:
        species = [apply_overrides(s, a) for s in SPECIES.values()]
    else:
        base = load_species_file(a.species_file) if a.species_file else SPECIES[a.species]
        species = [apply_overrides(base, a)]
    if not a.quiet:
        print(species_table(cfg.slope, cfg.infill_fraction, species))
        print()

    models = [run_species(sp, cfg, out / sp.key, not a.no_plots, a.quiet) for sp in species]
    if a.compare:
        summaries = {m.sp.key: m.summary() for m in models}
        with open(out / "comparison_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summaries, fh, indent=2)
        if not a.no_plots:
            plot_comparison(models, out / "species_comparison.png")
        if not a.quiet:
            print(f"\nComparison written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
