"""Equilibrium open-channel columns and synthetic river turbulence (SI units).

The advanced solver integrates the steady 1DV RANS momentum balance, not a
2D/3D Navier--Stokes system. See StreamBed.md for closures and verification.
"""
from __future__ import annotations

import math
import numpy as np

KAPPA = .41


def startup_factor(s, t):
    """C1-continuous forcing hydrograph, zero velocity/u* at flow time zero."""
    if s.STREAM_STARTUP_TIME <= 0:
        return 1.
    q = float(np.clip(t/s.STREAM_STARTUP_TIME, 0., 1.))
    return q*q*(3-2*q)


def log_mean_factor(depth, z0):
    """Exact depth integral of log(1+y/z0), divided by depth and kappa."""
    r = np.asarray(depth) / z0
    return ((1 + 1/r)*np.log1p(r) - 1)/KAPPA


def column_profile(ustar, depth, z0, viscosity, levels, solver):
    """Return u, du/dy, nu_t and total shear/rho on a stretched vertical grid.

    RANS: (nu + l^2 |du/dy|) du/dy = ustar^2 (1-y/H).
    l = kappa (y+z0) sqrt(1-y/H), a rough-wall mixing-length closure.
    Integration uses piecewise constant gradients on cell midpoints; it
    conserves the imposed linear total shear exactly at those midpoints.
    """
    h = np.asarray(depth, dtype=float)
    us = np.asarray(ustar, dtype=float)
    eta = np.linspace(0., 1., levels)**2
    y = h[:, None]*eta
    if solver == "simple":
        u = us[:, None]/KAPPA*np.log1p(y/z0)
        grad = us[:, None]/(KAPPA*(y+z0))
        nu_t = np.zeros_like(u)
        shear = us[:, None]**2*(1-eta)
        return eta, u, grad, nu_t, shear
    ym = .5*(y[:, 1:]+y[:, :-1])
    relative = np.maximum(0., 1-ym/h[:, None])
    l2 = KAPPA**2*(ym+z0)**2*relative
    shear = us[:, None]**2*relative
    # Rationalized positive root avoids cancellation for small shear.
    grad = 2*shear/(viscosity + np.sqrt(viscosity**2+4*l2*shear))
    nu_t = l2*grad
    u = np.column_stack((np.zeros(len(h)), np.cumsum(grad*np.diff(y), axis=1)))
    return eta, u, grad, nu_t, shear


class RiverHydraulics:
    """Hydrostatic equilibrium columns above an evolving, smoothed grain bed.

    A fixed water level and prescribed unit discharge are boundary conditions
    in velocity mode. Independent columns share q; no streamwise momentum or
    free-surface evolution is solved. Hydraulic refresh is independent of DEM dt.
    """
    def __init__(self, s, backend):
        self.s, self.backend = s, backend
        self.x = np.linspace(0., s.DOMAIN_X, s.FLOW_COLUMNS)
        self.water_level = s.SLOPE_HEIGHT+s.FLOW_DEPTH
        self.bed = np.full(len(self.x), s.SLOPE_HEIGHT)
        self.depth = np.full(len(self.x), s.FLOW_DEPTH)
        # Surface percentile is projected-area weighted; rectangles use b-axis,
        # consistent with the intermediate diameter used in gravel surveys.
        self.diameters = np.array([d if kind == "circle" else b for d, kind, b in
                                  zip(backend.particle_sizes, backend.particle_kinds,
                                      backend.particle_b_sizes)])
        self.d84 = float(np.quantile(self.diameters, .84))
        self.z0 = s.ROUGHNESS_D84_FACTOR*self.d84/30
        self.last_refresh = -math.inf
        self.time = None
        self.rng = np.random.default_rng(s.TURBULENCE_SEED)
        self.noise = None
        self.correlation_time = math.inf
        self.mean_tau = self.tau = np.zeros(len(self.x))
        self.ustar = np.zeros(len(self.x))
        self.fluctuation_u = self.fluctuation_w = np.zeros((len(self.x), s.FLOW_LEVELS))
        self.discharge_error, self.momentum_error = 0., math.nan

    def _solve(self, t, exposure, active):
        s = self.s
        ramp = startup_factor(s, t)
        weights = np.asarray(exposure)*np.asarray(self.backend.particle_planform_areas)*active
        if weights.sum() > 0:
            order = np.argsort(self.diameters)
            cdf = np.cumsum(weights[order])
            self.d84 = float(self.diameters[order][np.searchsorted(cdf, .84*cdf[-1])])
        self.z0 = s.ROUGHNESS_D84_FACTOR*self.d84/30
        self.depth = self.water_level-self.bed
        if np.any(self.depth <= 0):
            raise ValueError("The connected sediment bed reaches the specified water surface. Increase flow depth.")
        if s.STREAM_FORCING == "velocity":
            target = ramp*s.FLOW_MEAN_VELOCITY*s.FLOW_DEPTH/self.depth
            if s.HYDRAULIC_SOLVER == "simple":
                self.ustar = target/log_mean_factor(self.depth, self.z0)
            else:
                lo = np.zeros(len(self.x))
                hi = np.maximum(target, .01)
                for _ in range(12):
                    eta, u, *_ = column_profile(hi, self.depth, self.z0, s.WATER_VISCOSITY,
                                                s.FLOW_LEVELS, "advanced")
                    means = np.trapezoid(u, eta, axis=1)
                    if np.all(means >= target):
                        break
                    hi = np.where(means < target, hi*2, hi)
                else:
                    raise ValueError("Cannot bracket hydraulic friction velocity.")
                for _ in range(42):
                    mid = (lo+hi)/2
                    eta, u, *_ = column_profile(mid, self.depth, self.z0, s.WATER_VISCOSITY,
                                                s.FLOW_LEVELS, "advanced")
                    means = np.trapezoid(u, eta, axis=1)
                    lo, hi = np.where(means < target, mid, lo), np.where(means >= target, mid, hi)
                self.ustar = np.where(target > 0, (lo+hi)/2, 0.)
        else:
            tau = s.BED_STRESS_MEAN + (s.BED_STRESS_AMPLITUDE*math.sin(
                2*math.pi*s.BED_STRESS_FREQUENCY*t) if s.STREAM_FORCING == "harmonic" else 0.)
            self.ustar = np.full(len(self.x), ramp*math.sqrt(max(0., tau)/s.WATER_DENSITY))
        self.eta, self.u, self.gradient, self.eddy_viscosity, self.shear = column_profile(
            self.ustar, self.depth, self.z0, s.WATER_VISCOSITY, s.FLOW_LEVELS, s.HYDRAULIC_SOLVER)
        self.mean_velocity = (self.ustar*log_mean_factor(self.depth, self.z0) if s.HYDRAULIC_SOLVER == "simple"
                              else np.trapezoid(self.u, self.eta, axis=1))
        self.mean_tau = s.WATER_DENSITY*self.ustar**2
        self.unit_discharge = self.mean_velocity*self.depth
        self.energy_slope = self.ustar**2/(s.GRAVITY_MAG*self.depth)
        self.reynolds_number = self.mean_velocity*self.depth/s.WATER_VISCOSITY
        self.froude_number = self.mean_velocity/np.sqrt(s.GRAVITY_MAG*self.depth)
        self.roughness_reynolds_number = self.ustar*(30*self.z0)/s.WATER_VISCOSITY
        q = ramp*s.FLOW_MEAN_VELOCITY*s.FLOW_DEPTH
        self.discharge_error = (float(np.max(np.abs(self.unit_discharge-q)))/max(q, 1e-15)
                                if s.STREAM_FORCING == "velocity" else 0.)
        if s.HYDRAULIC_SOLVER == "advanced":
            modeled = (s.WATER_VISCOSITY+self.eddy_viscosity)*self.gradient
            self.momentum_error = float(np.max(np.abs(modeled-self.shear)))/max(float(np.max(self.ustar**2)), 1e-15)
        self.last_refresh = t

    def _innovation(self):
        """Unit-variance Gaussian fields with separable exponential covariance."""
        s = self.s
        h = float(self.depth.mean())
        ax = math.exp(-(self.x[1]-self.x[0])/(s.TURBULENCE_LENGTH_X*h))
        az = np.exp(-np.diff(self.eta)/s.TURBULENCE_LENGTH_Z)
        g = self.rng.normal(size=(3, len(self.x), len(self.eta)))
        for i in range(1, len(self.x)):
            g[:, i] = ax*g[:, i-1]+math.sqrt(1-ax*ax)*g[:, i]
        for j in range(1, len(self.eta)):
            g[:, :, j] = az[j-1]*g[:, :, j-1]+math.sqrt(1-az[j-1]**2)*g[:, :, j]
        return g

    def update(self, t, xs, bed, exposure, active):
        s = self.s
        # Smooth over a hydraulic column to avoid treating each grain-scale
        # peak as a separate channel constriction. Keep the bed-connected mean.
        edges = np.linspace(0., s.DOMAIN_X, len(self.x)+1)
        index = np.minimum(np.searchsorted(edges, xs, side="right")-1, len(self.x)-1)
        count = np.bincount(index, minlength=len(self.x))
        sampled = np.bincount(index, weights=bed, minlength=len(self.x))/np.maximum(count, 1)
        if t-self.last_refresh >= s.FLOW_UPDATE_INTERVAL-1e-12 or self.time is None:
            self.bed = np.where(count > 0, sampled, np.interp(self.x, xs, bed))
            self._solve(t, exposure, active)
        # Harmonic input is evaluated at every actual DEM timestep.
        elif s.STREAM_FORCING == "harmonic" or startup_factor(s, t) != startup_factor(s, self.last_refresh):
            self._solve(t, exposure, active)
        self.tau = self.mean_tau.copy()
        self.fluctuation_u = np.zeros_like(self.u)
        self.fluctuation_w = np.zeros_like(self.u)
        if s.SIMULATE_TURBULENCE:
            # Eulerian sweeping time: large-eddy length divided by convection
            # speed. Finite at zero mean flow; amplitudes then vanish with u*.
            h = float(self.depth.mean())
            self.correlation_time = s.TURBULENCE_TIME_FACTOR*h/max(float(self.mean_velocity.mean()),
                                                                  float(self.ustar.mean()), 1e-6)
            if self.noise is None:
                self.noise = self._innovation()
            elif t > self.time:
                a = math.exp(-(t-self.time)/self.correlation_time)
                self.noise = a*self.noise+math.sqrt(-math.expm1(-2*(t-self.time)/self.correlation_time))*self._innovation()
            eta = self.eta[None, :]
            su = 2.3*self.ustar[:, None]*np.exp(-eta)
            # Rigid-lid damping enforces zero fluctuating normal velocity at
            # the fixed free surface; the empirical exponential is an outer-
            # layer envelope, not a boundary condition by itself.
            sw = 1.27*self.ustar[:, None]*np.exp(-eta)*np.sqrt(1-eta)
            # Match the modeled Reynolds shear, not an arbitrary component
            # correlation. Viscous shear is excluded in the RANS option.
            if s.HYDRAULIC_SOLVER == "advanced":
                rs = self.eddy_viscosity*self.gradient
                rs = np.column_stack((rs[:, 0], .5*(rs[:, 1:]+rs[:, :-1]), np.zeros(len(rs))))
            else:
                rs = self.ustar[:, None]**2*(1-eta)
            corr = np.clip(-rs/np.maximum(su*sw, 1e-30), -.999, .999)
            amp = s.TURBULENCE_INTENSITY
            self.fluctuation_u = amp*su*self.noise[0]
            self.fluctuation_w = amp*sw*(corr*self.noise[0]+np.sqrt(1-corr*corr)*self.noise[1])
            self.fluctuation_u[:, 0] = 0.
            self.fluctuation_w[:, 0] = 0.
            # Positive intermittent stress with E[tau]=rho u*^2 and
            # std(tau)=CV rho u*^2, independent of grid or timestep.
            sigma = math.sqrt(math.log1p((amp*s.TURBULENCE_STRESS_CV)**2))
            if s.TURBULENCE_TARGET == "stress":
                self.tau *= np.exp(sigma*self.noise[2, :, 0]-.5*sigma*sigma)
        else:
            # Switching turbulence off discards its dormant eddies; enabling
            # it later starts a stationary realization at the current flow.
            self.noise = None
        self.time = t

    def sample(self, points):
        """Bilinear sampling in streamwise position and relative bed height."""
        p = np.asarray(points)
        j = np.clip(np.searchsorted(self.x, p[:, 0])-1, 0, len(self.x)-2)
        f = np.clip((p[:, 0]-self.x[j])/(self.x[j+1]-self.x[j]), 0, 1)
        bed = (1-f)*self.bed[j]+f*self.bed[j+1]
        h = self.water_level-bed
        y = p[:, 1]-bed
        # Sample exposed caps within the unresolved roughness layer at half
        # a surface diameter. Shielding is handled by the grain-force adapter.
        e = np.clip(np.maximum(y, np.minimum(.5*self.d84, .99*h))/h, 0, 1)
        k = np.clip(np.searchsorted(self.eta, e)-1, 0, len(self.eta)-2)
        g = (e-self.eta[k])/(self.eta[k+1]-self.eta[k])

        def interp(field):
            return (1-f)*((1-g)*field[j, k]+g*field[j, k+1])+f*((1-g)*field[j+1, k]+g*field[j+1, k+1])

        u = interp(self.u)
        du, w = interp(self.fluctuation_u), interp(self.fluctuation_w)
        velocity_turbulence = self.s.SIMULATE_TURBULENCE and self.s.TURBULENCE_TARGET == "velocity"
        return (u+du if velocity_turbulence else u), (w if velocity_turbulence else np.zeros(len(p))), np.interp(
            p[:, 0], self.x, self.tau if self.s.TURBULENCE_TARGET == "stress" else self.mean_tau)

    def snapshot(self):
        velocity = self.u.copy()
        vertical = np.zeros_like(velocity)
        if self.s.SIMULATE_TURBULENCE and self.s.TURBULENCE_TARGET == "velocity":
            velocity += self.fluctuation_u
            vertical += self.fluctuation_w
        return dict(x=self.x.copy(), eta=self.eta.copy(), bed=self.bed.copy(), depth=self.depth.copy(),
                    water_level=self.water_level, velocity=velocity, vertical_velocity=vertical,
                    mean_velocity=self.u.copy(), stress=self.tau.copy(), mean_stress=self.mean_tau.copy(),
                    ustar=self.ustar.copy(), z0=self.z0, d84=self.d84,
                    discharge=self.unit_discharge.copy(), energy_slope=self.energy_slope.copy(),
                    reynolds_number=self.reynolds_number.copy(), froude_number=self.froude_number.copy(),
                    roughness_reynolds_number=self.roughness_reynolds_number.copy(),
                    momentum_error=self.momentum_error, discharge_error=self.discharge_error,
                    startup_factor=startup_factor(self.s, self.time),
                    turbulence_active=bool(self.s.SIMULATE_TURBULENCE))
