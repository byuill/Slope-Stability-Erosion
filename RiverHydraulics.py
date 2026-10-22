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


class _ColumnGrid:
    """Geometry shared by profile construction and the friction-velocity solve."""
    def __init__(self, depth, z0, viscosity, levels):
        self.depth = np.asarray(depth, dtype=float)
        self.z0, self.viscosity = z0, viscosity
        self.eta = np.linspace(0., 1., levels)**2
        self.y = self.depth[:, None]*self.eta
        self.dy = np.diff(self.y, axis=1)
        self.mid_eta = .5*(self.eta[1:]+self.eta[:-1])
        self.relative = 1-self.mid_eta
        ym = self.depth[:, None]*self.mid_eta
        self.length2 = KAPPA**2*(ym+z0)**2*self.relative
        # A cell's gradient contributes to every velocity node above it.
        # This is exactly trapezoid(cumsum(gradient*dy), eta), without
        # constructing any profiles during root finding.
        self.mean_weights = self.dy*(1-self.mid_eta)

    def gradients(self, ustar):
        us = np.asarray(ustar)[:, None]
        shear = us**2*self.relative
        root = np.sqrt(self.viscosity**2+4*self.length2*shear)
        gradient = 2*shear/(self.viscosity+root)
        derivative = 2*us*self.relative/root
        return gradient, derivative

    def mean(self, ustar):
        gradient, derivative = self.gradients(ustar)
        return (np.sum(gradient*self.mean_weights, axis=1),
                np.sum(derivative*self.mean_weights, axis=1))

    def friction_velocity(self, target, previous=None):
        """Safeguarded Newton solve, with exact discrete discharge and a warm start."""
        target = np.asarray(target)
        lo = np.zeros_like(target)
        hi = np.maximum(target, .01)
        for _ in range(32):
            mean, _ = self.mean(hi)
            below = mean < target
            if not np.any(below):
                break
            hi = np.where(below, hi*2, hi)
        else:
            raise ValueError("Cannot bracket hydraulic friction velocity.")
        guess = target/log_mean_factor(self.depth, self.z0) if previous is None else previous
        us = np.clip(guess, lo, hi)
        tolerance = 1e-11*np.maximum(target, 1e-12)
        for _ in range(64):
            mean, derivative = self.mean(us)
            residual = mean-target
            done = (np.abs(residual) <= tolerance) | (target == 0)
            if np.all(done):
                return np.where(target > 0, us, 0.)
            lo = np.where(residual < 0, us, lo)
            hi = np.where(residual >= 0, us, hi)
            candidate = us-np.divide(residual, derivative, out=np.zeros_like(us), where=derivative > 0)
            inside = (candidate > lo) & (candidate < hi)
            us = np.where(done, us, np.where(inside, candidate, .5*(lo+hi)))
        raise ValueError("Hydraulic friction-velocity solve did not converge.")

    def profile(self, ustar, solver):
        us = np.asarray(ustar)[:, None]
        if solver == "simple":
            # Rough-wall log core with an outer-layer correction: no slip
            # at the bed and zero velocity gradient at the free surface.
            u = us/KAPPA*((1+self.z0/self.depth[:, None])*np.log1p(self.y/self.z0)-self.eta)
            gradient = us/KAPPA*(1-self.eta)/(self.y+self.z0)
            return self.eta, u, gradient, np.zeros_like(u), us**2*(1-self.eta)
        if solver != "advanced":
            raise ValueError("Flow solver must be simple or advanced.")
        gradient, _ = self.gradients(ustar)
        u = np.column_stack((np.zeros(len(us)), np.cumsum(gradient*self.dy, axis=1)))
        return self.eta, u, gradient, self.length2*gradient, us**2*self.relative


def column_profile(ustar, depth, z0, viscosity, levels, solver):
    """Build a no-slip, shear-free open-channel profile on a stretched grid.

    Advanced: (nu + l^2 |du/dy|) du/dy = ustar^2 (1-y/H),
    with l = kappa (y+z0) sqrt(1-y/H). Gradients/stress are cell-centered;
    simple-solver gradients/stress and all velocities are node-centered.
    """
    return _ColumnGrid(depth, z0, viscosity, levels).profile(ustar, solver)


class RiverHydraulics:
    """Hydrostatic equilibrium columns above an evolving, smoothed grain bed.

    Columns share discharge under a fixed water level. Stress inputs specify
    reference conditions at the initial depth. Mean vertical motion follows
    continuity; no streamwise momentum/free-surface evolution is solved.
    Bed geometry refresh is independent of DEM dt.
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
        self._diameter_order = np.argsort(self.diameters)
        self._particle_areas = np.asarray(backend.particle_planform_areas)
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
        self._grid = None
        self._sample_grid = None
        self.vertical_velocity = np.zeros_like(self.fluctuation_u)

    def _refresh_geometry(self, exposure, active):
        s = self.s
        weights = np.asarray(exposure)*self._particle_areas*active
        if weights.sum() > 0:
            order = self._diameter_order
            cdf = np.cumsum(weights[order])
            self.d84 = float(self.diameters[order][np.searchsorted(cdf, .84*cdf[-1])])
        self.z0 = s.ROUGHNESS_D84_FACTOR*self.d84/30
        self.depth = self.water_level-self.bed
        if np.any(self.depth <= 0):
            raise ValueError("The connected sediment bed reaches the specified water surface. Increase flow depth.")
        self._grid = _ColumnGrid(self.depth, self.z0, s.WATER_VISCOSITY, s.FLOW_LEVELS)
        self._reference_grid = _ColumnGrid([s.FLOW_DEPTH], self.z0, s.WATER_VISCOSITY, s.FLOW_LEVELS)
        unit = self._grid.profile(np.ones(len(self.x)), "simple")
        factor = np.trapezoid(unit[1], unit[0], axis=1)
        self._bed_slope = np.gradient(self.bed, self.x, edge_order=2)
        self._simple_ustar = 1/(self.depth*factor)
        self._simple_profile = self._grid.profile(self._simple_ustar, "simple")
        self._simple_vertical = self._vertical(self._simple_profile[1])
        ref_eta, ref_u, *_ = self._reference_grid.profile([1.], "simple")
        self._reference_factor = float(np.trapezoid(ref_u[0], ref_eta))

    def _vertical(self, velocity):
        # Streamfunction psi=integral_bed^z u dz. Differentiate at fixed z,
        # not fixed eta: w=-psi_x|eta + u*(1-eta)*bed_x. This turns flow
        # upward over rising beds and downward over falling beds while
        # conserving discharge. Geometry is quasi-steady between refreshes.
        eta = self._grid.eta
        integral = .5*(velocity[:, 1:]+velocity[:, :-1])*np.diff(eta)
        psi = self.depth[:, None]*np.column_stack((np.zeros(len(self.x)), np.cumsum(integral, axis=1)))
        vertical = (-np.gradient(psi, self.x, axis=0, edge_order=2)
                    +velocity*(1-eta)*self._bed_slope[:, None])
        vertical[:, [0, -1]] = 0.
        return vertical

    def _solve(self, t):
        s = self.s
        ramp = startup_factor(s, t)
        if s.STREAM_FORCING == "velocity":
            discharge = ramp*s.FLOW_MEAN_VELOCITY*s.FLOW_DEPTH
        else:
            # Prescribed stress is a reference value at the initial depth.
            # Carry its discharge through every column; imposing identical
            # local stress at different depths would violate continuity.
            tau = s.BED_STRESS_MEAN + (s.BED_STRESS_AMPLITUDE*math.sin(
                2*math.pi*s.BED_STRESS_FREQUENCY*t) if s.STREAM_FORCING == "harmonic" else 0.)
            reference_us = ramp*math.sqrt(max(0., tau)/s.WATER_DENSITY)
            reference_mean = (reference_us*self._reference_factor if s.HYDRAULIC_SOLVER == "simple"
                              else float(self._reference_grid.mean([reference_us])[0][0]))
            discharge = reference_mean*s.FLOW_DEPTH
        target = discharge/self.depth
        if s.HYDRAULIC_SOLVER == "simple":
            eta, unit_u, unit_grad, nu_t, unit_shear = self._simple_profile
            self.ustar = discharge*self._simple_ustar
            self.eta, self.u, self.gradient = eta, discharge*unit_u, discharge*unit_grad
            self.eddy_viscosity, self.shear = nu_t, discharge**2*unit_shear
            self.vertical_velocity = discharge*self._simple_vertical
        else:
            self.ustar = self._grid.friction_velocity(target, self.ustar if self.time is not None else None)
            self.eta, self.u, self.gradient, self.eddy_viscosity, self.shear = self._grid.profile(
                self.ustar, s.HYDRAULIC_SOLVER)
            self.vertical_velocity = self._vertical(self.u)
        self.mean_velocity = np.trapezoid(self.u, self.eta, axis=1)
        self.mean_tau = s.WATER_DENSITY*self.ustar**2
        self.unit_discharge = self.mean_velocity*self.depth
        self.energy_slope = self.ustar**2/(s.GRAVITY_MAG*self.depth)
        self.reynolds_number = self.mean_velocity*self.depth/s.WATER_VISCOSITY
        self.froude_number = self.mean_velocity/np.sqrt(s.GRAVITY_MAG*self.depth)
        self.roughness_reynolds_number = self.ustar*(30*self.z0)/s.WATER_VISCOSITY
        self.discharge_error = float(np.max(np.abs(self.unit_discharge-discharge)))/max(discharge, 1e-15)
        self.momentum_error = math.nan
        if s.HYDRAULIC_SOLVER == "advanced":
            modeled = (s.WATER_VISCOSITY+self.eddy_viscosity)*self.gradient
            self.momentum_error = float(np.max(np.abs(modeled-self.shear)))/max(float(np.max(self.ustar**2)), 1e-15)

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
        refresh = t-self.last_refresh >= s.FLOW_UPDATE_INTERVAL-1e-12 or self.time is None
        if refresh:
            # Aggregate bed geometry only at the hydraulic refresh interval.
            # The dense exposure grid is fixed during a run; reuse its bins.
            if self._sample_grid is not xs:
                edges = np.linspace(0., s.DOMAIN_X, len(self.x)+1)
                self._sample_index = np.clip(np.searchsorted(edges, xs, side="right")-1, 0, len(self.x)-1)
                self._sample_count = np.bincount(self._sample_index, minlength=len(self.x))
                self._sample_grid = xs
            sampled = np.bincount(self._sample_index, weights=bed, minlength=len(self.x))/np.maximum(self._sample_count, 1)
            self.bed = np.where(self._sample_count > 0, sampled, np.interp(self.x, xs, bed))
            self._refresh_geometry(exposure, active)
            self.last_refresh = t
        if refresh or s.STREAM_FORCING == "harmonic" or startup_factor(s, t) != startup_factor(s, self.time):
            self._solve(t)
        np.copyto(self.tau, self.mean_tau)
        if self.fluctuation_u.shape != self.u.shape:
            self.fluctuation_u = np.zeros_like(self.u)
            self.fluctuation_w = np.zeros_like(self.u)
        else:
            self.fluctuation_u.fill(0.)
            self.fluctuation_w.fill(0.)
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

    def sample(self, points, roughness_height=0.):
        """Bilinear sampling in streamwise position and relative bed height."""
        p = np.asarray(points)
        j = np.clip(np.searchsorted(self.x, p[:, 0])-1, 0, len(self.x)-2)
        f = np.clip((p[:, 0]-self.x[j])/(self.x[j+1]-self.x[j]), 0, 1)
        bed = (1-f)*self.bed[j]+f*self.bed[j+1]
        h = self.water_level-bed
        y = p[:, 1]-bed
        # The field itself is zero below the bed/above water. Only exposed
        # particle caps may request a roughness-layer sampling height.
        wet = (p[:, 1] <= self.water_level) & ((y >= 0) | (np.asarray(roughness_height) > 0))
        e = np.clip(np.maximum(y, np.minimum(roughness_height, .99*h))/h, 0, 1)
        k = np.clip(np.searchsorted(self.eta, e)-1, 0, len(self.eta)-2)
        g = (e-self.eta[k])/(self.eta[k+1]-self.eta[k])

        def interp(field):
            return (1-f)*((1-g)*field[j, k]+g*field[j, k+1])+f*((1-g)*field[j+1, k]+g*field[j+1, k+1])

        u = interp(self.u)
        w = interp(self.vertical_velocity)
        velocity_turbulence = self.s.SIMULATE_TURBULENCE and self.s.TURBULENCE_TARGET == "velocity"
        if velocity_turbulence:
            u += interp(self.fluctuation_u)
            w += interp(self.fluctuation_w)
        return np.where(wet, u, 0.), np.where(wet, w, 0.), np.interp(
            p[:, 0], self.x, self.tau if self.s.TURBULENCE_TARGET == "stress" else self.mean_tau)

    def snapshot(self):
        velocity = self.u.copy()
        vertical = self.vertical_velocity.copy()
        if self.s.SIMULATE_TURBULENCE and self.s.TURBULENCE_TARGET == "velocity":
            velocity += self.fluctuation_u
            vertical += self.fluctuation_w
        return dict(x=self.x.copy(), eta=self.eta.copy(), bed=self.bed.copy(), depth=self.depth.copy(),
                    water_level=self.water_level, velocity=velocity, vertical_velocity=vertical,
                    mean_velocity=self.u.copy(), mean_vertical_velocity=self.vertical_velocity.copy(),
                    stress=self.tau.copy(), mean_stress=self.mean_tau.copy(),
                    ustar=self.ustar.copy(), z0=self.z0, d84=self.d84,
                    discharge=self.unit_discharge.copy(), energy_slope=self.energy_slope.copy(),
                    reynolds_number=self.reynolds_number.copy(), froude_number=self.froude_number.copy(),
                    roughness_reynolds_number=self.roughness_reynolds_number.copy(),
                    momentum_error=self.momentum_error, discharge_error=self.discharge_error,
                    startup_factor=startup_factor(self.s, self.time),
                    turbulence_active=bool(self.s.SIMULATE_TURBULENCE))
