"""Hydraulic boundary, continuity, RANS and impulse regressions (SI units)."""
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import HillslopeSim as model
from RiverHydraulics import RiverHydraulics, column_profile
from DragIntegration import implicit_drag_force
from StreamBed import StreamForces, StreamPhysics


def configuration(solver="advanced", **changes):
    cfg = types.SimpleNamespace(**vars(model))
    cfg.DOMAIN_X, cfg.SLOPE_HEIGHT, cfg.FLOW_DEPTH = 1., .1, .3
    cfg.FLOW_COLUMNS, cfg.FLOW_LEVELS = 64, 128
    cfg.FLOW_MEAN_VELOCITY, cfg.STREAM_STARTUP_TIME = .5, 0.
    cfg.HYDRAULIC_SOLVER, cfg.STREAM_FORCING = solver, "velocity"
    cfg.SIMULATE_TURBULENCE = False
    cfg.__dict__.update(changes)
    return cfg


def hydraulics(cfg, slope=0.):
    backend = types.SimpleNamespace(particle_sizes=[.003]*6, particle_kinds=["circle"]*6,
                                    particle_b_sizes=[.003]*6, particle_planform_areas=[1e-5]*6)
    h = RiverHydraulics(cfg, backend)
    xs = np.linspace(0., cfg.DOMAIN_X, cfg.FLOW_COLUMNS)
    bed = cfg.SLOPE_HEIGHT+slope*(xs-.5*cfg.DOMAIN_X)
    h.update(.1, xs, bed, np.ones(6), np.ones(6, bool))
    return h, xs, bed


class HydraulicTests(unittest.TestCase):
    def test_no_slip_monotonicity_and_shear_balance(self):
        for solver in ("simple", "advanced"):
            with self.subTest(solver=solver):
                eta, u, gradient, nu_t, shear = column_profile([.03, .06], [.2, .4], .0003, 1e-6, 128, solver)
                np.testing.assert_array_equal(u[:, 0], 0.)
                self.assertTrue(np.all(np.diff(u, axis=1) >= 0))
                if solver == "advanced":
                    np.testing.assert_allclose((1e-6+nu_t)*gradient, shear, rtol=1e-12)
                    np.testing.assert_allclose(shear, np.array([.03, .06])[:, None]**2*(1-.5*(eta[1:]+eta[:-1])))
                else:
                    np.testing.assert_array_equal(gradient[:, -1], 0.)

    def test_advanced_laminar_limit(self):
        depth, viscosity, us = .01, .01, .0001
        eta, u, *_ = column_profile([us], [depth], 1e-8, viscosity, 128, "advanced")
        y = depth*eta
        expected = us**2/viscosity*(y-y*y/(2*depth))
        np.testing.assert_allclose(u[0], expected, rtol=1e-7, atol=1e-15)

    def test_discrete_discharge_on_variable_bed_for_all_forcings(self):
        for solver in ("simple", "advanced"):
            for forcing in ("velocity", "steady", "harmonic"):
                with self.subTest(solver=solver, forcing=forcing):
                    cfg = configuration(solver, STREAM_FORCING=forcing)
                    h, _, _ = hydraulics(cfg, slope=.04)
                    q = np.trapezoid(h.u, h.eta, axis=1)*h.depth
                    np.testing.assert_allclose(q, q.mean(), rtol=2e-10)
                    self.assertLess(h.discharge_error, 2e-10)
                    if solver == "advanced":
                        self.assertLess(h.momentum_error, 1e-12)

    def test_vertical_flow_direction_boundaries_and_continuity(self):
        for solver in ("simple", "advanced"):
            for slope in (-.04, .04):
                with self.subTest(solver=solver, slope=slope):
                    h, _, _ = hydraulics(configuration(solver, FLOW_LEVELS=256), slope=slope)
                    w = h.vertical_velocity
                    np.testing.assert_array_equal(w[:, [0, -1]], 0.)
                    self.assertGreater(float(w[3:-3, 20:-20].mean())*slope, 0.)
                    ux = (np.gradient(h.u, h.x, axis=0, edge_order=2)
                          -np.gradient(h.u, h.eta, axis=1, edge_order=2)*(1-h.eta)
                          *np.gradient(h.bed, h.x)[:, None]/h.depth[:, None])
                    wz = np.gradient(w, h.eta, axis=1, edge_order=2)/h.depth[:, None]
                    # Check differential continuity away from the unresolved wall layer.
                    self.assertLess(float(np.max(np.abs((ux+wz)[3:-3, 20:-20]))), .003)

    def test_sampling_preserves_wall_and_dry_boundaries(self):
        h, _, _ = hydraulics(configuration(), slope=.04)
        x = .5
        z = float(np.interp(x, h.x, h.bed))
        u, w, _ = h.sample([[x, z], [x, z-.1], [x, h.water_level+.01], [x, h.water_level]])
        np.testing.assert_allclose(u[:3], 0., atol=1e-12)
        np.testing.assert_allclose(w, 0., atol=1e-12)
        cap_u, _, _ = h.sample([[x, z-.001]], roughness_height=.0015)
        self.assertGreater(cap_u[0], 0.)

    def test_startup_and_geometry_refresh_are_independent(self):
        cfg = configuration(STREAM_STARTUP_TIME=1., FLOW_UPDATE_INTERVAL=.05)
        h, xs, bed = hydraulics(cfg)
        with patch.object(h, "_refresh_geometry", wraps=h._refresh_geometry) as refresh:
            for t in np.linspace(.101, .199, 99):
                h.update(t, xs, bed, np.ones(6), np.ones(6, bool))
            self.assertEqual(refresh.call_count, 1)
            self.assertLess(h.discharge_error, 2e-10)
        h, _, _ = hydraulics(configuration(STREAM_STARTUP_TIME=1., FLOW_MEAN_VELOCITY=0.))
        np.testing.assert_array_equal(h.u, 0.)
        np.testing.assert_array_equal(h.vertical_velocity, 0.)

    def test_turbulence_is_reproducible_with_impermeable_boundaries(self):
        cfg = configuration(SIMULATE_TURBULENCE=True, TURBULENCE_TARGET="velocity")
        a, xs, bed = hydraulics(cfg, slope=.02)
        b, _, _ = hydraulics(cfg, slope=.02)
        for t in (.2, .3):
            a.update(t, xs, bed, np.ones(6), np.ones(6, bool))
            b.update(t, xs, bed, np.ones(6), np.ones(6, bool))
        np.testing.assert_array_equal(a.snapshot()["vertical_velocity"], b.snapshot()["vertical_velocity"])
        np.testing.assert_array_equal(a.snapshot()["vertical_velocity"][:, [0, -1]], 0.)
        np.testing.assert_array_equal(a.snapshot()["velocity"][:, 0], 0.)


class ImpulseTests(unittest.TestCase):
    def test_backward_euler_equation_and_dissipation(self):
        rng = np.random.default_rng(7)
        count = 100
        relative = rng.normal(size=(count, 2))
        mass = np.exp(rng.uniform(-12, -3, count))
        inertia = .4*mass*.003**2
        arm = rng.uniform(-.003, .003, (count, 2))
        coefficient = np.exp(rng.uniform(-8, 0, count))
        for dt in (1e-6, .001, .1, 10.):
            with self.subTest(dt=dt):
                force = implicit_drag_force(relative, coefficient, mass, inertia, arm, dt)
                impulse = dt*force
                spin = (arm[:, 0]*impulse[:, 1]-arm[:, 1]*impulse[:, 0])/inertia
                response = impulse/mass[:, None]+spin[:, None]*np.column_stack((-arm[:, 1], arm[:, 0]))
                updated = relative-response
                expected = coefficient[:, None]*np.linalg.norm(updated, axis=1)[:, None]*updated
                np.testing.assert_allclose(force, expected, rtol=2e-7, atol=1e-12)
                energy_change = -np.sum(impulse*relative, axis=1)+.5*np.sum(impulse*response, axis=1)
                self.assertTrue(np.all(energy_change <= 1e-15))

    def test_zero_drag_and_zero_lever(self):
        relative = np.array([[10., -3.], [0., 0.]])
        force = implicit_drag_force(relative, [0., 1.], [1., 1.], [1., 1.], np.zeros((2, 2)), 1.)
        np.testing.assert_array_equal(force, 0.)
        force = implicit_drag_force(relative, [1., 1.], [1., 1.], [1., 1.], np.zeros((2, 2)), 10.)
        updated = relative-10*force
        self.assertGreater(float(np.dot(updated[0], relative[0])), 0.)
        np.testing.assert_allclose(force, np.linalg.norm(updated, axis=1)[:, None]*updated, atol=1e-12)


class CouplingTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.multiple(model, ENGINE="pymunk", STREAM_BED_SETUP=False,
            ADAPTIVE_DT=False, GRAVITY_PACKED=False, PARTICLE_SHAPE="circle", PARTICLE_SIZE=.02,
            PARTICLE_SIZE_INPUT="", _rect_b=.02, _rect_c=.02, DOMAIN_X=1., DOMAIN_Y=.2, DOMAIN_Z=1.,
            SLOPE_BASE=0., SLOPE_HEIGHT=.1, SLOPE_LENGTH=1., FLOW_DEPTH=.5, FLOW_COLUMNS=24,
            FLOW_LEVELS=96, GRAVITY_MAG=9.81, MATERIAL_TYPE="granular", STREAM_STARTUP_TIME=0.,
            SIMULATE_TURBULENCE=False, HYDRAULIC_LOAD="drag", HYDRAULIC_SOLVER="advanced",
            STREAM_DRAG_INTEGRATION="implicit", SIMULATE_BUOYANCY=True,
            SIMULATE_HYDRAULIC_FORCES=True, SIMULATE_LIFT=True)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def backend(self, engine="pymunk", z=.5):
        cls = model.PymunkBackend if engine == "pymunk" else model.PychronoBackend
        with contextlib.redirect_stdout(io.StringIO()):
            return cls([dict(x=.5, z=z, angle=0., size=.02, b=.02, c=.02, shape="circle")])

    def test_real_engine_linear_and_angular_momentum(self):
        for engine in ("pymunk", "pychrono"):
            with self.subTest(engine=engine), patch.object(model, "ENGINE", engine):
                backend = self.backend(engine)
                # Disable Pymunk's existing global damping for an isolated impulse balance.
                if hasattr(backend, "space"):
                    backend.space.damping = 1.
                    backend.space.gravity = (0., 0.)
                else:
                    backend.sys.SetGravitationalAcceleration(backend.chrono.ChVector3d(0., 0., 0.))
                physics = StreamPhysics(model, backend)
                arm = np.array([[.003, .008]])
                relative = np.array([[.3, -.1]])
                dt = .001
                force = implicit_drag_force(relative, [.1], physics.masses, physics.inertias, arm, dt)
                before = backend.get_velocities().copy()
                physics.apply(0, 0., force[0], backend.get_positions()[0]+arm[0])
                backend.step(dt)
                np.testing.assert_allclose(physics.masses[:, None]*(backend.get_velocities()-before), dt*force, rtol=1e-8, atol=1e-12)
                spin = (backend.particle_bodies[0].angular_velocity if engine == "pymunk"
                        else -backend.particle_bodies[0].GetAngVelParent().y)
                expected_torque = arm[0, 0]*force[0, 1]-arm[0, 1]*force[0, 0]
                self.assertAlmostEqual(physics.inertias[0]*spin, dt*expected_torque, delta=1e-12)

    def test_hydraulic_and_lift_switches_and_fluid_reaction(self):
        forcing = StreamForces(model, self.backend())
        with patch.object(model, "STREAM_LIFT_RATIO", .2):
            forcing.update(time=.1)
            with_lift = forcing.prepare_step(.001).copy()
            model.SIMULATE_LIFT = False
            forcing.update(time=.1)
            without_lift = forcing.prepare_step(.001).copy()
        self.assertGreater(with_lift[0, 1], without_lift[0, 1])
        forcing.accumulate_exchange(.001)
        np.testing.assert_allclose(forcing.fluid_impulse, -.001*without_lift.sum(axis=0))
        model.SIMULATE_HYDRAULIC_FORCES = False
        forcing.update(time=.2)
        np.testing.assert_array_equal(forcing.prepare_step(.001), 0.)

    def test_submergence_and_buoyancy_switch(self):
        with patch.object(model, "GRAVITY_MAG", 9.81):
            forcing = StreamForces(model, self.backend(z=.6))
            forcing.update(time=.1)
            np.testing.assert_allclose(forcing.buoyancy, forcing.full_buoyancy*.5, rtol=1e-12)
            model.SIMULATE_BUOYANCY = False
            forcing.update(time=.2)
            np.testing.assert_array_equal(forcing.buoyancy, 0.)
        dry = StreamForces(model, self.backend(z=.7))
        dry.update(time=.1)
        np.testing.assert_array_equal(dry.prepare_step(.001), 0.)


if __name__ == "__main__":
    unittest.main()
