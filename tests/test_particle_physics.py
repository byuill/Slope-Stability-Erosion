"""Contact switches, real particle dynamics, and configuration persistence."""
import contextlib
import functools
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))


import HillslopeSim as model


class ParticlePhysicsTests(unittest.TestCase):
    def setUp(self):
        settings = dict(
            STREAM_BED_SETUP=False, ENGINE="pymunk", ADAPTIVE_DT=False,
            GRAVITY_PACKED=False, PARTICLE_SHAPE="circle", PARTICLE_SIZE=0.1,
            _rect_b=0.1, _rect_c=0.1, DOMAIN_X=2.0, DOMAIN_Z=2.0,
            MATERIAL_TYPE="cohesive", FRICTION_COEFF=0.6, RESTITUTION=0.8,
            COHESION_FORCE=2.0, SOLVER_ITERS=30,
            SIMULATE_CONTACT_FRICTION=True, SIMULATE_RESTITUTION=True,
            SIMULATE_COHESION=True,
        )
        self.addCleanup(patch.stopall)
        patch.multiple(model, **settings).start()
        patch.object(model, "_size_distribution", return_value=types.SimpleNamespace(mode="single")).start()

    def backend(self, positions):
        specs = [dict(x=x, z=z, angle=0.0, size=0.1, c=0.1, b=0.1, shape="circle")
                 for x, z in positions]
        with contextlib.redirect_stdout(io.StringIO()):
            return model.PymunkBackend(specs)

    def test_contact_switches_preserve_coefficients(self):
        model.SIMULATE_CONTACT_FRICTION = model.SIMULATE_RESTITUTION = model.SIMULATE_COHESION = False
        self.assertEqual(model._contact_parameters(), (0.0, 0.0, 0.0))
        self.assertEqual((model.FRICTION_COEFF, model.RESTITUTION, model.COHESION_FORCE), (0.6, 0.8, 2.0))
        model.SIMULATE_CONTACT_FRICTION = model.SIMULATE_RESTITUTION = model.SIMULATE_COHESION = True
        self.assertEqual(model._contact_parameters(), (0.6, 0.8, 2.0))
        model.MATERIAL_TYPE = "granular"
        self.assertEqual(model._contact_parameters()[2], 0.0)

    def test_friction_changes_actual_sliding(self):
        speeds = []
        for enabled in (True, False):
            model.SIMULATE_CONTACT_FRICTION = enabled
            backend = self.backend([(0.5, 0.07)])
            body = backend.particle_bodies[0]
            body.velocity = (model.m2px(1.0), 0.0)
            for _ in range(300):
                backend.step(0.001)
            speeds.append(backend.get_velocities()[0, 0])
        self.assertLess(speeds[0], speeds[1] * 0.8)

    def test_restitution_changes_actual_rebound(self):
        upward_speeds = []
        for enabled in (True, False):
            model.SIMULATE_RESTITUTION = enabled
            backend = self.backend([(0.5, 0.2)])
            backend.particle_bodies[0].velocity = (0.0, -model.m2px(1.0))
            maximum = 0.0
            for _ in range(160):
                backend.step(0.001)
                maximum = max(maximum, backend.get_velocities()[0, 1])
            upward_speeds.append(maximum)
        self.assertGreater(upward_speeds[0], upward_speeds[1] + 0.2)

    def test_cohesion_controls_bond_creation(self):
        counts = []
        for enabled in (True, False):
            model.SIMULATE_COHESION = enabled
            backend = self.backend([(0.5, 0.5), (0.6, 0.5)])
            with contextlib.redirect_stdout(io.StringIO()):
                backend.setup_bonds()
            counts.append(len(backend.bond_springs))
        self.assertEqual(counts, [1, 0])
        self.assertEqual(model.COHESION_FORCE, 2.0)

    def test_chrono_particle_and_wall_materials(self):
        if importlib.util.find_spec("pychrono") is None:
            self.skipTest("Project Chrono is not installed")
        import pychrono
        backend = object.__new__(model.PychronoBackend)
        backend.chrono = pychrono
        for enabled in (True, False):
            model.SIMULATE_CONTACT_FRICTION = model.SIMULATE_RESTITUTION = model.SIMULATE_COHESION = enabled
            for wall in (False, True):
                material = backend._make_material(is_wall=wall)
                self.assertAlmostEqual(material.GetSlidingFriction(), 0.6 if enabled else 0.0)
                self.assertAlmostEqual(material.GetRestitution(), 0.8 if enabled else 0.0)
                self.assertAlmostEqual(material.GetCohesion(), 2.0 if enabled and not wall else 0.0)

    def test_disabled_cohesion_survives_auto_stabilization(self):
        model.SIMULATE_COHESION = False
        with patch.object(model, "AUTO_STABILIZE", True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(model._stabilize_initial_slope(), 0.0)
        self.assertEqual(model.COHESION_FORCE, 2.0)

    def test_stability_uses_enabled_processes(self):
        positions = model.np.array([[0.4, 0.2], [0.6, 0.2]])
        self.assertGreater(model.estimate_factor_of_safety(positions, 30.0, 2.0), 0.0)
        model.SIMULATE_CONTACT_FRICTION = model.SIMULATE_COHESION = False
        self.assertEqual(model.estimate_factor_of_safety(positions, 30.0, 2.0), 0.0)

    def gui(self):
        import tkinter as tk
        interpreter = tk.Tcl()
        gui = object.__new__(model.HillslopeSimGUI)
        gui.tk = types.SimpleNamespace(**{name: functools.partial(getattr(tk, name), master=interpreter)
                                        for name in ("BooleanVar", "StringVar", "DoubleVar", "IntVar")})
        gui._init_variables()
        gui._update_recording_controls = lambda: None
        gui._update_stream_controls = lambda: None
        gui.is_running = False
        gui.messagebox = types.SimpleNamespace(showerror=lambda *a: self.fail(str(a)))
        return gui

    def test_configuration_round_trip_and_legacy_defaults(self):
        gui = self.gui()
        gui.particle_physics_vars["SIMULATE_CONTACT_FRICTION"].set(False)
        gui.particle_physics_vars["SIMULATE_COHESION"].set(False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            gui.filedialog = types.SimpleNamespace(asksaveasfilename=lambda **kw: str(path))
            gui.apply_model_parameters = lambda **kw: True
            gui.lbl_status = types.SimpleNamespace(config=lambda **kw: None)
            with patch.multiple(model, SIMULATE_CONTACT_FRICTION=False, SIMULATE_COHESION=False):
                gui.save_config_json()
            cfg = json.loads(path.read_text())
        gui._apply_dict_to_vars(cfg)
        self.assertFalse(gui.particle_physics_vars["SIMULATE_CONTACT_FRICTION"].get())
        self.assertFalse(gui.particle_physics_vars["SIMULATE_COHESION"].get())
        self.assertTrue(gui.particle_physics_vars["SIMULATE_RESTITUTION"].get())
        gui._apply_dict_to_vars({})
        self.assertTrue(all(var.get() for var in gui.particle_physics_vars.values()))

    def test_gui_switches_reach_backend_configuration(self):
        gui = self.gui()
        gui.var_stream_bed.set(False)
        for var in gui.particle_physics_vars.values():
            var.set(False)
        # Isolate unavailable size/recording/flow helpers; exercise the real GUI binding.
        with patch.object(model, "parse_size_input", return_value=types.SimpleNamespace(mode="single", representative=0.1)), \
             patch.object(model, "_recompute_derived_quantities"):
            self.assertTrue(gui._apply_gui_to_globals())
        self.assertEqual(model._contact_parameters(), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
