"""Short complete stream runs exercise both flow solvers and both particle engines."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import HillslopeSim as model


class StreamIntegrationTests(unittest.TestCase):
    def run_stream(self, engine, solver):
        settings = dict(
            ENGINE=engine, HYDRAULIC_SOLVER=solver, STREAM_BED_SETUP=True,
            STREAM_DOMAIN_SCALE=1., DOMAIN_X=.08, DOMAIN_Y=.02, DOMAIN_Z=.08,
            SLOPE_HEIGHT=.015, SLOPE_LENGTH=.08, SLOPE_BASE=0.,
            PARTICLE_SHAPE="circle", PARTICLE_SIZE=.005, PARTICLE_SIZE_INPUT="",
            FRACTURE_MODE="grid", CIRCLE_DIAMETER=.005, PARTICLE_B_AXIS=.005,
            PARTICLE_C_AXIS=None, INITIAL_CLEARANCE=.00001, FLOW_DEPTH=.04,
            FLOW_MEAN_VELOCITY=.05, STREAM_FORCING="velocity", STREAM_STARTUP_TIME=.01,
            STREAM_DURATION=.02, SETTLE_TIME=.005, DT=.0002, DT_MIN=1e-7,
            FLOW_COLUMNS=12, FLOW_LEVELS=48, FLOW_UPDATE_INTERVAL=.005,
            GRAVITY_PACKED=True, FILL_VOID=False, HOTSTART_PACK=False,
            FRAME_INTERVAL_MODE="seconds", FRAME_INTERVAL_SECONDS=.005,
            ANIMATION_START_TIME=0., ANIMATION_STOP_TIME=None,
            SIMULATE_TURBULENCE=False, SIMULATE_HYDRAULIC_FORCES=True,
        )
        # Restore generated caches/derived globals as well as the input values.
        with patch.dict(model.__dict__, settings), contextlib.redirect_stdout(io.StringIO()):
            data = model.run_simulation(save_video=False, return_data=True)
        self.assertEqual(data["engine"], engine, data["engine_fallback_reason"])
        self.assertEqual(data["calculated_time"], .02)
        self.assertEqual(len(data["computational_timesteps"]), 100)
        self.assertEqual(len(data["arr_time"]), 5)
        self.assertTrue(np.isfinite(data["final_positions"]).all())
        self.assertTrue(np.isfinite(data["frames_fluid_impulse"]).all())
        for h in data["frames_hydraulics"]:
            self.assertLess(h["discharge_error"], 2e-9)
            self.assertTrue(np.isfinite(h["vertical_velocity"]).all())
        self.assertTrue(data["particle_physics_parameters"]["SIMULATE_HYDRAULIC_FORCES"])

    def test_pymunk_simple(self):
        self.run_stream("pymunk", "simple")

    def test_pymunk_advanced(self):
        self.run_stream("pymunk", "advanced")

    def test_chrono_simple(self):
        self.run_stream("pychrono", "simple")

    def test_chrono_advanced(self):
        self.run_stream("pychrono", "advanced")
