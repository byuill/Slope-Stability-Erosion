"""Mass conservation, transport and species-parameter checks for TreeThrow.py."""
import contextlib
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import TreeThrow as tt


def model(species="pine", **changes):
    cfg = tt.HillslopeConfig(length=20.0, width=10.0, slope=0.3, dx=0.25, years=10.0, seed=1)
    for k, v in changes.items():
        setattr(cfg, k, v)
    return tt.TreeThrowModel(tt.SPECIES[species], cfg)


class SpeciesTests(unittest.TestCase):
    def test_builtin_species_are_valid_and_distinct(self):
        self.assertEqual(set(tt.SPECIES), {"pine", "deciduous", "aspen"})
        for sp in tt.SPECIES.values():
            sp.validate()
        sp = tt.SPECIES
        self.assertGreater(sp["aspen"].density, sp["pine"].density)
        self.assertLess(sp["aspen"].lifespan_mean, sp["pine"].lifespan_mean)
        self.assertGreater(sp["deciduous"].root_plate_diameter, sp["pine"].root_plate_diameter)
        self.assertGreater(sp["pine"].root_plate_depth, sp["deciduous"].root_plate_depth)

    def test_invalid_species_rejected(self):
        bad = tt.TreeSpecies(**{**tt.SPECIES["pine"].__dict__, "uproot_fraction": 1.5})
        with self.assertRaises(ValueError):
            bad.validate()

    def test_expected_diffusivity_positive_and_slope_dependent(self):
        for sp in tt.SPECIES.values():
            lo = tt.expected_tree_throw(sp, 0.1, n_samples=20000)
            hi = tt.expected_tree_throw(sp, 0.6, n_samples=20000)
            self.assertGreater(lo["diffusivity_m2_per_yr"], 0)
            self.assertGreater(hi["flux_m2_per_yr"], lo["flux_m2_per_yr"])

    def test_bessel_ratio(self):
        self.assertAlmostEqual(tt._bessel_ratio(0.0), 0.0)
        self.assertAlmostEqual(tt._bessel_ratio(1.0), 0.44639, places=4)
        self.assertGreater(tt._bessel_ratio(50.0), 0.98)


class UprootTests(unittest.TestCase):
    def test_single_event_conserves_volume_without_bedrock(self):
        m = model(soil_model=False, infill_fraction=0.0)
        z0 = m.z.copy()
        R, H = 1.2, 0.8
        ev = m.uproot(10.0, 5.0, R, H, 0.0)
        self.assertAlmostEqual(ev["volume"], 0.5 * math.pi * R * R * H, places=10)
        self.assertAlmostEqual(np.sum(m.z - z0) * m.cell_area, 0.0, places=10)
        self.assertEqual(ev["export"], 0.0)
        # Soil centroid moves about R + H/3 downslope.
        self.assertAlmostEqual(ev["moment"] / ev["volume"], R + H / 3, delta=m.cfg.dx)
        self.assertLess(m.z[int(10.0 / m.cfg.dx), int(5.0 / m.cfg.dx)], z0[int(10.0 / m.cfg.dx), int(5.0 / m.cfg.dx)])

    def test_infill_reduces_transport(self):
        a = model(soil_model=False, infill_fraction=0.0).uproot(10.0, 5.0, 1.0, 0.6, 0.0)
        b = model(soil_model=False, infill_fraction=0.5).uproot(10.0, 5.0, 1.0, 0.6, 0.0)
        self.assertAlmostEqual(b["moment"], 0.5 * a["moment"], delta=0.05 * a["moment"])

    def test_bedrock_plucking_dilates_volume(self):
        m = model(initial_soil_depth=0.3, bedrock_pluck_fraction=0.5, infill_fraction=0.0)
        z0, zb0 = m.z.copy(), m.zb.copy()
        m.uproot(10.0, 5.0, 1.0, 1.0, math.pi / 2)
        rock = np.sum(zb0 - m.zb) * m.cell_area
        self.assertGreater(rock, 0)
        self.assertGreaterEqual(m.h.min(), 0.0)
        gain = np.sum(m.z - z0) * m.cell_area
        self.assertAlmostEqual(gain, rock * (m.cfg.rock_to_soil_dilation - 1), places=10)

    def test_soil_thrown_into_channel_is_exported(self):
        m = model(soil_model=False, infill_fraction=0.0)
        z0 = m.z.copy()
        x0 = m.L_active - 0.6
        ev = m.uproot(x0, 5.0, 1.0, 0.5, 0.0)
        self.assertGreater(ev["export"], 0.0)
        change = np.sum((m.z - z0)[:-1]) * m.cell_area
        self.assertAlmostEqual(change, -ev["export"], places=10)

    def test_soil_thrown_over_divide_is_mirrored(self):
        m = model(soil_model=False, infill_fraction=0.0)
        z0 = m.z.copy()
        ev = m.uproot(0.6, 5.0, 1.0, 0.5, math.pi)  # fall upslope over the ridge
        self.assertEqual(ev["export"], 0.0)
        self.assertAlmostEqual(np.sum(m.z - z0) * m.cell_area, 0.0, places=10)
        self.assertLess(ev["moment"], 0.0)


class DiffusionTests(unittest.TestCase):
    def test_diffusion_volume_balance(self):
        m = model(tree_throw=False, soil_model=False, diffusivity=0.01)
        m.uproot(10.0, 5.0, 1.0, 0.5, 0.0)   # add roughness
        v0 = np.sum(m.z[:-1]) * m.cell_area
        for _ in range(50):
            m.step()
        v1 = np.sum(m.z[:-1]) * m.cell_area
        self.assertGreater(m.export_diff, 0)
        self.assertAlmostEqual(v1 - v0, -m.export_diff, places=8)

    def test_linear_flux_matches_D_times_S(self):
        m = model(tree_throw=False, soil_model=False, diffusivity=0.005)
        m.step()
        q = m.moment_diff / (m.L_active * m.W * m.time)
        self.assertAlmostEqual(q, 0.005 * 0.3, delta=1e-6)

    def test_nonlinear_enhances_flux(self):
        lin = model(tree_throw=False, soil_model=False, slope=0.8)
        non = model(tree_throw=False, soil_model=False, slope=0.8, transport_law="nonlinear")
        lin.step(), non.step()
        self.assertGreater(non.moment_diff, 1.5 * lin.moment_diff)

    def test_ridge_rounds_into_convex_form(self):
        m = model(tree_throw=False, soil_model=False, diffusivity=0.05, years=200)
        m.run(progress=False)
        prof = m.z.mean(axis=1)
        self.assertLess(prof[0], m.z_initial[0, 0])
        curv = np.diff(prof[:20], 2)
        self.assertTrue(np.all(curv < 1e-9))


class RunTests(unittest.TestCase):
    def test_simulated_tree_throw_diffusivity_matches_analytic(self):
        cfg = tt.HillslopeConfig(length=40, width=40, slope=0.3, dx=0.5, years=1500,
                                 diffusivity=0.0, soil_model=False, seed=3, n_snapshots=2)
        m = tt.TreeThrowModel(tt.SPECIES["aspen"], cfg)
        m.run(progress=False)
        s = m.summary()
        self.assertGreater(s["uprooting_events"], 300)
        rel = s["effective_D_tree_throw_m2_yr"] / s["expected_D_tree_throw_m2_yr"]
        self.assertAlmostEqual(rel, 1.0, delta=0.3)
        rate = s["uprooting_rate_per_ha_yr"] / tt.SPECIES["aspen"].uproot_rate_per_ha
        self.assertAlmostEqual(rate, 1.0, delta=0.2)

    def test_seed_reproducible(self):
        a, b = model(years=200), model(years=200)
        a.run(progress=False), b.run(progress=False)
        np.testing.assert_array_equal(a.z, b.z)
        self.assertEqual(a.n_events, b.n_events)

    def test_cli_compare_writes_outputs(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            rc = tt.main(["--compare", "--length", "12", "--width", "6", "--slope-deg", "20",
                          "--years", "60", "--dx", "0.5", "--output", d, "--quiet"])
            self.assertEqual(rc, 0)
            for sp in tt.SPECIES:
                for f in ("events.csv", "summary.json", "final_state.npz", "hillslope_evolution.png"):
                    self.assertTrue((Path(d) / sp / f).exists(), f"{sp}/{f}")
            self.assertTrue((Path(d) / "species_comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
