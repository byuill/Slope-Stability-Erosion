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
        self.assertEqual(set(tt.SPECIES), {"pine", "deciduous", "aspen", "spruce"})
        for sp in tt.SPECIES.values():
            sp.validate()
        sp = tt.SPECIES
        self.assertGreater(sp["aspen"].density, sp["pine"].density)
        self.assertLess(sp["aspen"].lifespan_mean, sp["pine"].lifespan_mean)
        self.assertGreater(sp["deciduous"].root_plate_diameter, sp["pine"].root_plate_diameter)
        self.assertGreater(sp["pine"].root_plate_depth, sp["deciduous"].root_plate_depth)
        self.assertGreater(sp["spruce"].root_plate_diameter, sp["deciduous"].root_plate_diameter)
        for s in sp.values():
            self.assertLess(s.size_depth_correlation, 0)
            self.assertGreater(s.pit_aspect, 1)

    def test_size_depth_anticorrelated(self):
        rng = np.random.default_rng(0)
        r, h = tt.draw_mature_sizes(rng, tt.SPECIES["spruce"], 50000)
        self.assertLess(np.corrcoef(np.log(r), np.log(h))[0, 1], -0.7)

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
        self.assertAlmostEqual(ev["volume"], tt.pit_volume_factor(m.sp.pit_shape_exponent) * math.pi * R * R * H,
                               places=10)
        self.assertAlmostEqual(np.sum(m.z - z0) * m.cell_area, 0.0, places=10)
        self.assertEqual(ev["export"], 0.0)
        # Soil centroid moves about a + H/3 downslope, a = semi-axis along the fall line.
        a = R / math.sqrt(m.sp.pit_aspect)
        c = tt.pit_centroid_depth_fraction(m.sp.pit_shape_exponent)
        self.assertAlmostEqual(ev["moment"] / ev["volume"], a + c * H, delta=0.1)
        self.assertLess(m.z[int(10.0 / m.cfg.dx), int(5.0 / m.cfg.dx)], z0[int(10.0 / m.cfg.dx), int(5.0 / m.cfg.dx)])

    def test_infill_reduces_transport(self):
        a = model(soil_model=False, infill_fraction=0.0).uproot(10.0, 5.0, 1.0, 0.6, 0.0, s_fall=0.0)
        b = model(soil_model=False, infill_fraction=0.5).uproot(10.0, 5.0, 1.0, 0.6, 0.0, s_fall=0.0)
        self.assertAlmostEqual(b["moment"], 0.5 * a["moment"], delta=0.05 * a["moment"])

    def test_infill_depends_on_fall_direction(self):
        down = model(soil_model=False, infill_fraction=0.3).uproot(10.0, 5.0, 1.0, 0.6, 0.0)
        up = model(soil_model=False, infill_fraction=0.3).uproot(10.0, 5.0, 1.0, 0.6, math.pi)
        self.assertLess(down["infill"], 0.3)
        self.assertGreater(up["infill"], 0.3)
        self.assertGreater(down["displaced"], up["displaced"])
        self.assertAlmostEqual(float(tt.infill_at_slope(0.3, 1.07, 1.07)), 0.0)

    def test_pit_shape_factors(self):
        # Numerical integration of H (1 - rho^n) over a unit disc.
        rho = np.linspace(0, 1, 200001)
        for n in (2.0, 4.0, 7.0):
            d = 1 - rho ** n
            vol = np.trapezoid(d * 2 * np.pi * rho, rho)
            self.assertAlmostEqual(vol / np.pi, tt.pit_volume_factor(n), places=6)
            cen = np.trapezoid(0.5 * d * d * 2 * np.pi * rho, rho) / vol
            self.assertAlmostEqual(cen, tt.pit_centroid_depth_fraction(n), places=6)
        self.assertAlmostEqual(tt.pit_volume_factor(2), 0.5)
        self.assertAlmostEqual(tt.pit_centroid_depth_fraction(2), 1 / 3)

    def test_flat_floored_pit(self):
        """A larger exponent keeps the pit floor near full depth further from the centre."""
        depths = {}
        for n in (2.0, 6.0):
            sp = tt.replace(tt.SPECIES["pine"], pit_aspect=1.0, pit_shape_exponent=n)
            cfg = tt.HillslopeConfig(length=20.0, width=10.0, slope=0.0, dx=0.1, years=1, soil_model=False,
                                     infill_fraction=0.0, fresh_diffusivity=0.0, seed=1)
            m = tt.TreeThrowModel(sp, cfg)
            z0 = m.z.copy()
            m.uproot(10.0, 5.0, 1.5, 0.8, 0.0)
            pit = -(m.z - z0)[:, 50]
            i0 = int(np.argmax(pit))
            depths[n] = pit[i0 - 7] / pit[i0]          # toward the upslope rim (mound lies downslope)
            dist = abs((i0 - 7 + 0.5) * 0.1 - 10.0) / 1.5
            centre = 1 - (abs((i0 + 0.5) * 0.1 - 10.0) / 1.5) ** n
            self.assertAlmostEqual(depths[n], (1 - dist ** n) / centre, delta=0.02)
        self.assertLess(depths[2.0], 0.8)
        self.assertGreater(depths[6.0], 0.95)

    def test_elliptical_pit_volume_and_shape(self):
        m = model(soil_model=False, infill_fraction=0.0, fresh_diffusivity=0.0)
        z0 = m.z.copy()
        ev = m.uproot(10.0, 5.0, 1.2, 0.8, 0.0)
        self.assertAlmostEqual(ev["volume"], tt.pit_volume_factor(m.sp.pit_shape_exponent) * math.pi * 1.2 ** 2 * 0.8,
                               places=10)
        pit = (m.z - z0) < -0.05
        rows, cols = np.nonzero(pit)
        along = (rows.max() - rows.min() + 1) * m.cfg.dx
        across = (cols.max() - cols.min() + 1) * m.cfg.dx
        self.assertGreater(across / along, 1.2)

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


class FreshCollapseTests(unittest.TestCase):
    def test_fresh_collapse_conserves_volume_and_smooths(self):
        m = model(soil_model=False, tree_throw=False, diffusivity=0.0, infill_fraction=0.0)
        z0 = m.z.copy()
        m.uproot(10.0, 5.0, 1.5, 0.9, 0.0)
        z1 = m.z.copy()
        depth1, mound1 = -(z1 - z0).min(), (z1 - z0).max()
        for _ in range(3):
            m.step()
        dz = m.z - z0
        self.assertAlmostEqual(np.sum(dz) * m.cell_area, 0.0, places=9)
        self.assertLess(-dz.min(), 0.85 * depth1)
        self.assertLess(dz.max(), 0.85 * mound1)

    def test_collapse_switches_off_with_age(self):
        m = model(soil_model=False, tree_throw=False, diffusivity=0.0, fresh_decay_time=1.0)
        m.uproot(10.0, 5.0, 1.5, 0.9, 0.0)
        for _ in range(20):
            m.step()
        z20 = m.z.copy()
        m.step()
        self.assertLess(np.abs(m.z - z20).max(), 1e-6)
        self.assertEqual(m.fresh_windows, [])

    def test_no_collapse_when_disabled(self):
        m = model(soil_model=False, tree_throw=False, diffusivity=0.0, fresh_diffusivity=0.0)
        m.uproot(10.0, 5.0, 1.5, 0.9, 0.0)
        z1 = m.z.copy()
        m.step()
        np.testing.assert_array_equal(m.z, z1)


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
