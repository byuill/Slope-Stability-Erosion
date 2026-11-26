"""Mass conservation, transport and species-parameter checks for TreeThrow.py."""
import contextlib
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertEqual(set(tt.SPECIES), {"pine", "deciduous", "aspen", "spruce", "douglas_fir", "oak",
                                           "lodgepole_pine", "ponderosa_pine", "eastern_hemlock"})
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
        self.assertGreaterEqual(m.time - m.fresh_last, 6 * m.cfg.fresh_decay_time)

    def test_no_collapse_when_disabled(self):
        m = model(soil_model=False, tree_throw=False, diffusivity=0.0, fresh_diffusivity=0.0)
        m.uproot(10.0, 5.0, 1.5, 0.9, 0.0)
        z1 = m.z.copy()
        m.step()
        np.testing.assert_array_equal(m.z, z1)


class ForestDynamicsTests(unittest.TestCase):
    def test_stand_rates_match_simulation(self):
        sp = tt.SPECIES["spruce"]
        cfg = tt.HillslopeConfig(length=40, width=40, slope=0.3, dx=0.5, years=600, diffusivity=0.0,
                                 soil_model=False, fresh_diffusivity=0.0, seed=4, n_snapshots=2)
        m = tt.TreeThrowModel(sp, cfg)
        m.run(progress=False)
        s = m.summary()
        r = tt.stand_rates(sp)
        self.assertAlmostEqual(s["death_rate_per_ha_yr"] / r["deaths_per_ha_yr"], 1.0, delta=0.12)
        self.assertAlmostEqual(s["uprooting_share_of_deaths"], r["uprooting_share_of_deaths"], delta=0.05)

    def test_background_mortality_kills_young_trees(self):
        sp = tt.replace(tt.SPECIES["pine"], background_mortality=0.02, min_uproot_age=0.0)
        rng = np.random.default_rng(0)
        life, a = tt.draw_ages_at_death(rng, sp, 20000)
        self.assertTrue(np.all(a <= life))
        self.assertAlmostEqual(np.mean(a < life), 1 - np.mean(np.exp(-0.0202 * life)), delta=0.02)

    def test_min_uproot_age_respected(self):
        sp = tt.replace(tt.SPECIES["aspen"], min_uproot_age=60.0, background_mortality=0.05)
        cfg = tt.HillslopeConfig(length=30, width=30, slope=0.3, dx=0.5, years=200, diffusivity=0.0,
                                 soil_model=False, seed=2, n_snapshots=2)
        m = tt.TreeThrowModel(sp, cfg)
        m.run(progress=False)
        Rmin = 0.5 * sp.root_plate_diameter * tt.growth_factor(60.0, sp.maturity_age) * 0.3
        ev = np.array(m.event_log)
        self.assertGreater(len(ev), 0)
        self.assertGreater(ev[:, 3].min(), Rmin)       # no plates from trees younger than 60 yr
        self.assertLess(tt.stand_rates(sp)["uprooting_share_of_deaths"], sp.uproot_fraction)

    def test_windstorm_kills_large_trees_preferentially(self):
        cfg = tt.HillslopeConfig(length=80, width=60, slope=0.3, dx=0.5, years=1, diffusivity=0.0, soil_model=False,
                                 seed=5, n_snapshots=2, disturbance="windstorm", first_disturbance=0.0,
                                 disturbance_severity=0.3, storm_size_exponent=2.0)
        m = tt.TreeThrowModel(tt.SPECIES["spruce"], cfg)
        live = m.age >= 0
        R_all = tt.plate_at_age(m.sp, m.R_mature[live], m.H_mature[live], m.age[live])[0].mean()
        m.step()
        ev = np.array(m.event_log)
        storm = ev[ev[:, 9] == 1]
        self.assertEqual(len(m.disturbance_log), 1)
        self.assertGreater(len(storm), 10)
        self.assertGreater(storm[:, 3].mean(), R_all)

    def test_fire_snags_fall_later_with_shrinking_root_plates(self):
        sp = tt.replace(tt.SPECIES["pine"], min_uproot_age=0.0, uproot_fraction=1.0, background_mortality=0.0,
                        root_plate_diameter_cv=0.0, root_plate_depth_cv=0.0, lifespan_sd=0.0)
        cfg = tt.HillslopeConfig(length=40, width=40, slope=0.3, dx=0.5, years=60, diffusivity=0.0, soil_model=False,
                                 seed=6, n_snapshots=2, disturbance="fire", first_disturbance=0.0,
                                 disturbance_severity=1.0, snag_fall_rate=0.1, root_decay_time=15.0,
                                 stand_structure="even-aged")
        m = tt.TreeThrowModel(sp, cfg)
        m.age[:] = 150.0                                  # a mature, uniform stand
        n_trees = m.age.size
        m.run(progress=False)
        ev = np.array(m.event_log)
        snag = ev[ev[:, 9] == 3]                          # falls of fire-killed snags
        self.assertGreater(len(snag), 0.8 * n_trees)      # nearly all snags have fallen after 60 yr
        early, late = snag[snag[:, 0] < 10], snag[snag[:, 0] > 25]
        self.assertGreater(early[:, 6].mean(), 1.5 * late[:, 6].mean())   # decayed plates hold less soil
        self.assertGreater(len(early), len(late))

    def test_thin_soil_makes_shallower_plates(self):
        vols = {}
        for h in (1.0, 0.1):
            cfg = tt.HillslopeConfig(length=20, width=10, slope=0.3, dx=0.25, years=1, soil_model=True,
                                     initial_soil_depth=h, bedrock_pluck_fraction=1.0, rock_to_soil_dilation=1.0,
                                     infill_fraction=0.0, fresh_diffusivity=0.0, seed=1)
            m = tt.TreeThrowModel(tt.SPECIES["pine"], cfg)
            vols[h] = m.uproot(10.0, 5.0, 1.2, 0.8, 0.0)["volume"]
        self.assertAlmostEqual(vols[0.1] / vols[1.0], 0.5 + 0.5 * 0.1 / 0.5, delta=0.02)

    def test_root_cycle_matches_hoffman_anderson(self):
        th, a = math.radians(15), (4.4, 0.45, 400.0)
        betas = np.linspace(0.5, 0.999, 300)
        self.assertLess(max(tt.root_cycle_flux(th, *a, b, 0.1) for b in betas), 4e-5)
        self.assertAlmostEqual(tt.root_cycle_flux(th, *a, 0.98, 1.0), 3.8e-4, delta=0.1e-4)
        self.assertLess(tt.root_cycle_flux(th, *a, 0.8, 1.0), 6e-5)
        median = a[0] * a[1] / a[2] * math.sin(th) * math.log(0.5) / (100 * math.log(0.95))
        self.assertAlmostEqual(tt.root_cycle_flux(th, *a, 0.95, None) / median, 1 / math.log(2), places=6)
        m = model(tree_throw=False, soil_model=False, diffusivity=0.0, root_cycle=True)
        self.assertGreater(m.D_root, 0)
        m.step()
        self.assertGreater(m.moment_diff, 0)


class KeystoneSpeciesTests(unittest.TestCase):
    """Snag pathway, wind vulnerability, outbreaks and the contrasting keystone presets."""

    def test_presets_express_contrasting_dynamics(self):
        sp = tt.SPECIES
        rates = {k: tt.stand_rates(s) for k, s in sp.items()}
        # Taproots resist wind; shallow plates do not.
        self.assertLess(sp["oak"].wind_vulnerability, 1.0)
        self.assertLess(sp["ponderosa_pine"].wind_vulnerability, 1.0)
        self.assertGreater(sp["eastern_hemlock"].wind_vulnerability, 1.0)
        self.assertGreater(sp["lodgepole_pine"].wind_vulnerability, 1.0)
        # Douglas fir, oak and ponderosa mostly die standing; uprooting is a small share of deaths.
        for k in ("douglas_fir", "oak", "ponderosa_pine"):
            self.assertGreater(sp[k].snag_fraction, 0.5)
            self.assertLess(rates[k]["uprooting_share_of_deaths"], 0.35)
        self.assertAlmostEqual(rates["douglas_fir"]["uprooting_share_of_deaths"], 0.15, delta=0.02)  # Gabet & Mudd
        self.assertAlmostEqual(rates["douglas_fir"]["deaths_per_ha_yr"] / sp["douglas_fir"].density, 1 / 330,
                               delta=0.0006)
        self.assertGreater(rates["oak"]["snag_share_of_uprootings"], 0.5)       # oaks topple as decayed snags
        self.assertGreater(rates["eastern_hemlock"]["uprooting_share_of_deaths"], 0.5)
        # Ponderosa is the rarest thrower; lodgepole the densest conifer with the smallest plates.
        self.assertEqual(min(rates, key=lambda k: rates[k]["uprootings_per_ha_yr"]), "ponderosa_pine")
        self.assertEqual(max((k for k in sp if k != "aspen"), key=lambda k: sp[k].density), "lodgepole_pine")
        vols = {k: tt.expected_tree_throw(s, 0.35, n_samples=20000)["mean_event_volume_m3"] for k, s in sp.items()}
        self.assertLess(vols["oak"], 0.2 * vols["douglas_fir"])
        self.assertLess(vols["lodgepole_pine"], vols["aspen"])

    def test_snag_pathway_matches_stand_rates(self):
        sp = tt.replace(tt.SPECIES["oak"], snag_fall_rate=0.1)
        cfg = tt.HillslopeConfig(length=60, width=60, slope=0.3, dx=0.5, years=800, diffusivity=0.0,
                                 soil_model=False, seed=8, n_snapshots=2)
        m = tt.TreeThrowModel(sp, cfg)
        standing = []
        for _ in range(800):
            m.step()
            if m.time > 200:
                standing.append(m.snags["x"].size)
        s, r = m.summary(), tt.stand_rates(sp)
        ev = np.array(m.event_log)
        self.assertAlmostEqual(s["uprooting_share_of_deaths"] / r["uprooting_share_of_deaths"], 1.0, delta=0.15)
        self.assertAlmostEqual(np.mean(ev[:, 9] == 2) / r["snag_share_of_uprootings"], 1.0, delta=0.15)
        n_standing = np.mean(standing) / (m.L_active * m.W / 1e4)
        self.assertAlmostEqual(n_standing / r["standing_dead_per_ha"], 1.0, delta=0.2)
        # Snag root plates have decayed: smaller than immediate uprootings.
        self.assertLess(ev[ev[:, 9] == 2, 6].mean(), 0.8 * ev[ev[:, 9] == 0, 6].mean())
        rel = s["effective_D_tree_throw_m2_yr"] / s["expected_D_tree_throw_m2_yr"]
        self.assertAlmostEqual(rel, 1.0, delta=0.3)

    def test_expected_volume_accounts_for_snag_decay(self):
        base = tt.replace(tt.SPECIES["pine"], uproot_fraction=0.0, snag_fraction=1.0, snag_uproot_fraction=1.0)
        fast = tt.expected_tree_throw(tt.replace(base, snag_fall_rate=10.0), 0.3, n_samples=50000)
        slow = tt.expected_tree_throw(tt.replace(base, snag_fall_rate=0.05), 0.3, n_samples=50000, root_decay_time=15.0)
        # E[exp(-T/tau)] for T ~ Exp(k) is k / (k + 1/tau).
        k, tau = 0.05, 15.0
        self.assertAlmostEqual(slow["mean_event_volume_m3"] / fast["mean_event_volume_m3"],
                               k / (k + 1 / tau) / (10.0 / (10.0 + 1 / tau)), delta=0.02)

    def test_wind_vulnerability_scales_storm_kills(self):
        kills = {}
        for v in (0.0, 0.5, 1.0):
            sp = tt.replace(tt.SPECIES["spruce"], wind_vulnerability=v)
            cfg = tt.HillslopeConfig(length=200, width=200, slope=0.3, dx=1.0, years=1, diffusivity=0.0,
                                     soil_model=False, seed=5, n_snapshots=2, disturbance="windstorm",
                                     first_disturbance=0.0, disturbance_severity=0.08)
            m = tt.TreeThrowModel(sp, cfg)
            m.step()
            kills[v] = m.disturbance_log[0][2]
        self.assertEqual(kills[0.0], 0)
        self.assertAlmostEqual(kills[0.5] / kills[1.0], 0.5, delta=0.1)

    def test_outbreak_kills_large_trees_standing(self):
        cfg = tt.HillslopeConfig(length=60, width=60, slope=0.3, dx=0.5, years=40, diffusivity=0.0,
                                 soil_model=False, seed=9, n_snapshots=2, disturbance="outbreak",
                                 first_disturbance=0.0, disturbance_interval=0.0, disturbance_severity=0.6)
        m = tt.TreeThrowModel(tt.SPECIES["eastern_hemlock"], cfg)
        R_all = tt.plate_at_age(m.sp, m.R_mature, m.H_mature, m.age)[0].mean()
        m.step()
        killed = m.snags["killed"] == 1
        self.assertGreater(killed.sum(), 0.3 * m.age.size)
        self.assertGreater(m.snags["R"][killed].mean(), R_all)      # large trees attacked preferentially
        self.assertEqual(m.n_events_by_cause["windstorm"], 0)       # nothing is thrown by the outbreak itself
        m.run(progress=False)
        self.assertGreater(m.n_events_by_cause["disturbance_snag"], 0.2 * killed.sum())


class QAQCRegressionTests(unittest.TestCase):
    """Defects found in the QA/QC review."""

    @staticmethod
    def budget(m, z0):
        dil = m.cfg.rock_to_soil_dilation
        dz = float((m.z[:-1] - z0[:-1]).sum()) * m.cell_area
        gain = (dil - 1) / dil * (m.rock_to_soil_tt + m.rock_to_soil_sp) if m.cfg.soil_model else 0.0
        return dz, gain - m.export_tt - m.export_diff + m.clip_error

    def test_global_volume_budget_closes(self):
        for kw in (dict(), dict(disturbance="windstorm", disturbance_interval=50, disturbance_severity=0.4),
                   dict(transport_law="nonlinear", slope=0.9, initial_soil_depth=0.2)):
            cfg = tt.HillslopeConfig(length=24, width=10, years=300, seed=4, **{"slope": 0.35, **kw})
            m = tt.TreeThrowModel(tt.SPECIES["spruce"], cfg)
            z0 = m.z.copy()
            m.run(progress=False)
            dz, expected = self.budget(m, z0)
            self.assertAlmostEqual(dz, expected, places=8)
            self.assertLess(m.clip_error, 1e-9)          # soil never driven negative (beyond round-off)
            self.assertGreaterEqual(m.h[:-1].min(), 0.0)

    def test_pit_wider_than_plot_does_not_overcut(self):
        cfg = tt.HillslopeConfig(length=20, width=2.0, slope=0.3, dx=0.25, years=1, initial_soil_depth=0.3,
                                 bedrock_pluck_fraction=0.0, infill_fraction=0.0, fresh_diffusivity=0.0, seed=1)
        m = tt.TreeThrowModel(tt.SPECIES["spruce"], cfg)
        h0 = m.h.copy()
        ev = m.uproot(10.0, 1.0, 1.65, 1.2, 0.0)
        self.assertGreaterEqual(m.h.min(), -1e-12)
        removed_soil = float((h0 - m.h).clip(min=0).sum()) * m.cell_area
        self.assertLessEqual(ev["volume"], float(h0[:-1].sum()) * m.cell_area)
        self.assertGreater(removed_soil, 0)

    def test_overlapping_disturbances_collapse_once(self):
        ratios = []
        for n_ev in (1, 2):
            cfg = tt.HillslopeConfig(length=16, width=12, slope=0.35, dx=0.25, years=3, soil_model=False,
                                     tree_throw=False, diffusivity=0.0, infill_fraction=0.0, seed=0)
            m = tt.TreeThrowModel(tt.SPECIES["spruce"], cfg)
            z0 = m.z.copy()
            for _ in range(n_ev):
                m.uproot(6.0, 6.0, 1.6, 1.2, 0.0)
            d0 = -(m.z - z0).min()
            for _ in range(3):
                m.step()
            ratios.append(-(m.z - z0).min() / d0)
        self.assertAlmostEqual(ratios[0], ratios[1], places=6)

    def test_collapse_across_periodic_edge(self):
        """A pit straddling the side boundary collapses like one in the middle of the plot."""
        res = []
        for y in (6.0, 0.1):
            cfg = tt.HillslopeConfig(length=16, width=12, slope=0.35, dx=0.25, years=3, soil_model=False,
                                     tree_throw=False, diffusivity=0.0, infill_fraction=0.0, seed=0)
            m = tt.TreeThrowModel(tt.SPECIES["spruce"], cfg)
            z0 = m.z.copy()
            m.uproot(6.0, y, 1.6, 1.2, math.pi / 2)
            d0 = -(m.z - z0).min()
            for _ in range(3):
                m.step()
            self.assertAlmostEqual(float((m.z - z0).sum()), 0.0, places=8)
            res.append(-(m.z - z0).min() / d0)
        self.assertAlmostEqual(res[0], res[1], delta=0.03)

    def test_limiter_handles_roundoff_negative_soil(self):
        cfg = tt.HillslopeConfig(length=10, width=5, years=1, seed=1)
        m = tt.TreeThrowModel(tt.SPECIES["pine"], cfg)
        h = np.full((4, 6), 0.2)
        h[1, 2] = -1e-20                                  # round-off left by an earlier step
        qx = np.zeros((3, 6))
        qy = np.zeros((4, 6))
        qx[0, 0] = 1.0                                    # one active face forces the full limiter path
        qx2, qy2 = m._limit_outflow(qx, qy, h, 1.0)
        self.assertTrue(np.isfinite(qx2).all() and np.isfinite(qy2).all())
        for periodic in (True, False):
            qy = np.zeros((4, 6 if periodic else 5))
            qx2, qy2 = m._limit_outflow(qx, qy, h, 1.0, periodic=periodic)
            self.assertTrue(np.isfinite(qx2).all() and np.isfinite(qy2).all())

    def test_config_validation_rejects_bad_values(self):
        for bad in (dict(diffusivity=-1e-3), dict(record_interval=0), dict(a_horizon_depth=0.0),
                    dict(rock_to_soil_dilation=0.5), dict(storm_min_age=-1), dict(first_disturbance=-5)):
            with self.assertRaises(ValueError):
                tt.HillslopeConfig(**bad).validate()


class SecondReviewRegressionTests(unittest.TestCase):
    def test_periodic_translation_preserves_entire_collapse_surface(self):
        runs = []
        for y in (6.0, 0.25):
            m = model("spruce", length=16, width=12, years=3, soil_model=False,
                      tree_throw=False, diffusivity=0, infill_fraction=0)
            m.uproot(6, y, 1.6, 1.2, math.pi / 2)
            runs.append(m)
        np.testing.assert_array_equal(np.roll(runs[0].h, -23, axis=1), runs[1].h)
        for m in runs:
            for _ in range(3):
                m.step()
        np.testing.assert_allclose(np.roll(runs[0].z, -23, axis=1), runs[1].z, rtol=0, atol=1e-12)

    def test_patches_sharing_transport_cells_are_solved_together(self):
        m = model()
        for cols in ((0, 11), (4, 6)):
            Dc = np.zeros((8, 12))
            Dc[3:5, list(cols)] = 0.2
            patches = list(m._fresh_patches(Dc))
            self.assertEqual(len(patches), 1)
            self.assertEqual(np.count_nonzero(patches[0][3]), 4)

    def test_disjoint_patches_equal_one_band_solve(self):
        m = model(soil_model=False)
        rng = np.random.default_rng(17)
        for shared_halo in (False, True):
            h = rng.normal(size=(12, 24))
            Dc = np.zeros_like(h)
            if shared_halo:
                Dc[3, 4] = Dc[3, 6] = 0.2
            else:
                Dc[2:4, 0:2] = Dc[2:4, -2:] = 0.2
                Dc[7:9, 10:12] = 0.2
            ref = h.copy()
            m._collapse_patch(ref, np.zeros_like(h), Dc, 0.3, True)
            for rows, cols, periodic, D in m._fresh_patches(Dc):
                ix = np.ix_(rows, cols)
                local = h[ix]
                m._collapse_patch(local, np.zeros_like(local), D, 0.3, periodic)
                h[ix] = local
            np.testing.assert_allclose(h, ref, rtol=0, atol=1e-14)

    def test_flux_limiter_keeps_soil_positive_and_balances_boundary(self):
        m = model()
        rng = np.random.default_rng(81)
        for periodic in (False, True):
            for fixed in (False, True):
                h = rng.uniform(0, 0.01, (7, 9))
                h[0, 0] = 0
                qx = rng.normal(0, 10, (6, 9))
                qy = rng.normal(0, 10, (7, 9 if periodic else 8))
                qx, qy = m._limit_outflow(qx, qy, h, 2, fixed_last_row=fixed, periodic=periodic)
                div = np.zeros_like(h)
                div[:-1] += qx
                div[1:] -= qx
                if periodic:
                    div += qy - np.roll(qy, 1, axis=1)
                else:
                    div[:, :-1] += qy
                    div[:, 1:] -= qy
                final = h - div * 2 / m.cfg.dx
                active = slice(None, -1) if fixed else slice(None)
                self.assertGreaterEqual(final[active].min(), -1e-14)
                change = (final[active] - h[active]).sum() * m.cell_area
                expected = -qx[-1].sum() * m.cfg.dx * 2 if fixed else 0
                self.assertAlmostEqual(change, expected, places=12)

    def test_nonlinear_diffusion_damps_small_grid_scale_noise(self):
        runs = [model(length=20, width=10, slope=0.95, diffusivity=0.02,
                      transport_law="nonlinear", soil_model=False, tree_throw=False,
                      fresh_diffusivity=0) for _ in range(2)]
        runs[1].h[20:60] += (1e-6 * (-1.0) ** np.arange(40))[:, None]
        for m in runs:
            m._diffusion_step(1)
        # Inspect cells far from the ridge/channel and perturbation edges.
        noise = (runs[1].z - runs[0].z)[30:50]
        self.assertLess(np.abs(noise).max(), 1e-6)

    def test_snags_use_age_at_death_cause_and_configured_decay(self):
        m = model("oak", snag_fall_rate=100, root_decay_time=6)
        m._add_snags([3, 4], [2, 2], [1, 1], [0.6, 0.6], [100, 0], killed=True)
        m.time = 12
        # Force both topplings and the conditional uprooting draws.
        with patch.object(m, "_throw") as throw, patch.object(m, "rng") as rng:
            rng.random.side_effect = [np.zeros(2), np.zeros(2)]
            m._snag_step(1)
        self.assertEqual(throw.call_count, 1)
        args = throw.call_args.args
        self.assertAlmostEqual(args[2], math.exp(-12 / 6) ** (1 / 3))
        self.assertAlmostEqual(args[3], 0.6 * math.exp(-12 / 6) ** (1 / 3))
        self.assertEqual(args[-1], "disturbance_snag")
        self.assertEqual(m.snags["x"].size, 0)

    def test_run_ends_at_requested_duration(self):
        m = model(years=2.5, dt=2, tree_throw=False)
        m.run(progress=False)
        self.assertEqual(m.time, 2.5)
        self.assertEqual(m.record.time[-1], 2.5)
        m.run(years=0.75, progress=False)
        self.assertEqual(m.time, 3.25)

    def test_invalid_step_or_duration_does_not_mutate_model(self):
        m = model()
        h0, age0 = m.h.copy(), m.age.copy()
        for bad in (0, -1, float("nan"), float("inf")):
            with self.subTest(dt=bad), self.assertRaises(ValueError):
                m.step(bad)
            with self.subTest(years=bad), self.assertRaises(ValueError):
                m.run(years=bad, progress=False)
        np.testing.assert_array_equal(m.h, h0)
        np.testing.assert_array_equal(m.age, age0)
        self.assertEqual(m.time, 0)

    def test_nonfinite_parameters_rejected_early(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for name in ("length", "slope", "dt", "years", "wind_direction_deg", "record_interval"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    tt.HillslopeConfig(**{name: value}).validate()
            for name in ("lifespan_mean", "lifespan_sd", "density", "root_plate_depth"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    tt.replace(tt.SPECIES["pine"], **{name: value}).validate()

    def test_integer_soil_depth_allocates_floating_point_state(self):
        for depth in (0, 1):
            m = model(initial_soil_depth=depth, tree_throw=False)
            self.assertEqual(m.h.dtype.kind, "f")
            m.step()
            self.assertTrue(np.isfinite(m.z).all())

    def test_zero_event_expectation_is_well_defined(self):
        for sp in (tt.replace(tt.SPECIES["pine"], uproot_fraction=0, snag_fraction=0),
                   tt.replace(tt.SPECIES["pine"], min_uproot_age=1e6)):
            with np.errstate(all="raise"):
                e = tt.expected_tree_throw(sp, 0.35, n_samples=1000)
            self.assertEqual(e["events_per_ha_yr"], 0)
            self.assertEqual(e["flux_m2_per_yr"], 0)
            self.assertEqual(e["diffusivity_m2_per_yr"], 0)
            self.assertEqual(e["surface_turnover_yr"], math.inf)
            self.assertTrue(math.isnan(e["mean_event_volume_m3"]))

    def test_zero_snag_fall_rate_removes_only_snag_events(self):
        sp = tt.SPECIES["oak"]
        e = tt.expected_tree_throw(sp, 0.35, n_samples=2000, snag_fall_rate=0)
        _, ages = tt.draw_ages_at_death(np.random.default_rng(12345), sp, 100000)
        expected = tt.stand_rates(sp)["deaths_per_ha_yr"] * np.mean(ages >= sp.min_uproot_age) * sp.uproot_fraction
        self.assertAlmostEqual(e["events_per_ha_yr"], expected)
        self.assertGreater(e["flux_m2_per_yr"], 0)
        all_snags = tt.replace(sp, uproot_fraction=0, snag_fraction=1)
        e = tt.expected_tree_throw(all_snags, 0.35, n_samples=2000, snag_fall_rate=0)
        self.assertEqual(e["flux_m2_per_yr"], 0)

    def test_run_handles_roundoff_in_requested_step_count(self):
        m = model(years=0.1 + 0.2, dt=0.1, tree_throw=False)
        m.run(progress=False)
        self.assertEqual(m.time, m.cfg.years)
        self.assertEqual(m.record.time[-1], m.time)

    def test_expected_samples_must_be_positive_integer(self):
        for n in (0, -1, 1.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                tt.expected_tree_throw(tt.SPECIES["pine"], 0.35, n_samples=n)
        for bad in (dict(soil_depth=-1), dict(infill_critical_slope=0), dict(root_decay_time=0),
                    dict(rock_to_soil_dilation=0.5), dict(snag_fall_rate=-1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tt.expected_tree_throw(tt.SPECIES["pine"], 0.35, **bad)
        with self.assertRaises(ValueError):
            tt.HillslopeConfig(n_snapshots=1.5).validate()

    def test_expected_cache_is_reused_without_exposing_mutable_results(self):
        sp = tt.replace(tt.SPECIES["pine"], key="cache_regression")
        with patch.object(tt, "draw_mature_sizes", wraps=tt.draw_mature_sizes) as draw:
            first = tt.expected_tree_throw(sp, 0.35, n_samples=1200)
            original = dict(first)
            first["flux_m2_per_yr"] = -999
            second = tt.expected_tree_throw(sp, 0.35, n_samples=1200)
            self.assertEqual(second, original)
            self.assertEqual(draw.call_count, 1)
            tt.expected_tree_throw(sp, 0.4, n_samples=1200)
            self.assertEqual(draw.call_count, 2)
            tt.expected_tree_throw(sp, 0.4, n_samples=1200, seed=None)
            tt.expected_tree_throw(sp, 0.4, n_samples=1200, seed=None)
            self.assertEqual(draw.call_count, 4)

    def test_qa_budget_gate_rejects_failures(self):
        from validation.qaqc_checks import check_budget
        good = ("probe", 1.0, 1.0, 0.0, 0, 0.0, True, 0.0, True)
        check_budget(good)
        for index, value in ((1, 1.0 + 1e-9), (3, 1e-8), (6, False), (7, -1e-6)):
            bad = list(good)
            bad[index] = value
            with self.subTest(index=index), self.assertRaises(AssertionError):
                check_budget(bad)

    def test_summary_preserves_model_rng_state(self):
        m = model()
        state = m.rng.bit_generator.state
        m.summary()
        m.summary()
        self.assertEqual(m.rng.bit_generator.state, state)


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
