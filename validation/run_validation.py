#!/usr/bin/env python3
"""
run_validation.py — literature validation suite for TreeThrow.py
================================================================
Runs a set of numerical experiments with the tree-throw model and compares
the results with published field and modelling values.  Writes figures, a
machine-readable results file and a Markdown report to validation/results/.

    python validation/run_validation.py            # full suite (~3-5 min on 4 cores)
    python validation/run_validation.py --quick    # smaller domains, fewer seeds

IMPORTANT: the benchmark values in BENCHMARKS were transcribed from abstracts
and search-engine summaries of the cited papers, because the publishers' sites
could not be reached from the environment in which the suite was built.  Each
benchmark carries a `status` field; values marked "secondary" must be checked
against the original papers before the verdicts are relied on.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import TreeThrow as tt  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import zadrozny  # noqa: E402
import doane  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

# ----------------------------------------------------------------------------
# Literature benchmarks
# ----------------------------------------------------------------------------
BENCHMARKS = {
    "boubin_pit_mound_volume": {
        "value": {"mean_m3": 2.8, "min_m3": 0.1, "max_m3": 5.6},
        "source": "Šamonil et al., Boubínský old-growth forest (Fagus/Picea), as cited in "
                  "Geographica Helvetica 76:319 (2021), Jura Mountains tree-throw study",
        "url": "https://gh.copernicus.org/articles/76/319/2021/",
        "status": "secondary",
    },
    "norman_1995_slope_trend": {
        "value": {"finding": "pit and mound volumes increase with slope angle; 189 pit-mound pairs, "
                             "slopes 0-54 %, northern Lower Michigan, sandy soils"},
        "source": "Norman, Schaetzl & Small (1995) Geomorphology 14:19-27",
        "url": "https://people.geo.msu.edu/schaetzl/PDFs/Norman-treethrow.pdf",
        "status": "secondary",
    },
    "gallaway_2009_slope_trend": {
        "value": {"finding": "significant relation between gradient and root-plate volume; sediment "
                             "disturbance higher on steeper gradients",
                  "flux_vs_creep": "root-throw transport ~two orders of magnitude below creep "
                                   "(2007 AGU abstract)"},
        "source": "Gallaway, Martin & Johnson (2009) ESPL 34:1255-1269; 2007 AGU abstract H53C-1401",
        "url": "https://ui.adsabs.harvard.edu/abs/2007AGUFM.H53C1401G/abstract",
        "status": "secondary",
    },
    "zadrozny_2017_spruce": {
        "value": {"pit_1yr_LxWxD_m": [2.74, 4.23, 0.95],
                  "pit_depth_vs_root_plate_height": "depth = -0.3758*PR + 2.049, R2 = 0.64",
                  "finding": "slope steepness has the largest influence on soil dislocation (RDA)"},
        "source": "Zadrożny et al. (2017) iForest 10:783-787, doi:10.3832/ifor1959-010 "
                  "(see validation/zadrozny.py for Table 1)",
        "url": "https://iforest.sisef.org/contents/?id=ifor1959-010",
        "status": "primary (full text read)",
    },
    "pawlik_2016_sudetes_root_plate": {
        "value": {"spruce_site_mean_min_m3": 0.4, "spruce_site_mean_max_m3": 4.2, "beech_mean_m3": 2.4},
        "source": "Pawlik et al. (2016) ESPL, Sudetes Mountains, doi:10.1002/esp.3950",
        "url": "https://onlinelibrary.wiley.com/doi/10.1002/esp.3950",
        "status": "secondary",
    },
    "carpathian_root_plate": {
        "value": {"mean_min_m3": 2.2, "mean_max_m3": 3.35},
        "source": "Uprooting study, Polish Carpathians (as summarised in search results; citation to confirm)",
        "url": "https://www.researchgate.net/publication/309006609_Domination_of_hillslope_denudation_by_tree_uprooting_in_an_old-growth_forest",
        "status": "secondary (citation unconfirmed)",
    },
    "spruce_40cm_dbh_volume": {
        "value": {"dbh_cm": 40, "volume_m3": 1.0, "sd_m3": 0.25},
        "source": "Review estimate for a ~100-150-yr-old Norway spruce (as summarised in search results; "
                  "citation to confirm)",
        "url": "https://doi.org/10.3390/f11111143",
        "status": "secondary (citation unconfirmed)",
    },
    "doane_2021_flux_share": {
        "value": {"share_min": 0.11, "share_max": 0.18, "R_min": 0.12, "R_max": 0.22,
                  "aspect_effect": "~50 % higher tree-throw flux on east-facing slopes",
                  "variance_m2": [0.001, 0.03], "couplet": "mu_A = 0.68 m2, sigma_A2 = 0.05, phi = 0.83"},
        "source": "Doane et al. (2021) GRL 48, e2021GL094987, doi:10.1029/2021GL094987 "
                  "(Brown County, southern Indiana hardwoods; see validation/doane.py)",
        "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL094987",
        "status": "primary (full text read)",
    },
    "constantine_2012_flux": {
        "value": {"flux_m2_yr": 2.0e-4, "peak_D_m2_yr": 3.5e-3},
        "source": "Constantine, Schelhaas, Gabet & Mudd (2012) Geomorphology (windthrow model)",
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S0169555X12003066",
        "status": "secondary",
    },
    "razula_2017_rate_coverage": {
        "value": {"uprootings_per_ha_yr": 1.5, "pit_mound_cover_min": 0.14},
        "source": "Razula old-growth forest, Western Carpathians (Fagus/Abies/Picea), "
                  "Geomorphology 2017 (Šamonil group)",
        "url": "https://is.muni.cz/publication/1402037",
        "status": "secondary",
    },
    "jura_2021_mound_persistence": {
        "value": {"mound_height_m": 0.5, "persistence_yr_min": 200, "persistence_yr_max": 250},
        "source": "Geographica Helvetica 76:319 (2021), Jura Mountains tree-throw study",
        "url": "https://gh.copernicus.org/articles/76/319/2021/",
        "status": "secondary",
    },
    "martin_2013_pitmound_D": {
        "value": {"D_min_m2_yr": 0.01, "D_max_m2_yr": 0.1},
        "source": "Martin et al. (2013) Geomorphology, Canadian Rockies pit-mound degradation modelling",
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S0169555X13002237",
        "status": "secondary",
    },
}

# ----------------------------------------------------------------------------
# Run helpers (top level so they can be pickled for multiprocessing)
# ----------------------------------------------------------------------------


def _run(job):
    key, sp_overrides, cfg_kw = job
    sp = replace(tt.SPECIES[key], **sp_overrides) if sp_overrides else tt.SPECIES[key]
    cfg = tt.HillslopeConfig(**cfg_kw)
    m = tt.TreeThrowModel(sp, cfg)
    m.run(progress=False)
    s = m.summary()
    ev = np.array(m.event_log) if m.event_log else np.zeros((0, 9))
    return {
        "species": key, "slope": cfg.slope, "seed": cfg.seed, "overrides": sp_overrides,
        "summary": {k: v for k, v in s.items() if k not in ("species", "config", "expected")},
        "expected": s["expected"],
        "event_volume": ev[:, 7].tolist(),          # net displaced volume (excavated minus infill)
        "excavated_volume": ev[:, 6].tolist(),
        "coverage": pit_mound_coverage(m),
        "doane_variance": doane.topographic_variance(m),
    }


def pit_mound_coverage(m: "tt.TreeThrowModel", sigma_m: float = 3.0):
    """Fraction of hillslope surface whose relief relative to a smoothed surface exceeds
    thresholds of 5, 10 and 20 cm (an objective proxy for mapped pit-mound cover)."""
    from scipy.ndimage import gaussian_filter
    z = m.z[:-1]
    s = sigma_m / m.cfg.dx
    smooth = gaussian_filter(z, sigma=(s, s), mode=("nearest", "wrap"))
    rel = np.abs(z - smooth)
    # Exclude the ridge and channel margins, where the smoothing is biased.
    k = int(round(2 * sigma_m / m.cfg.dx))
    rel = rel[k:-k] if rel.shape[0] > 2 * k + 4 else rel
    return {f"{int(t * 100)}cm": float(np.mean(rel > t)) for t in (0.05, 0.10, 0.20)}


def base_cfg(quick: bool, **kw):
    c = dict(length=30.0 if quick else 40.0, width=30.0 if quick else 40.0, slope=0.35, dx=0.25,
             years=1500.0 if quick else 3000.0, diffusivity=1.0e-3, n_snapshots=2, record_interval=50.0)
    c.update(kw)
    return c


def run_jobs(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run, jobs))


def verdict(ok, partial=False):
    return "PASS" if ok else ("PARTIAL" if partial else "FAIL")


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_internal_consistency(res_species):
    """V0: simulated tree-throw diffusivity vs. the analytic (Monte-Carlo) expectation."""
    out = {}
    for key in tt.SPECIES:
        rs = [r for r in res_species if r["species"] == key]
        ratio = np.array([r["summary"]["effective_D_tree_throw_m2_yr"] / r["expected"]["diffusivity_m2_per_yr"]
                          for r in rs])
        mean, se = float(ratio.mean()), float(ratio.std(ddof=1) / math.sqrt(len(ratio)))
        out[key] = {"ratio_mean": mean, "ratio_se": se, "ratios": ratio.tolist(),
                    "verdict": verdict(abs(mean - 1) <= max(2 * se, 0.1))}
    return out


def test_event_volume(res_species):
    """V1: uprooting event volume distribution vs. Boubín pit-mound volumes."""
    b = BENCHMARKS["boubin_pit_mound_volume"]["value"]
    out = {}
    for key in tt.SPECIES:
        vols = np.concatenate([np.array(r["event_volume"]) for r in res_species if r["species"] == key])
        p5, p50, p95 = np.percentile(vols, [5, 50, 95])
        mean = float(vols.mean())
        inside = float(np.mean((vols >= b["min_m3"]) & (vols <= b["max_m3"])))
        ok = abs(mean / b["mean_m3"] - 1) <= 0.5
        extra = {}
        if key == "deciduous":
            extra["ratio_to_sudetes_beech_mean"] = mean / BENCHMARKS["pawlik_2016_sudetes_root_plate"]["value"]["beech_mean_m3"]
            cb = BENCHMARKS["carpathian_root_plate"]["value"]
            extra["within_carpathian_mean_range"] = bool(cb["mean_min_m3"] <= mean <= cb["mean_max_m3"])
        if key == "pine":
            ps = BENCHMARKS["pawlik_2016_sudetes_root_plate"]["value"]
            extra["within_sudetes_spruce_site_means"] = bool(ps["spruce_site_mean_min_m3"] <= mean <= ps["spruce_site_mean_max_m3"])
            extra["ratio_to_spruce_40cm_dbh"] = mean / BENCHMARKS["spruce_40cm_dbh_volume"]["value"]["volume_m3"]
        out[key] = {"n_events": int(vols.size), "mean_m3": mean, "median_m3": float(p50), **extra,
                    "p5_m3": float(p5), "p95_m3": float(p95), "max_m3": float(vols.max()),
                    "fraction_within_published_range": inside,
                    "ratio_to_benchmark_mean": mean / b["mean_m3"],
                    "applicable": key in ("deciduous", "pine", "spruce"),
                    "verdict": verdict(ok) if key in ("deciduous", "pine", "spruce") else "N/A (no aspen benchmark)"}
    return out


def test_slope_dependence(res_slope):
    """V2: slope dependence of event volume and downslope transport (Norman 1995; Gallaway 2009)."""
    slopes = sorted({r["slope"] for r in res_slope})
    rows = []
    for s in slopes:
        rs = [r for r in res_slope if r["slope"] == s]
        vols = np.concatenate([np.array(r["event_volume"]) for r in rs])
        rows.append({
            "slope": s,
            "mean_event_volume_m3": float(vols.mean()),
            "se_event_volume_m3": float(vols.std(ddof=1) / math.sqrt(vols.size)),
            "D_tt_m2_yr": float(np.mean([r["summary"]["effective_D_tree_throw_m2_yr"] for r in rs])),
            "D_tt_expected_m2_yr": rs[0]["expected"]["diffusivity_m2_per_yr"],
            "q_tt_m2_yr": float(np.mean([r["summary"]["mean_flux_tree_throw_m2_yr"] for r in rs])),
            "q_tt_expected_m2_yr": rs[0]["expected"]["flux_m2_per_yr"],
            "mean_cos_fall": rs[0]["expected"]["mean_cos_fall"],
            "tree_throw_share": float(np.mean([r["summary"]["tree_throw_share_of_flux"] for r in rs])),
        })
    S = np.array([r["slope"] for r in rows])
    V = np.array([r["mean_event_volume_m3"] for r in rows])
    Q = np.array([r["q_tt_m2_yr"] for r in rows])
    vol_trend = float(np.polyfit(S, V / V.mean(), 1)[0])       # fractional change per unit gradient
    flux_trend = float(np.polyfit(S, Q / Q.mean(), 1)[0])
    return {
        "rows": rows,
        "event_volume_fractional_trend_per_unit_slope": vol_trend,
        "flux_fractional_trend_per_unit_slope": flux_trend,
        "verdict_volume_increases_with_slope": verdict(vol_trend > 0.2),
        "verdict_flux_increases_with_slope": verdict(flux_trend > 0.2),
    }


def test_flux_share(slope_tests, res_species):
    """V3: tree-throw share of total hillslope flux vs. Doane et al. (2021)."""
    b = BENCHMARKS["doane_2021_flux_share"]["value"]
    dec = {r["slope"]: r["tree_throw_share"] for r in slope_tests["rows"]}
    sp = {}
    for key in tt.SPECIES:
        sh = [r["summary"]["tree_throw_share_of_flux"] for r in res_species if r["species"] == key]
        sp[key] = float(np.mean(sh))
    lo, hi = b["share_min"], b["share_max"]
    d35 = sp["deciduous"]
    ok = lo <= d35 <= hi
    near = (0.5 * lo) <= d35 <= (1.5 * hi)
    return {"deciduous_share_by_slope": dec, "share_by_species_at_S0.35": sp,
            "verdict": verdict(ok, partial=near)}


def test_flux_magnitude(res_species):
    """V4: tree-throw flux magnitude vs. Constantine et al. (2012) and Gallaway et al. (2009)."""
    b = BENCHMARKS["constantine_2012_flux"]["value"]
    out = {}
    for key in tt.SPECIES:
        rs = [r for r in res_species if r["species"] == key]
        q = float(np.mean([r["summary"]["mean_flux_tree_throw_total_m2_yr"] for r in rs]))
        qc = float(np.mean([r["summary"]["mean_flux_pit_mound_collapse_m2_yr"] for r in rs]))
        qd = float(np.mean([r["summary"]["mean_flux_diffusion_m2_yr"] for r in rs]))
        D = float(np.mean([r["summary"]["effective_D_tree_throw_total_m2_yr"] for r in rs]))
        ratio = q / b["flux_m2_yr"]
        out[key] = {"q_tt_m2_yr": q, "q_collapse_m2_yr": qc, "D_tt_m2_yr": D, "q_tt_over_q_creep": q / qd,
                    "ratio_to_constantine_flux": ratio,
                    "ratio_to_constantine_peak_D": D / b["peak_D_m2_yr"],
                    "verdict_vs_constantine": verdict(0.1 <= ratio <= 10, partial=False),
                    "verdict_vs_gallaway_two_orders_below_creep": verdict(q / qd <= 0.03)}
    return out


def test_rate_coverage(res_species, res_razula):
    """V5: uprooting rate and pit-mound ground cover vs. the Razula old-growth forest."""
    b = BENCHMARKS["razula_2017_rate_coverage"]["value"]
    def agg(rs):
        cov = {k: float(np.mean([r["coverage"][k] for r in rs])) for k in rs[0]["coverage"]}
        rate = float(np.mean([r["summary"]["uprooting_rate_per_ha_yr"] for r in rs]))
        return {"uprootings_per_ha_yr": rate, "cover": cov}
    out = {k: agg([r for r in res_species if r["species"] == k]) for k in tt.SPECIES}
    out["deciduous_at_razula_rate"] = agg(res_razula)
    dec = out["deciduous"]
    out["verdict_rate_deciduous"] = verdict(abs(dec["uprootings_per_ha_yr"] / b["uprootings_per_ha_yr"] - 1) <= 0.5)
    cov = out["deciduous_at_razula_rate"]["cover"]["10cm"]
    out["verdict_cover_at_razula_rate"] = verdict(cov >= b["pit_mound_cover_min"],
                                                  partial=cov >= 0.5 * b["pit_mound_cover_min"])
    return out


def test_mound_persistence(quick: bool):
    """V6: decay of an isolated pit-mound pair by background diffusion."""
    bj = BENCHMARKS["jura_2021_mound_persistence"]["value"]
    sp = tt.SPECIES["deciduous"]
    R, H = 0.5 * sp.root_plate_diameter, sp.root_plate_depth
    results = {}
    D0 = 1.0e-3
    for D in (D0, 1.0e-2):
        cfg = dict(length=16.0, width=10.0, slope=0.35, dx=0.1, years=1.0, diffusivity=D,
                   soil_model=False, tree_throw=False, seed=0)
        ref = tt.TreeThrowModel(sp, tt.HillslopeConfig(**cfg))
        m = tt.TreeThrowModel(sp, tt.HillslopeConfig(**cfg))
        m.uproot(8.0, 5.0, R, H, 0.0)
        rel0 = m.z - ref.z
        h0 = float(rel0.max())
        t, hist = 0.0, [(0.0, h0, float(-rel0.min()))]
        tmax = 20000.0 if D < 1e-2 else 2000.0
        t_half = t_5cm = t_50cm = None
        while t < tmax:
            # Annual steps while fresh soil is collapsing, then longer steps.
            dt = 1.0 if t < 30 else (10.0 if D < 1e-2 else 2.0)
            m.step(dt)
            ref.step(dt)
            t += dt
            rel = m.z - ref.z
            hm, pd = float(rel.max()), float(-rel.min())
            hist.append((t, hm, pd))
            if t_half is None and hm <= 0.5 * h0:
                t_half = t
            if t_50cm is None and hm <= bj["mound_height_m"]:
                t_50cm = t
            if t_5cm is None and hm <= 0.05:
                t_5cm = t
                break
        results[f"D={D:g}"] = {"D_m2_yr": D, "initial_mound_height_m": h0,
                               "time_to_half_height_yr": t_half, "time_to_5cm_yr": t_5cm,
                               "time_to_50cm_yr": t_50cm,
                               "persistence_50cm_to_5cm_yr": (t_5cm - t_50cm) if (t_5cm and t_50cm is not None) else None,
                               "history": hist[:: max(len(hist) // 200, 1)]}
    # Jura benchmark: a 50 cm mound persists 200-250 yr.  Measure the time from the moment the
    # model mound is 50 cm high until it is below 5 cm.  This happens after fresh collapse has
    # ended, so it scales as 1/D and gives the background D that reproduces the benchmark.
    r0 = results[f"D={D0:g}"]
    t_bench = 0.5 * (bj["persistence_yr_min"] + bj["persistence_yr_max"])
    pers = r0["persistence_50cm_to_5cm_yr"]
    D_implied = D0 * pers / t_bench if pers else None
    bm = BENCHMARKS["martin_2013_pitmound_D"]["value"]
    lo, hi = bj["persistence_yr_min"] / 2, bj["persistence_yr_max"] * 2
    return {
        "runs": results,
        "root_plate": {"R_m": R, "H_m": H},
        "benchmark_persistence_yr": [bj["persistence_yr_min"], bj["persistence_yr_max"]],
        "persistence_50cm_to_5cm_yr": pers,
        "D_implied_by_jura_m2_yr": D_implied,
        "martin_2013_D_range_m2_yr": [bm["D_min_m2_yr"], bm["D_max_m2_yr"]],
        "verdict_default_D": verdict(pers is not None and lo <= pers <= hi,
                                     partial=pers is not None and lo / 2 <= pers <= hi * 2),
    }


# ----------------------------------------------------------------------------
# Figures and report
# ----------------------------------------------------------------------------


def make_figures(R):
    plt = tt._import_pyplot()
    colors = {"pine": "tab:green", "deciduous": "tab:orange", "aspen": "tab:olive", "spruce": "tab:blue"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("TreeThrow.py — literature validation", fontsize=13)

    # V1 event volumes
    a = ax[0, 0]
    b = BENCHMARKS["boubin_pit_mound_volume"]["value"]
    data = [np.array(R["_vols"][k]) for k in tt.SPECIES]
    a.boxplot(data, tick_labels=list(tt.SPECIES), showfliers=False, whis=(5, 95))
    for i, k in enumerate(tt.SPECIES):
        a.plot(i + 1, data[i].mean(), "o", color=colors[k])
    a.axhspan(b["min_m3"], b["max_m3"], color="gray", alpha=0.15, label="Boubín range 0.1–5.6 m³")
    a.axhline(b["mean_m3"], color="k", ls="--", lw=1, label="Boubín mean 2.8 m³")
    a.set(title="V1  Displaced volume per uprooting (5–95 %, dot = mean)", ylabel="displaced soil (m³)")
    a.legend(fontsize=8)

    # V2 slope dependence
    a = ax[0, 1]
    rows = R["V2_slope_dependence"]["rows"]
    S = np.array([r["slope"] for r in rows])
    V = np.array([r["mean_event_volume_m3"] for r in rows])
    E = np.array([r["se_event_volume_m3"] for r in rows])
    a.errorbar(S, V / V.mean(), yerr=E / V.mean(), fmt="o-", color="tab:orange", label="model event volume")
    q = np.array([r["q_tt_m2_yr"] for r in rows])
    a.plot(S, q / q.mean(), "s-", color="tab:green", label="model tree-throw flux")
    qe = np.array([r["q_tt_expected_m2_yr"] for r in rows])
    a.plot(S, qe / q.mean(), ":", color="tab:green", label="analytic flux")
    a.axhline(1, color="k", lw=0.5)
    a.set(title="V2  Slope dependence (deciduous)\nNorman 1995 / Gallaway 2009 / Zadrożny 2017: volume ↑ with slope",
          xlabel="gradient (m/m)", ylabel="value / mean")
    a.legend(fontsize=8)

    # V3 flux share
    a = ax[0, 2]
    bd = BENCHMARKS["doane_2021_flux_share"]["value"]
    sh = R["V3_flux_share"]["deciduous_share_by_slope"]
    a.plot(list(map(float, sh)), [100 * v for v in sh.values()], "o-", color="tab:orange", label="deciduous (model)")
    for k in tt.SPECIES:
        a.plot(0.35, 100 * R["V3_flux_share"]["share_by_species_at_S0.35"][k], "D", color=colors[k], label=f"{k} S=0.35")
    a.axhspan(100 * bd["share_min"], 100 * bd["share_max"], color="gray", alpha=0.2, label="Doane 2021: 11–18 %")
    a.set(title="V3  Tree-throw share of total flux", xlabel="gradient (m/m)", ylabel="share (%)")
    a.legend(fontsize=8)

    # V4 magnitude
    a = ax[1, 0]
    v4 = R["V4_flux_magnitude"]
    ks = list(tt.SPECIES)
    a.bar(ks, [v4[k]["q_tt_m2_yr"] for k in ks], color=[colors[k] for k in ks],
          label="model q_tt incl. collapse (S=0.35)")
    bc = BENCHMARKS["constantine_2012_flux"]["value"]["flux_m2_yr"]
    a.axhline(bc, color="k", ls="--", label="Constantine 2012 ≈ 2×10⁻⁴")
    qd = np.mean([v4[k]["q_tt_m2_yr"] / v4[k]["q_tt_over_q_creep"] for k in ks])
    a.axhline(qd, color="tab:brown", ls=":", label="model creep flux")
    a.axhline(qd / 100, color="tab:red", ls="-.", label="Gallaway: creep / 100")
    a.set_yscale("log")
    a.set(title="V4  Tree-throw flux magnitude", ylabel="flux (m² / yr)")
    a.legend(fontsize=8)

    # V5 rate & coverage
    a = ax[1, 1]
    v5 = R["V5_rate_coverage"]
    br = BENCHMARKS["razula_2017_rate_coverage"]["value"]
    labels = ks + ["deciduous\n@1.5/ha/yr"]
    keys = ks + ["deciduous_at_razula_rate"]
    rate = [v5[k]["uprootings_per_ha_yr"] for k in keys]
    cov = [100 * v5[k]["cover"]["10cm"] for k in keys]
    xx = np.arange(len(keys))
    a.bar(xx - 0.2, rate, 0.4, color="tab:blue", label="uprootings /ha/yr")
    a.axhline(br["uprootings_per_ha_yr"], color="tab:blue", ls="--", lw=1)
    a2 = a.twinx()
    a2.bar(xx + 0.2, cov, 0.4, color="tab:orange", label="cover >10 cm relief (%)")
    a2.axhline(100 * br["pit_mound_cover_min"], color="tab:orange", ls="--", lw=1)
    a.set_xticks(xx, labels, fontsize=8)
    a.set(title="V5  Uprooting rate & pit-mound cover\n(dashed = Razula: 1.5 /ha/yr, >14 %)",
          ylabel="uprootings / ha / yr")
    a2.set_ylabel("ground cover (%)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # V6 persistence
    a = ax[1, 2]
    v6 = R["V6_mound_persistence"]
    for name, run in v6["runs"].items():
        h = np.array(run["history"])
        a.plot(h[:, 0], h[:, 1], label=f"mound height, {name} m²/yr")
    bj = v6["benchmark_persistence_yr"]
    a.axvspan(bj[0], bj[1], color="gray", alpha=0.2, label="Jura: 50 cm mound persists 200–250 yr")
    a.axhline(0.05, color="k", lw=0.5, ls=":")
    a.set_xscale("log")
    a.set(title="V6  Mound decay by diffusion (deciduous root plate)", xlabel="years", ylabel="mound height (m)")
    a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "validation_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fmt(x, p=2):
    if x is None:
        return "—"
    if isinstance(x, float) and (abs(x) < 1e-2 or abs(x) >= 1e4):
        return f"{x:.{p}e}"
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


BASELINE_METRICS = [
    ("V1 mean displaced volume, deciduous (m³)", ("V1_event_volume", "deciduous", "mean_m3"), "{:.2f}", "≈ 2.4–2.8"),
    ("V1 mean displaced volume, pine (m³)", ("V1_event_volume", "pine", "mean_m3"), "{:.2f}", "≈ 2.8"),
    ("V2 displaced volume trend with slope (% per unit gradient)",
     ("V2_slope_dependence", "event_volume_fractional_trend_per_unit_slope"), "pct", "> 0"),
    ("V3 tree-throw share of flux, deciduous", ("V3_flux_share", "share_by_species_at_S0.35", "deciduous"),
     "pct", "11–18 %"),
    ("V4 tree-throw flux, deciduous (m²/yr)", ("V4_flux_magnitude", "deciduous", "q_tt_m2_yr"), "{:.1e}", "2×10⁻⁴"),
    ("V5 uprooting rate, deciduous (/ha/yr)", ("V5_rate_coverage", "deciduous", "uprootings_per_ha_yr"), "{:.2f}", "1.5"),
    ("V5 pit-mound cover > 10 cm, deciduous", ("V5_rate_coverage", "deciduous", "cover", "10cm"), "pct", "> 14 %"),
    ("V5 pit-mound cover > 10 cm at Razula rate", ("V5_rate_coverage", "deciduous_at_razula_rate", "cover", "10cm"),
     "pct", "> 14 %"),
    ("V6 fresh mound half-height time (yr)", ("V6_mound_persistence", "runs", "D=0.001", "time_to_half_height_yr"),
     "{:.0f}", "≈ 2–3 (Zadrożny mound 0.71→0.42 m in 2 yr)"),
    ("Z2 size–depth correlation, pine", ("Z_zadrozny_2017", "Z2_size_depth", "pine", "r_model"), "{:+.2f}", "−0.80"),
]


def baseline_comparison(R):
    path = OUT / "baseline_results.json"
    if not path.exists():
        return []
    base = json.loads(path.read_text(encoding="utf-8"))["results"]
    L = ["## Change from the previous model\n",
         "Baseline: the model and suite before the root-plate, infill and pit-mound-collapse changes "
         "(`baseline_results.json`). Pit depth after 3 / 5 yr is shown relative to the 1-yr pit.\n",
         "| metric | literature | previous model | updated model |", "|---|---|---|---|"]

    def f(v, fmt):
        if v is None:
            return "—"
        return f"{100 * v:.0f} %" if fmt == "pct" else fmt.format(v)

    for label, p, fmt, lit in BASELINE_METRICS:
        L.append(f"| {label} | {lit} | {f(_get(base, *p), fmt)} | {f(_get(R, *p), fmt)} |")
    def persistence(runs):
        h = np.array(_get(runs, "runs", "D=0.001", "history") or [])
        if not len(h):
            return None
        t50 = h[np.argmax(h[:, 1] <= 0.5), 0] if (h[:, 1] <= 0.5).any() else None
        t5 = h[np.argmax(h[:, 1] <= 0.05), 0] if (h[:, 1] <= 0.05).any() else None
        return (t5 - t50) if (t5 is not None and t50 is not None) else None
    pb, pn = persistence(base.get("V6_mound_persistence", {})), R["V6_mound_persistence"]["persistence_50cm_to_5cm_yr"]
    L.append(f"| V6 persistence of a 50 cm mound to < 5 cm (yr) | 200–250 | {f(pb, '{:.0f}')} | {f(pn, '{:.0f}')} |")
    old = _get(base, "Z_zadrozny_2017", "Z4_degradation", "pred_default_D", "pit")
    new = R["Z_zadrozny_2017"]["Z4_degradation"]["comparison"]["pit depth"]
    if old:
        L.append(f"| Z4 spruce pit depth at 3 / 5 yr, relative to 1 yr | 0.68 / 0.55 | {old[0]:.2f} / {old[1]:.2f} | "
                 f"{new[1]['model'] / new[0]['model']:.2f} / {new[2]['model'] / new[0]['model']:.2f} |")
    L.append("")
    return L


def write_report(R, runtime, quick):
    v0, v1, v2, v3, v4, v5, v6 = (R[k] for k in ("V0_internal", "V1_event_volume", "V2_slope_dependence",
                                                 "V3_flux_share", "V4_flux_magnitude", "V5_rate_coverage",
                                                 "V6_mound_persistence"))
    L = []
    L.append("# TreeThrow.py — validation results\n")
    L.append(f"Generated by `validation/run_validation.py`{' --quick' if quick else ''} "
             f"in {runtime / 60:.1f} min. Figures: `validation_summary.png`; data: `validation_results.json`.\n")
    L.append("> **Benchmark caveat.** Zadrożny et al. (2017) (Z rows) and Doane et al. (2021) (D rows, V3) were read "
             "from the full papers. "
             "Every other benchmark comes from abstracts or search summaries (status *secondary*) because the "
             "publishers' websites could not be reached; check them against the original papers before relying "
             "on those verdicts.\n")
    L.append("![summary](validation_summary.png)\n")

    L.append("## Scorecard\n")
    L.append("| Test | Benchmark | Model | Verdict |")
    L.append("|---|---|---|---|")
    for k in tt.SPECIES:
        L.append(f"| V0 internal: simulated / analytic D_tt ({k}) | 1.0 | "
                 f"{v0[k]['ratio_mean']:.2f} ± {v0[k]['ratio_se']:.2f} (SE) | {v0[k]['verdict']} |")
    for k in ("deciduous", "pine", "spruce"):
        L.append(f"| V1 mean displaced volume ({k}) | 2.8 m³ (Boubín) | {v1[k]['mean_m3']:.2f} m³ | {v1[k]['verdict']} |")
    d, pn = v1["deciduous"], v1["pine"]
    L.append(f"| V1b deciduous vs. Sudetes beech root plates | 2.4 m³ (Pawlik 2016) | {d['mean_m3']:.2f} m³ "
             f"(×{d['ratio_to_sudetes_beech_mean']:.2f}) | {verdict(abs(d['ratio_to_sudetes_beech_mean'] - 1) <= 0.5)} |")
    L.append(f"| V1b deciduous vs. Carpathian root plates | 2.2–3.35 m³ | {d['mean_m3']:.2f} m³ | "
             f"{verdict(d['within_carpathian_mean_range'], partial=d['mean_m3'] <= 1.5 * 3.35)} |")
    L.append(f"| V1b pine vs. Sudetes spruce site means (conifer proxy) | 0.4–4.2 m³ (Pawlik 2016) | "
             f"{pn['mean_m3']:.2f} m³ | {verdict(pn['within_sudetes_spruce_site_means'])} |")
    L.append(f"| V1b pine vs. 40 cm DBH spruce (unconfirmed review value) | 1.0 ± 0.25 m³ | {pn['mean_m3']:.2f} m³ "
             f"(×{pn['ratio_to_spruce_40cm_dbh']:.1f}) | N/A — contradicted by primary spruce data (Z1) |")
    L.append(f"| V2 event volume increases with slope | yes (Norman 1995; Gallaway 2009; Zadrożny 2017) | "
             f"{100 * v2['event_volume_fractional_trend_per_unit_slope']:+.0f} % per unit gradient | "
             f"{v2['verdict_volume_increases_with_slope']} |")
    L.append(f"| V2 tree-throw flux increases with slope | yes | "
             f"{100 * v2['flux_fractional_trend_per_unit_slope']:+.0f} % per unit gradient | "
             f"{v2['verdict_flux_increases_with_slope']} |")
    L.append(f"| V3 tree-throw share of flux (deciduous, S = 0.35) | 11–18 % (Doane 2021, primary) | "
             f"{100 * v3['share_by_species_at_S0.35']['deciduous']:.0f} % | {v3['verdict']} |")
    for k in tt.SPECIES:
        L.append(f"| V4 flux within 10× of Constantine 2012 ({k}) | 2×10⁻⁴ m²/yr | "
                 f"{v4[k]['q_tt_m2_yr']:.1e} m²/yr | {v4[k]['verdict_vs_constantine']} |")
    L.append(f"| V4 tree throw ≈ 1 % of creep (deciduous) | ≤ ~1–3 % (Gallaway 2007/2009) | "
             f"{100 * v4['deciduous']['q_tt_over_q_creep']:.0f} % | "
             f"{v4['deciduous']['verdict_vs_gallaway_two_orders_below_creep']} |")
    L.append(f"| V5 uprooting rate (deciduous) | 1.5 /ha/yr (Razula) | "
             f"{v5['deciduous']['uprootings_per_ha_yr']:.2f} /ha/yr | {v5['verdict_rate_deciduous']} |")
    L.append(f"| V5 uprooting rate (spruce) | 1.5 /ha/yr (Razula, mixed beech–fir–spruce) | "
             f"{v5['spruce']['uprootings_per_ha_yr']:.2f} /ha/yr | "
             f"{verdict(abs(v5['spruce']['uprootings_per_ha_yr'] / 1.5 - 1) <= 0.5)} |")
    L.append(f"| V5 pit-mound cover at Razula rate | > 14 % | "
             f"{100 * v5['deciduous_at_razula_rate']['cover']['10cm']:.0f} % (>10 cm relief) | "
             f"{v5['verdict_cover_at_razula_rate']} |")
    r0 = v6["runs"]["D=0.001"]
    L.append(f"| V6 persistence of a 50 cm mound (to < 5 cm), default D | 200–250 yr (Jura) | "
             f"{fmt(v6['persistence_50cm_to_5cm_yr'], 0)} yr (fresh mound halves in "
             f"{fmt(r0['time_to_half_height_yr'], 0)} yr) | {v6['verdict_default_D']} |")
    L.extend(zadrozny.scorecard_rows(R["Z_zadrozny_2017"], v2))
    L.extend(doane.scorecard_rows(R["D_doane_2021"], v3["share_by_species_at_S0.35"]["deciduous"]))
    L.append("")
    L.extend(baseline_comparison(R))

    L.append("## Details\n")
    L.append("### V0 — internal consistency\n")
    L.append("The tree-throw diffusivity produced by the simulation is compared with the diffusivity computed directly "
             "from the species parameters (Monte-Carlo expectation for an infinite planar slope). "
             "Ratios per seed:\n")
    for k in tt.SPECIES:
        L.append(f"* **{k}**: " + ", ".join(f"{x:.2f}" for x in v0[k]["ratios"]))
    L.append("")

    L.append("### V1 — event volume\n")
    L.append("| species | events | mean | median | 5th pct | 95th pct | within 0.1–5.6 m³ |")
    L.append("|---|---|---|---|---|---|---|")
    for k in tt.SPECIES:
        r = v1[k]
        L.append(f"| {k} | {r['n_events']} | {r['mean_m3']:.2f} | {r['median_m3']:.2f} | {r['p5_m3']:.2f} | "
                 f"{r['p95_m3']:.2f} | {100 * r['fraction_within_published_range']:.0f} % |")
    L.append("")

    L.append("### V2 — slope dependence (deciduous)\n")
    L.append("| gradient | mean event volume (m³) | q_tt sim (m²/yr) | q_tt analytic | D_tt sim (m²/yr) | mean cos(fall) | tree-throw share |")
    L.append("|---|---|---|---|---|---|---|")
    for r in v2["rows"]:
        L.append(f"| {r['slope']:.2f} | {r['mean_event_volume_m3']:.2f} ± {r['se_event_volume_m3']:.2f} | "
                 f"{r['q_tt_m2_yr']:.2e} | {r['q_tt_expected_m2_yr']:.2e} | {r['D_tt_m2_yr']:.2e} | "
                 f"{r['mean_cos_fall']:.2f} | {100 * r['tree_throw_share']:.0f} % |")
    L.append("")

    L.append("### V4 — flux magnitude\n")
    L.append("| species | q_tt (m²/yr) | D_tt (m²/yr) | q_tt / creep | q_tt / Constantine | D_tt / Constantine peak D |")
    L.append("|---|---|---|---|---|---|")
    for k in tt.SPECIES:
        r = v4[k]
        L.append(f"| {k} | {r['q_tt_m2_yr']:.2e} | {r['D_tt_m2_yr']:.2e} | {100 * r['q_tt_over_q_creep']:.0f} % | "
                 f"{r['ratio_to_constantine_flux']:.2f} | {r['ratio_to_constantine_peak_D']:.3f} |")
    L.append("")

    L.append("### V5 — uprooting rate and pit-mound cover\n")
    L.append("Cover is the fraction of the hillslope whose elevation differs from a 3 m Gaussian-smoothed surface "
             "by more than the threshold.\n")
    L.append("| case | uprootings /ha/yr | cover > 5 cm | > 10 cm | > 20 cm |")
    L.append("|---|---|---|---|---|")
    for k, lab in [(k, k) for k in tt.SPECIES] + [("deciduous_at_razula_rate", "deciduous at 1.5 /ha/yr")]:
        r = v5[k]
        c = r["cover"]
        L.append(f"| {lab} | {r['uprootings_per_ha_yr']:.2f} | {100 * c['5cm']:.0f} % | {100 * c['10cm']:.0f} % | "
                 f"{100 * c['20cm']:.0f} % |")
    L.append("")

    L.append("### V6 — mound persistence\n")
    L.append(f"A single mature deciduous root plate (R = {v6['root_plate']['R_m']:.2f} m, "
             f"H = {v6['root_plate']['H_m']:.2f} m) is uprooted downslope and then left to diffuse.\n")
    L.append("| D (m²/yr) | initial mound height (m) | time to half height (yr) | time to 50 cm (yr) | time to < 5 cm (yr) |")
    L.append("|---|---|---|---|---|")
    for name, r in v6["runs"].items():
        L.append(f"| {r['D_m2_yr']:g} | {r['initial_mound_height_m']:.2f} | {fmt(r['time_to_half_height_yr'], 0)} | "
                 f"{fmt(r['time_to_50cm_yr'], 0)} | {fmt(r['time_to_5cm_yr'], 0)} |")
    L.append(f"\nThe fresh mound first collapses quickly (fresh-soil diffusivity), then decays by background "
             f"diffusion. Time for the mound to go from 50 cm to below 5 cm at D = 0.001 m²/yr: "
             f"**{fmt(v6['persistence_50cm_to_5cm_yr'], 0)} yr** (Jura benchmark 200–250 yr). Background D that "
             f"would reproduce the benchmark: **{fmt(v6['D_implied_by_jura_m2_yr'], 2)} m²/yr**. "
             f"Martin et al. (2013) inferred 0.01–0.1 m²/yr for pit-mound degradation in the Canadian Rockies.\n")

    L.extend(zadrozny.detail_lines(R["Z_zadrozny_2017"]))
    L.append("![zadrozny](zadrozny_validation.png)\n")
    L.extend(doane.detail_lines(R["D_doane_2021"]))
    L.append("![doane](doane_validation.png)\n")

    L.append("## Benchmarks used\n")
    L.append("| key | value | source | status |")
    L.append("|---|---|---|---|")
    for k, b in BENCHMARKS.items():
        L.append(f"| `{k}` | {json.dumps(b['value'], ensure_ascii=False)} | [{b['source']}]({b['url']}) | {b['status']} |")
    L.append("")
    (OUT / "VALIDATION_REPORT.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller domains and fewer seeds")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = range(3) if a.quick else range(6)

    print("Running species ensemble (S = 0.35) ...", flush=True)
    jobs = [(k, {}, base_cfg(a.quick, seed=100 + s)) for k in tt.SPECIES for s in seeds]
    res_species = run_jobs(jobs, a.workers)

    print("Running deciduous slope sweep ...", flush=True)
    slopes = [0.05, 0.15, 0.25, 0.35, 0.45, 0.54, 0.7]
    jobs = [("deciduous", {}, base_cfg(a.quick, slope=S, seed=200 + s)) for S in slopes for s in seeds]
    res_slope = run_jobs(jobs, a.workers)

    print("Running deciduous stand at the Razula uprooting rate ...", flush=True)
    dec = tt.SPECIES["deciduous"]
    target = BENCHMARKS["razula_2017_rate_coverage"]["value"]["uprootings_per_ha_yr"]
    dens = target * (dec.lifespan_mean + dec.regeneration_delay) / dec.uproot_fraction
    jobs = [("deciduous", {"density": dens}, base_cfg(a.quick, seed=300 + s)) for s in seeds]
    res_razula = run_jobs(jobs, a.workers)

    print("Running mound persistence experiment ...", flush=True)
    R = {}
    R["V0_internal"] = test_internal_consistency(res_species)
    R["V1_event_volume"] = test_event_volume(res_species)
    R["V2_slope_dependence"] = test_slope_dependence(res_slope)
    R["V3_flux_share"] = test_flux_share(R["V2_slope_dependence"], res_species)
    R["V4_flux_magnitude"] = test_flux_magnitude(res_species)
    R["V5_rate_coverage"] = test_rate_coverage(res_species, res_razula)
    R["V5_rate_coverage"]["razula_density_stems_per_ha"] = dens
    R["V6_mound_persistence"] = test_mound_persistence(a.quick)
    print("Running Zadrożny et al. (2017) comparison ...", flush=True)
    R["Z_zadrozny_2017"] = {
        "Z1_geometry": zadrozny.test_geometry(),
        "Z2_size_depth": zadrozny.test_size_depth_relation(),
        "Z4_degradation": zadrozny.test_degradation(),
    }
    R["_vols"] = {k: np.concatenate([np.array(r["event_volume"]) for r in res_species if r["species"] == k]).tolist()
                  for k in tt.SPECIES}

    print("Running Doane et al. (2021) comparison ...", flush=True)
    R["D_doane_2021"] = {
        "D2_variance": doane.test_variance(res_slope, res_species),
        "D4_couplet": doane.test_couplet(zadrozny.single_event_series),
    }
    make_figures(R)
    doane.plot(R["D_doane_2021"], OUT / "doane_validation.png")
    zadrozny.plot(R["Z_zadrozny_2017"], OUT / "zadrozny_validation.png")
    runtime = time.time() - t0
    write_report(R, runtime, a.quick)
    R.pop("_vols")
    for k in ("spruce", "pine", "deciduous"):
        for f in ("sample_diam", "sample_depth"):
            R["Z_zadrozny_2017"]["Z2_size_depth"][k].pop(f, None)
    meta = {"benchmarks": BENCHMARKS, "seeds": list(seeds), "quick": a.quick, "runtime_s": runtime,
            "config": base_cfg(a.quick)}
    with open(OUT / "validation_results.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": R}, fh, indent=2, default=float)
    print(f"Done in {runtime / 60:.1f} min. Report: {OUT / 'VALIDATION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
