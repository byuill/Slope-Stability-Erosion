"""
zadrozny.py — validation against Zadrożny et al. (2017), iForest 10:783-787
==========================================================================
"Vertical pit-mounds distribution of uprooted Norway spruce (Picea abies L.):
field evidence in the upper mountain belt", doi:10.3832/ifor1959-010.

Values below were read from the full text (Table 1, Fig. 3, Fig. 4 and the
site description).  The site is old-growth Norway spruce in Babia Góra
National Park, Poland: 1135-1198 m a.s.l., slopes 10-40°, stand age 110 yr,
mean DBH 33 cm, stems uprooted by a windstorm in November 2004; 75 pit-mounds
(5 in each of 15 plots) surveyed 1, 3 and 5 years after uprooting.

The paper does not tabulate displaced-soil volumes.  Pit volumes quoted here
are derived from the Table 1 pit dimensions, assuming a pit shape between a
paraboloid and a half-ellipsoid; that assumption is ours, not the authors'.

The model's 'spruce' preset and its fresh pit-mound collapse parameters were
calibrated to these data, so Z1 and Z4 for spruce are calibration checks; the
pine and deciduous rows and Z2-Z3 are independent tests.

Tests
-----
Z1  pit size one year after uprooting (single-event model runs)
Z2  sign of the root-plate size vs. pit-depth relationship (Fig. 3)
Z3  slope dependence of soil dislocation (Fig. 4 RDA) — uses the V2 sweep
Z4  pit and mound degradation from 1 to 5 years (Table 1 chronosequence)
"""

from __future__ import annotations

import math

import numpy as np

import TreeThrow as tt

ZADROZNY = {
    "citation": "Zadrożny P, Halecki W, Gąsiorek M, Nicia P, Lamorski T (2017) iForest 10:783-787, "
                "doi:10.3832/ifor1959-010",
    "url": "https://iforest.sisef.org/contents/?id=ifor1959-010",
    "status": "primary (full text read)",
    "site": {"species": "Picea abies", "stand_age_yr": 110, "dbh_cm": 33, "height_m": 30,
             "slope_deg_min": 10, "slope_deg_max": 40, "elevation_m": [1135, 1198], "n_pit_mounds": 75,
             "soils": "Skeletic Podzols, Dystric Skeletic Cambisols, Dystric Skeletic Regosols on sandstone"},
    # Table 1: mean ± standard error, by years since uprooting
    "table1": {
        "age_yr": [1, 3, 5],
        "pit_length_m": [(2.74, 0.29), (2.26, 0.05), (2.18, 0.09)],
        "pit_width_m": [(4.23, 0.24), (3.68, 0.12), (3.28, 0.28)],
        "pit_depth_m": [(0.95, 0.09), (0.65, 0.02), (0.52, 0.12)],
        "root_plate_height_m": [(2.35, 0.15), (1.98, 0.34), (1.82, 0.45)],
        "mineral_mound_height_m": [(0.71, 0.22), (0.42, 0.11), (0.44, 0.19)],
        "organic_mound_height_m": [(0.33, 0.09), (0.32, 0.07), (0.38, 0.08)],
    },
    # Fig. 3: pit depth (m) = -0.3758 * root plate height (m) + 2.049, R^2 = 0.64
    "fig3": {"slope": -0.3758, "intercept": 2.049, "r2": 0.64, "x_range_m": [1.2, 2.4]},
    "fig4": "RDA: slope steepness has the largest influence on soil dislocation; pit depth and root-plate "
            "height are negatively associated with slope steepness; volume of dislocated soil loads with "
            "slope steepness on RDA axis 1.",
}
SITE_SLOPE_DEG = 25.0   # mid-point of the 10-40° range
EDGE_THRESHOLD_M = 0.10  # pit edge: >= 10 cm below the pre-uproot surface (~1/10 of the 1-yr pit depth)


def _verdict(ok, partial=False):
    return "PASS" if ok else ("PARTIAL" if partial else "FAIL")


def pit_volume_range(L, W, D):
    """Pit volume (m^3) for an elliptical planform L x W and depth D: paraboloid .. half-ellipsoid."""
    return math.pi / 8 * L * W * D, math.pi / 6 * L * W * D


def obs(name, k):
    return ZADROZNY["table1"][name][k]


# ----------------------------------------------------------------------------
# Single-event model experiment
# ----------------------------------------------------------------------------


def single_event_series(key, years=10, slope_deg=SITE_SLOPE_DEG, dx=0.25, R=None, H=None, **cfg_kw):
    """Uproot one mature tree (downslope fall) and record pit/mound geometry each year.

    Returns a dict of per-year lists: time, pit depth, mound height, pit volume,
    pit length (along the fall line) and pit width (across it), measured relative
    to an identical hillslope without the event.  Pit length and width are the
    extent of ground at least EDGE_THRESHOLD_M below the pre-uproot surface (and,
    for comparison, 5 cm below it).
    """
    sp = tt.SPECIES[key]
    R = 0.5 * sp.root_plate_diameter if R is None else R
    H = sp.root_plate_depth if H is None else H
    cfg = dict(length=16.0, width=12.0, slope=math.tan(math.radians(slope_deg)), dx=dx, years=years,
               diffusivity=1e-3, soil_model=False, tree_throw=False, seed=0)
    cfg.update(cfg_kw)
    ref = tt.TreeThrowModel(sp, tt.HillslopeConfig(**cfg))
    m = tt.TreeThrowModel(sp, tt.HillslopeConfig(**cfg))
    m.uproot(6.0, 6.0, R, H, 0.0)
    out = {k: [] for k in ("t", "depth", "mound", "pit_volume", "mound_volume", "pit_length", "pit_width",
                           "pit_length_5cm", "pit_width_5cm")}

    def record(t):
        rel = m.z - ref.z
        for suffix, thr in (("", EDGE_THRESHOLD_M), ("_5cm", 0.05)):
            rows, cols = np.nonzero(rel < -thr)
            out["pit_length" + suffix].append(float((rows.max() - rows.min() + 1) * dx) if rows.size else 0.0)
            out["pit_width" + suffix].append(float((cols.max() - cols.min() + 1) * dx) if cols.size else 0.0)
        out["t"].append(t)
        out["depth"].append(float(-rel.min()))
        out["mound"].append(float(rel.max()))
        out["pit_volume"].append(float(-rel[rel < 0].sum() * m.cell_area))
        out["mound_volume"].append(float(rel[rel > 0].sum() * m.cell_area))

    record(0)
    for t in range(1, years + 1):
        m.step()
        ref.step()
        record(t)
    return out


# ----------------------------------------------------------------------------
# Z1 / Z2 — geometry
# ----------------------------------------------------------------------------


def test_geometry():
    L, W, D = obs("pit_length_m", 0), obs("pit_width_m", 0), obs("pit_depth_m", 0)
    vmin, vmax = pit_volume_range(L[0], W[0], D[0])
    out = {"observed_1yr": {"pit_length_m": L, "pit_width_m": W, "pit_depth_m": D,
                            "width_to_length": W[0] / L[0], "implied_pit_volume_m3": [vmin, vmax]}}
    for key in ("spruce", "pine", "deciduous"):
        s = single_event_series(key, years=1)
        r = {"fresh_depth_m": s["depth"][0], "fresh_pit_volume_m3": s["pit_volume"][0],
             "depth_1yr_m": s["depth"][1], "pit_volume_1yr_m3": s["pit_volume"][1],
             "length_1yr_m": s["pit_length"][1], "width_1yr_m": s["pit_width"][1],
             "length_1yr_5cm_m": s["pit_length_5cm"][1], "width_1yr_5cm_m": s["pit_width_5cm"][1]}
        r["width_to_length_1yr"] = r["width_1yr_m"] / r["length_1yr_m"]
        r["verdict_depth"] = _verdict(abs(r["depth_1yr_m"] - D[0]) <= 2 * D[1])
        r["verdict_volume"] = _verdict(vmin <= r["pit_volume_1yr_m3"] <= vmax,
                                       partial=0.67 * vmin <= r["pit_volume_1yr_m3"] <= 1.5 * vmax)
        r["verdict_length"] = _verdict(abs(r["length_1yr_m"] - L[0]) <= 2 * L[1])
        r["verdict_width"] = _verdict(abs(r["width_1yr_m"] - W[0]) <= 2 * W[1])
        out[key] = r
    return out


def test_size_depth_relation(n=20000, seed=0):
    """Correlation between root-plate diameter and pit depth for trees at death."""
    rng = np.random.default_rng(seed)
    out = {}
    f3 = ZADROZNY["fig3"]
    r_obs = -math.sqrt(f3["r2"])
    for key in ("spruce", "pine", "deciduous"):
        sp = tt.SPECIES[key]
        life = tt.draw_lifespans(rng, sp, n)
        r, h = tt.draw_mature_sizes(rng, sp, n)
        r, h = tt.plate_at_age(sp, r, h, life)
        diam, depth = 2 * r, h
        r_mod = float(np.corrcoef(diam, depth)[0, 1])
        out[key] = {"r_model": r_mod, "sample_diam": diam[:600].tolist(), "sample_depth": depth[:600].tolist(),
                    "verdict": _verdict(np.sign(r_mod) == np.sign(r_obs), partial=False)}
    out["r_observed"] = r_obs
    return out


# ----------------------------------------------------------------------------
# Z4 — degradation through time
# ----------------------------------------------------------------------------


def test_degradation():
    s = single_event_series("spruce", years=10)
    out = {"series": s, "comparison": {}}
    om = ZADROZNY["table1"]["organic_mound_height_m"]
    for name, key, label in (("pit_depth_m", "depth", "pit depth"),
                             ("pit_length_m", "pit_length", "pit length"),
                             ("pit_width_m", "pit_width", "pit width"),
                             ("mineral_mound_height_m", "mound", "mineral mound height"),
                             ("mineral+organic", "mound", "mineral + organic mound height")):
        rows = []
        for k, age in enumerate((1, 3, 5)):
            if name == "mineral+organic":
                mm = ZADROZNY["table1"]["mineral_mound_height_m"][k]
                m_, se = mm[0] + om[k][0], math.hypot(mm[1], om[k][1])
            else:
                m_, se = ZADROZNY["table1"][name][k]
            mod = s[key][age]
            rows.append({"age_yr": age, "observed": m_, "se": se, "model": mod,
                         "z": (mod - m_) / se if se > 0 else float("nan")})
        out["comparison"][label] = rows
    depth_ok = all(abs(r["z"]) <= 2 for r in out["comparison"]["pit depth"])
    out["verdict_pit_depth"] = _verdict(depth_ok)
    mm_ok = all(abs(r["z"]) <= 2 for r in out["comparison"]["mineral mound height"])
    mo_ok = all(abs(r["z"]) <= 2 for r in out["comparison"]["mineral + organic mound height"])
    out["verdict_mound"] = _verdict(mm_ok and mo_ok, partial=mm_ok or mo_ok)
    # Same experiment without the fresh-collapse process, for contrast.
    s0 = single_event_series("spruce", years=5, fresh_diffusivity=0.0)
    out["no_collapse_depth_ratio_5_to_1"] = s0["depth"][5] / s0["depth"][1]
    out["observed_depth_ratio_5_to_1"] = obs("pit_depth_m", 2)[0] / obs("pit_depth_m", 0)[0]
    out["model_depth_ratio_5_to_1"] = s["depth"][5] / s["depth"][1]
    return out


# ----------------------------------------------------------------------------
# Figure and report
# ----------------------------------------------------------------------------


def plot(res, path):
    plt = tt._import_pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Validation against Zadrożny et al. (2017) — uprooted Norway spruce, Babia Góra", fontsize=12)

    # Z1 geometry at 1 yr
    a = ax[0]
    g = res["Z1_geometry"]
    labels = ["pit length\n(along fall)", "pit width\n(across fall)", "pit depth"]
    o = [obs("pit_length_m", 0), obs("pit_width_m", 0), obs("pit_depth_m", 0)]
    x = np.arange(3)
    a.bar(x - 0.3, [v[0] for v in o], 0.2, yerr=[v[1] for v in o], color="gray", capsize=3,
          label="spruce, 1 yr (Zadrożny)")
    for i, k in enumerate(("spruce", "pine", "deciduous")):
        r = g[k]
        a.bar(x - 0.1 + 0.2 * i, [r["length_1yr_m"], r["width_1yr_m"], r["depth_1yr_m"]], 0.2,
              color=tt.SPECIES_COLORS.get(k, "tab:gray"), label=f"model {k}, 1 yr")
    a.set_xticks(x, labels)
    a.set(title="Z1  Pit size one year after uprooting", ylabel="m")
    a.legend(fontsize=8)

    # Z2 size-depth
    a = ax[1]
    z2 = res["Z2_size_depth"]
    for k in ("spruce", "pine", "deciduous"):
        a.scatter(z2[k]["sample_diam"], z2[k]["sample_depth"], s=4, alpha=0.3, color=tt.SPECIES_COLORS.get(k, "tab:gray"),
                  label=f"model {k} (r = {z2[k]['r_model']:+.2f})")
    f3 = ZADROZNY["fig3"]
    xx = np.linspace(*f3["x_range_m"], 10)
    a.plot(xx, f3["slope"] * xx + f3["intercept"], "k-", lw=2,
           label=f"Zadrożny Fig. 3 (r = {z2['r_observed']:.2f})")
    a.set(title="Z2  Root-plate size vs. depth at death", xlabel="root-plate diameter (m)",
          ylabel="root-plate / pit depth (m)", xlim=(0, 5.5), ylim=(0, 2.2))
    a.legend(fontsize=7)

    # Z4 time series
    a = ax[2]
    z4 = res["Z4_degradation"]
    s = z4["series"]
    a.plot(s["t"], s["depth"], "-", color="tab:blue", label="model pit depth")
    a.plot(s["t"], s["mound"], "-", color="tab:brown", label="model mound height")
    for lab, c, mk in (("pit depth", "tab:blue", "o"), ("mineral mound height", "tab:brown", "s"),
                       ("mineral + organic mound height", "tab:brown", "^")):
        rows = z4["comparison"][lab]
        a.errorbar([r["age_yr"] for r in rows], [r["observed"] for r in rows], yerr=[r["se"] for r in rows],
                   fmt=mk, color=c, mfc="white" if "organic" in lab else c, capsize=3, label=f"observed {lab}")
    a.set(title="Z4  Spruce pit and mound through time", xlabel="years since uprooting", ylabel="m",
          xlim=(-0.3, 10.3))
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def scorecard_rows(res, v2):
    g, z2, z4 = res["Z1_geometry"], res["Z2_size_depth"], res["Z4_degradation"]
    o = g["observed_1yr"]
    vmin, vmax = o["implied_pit_volume_m3"]
    sp = g["spruce"]
    rows = [
        f"| Z1 spruce pit depth at 1 yr (calibrated) | 0.95 ± 0.09 m | {sp['depth_1yr_m']:.2f} m | {sp['verdict_depth']} |",
        f"| Z1 spruce pit volume at 1 yr (calibrated) | {vmin:.1f}–{vmax:.1f} m³ | {sp['pit_volume_1yr_m3']:.2f} m³ | "
        f"{sp['verdict_volume']} |",
        f"| Z1 spruce pit length / width at 1 yr | 2.74 / 4.23 m | {sp['length_1yr_m']:.2f} / {sp['width_1yr_m']:.2f} m | "
        f"{'PASS' if sp['verdict_length'] == 'PASS' and sp['verdict_width'] == 'PASS' else 'PARTIAL' if 'PASS' in (sp['verdict_length'], sp['verdict_width']) else 'FAIL'} |",
    ]
    for k in ("spruce", "pine", "deciduous"):
        rows.append(f"| Z2 sign of size–depth correlation ({k}) | r = {z2['r_observed']:.2f} (Fig. 3) | "
                    f"r = {z2[k]['r_model']:+.2f} | {z2[k]['verdict']} |")
    rows.append(f"| Z3 soil dislocation increases with slope | yes (RDA, Fig. 4) | "
                f"{100 * v2['event_volume_fractional_trend_per_unit_slope']:+.0f} % per unit gradient | "
                f"{v2['verdict_volume_increases_with_slope']} |")
    d = z4["comparison"]["pit depth"]
    rows.append(f"| Z4 spruce pit depth at 3 / 5 yr | {d[1]['observed']:.2f} / {d[2]['observed']:.2f} m | "
                f"{d[1]['model']:.2f} / {d[2]['model']:.2f} m | {z4['verdict_pit_depth']} |")
    mm = z4["comparison"]["mineral mound height"]
    mo = z4["comparison"]["mineral + organic mound height"]
    rows.append(f"| Z4 spruce mound height at 1 / 3 / 5 yr | mineral {mm[0]['observed']:.2f} / {mm[1]['observed']:.2f} / "
                f"{mm[2]['observed']:.2f} m; +organic {mo[0]['observed']:.2f} / {mo[1]['observed']:.2f} / "
                f"{mo[2]['observed']:.2f} m | {mm[0]['model']:.2f} / {mm[1]['model']:.2f} / {mm[2]['model']:.2f} m | "
                f"{z4['verdict_mound']} |")
    return rows


def detail_lines(res):
    g, z2, z4 = res["Z1_geometry"], res["Z2_size_depth"], res["Z4_degradation"]
    o = g["observed_1yr"]
    L = ["### Z — Zadrożny et al. (2017), uprooted Norway spruce (primary data)\n",
         f"Source: [{ZADROZNY['citation']}]({ZADROZNY['url']}); the full text was read. "
         "Old-growth *Picea abies*, 110 yr, mean DBH 33 cm, slopes 10–40°, 75 pit-mounds surveyed 1, 3 and 5 yr "
         "after a November 2004 windstorm. The model's spruce preset and its fresh pit-mound collapse "
         "parameters were calibrated to these data, so the spruce rows of Z1 and Z4 are calibration checks; the "
         "pine and deciduous rows and Z2–Z3 are independent.\n",
         "**Table 1 (mean ± SE)**\n",
         "| parameter | 1 yr | 3 yr | 5 yr |", "|---|---|---|---|"]
    for k, v in ZADROZNY["table1"].items():
        if k == "age_yr":
            continue
        L.append(f"| {k.replace('_', ' ')} | " + " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in v) + " |")
    L.append("")
    L.append("**Z1 — pit one year after uprooting** (single mature tree falling downslope on a 25° slope). "
             f"Pit length and width are the extent of ground at least {EDGE_THRESHOLD_M * 100:.0f} cm below the "
             "pre-uproot surface; values for a 5 cm edge are given in brackets. "
             f"Observed pit volume implied by Table 1: {o['implied_pit_volume_m3'][0]:.1f}–"
             f"{o['implied_pit_volume_m3'][1]:.1f} m³ (our paraboloid-to-half-ellipsoid conversion).\n")
    L.append("| model | fresh open pit depth (m) | depth 1 yr (m) | length 1 yr (m) | width 1 yr (m) | width/length | pit volume 1 yr (m³) |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| observed spruce | — | {o['pit_depth_m'][0]:.2f} | {o['pit_length_m'][0]:.2f} | {o['pit_width_m'][0]:.2f} | "
             f"{o['width_to_length']:.2f} | {o['implied_pit_volume_m3'][0]:.1f}–{o['implied_pit_volume_m3'][1]:.1f} |")
    for k in ("spruce", "pine", "deciduous"):
        r = g[k]
        L.append(f"| {k} | {r['fresh_depth_m']:.2f} | {r['depth_1yr_m']:.2f} | {r['length_1yr_m']:.2f} "
                 f"({r['length_1yr_5cm_m']:.2f}) | {r['width_1yr_m']:.2f} ({r['width_1yr_5cm_m']:.2f}) | "
                 f"{r['width_to_length_1yr']:.2f} | {r['pit_volume_1yr_m3']:.2f} |")
    L.append("")
    L.append(f"**Z2 — size–depth relation.** Zadrożny Fig. 3: pit depth = −0.3758 × root-plate height + 2.049 "
             f"(R² = 0.64, r = {z2['r_observed']:.2f}). Model correlation of plate diameter and depth at death: "
             + ", ".join(f"{k} {z2[k]['r_model']:+.2f}" for k in ("spruce", "pine", "deciduous")) + ".\n")
    L.append("**Z4 — degradation.** Model spruce pit and mound (fresh collapse D = "
             f"{tt.HillslopeConfig().fresh_diffusivity} m²/yr decaying with e-folding time "
             f"{tt.HillslopeConfig().fresh_decay_time} yr, plus background D = 0.001 m²/yr) against the survey. "
             "The survey is a chronosequence (different trees in each age class). The model mound contains "
             "all root-plate soil, so it is compared with both the mineral mound and mineral + organic mound.\n")
    L.append("| quantity | age (yr) | observed | model | (model − obs) / SE |")
    L.append("|---|---|---|---|---|")
    for lab, rows in z4["comparison"].items():
        for r in rows:
            L.append(f"| {lab} | {r['age_yr']} | {r['observed']:.2f} ± {r['se']:.2f} | {r['model']:.2f} | {r['z']:+.1f} |")
    ln, wd = z4["comparison"]["pit length"], z4["comparison"]["pit width"]
    L.append(f"\nPit size through time: observed length {ln[0]['observed']:.2f} → {ln[2]['observed']:.2f} m and width "
             f"{wd[0]['observed']:.2f} → {wd[2]['observed']:.2f} m from 1 to 5 yr (pits narrow); model "
             f"{ln[0]['model']:.2f} → {ln[2]['model']:.2f} m and {wd[0]['model']:.2f} → {wd[2]['model']:.2f} m. "
             "Diffusive collapse lowers the rim and widens the depression, whereas the surveyed pits narrow (rim "
             "slumping and litter infill at the edges, or partly the chronosequence: the 1-yr cohort also has taller "
             "root plates). This is a structural limitation of the diffusive collapse.\n")
    L.append(f"\nPit depth at 5 yr / 1 yr: observed {z4['observed_depth_ratio_5_to_1']:.2f}, model "
             f"{z4['model_depth_ratio_5_to_1']:.2f}, model without fresh collapse "
             f"{z4['no_collapse_depth_ratio_5_to_1']:.2f}.\n")
    return L
