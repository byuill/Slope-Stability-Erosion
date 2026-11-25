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

Tests
-----
Z1  pit size (1-yr pits) vs. the model's conifer and hardwood root plates
Z2  sign of the root-plate size vs. pit-depth relationship (Fig. 3)
Z3  slope dependence of soil dislocation (Fig. 4 RDA) — uses the V2 sweep
Z4  short-term pit and mound degradation (Table 1 chronosequence) vs. diffusion
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


def _verdict(ok, partial=False):
    return "PASS" if ok else ("PARTIAL" if partial else "FAIL")


def pit_volume_range(L, W, D):
    """Pit volume (m^3) for an elliptical planform L x W and depth D: paraboloid .. half-ellipsoid."""
    return math.pi / 8 * L * W * D, math.pi / 6 * L * W * D


# ----------------------------------------------------------------------------
# Z1 / Z2 — geometry
# ----------------------------------------------------------------------------


def test_geometry(event_volume_results):
    t1 = ZADROZNY["table1"]
    L, W, D = t1["pit_length_m"][0][0], t1["pit_width_m"][0][0], t1["pit_depth_m"][0][0]
    vmin, vmax = pit_volume_range(L, W, D)
    d_eq = math.sqrt(L * W)
    out = {"observed_1yr": {"pit_length_m": L, "pit_width_m": W, "pit_depth_m": D,
                            "equivalent_diameter_m": d_eq, "width_to_length": W / L,
                            "implied_pit_volume_m3": [vmin, vmax]}}
    for key in ("pine", "deciduous"):
        sp = tt.SPECIES[key]
        mean_v = event_volume_results[key]["mean_m3"]
        out[key] = {
            "mature_diameter_m": sp.root_plate_diameter,
            "mature_depth_m": sp.root_plate_depth,
            "diameter_ratio": sp.root_plate_diameter / d_eq,
            "depth_ratio": sp.root_plate_depth / D,
            "mean_event_volume_m3": mean_v,
            "volume_ratio_to_midpoint": mean_v / (0.5 * (vmin + vmax)),
            "verdict_volume": _verdict(vmin <= mean_v <= vmax, partial=0.5 * vmin <= mean_v <= 1.5 * vmax),
            "verdict_depth": _verdict(abs(sp.root_plate_depth / D - 1) <= 0.25),
            "verdict_diameter": _verdict(abs(sp.root_plate_diameter / d_eq - 1) <= 0.25),
        }
    return out


def test_size_depth_relation(n=20000, seed=0):
    """Model correlation between root-plate diameter and pit depth for dying trees."""
    rng = np.random.default_rng(seed)
    out = {}
    f3 = ZADROZNY["fig3"]
    r_obs = -math.sqrt(f3["r2"])
    for key in ("pine", "deciduous"):
        sp = tt.SPECIES[key]
        life = tt.draw_lifespans(rng, sp, n)
        r, h = tt.draw_mature_sizes(rng, sp, n)
        g = tt.growth_factor(life, sp.maturity_age)
        diam, depth = 2 * r * g, h * g
        r_mod = float(np.corrcoef(diam, depth)[0, 1])
        out[key] = {"r_model": r_mod, "sample_diam": diam[:600].tolist(), "sample_depth": depth[:600].tolist(),
                    "verdict": _verdict(np.sign(r_mod) == np.sign(r_obs))}
    out["r_observed"] = r_obs
    return out


# ----------------------------------------------------------------------------
# Z4 — short-term degradation
# ----------------------------------------------------------------------------


def test_degradation(slope_deg=25.0, D0=1.0, tau_max=12.0, dx=0.1):
    """Decay of an isolated pit and mound under linear diffusion.

    For linear diffusion the relative decay depends only on tau = D * t, so one
    run with D0 = 1 m^2/yr gives the curve for every D.  The fresh model pit is
    matched to the 1-yr-old observed pit (equivalent radius, depth 0.95 m) and the
    3- and 5-yr observations are compared at t = 2 and 4 yr.
    """
    t1 = ZADROZNY["table1"]
    L, W = t1["pit_length_m"][0][0], t1["pit_width_m"][0][0]
    R, H = 0.5 * math.sqrt(L * W), t1["pit_depth_m"][0][0]
    S = math.tan(math.radians(slope_deg))
    cfg = dict(length=14.0, width=10.0, slope=S, dx=dx, years=1.0, diffusivity=D0,
               soil_model=False, tree_throw=False, seed=0)
    sp = tt.SPECIES["pine"]
    ref = tt.TreeThrowModel(sp, tt.HillslopeConfig(**cfg))
    m = tt.TreeThrowModel(sp, tt.HillslopeConfig(**cfg))
    m.uproot(6.0, 5.0, R, H, 0.0)
    rel = m.z - ref.z
    pit0, mound0 = float(-rel.min()), float(rel.max())
    dt = 0.2 * dx * dx / D0
    taus, pit, mound = [0.0], [1.0], [1.0]
    t = 0.0
    while t < tau_max / D0:
        m._diffusion_step(dt)
        ref._diffusion_step(dt)
        t += dt
        rel = m.z - ref.z
        taus.append(t * D0)
        pit.append(float(-rel.min()) / pit0)
        mound.append(float(rel.max()) / mound0)
    taus, pit, mound = map(np.array, (taus, pit, mound))

    def obs_ratio(name):
        (m0, s0), (m1, s1), (m2, s2) = t1[name]
        r1, r2 = m1 / m0, m2 / m0
        e1 = r1 * math.hypot(s1 / m1, s0 / m0)
        e2 = r2 * math.hypot(s2 / m2, s0 / m0)
        return np.array([r1, r2]), np.array([e1, e2])

    obs = {"pit": obs_ratio("pit_depth_m"), "mound": obs_ratio("mineral_mound_height_m")}
    curves = {"pit": pit, "mound": mound}
    t_obs = np.array([2.0, 4.0])
    Ds = np.logspace(-3, math.log10(tau_max / 4), 400)

    def pred(name, D):
        return np.interp(D * t_obs, taus, curves[name])

    fits = {}
    for name in ("pit", "mound", "both"):
        names = ("pit", "mound") if name == "both" else (name,)
        cost = np.array([sum(np.sum(((pred(n, D) - obs[n][0]) / obs[n][1]) ** 2) for n in names) for D in Ds])
        k = int(np.argmin(cost))
        fits[name] = {"D_best_m2_yr": float(Ds[k]), "chi2": float(cost[k]),
                      "pred_at_best": {n: pred(n, Ds[k]).tolist() for n in names}}
    default_D = 1.0e-3
    out = {
        "pit_radius_m": R, "pit_depth_m": H, "slope_deg": slope_deg,
        "model_initial": {"pit_depth_m": pit0, "mound_height_m": mound0},
        "observed_ratios": {k: {"t_yr": t_obs.tolist(), "ratio": v[0].tolist(), "err": v[1].tolist()}
                            for k, v in obs.items()},
        "fits": fits,
        "pred_default_D": {n: pred(n, default_D).tolist() for n in ("pit", "mound")},
        "curve": {"tau": taus[::max(len(taus) // 400, 1)].tolist(),
                  "pit": pit[::max(len(taus) // 400, 1)].tolist(),
                  "mound": mound[::max(len(taus) // 400, 1)].tolist()},
    }
    d_pd = pred("pit", default_D)
    out["verdict_default_D"] = _verdict(np.all(np.abs(d_pd - obs["pit"][0]) <= 2 * obs["pit"][1]))
    # Can one D reproduce both the 2-yr and 4-yr pit ratios within 2 SE?
    p = fits["pit"]["pred_at_best"]["pit"]
    out["verdict_diffusive_shape"] = _verdict(np.all(np.abs(np.array(p) - obs["pit"][0]) <= 2 * obs["pit"][1]))
    return out


# ----------------------------------------------------------------------------
# Figure and report
# ----------------------------------------------------------------------------


def plot(res, path):
    plt = tt._import_pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Validation against Zadrożny et al. (2017) — uprooted Norway spruce, Babia Góra", fontsize=12)

    # Z1 geometry
    a = ax[0]
    g = res["Z1_geometry"]
    o = g["observed_1yr"]
    labels = ["pit length\n(along fall)", "pit width\n(across fall)", "pit depth"]
    obs = [o["pit_length_m"], o["pit_width_m"], o["pit_depth_m"]]
    se = [ZADROZNY["table1"][k][0][1] for k in ("pit_length_m", "pit_width_m", "pit_depth_m")]
    x = np.arange(3)
    a.bar(x - 0.27, obs, 0.27, yerr=se, color="gray", label="spruce, 1 yr (Zadrożny)", capsize=3)
    for i, (k, c) in enumerate((("pine", "tab:green"), ("deciduous", "tab:orange"))):
        sp = tt.SPECIES[k]
        a.bar(x + 0.27 * i, [sp.root_plate_diameter, sp.root_plate_diameter, sp.root_plate_depth], 0.27,
              color=c, label=f"model {k} (mature)")
    a.set_xticks(x, labels)
    a.set(title="Z1  Pit size", ylabel="m")
    a.legend(fontsize=8)

    # Z2 size-depth
    a = ax[1]
    z2 = res["Z2_size_depth"]
    for k, c in (("pine", "tab:green"), ("deciduous", "tab:orange")):
        a.scatter(z2[k]["sample_diam"], z2[k]["sample_depth"], s=4, alpha=0.3, color=c,
                  label=f"model {k} (r = {z2[k]['r_model']:+.2f})")
    f3 = ZADROZNY["fig3"]
    xx = np.linspace(*f3["x_range_m"], 10)
    a.plot(xx, f3["slope"] * xx + f3["intercept"], "k-", lw=2,
           label=f"Zadrożny Fig. 3: depth = {f3['slope']}·PR + {f3['intercept']} (R² = {f3['r2']})")
    a.set(title="Z2  Root-plate size vs. pit depth", xlabel="root-plate diameter / height (m)",
          ylabel="pit depth (m)", xlim=(0, 5), ylim=(0, 2))
    a.legend(fontsize=7)

    # Z4 degradation
    a = ax[2]
    z4 = res["Z4_degradation"]
    tau = np.array(z4["curve"]["tau"])
    t = np.linspace(0, 6, 300)
    for name, c in (("pit", "tab:blue"), ("mound", "tab:brown")):
        ob = z4["observed_ratios"][name]
        a.errorbar([0] + ob["t_yr"], [1] + ob["ratio"], yerr=[0] + ob["err"], fmt="o", color=c, capsize=3,
                   label=f"observed {name} ({'depth' if name == 'pit' else 'mineral mound height'})")
        Db = z4["fits"][name]["D_best_m2_yr"]
        a.plot(t, np.interp(Db * t, tau, z4["curve"][name]), color=c,
               label=f"diffusion, best D = {Db:.2g} m²/yr")
        a.plot(t, np.interp(1e-3 * t, tau, z4["curve"][name]), color=c, ls=":",
               label="diffusion, default D = 0.001" if name == "pit" else None)
    a.set(title="Z4  Pit and mound degradation (relative to 1-yr survey)",
          xlabel="years after the 1-yr survey", ylabel="height or depth / initial", ylim=(0, 1.15))
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def scorecard_rows(res, v2):
    g, z2, z4 = res["Z1_geometry"], res["Z2_size_depth"], res["Z4_degradation"]
    o = g["observed_1yr"]
    vmin, vmax = o["implied_pit_volume_m3"]
    rows = []
    for k in ("pine", "deciduous"):
        r = g[k]
        rows.append(f"| Z1 pit volume, model {k} vs. 1-yr spruce pits | {vmin:.1f}–{vmax:.1f} m³ (from Table 1) | "
                    f"{r['mean_event_volume_m3']:.2f} m³ | {r['verdict_volume']} |")
    rows.append(f"| Z1 pit depth, model pine vs. spruce | 0.95 ± 0.09 m | {g['pine']['mature_depth_m']:.2f} m | "
                f"{g['pine']['verdict_depth']} |")
    rows.append(f"| Z1 pit diameter, model pine vs. spruce (√(L·W)) | {o['equivalent_diameter_m']:.2f} m | "
                f"{g['pine']['mature_diameter_m']:.2f} m | {g['pine']['verdict_diameter']} |")
    rows.append(f"| Z2 sign of size–depth correlation | r = {z2['r_observed']:.2f} (Fig. 3) | "
                f"r = {z2['pine']['r_model']:+.2f} (pine) | {z2['pine']['verdict']} |")
    rows.append(f"| Z3 soil dislocation increases with slope | yes (RDA, Fig. 4) | "
                f"{100 * v2['event_volume_fractional_trend_per_unit_slope']:+.0f} % per unit gradient | "
                f"{v2['verdict_volume_increases_with_slope']} |")
    pd = z4["pred_default_D"]["pit"]
    ob = z4["observed_ratios"]["pit"]["ratio"]
    rows.append(f"| Z4 pit depth after 2 / 4 yr, default D | {ob[0]:.2f} / {ob[1]:.2f} of initial | "
                f"{pd[0]:.3f} / {pd[1]:.3f} | {z4['verdict_default_D']} |")
    fp = z4["fits"]["pit"]
    rows.append(f"| Z4 one diffusivity fits both 2- and 4-yr pit data | within 2 SE | best D = "
                f"{fp['D_best_m2_yr']:.2g} m²/yr → {fp['pred_at_best']['pit'][0]:.2f} / "
                f"{fp['pred_at_best']['pit'][1]:.2f} | {z4['verdict_diffusive_shape']} |")
    return rows


def detail_lines(res):
    g, z2, z4 = res["Z1_geometry"], res["Z2_size_depth"], res["Z4_degradation"]
    o = g["observed_1yr"]
    L = ["### Z — Zadrożny et al. (2017), uprooted Norway spruce (primary data)\n",
         f"Source: [{ZADROZNY['citation']}]({ZADROZNY['url']}); the full text was read. "
         "Old-growth *Picea abies*, 110 yr, mean DBH 33 cm, slopes 10–40°, 75 pit-mounds surveyed 1, 3 and 5 yr "
         "after a November 2004 windstorm.\n",
         "**Table 1 (mean ± SE)**\n",
         "| parameter | 1 yr | 3 yr | 5 yr |", "|---|---|---|---|"]
    for k, v in ZADROZNY["table1"].items():
        if k == "age_yr":
            continue
        L.append(f"| {k.replace('_', ' ')} | " + " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in v) + " |")
    L.append("")
    L.append(f"**Z1 geometry.** The 1-yr pits are {o['pit_length_m']:.2f} m long × {o['pit_width_m']:.2f} m wide × "
             f"{o['pit_depth_m']:.2f} m deep (width/length = {o['width_to_length']:.2f}; equivalent diameter "
             f"{o['equivalent_diameter_m']:.2f} m). Pit volume implied by these dimensions is "
             f"{o['implied_pit_volume_m3'][0]:.1f}–{o['implied_pit_volume_m3'][1]:.1f} m³ "
             "(paraboloid to half-ellipsoid; this conversion is ours, the paper gives no volumes). "
             "The model's pits are circular, whereas the observed pits are about 1.5× wider across the fall line "
             "than along it.\n")
    L.append("| model | mature diameter (m) | mature depth (m) | mean event volume (m³) | volume / observed midpoint |")
    L.append("|---|---|---|---|---|")
    for k in ("pine", "deciduous"):
        r = g[k]
        L.append(f"| {k} | {r['mature_diameter_m']:.2f} | {r['mature_depth_m']:.2f} | {r['mean_event_volume_m3']:.2f} | "
                 f"{r['volume_ratio_to_midpoint']:.2f} |")
    L.append("")
    L.append(f"**Z2 size–depth relation.** Zadrożny Fig. 3: pit depth = −0.3758 × root-plate height + 2.049 "
             f"(R² = 0.64, r = {z2['r_observed']:.2f}); taller root plates have shallower pits, which the authors "
             "attribute to backward rotation of the uprooting tree. The model draws diameter and depth with a "
             f"positive correlation (r = {z2['pine']['r_model']:+.2f} pine, {z2['deciduous']['r_model']:+.2f} "
             "deciduous), so it has the opposite sign.\n")
    L.append("**Z4 short-term degradation.** A fresh model pit matched to the 1-yr pit (radius "
             f"{z4['pit_radius_m']:.2f} m, depth {z4['pit_depth_m']:.2f} m, slope {z4['slope_deg']:.0f}°) is left to "
             "diffuse; the 3- and 5-yr survey means are compared at t = 2 and 4 yr. The survey is a "
             "chronosequence (different trees in each age class), and the 3- vs 5-yr differences are mostly not "
             "significant (Table 1 letters).\n")
    L.append("| quantity | observed ratio 2 yr | 4 yr | best-fit D (m²/yr) | model at best D (2 / 4 yr) | model at D = 0.001 |")
    L.append("|---|---|---|---|---|---|")
    for n, lab in (("pit", "pit depth"), ("mound", "mineral mound height")):
        ob = z4["observed_ratios"][n]
        f = z4["fits"][n]
        pb = f["pred_at_best"][n]
        pdf = z4["pred_default_D"][n]
        L.append(f"| {lab} | {ob['ratio'][0]:.2f} ± {ob['err'][0]:.2f} | {ob['ratio'][1]:.2f} ± {ob['err'][1]:.2f} | "
                 f"{f['D_best_m2_yr']:.2g} | {pb[0]:.2f} / {pb[1]:.2f} | {pdf[0]:.3f} / {pdf[1]:.3f} |")
    Db = z4["fits"]["both"]["D_best_m2_yr"]
    L.append(f"\nJoint fit (pit and mound): D = {Db:.2g} m²/yr, i.e. {Db / 1e-3:.0f}× the model's default "
             "hillslope diffusivity of 0.001 m²/yr.\n")
    return L
