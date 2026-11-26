"""
doane.py — validation against Doane et al. (2021), GRL 48, e2021GL094987
=======================================================================
"Topographic roughness on forested hillslopes: a theoretical approach for
quantifying hillslope sediment flux from tree throw", doi:10.1029/2021GL094987.

Values below were read from the full text.  Study area: 1,910 forested
hillslopes in Brown County, southern Indiana (hardwood forest, Borden Group
siltstone, unglaciated).  Roughness is the variance of a Gaussian high-pass
filter (length scale 3.8 m = 5 pixels) applied to a 0.76 m lidar DEM; the
pit-mound couplet geometry comes from 101 couplets in 0.25 m drone lidar, of
which the 50 freshest give mu_A = 0.68 m^2, sigma_A^2 = 0.05, phi = l/w = 0.83.
Their idealised couplet (Eq. 2) represents a pit-mound pair once the roots
have rotted, 5-10 yr after toppling; their simulations give l ~ 1 + S (m).

Tests
-----
D1  tree-throw share of total hillslope flux, 11-18 % (primary; see V3)
D2  topographic variance of synthetic hillslopes processed like the lidar DEM
    (0.001-0.03 m^2, increasing with slope)
D3  Doane's Eq. 9 applied to the model's own topography vs. the model's true
    flux ratio R = q_TT / q_c (a consistency check of model and theory)
D4  pit-mound couplet geometry 5-10 yr after uprooting vs. the fitted couplets
"""

from __future__ import annotations

import math

import numpy as np

import TreeThrow as tt

DOANE = {
    "citation": "Doane TH, Edmonds D, Yanites BJ, Lewis Q (2021) Geophys. Res. Lett. 48, e2021GL094987, "
                "doi:10.1029/2021GL094987",
    "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL094987",
    "status": "primary (full text read)",
    "share_min": 0.11, "share_max": 0.18,          # q_TT / (q_c + q_TT), west- vs east-facing
    "R_min": 0.12, "R_max": 0.22,                  # q_TT / q_c, west- vs east-facing (area-weighted)
    "variance_min_m2": 0.001, "variance_max_m2": 0.03,
    "dem_resolution_m": 0.76, "highpass_length_m": 3.8,
    "mu_A_m2": 0.68, "var_A": 0.05, "phi": 0.83,   # 50 freshest couplets
    "slope_range": [0.1, 0.6],                     # approximate range of sampled hillslopes (Fig. 4b, 4d)
}


def _verdict(ok, partial=False):
    return "PASS" if ok else ("PARTIAL" if partial else "FAIL")


def topographic_variance(m: "tt.TreeThrowModel", dem_res=DOANE["dem_resolution_m"],
                         hp_length=DOANE["highpass_length_m"]) -> float:
    """Variance of a Gaussian high-pass filtered DEM, following Doane et al. (2021).

    The model surface is block-averaged to roughly the lidar DEM resolution
    (0.75 m at dx = 0.25 m) and the high-pass length scale is used as the
    Gaussian sigma.  Rows within two filter lengths of the ridge and channel
    are excluded.
    """
    from scipy.ndimage import gaussian_filter
    z = m.z[:-1]
    k = max(int(round(dem_res / m.cfg.dx)), 1)
    nx, ny = (z.shape[0] // k) * k, (z.shape[1] // k) * k
    zc = z[:nx, :ny].reshape(nx // k, k, ny // k, k).mean(axis=(1, 3))
    res = k * m.cfg.dx
    s = hp_length / res
    hp = zc - gaussian_filter(zc, s, mode=("nearest", "wrap"))
    margin = int(math.ceil(2 * s))
    if hp.shape[0] > 2 * margin + 4:
        hp = hp[margin:-margin]
    return float(hp.var())


def doane_R(variance, slope, mu_A=DOANE["mu_A_m2"], var_A=DOANE["var_A"], phi=DOANE["phi"]):
    """Doane et al. Eq. 9: R = q_TT / q_c = 4 sqrt(2) mu_A (phi + 1) / (mu_A^2 + sigma_A^2) * var / |S|."""
    return 4 * math.sqrt(2) * mu_A * (phi + 1) / (mu_A ** 2 + var_A) * variance / abs(slope)


def couplet_reference(slope):
    """Volume and peak height of Doane's idealised mound for the mean fitted couplet.

    zeta' = 2 A x / l^2 exp(-(x^2/l^2 + y^2/w^2)); mound volume = A sqrt(pi) w and peak
    height = sqrt(2) A / l * exp(-1/2), with l = 1 + S (Doane's simulations) and w = l / phi.
    The volume conversion is ours.
    """
    l = 1.0 + slope
    w = l / DOANE["phi"]
    A = DOANE["mu_A_m2"]
    return {"l_m": l, "w_m": w, "mound_volume_m3": A * math.sqrt(math.pi) * w,
            "mound_height_m": math.sqrt(2) * A / l * math.exp(-0.5)}


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_variance(res_slope, res_species):
    rows = []
    for S in sorted({r["slope"] for r in res_slope}):
        rs = [r for r in res_slope if r["slope"] == S]
        v = np.array([r["doane_variance"] for r in rs])
        R_true = np.array([r["summary"]["tree_throw_share_of_flux"] / (1 - r["summary"]["tree_throw_share_of_flux"])
                           for r in rs])
        R_est = np.array([doane_R(r["doane_variance"], S) for r in rs])
        rows.append({"slope": S, "variance_mean": float(v.mean()), "variance_min": float(v.min()),
                     "variance_max": float(v.max()), "R_true": float(R_true.mean()),
                     "R_doane_eq9": float(R_est.mean()), "values": v.tolist()})
    species = {}
    for key in tt.SPECIES:
        rs = [r for r in res_species if r["species"] == key]
        v = np.array([r["doane_variance"] for r in rs])
        R_true = np.array([r["summary"]["tree_throw_share_of_flux"] / (1 - r["summary"]["tree_throw_share_of_flux"])
                           for r in rs])
        species[key] = {"variance_mean": float(v.mean()), "R_true": float(R_true.mean()),
                        "R_doane_eq9": float(np.mean([doane_R(x, 0.35) for x in v]))}
    lo, hi = DOANE["slope_range"]
    inrange = [r for r in rows if lo <= r["slope"] <= hi]
    ok_range = all(DOANE["variance_min_m2"] <= r["variance_mean"] <= DOANE["variance_max_m2"] for r in inrange)
    S = np.array([r["slope"] for r in inrange])
    V = np.array([r["variance_mean"] for r in inrange])
    trend = float(np.polyfit(S, V / V.mean(), 1)[0])
    # Eq. 9 assumes every tree falls straight downslope; compare where Doane sampled most hillslopes.
    eq9_rows = [r for r in rows if 0.2 <= r["slope"] <= hi]
    ratio = np.array([r["R_doane_eq9"] / r["R_true"] for r in eq9_rows])
    return {"rows": rows, "species_at_S0.35": species, "variance_trend_per_unit_slope": trend,
            "verdict_range": _verdict(ok_range),
            "verdict_trend": _verdict(trend > 0.2),
            "eq9_over_true_mean": float(ratio.mean()),
            "verdict_eq9": _verdict(0.5 <= ratio.mean() <= 2.0, partial=0.25 <= ratio.mean() <= 4.0)}


def test_couplet(single_event_series):
    S = 0.35
    ref = couplet_reference(S)
    s = single_event_series("deciduous", years=10, slope_deg=math.degrees(math.atan(S)))
    win = [5, 7, 10]
    vol = [s["mound_volume"][t] for t in win]
    hgt = [s["mound"][t] for t in win]
    out = {"reference": ref, "years": win, "model_mound_volume_m3": vol, "model_mound_height_m": hgt,
           "series": {k: s[k] for k in ("t", "mound", "mound_volume", "depth")},
           "model_phi": 1.0 / tt.SPECIES["deciduous"].pit_aspect, "doane_phi": DOANE["phi"],
           "zadrozny_L_over_W": 2.74 / 4.23}
    v7, h7 = vol[1], hgt[1]
    out["verdict_volume"] = _verdict(abs(v7 / ref["mound_volume_m3"] - 1) <= 0.25,
                                     partial=abs(v7 / ref["mound_volume_m3"] - 1) <= 0.5)
    out["verdict_height"] = _verdict(abs(h7 / ref["mound_height_m"] - 1) <= 0.25,
                                     partial=abs(h7 / ref["mound_height_m"] - 1) <= 0.5)
    out["verdict_phi"] = _verdict(abs(out["model_phi"] / DOANE["phi"] - 1) <= 0.25)
    return out


# ----------------------------------------------------------------------------
# Figure and report
# ----------------------------------------------------------------------------


def plot(res, path):
    plt = tt._import_pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Validation against Doane et al. (2021) — forested hillslopes, Brown County, Indiana", fontsize=12)
    v = res["D2_variance"]

    a = ax[0]
    for r in v["rows"]:
        a.scatter([r["slope"]] * len(r["values"]), r["values"], s=10, color="tab:orange", alpha=0.4)
    a.plot([r["slope"] for r in v["rows"]], [r["variance_mean"] for r in v["rows"]], "o-", color="tab:orange",
           label="model deciduous (mean of seeds)")
    for k, d in v["species_at_S0.35"].items():
        if k != "deciduous":
            a.plot(0.35, d["variance_mean"], "D", color=tt.SPECIES_COLORS.get(k, "tab:gray"), label=f"model {k}")
    a.axhspan(DOANE["variance_min_m2"], DOANE["variance_max_m2"], color="gray", alpha=0.15,
              label="Doane: 0.001–0.03 m² (1,910 hillslopes)")
    a.set_yscale("log")
    a.set(title="D2  Topographic variance (0.76 m DEM, 3.8 m high-pass)", xlabel="gradient (m/m)",
          ylabel="variance (m²)")
    a.legend(fontsize=7)

    a = ax[1]
    Rt = [r["R_true"] for r in v["rows"]]
    Re = [r["R_doane_eq9"] for r in v["rows"]]
    a.scatter(Rt, Re, c=[r["slope"] for r in v["rows"]], cmap="viridis")
    for r in v["rows"]:
        a.annotate(f"S={r['slope']:.2f}", (r["R_true"], r["R_doane_eq9"]), fontsize=7, xytext=(3, 3),
                   textcoords="offset points")
    mx = max(max(Rt), max(Re)) * 1.2
    a.plot([0, mx], [0, mx], "k--", lw=1, label="1 : 1")
    a.axhspan(DOANE["R_min"], DOANE["R_max"], color="gray", alpha=0.15, label="Doane field R = 0.12–0.22")
    a.set(title="D3  Doane Eq. 9 on model topography vs. model's true R", xlabel="model R = q_TT / q_c",
          ylabel="R from Eq. 9 (roughness)", xlim=(0, mx), ylim=(0, mx))
    a.legend(fontsize=7)

    a = ax[2]
    c = res["D4_couplet"]
    s = c["series"]
    a.plot(s["t"], s["mound_volume"], color="tab:orange", label="model deciduous mound volume (m³)")
    a.plot(s["t"], s["mound"], color="tab:brown", label="model deciduous mound height (m)")
    ref = c["reference"]
    a.axhline(ref["mound_volume_m3"], color="tab:orange", ls="--", label="Doane couplet volume (derived)")
    a.axhline(ref["mound_height_m"], color="tab:brown", ls="--", label="Doane couplet peak height (derived)")
    a.axvspan(5, 10, color="gray", alpha=0.12, label="5–10 yr (roots rotted)")
    a.set(title="D4  Pit-mound couplet after root decay (S = 0.35)", xlabel="years since uprooting")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def scorecard_rows(res, share_deciduous):
    v, c = res["D2_variance"], res["D4_couplet"]
    inrange = [r for r in v["rows"] if DOANE["slope_range"][0] <= r["slope"] <= DOANE["slope_range"][1]]
    lo_v = min(r["variance_mean"] for r in inrange)
    hi_v = max(r["variance_mean"] for r in inrange)
    ref = c["reference"]
    return [
        f"| D1 tree-throw share of flux (deciduous, S = 0.35) | 11–18 % (Doane 2021, primary) | "
        f"{100 * share_deciduous:.0f} % | {_verdict(0.11 <= share_deciduous <= 0.18, partial=0.08 <= share_deciduous <= 0.25)} |",
        f"| D2 topographic variance, deciduous, S = 0.15–0.54 | 0.001–0.03 m² | {lo_v:.4f}–{hi_v:.4f} m² | {v['verdict_range']} |",
        f"| D2 variance increases with slope | yes (Fig. 4b) | {100 * v['variance_trend_per_unit_slope']:+.0f} % per unit gradient | "
        f"{v['verdict_trend']} |",
        f"| D3 Doane Eq. 9 on model DEM vs. model's true R (S = 0.25–0.54) | ratio 1 | {v['eq9_over_true_mean']:.2f} | "
        f"{v['verdict_eq9']} |",
        f"| D4 couplet mound volume at 7 yr (deciduous) | ≈ {ref['mound_volume_m3']:.1f} m³ (derived) | "
        f"{c['model_mound_volume_m3'][1]:.2f} m³ | {c['verdict_volume']} |",
        f"| D4 couplet mound height at 7 yr (deciduous) | ≈ {ref['mound_height_m']:.2f} m (derived) | "
        f"{c['model_mound_height_m'][1]:.2f} m | {c['verdict_height']} |",
        f"| D4 couplet length / width φ | 0.83 (Doane); 0.65 (Zadrożny pits) | {c['model_phi']:.2f} | {c['verdict_phi']} |",
    ]


def detail_lines(res):
    v, c = res["D2_variance"], res["D4_couplet"]
    ref = c["reference"]
    L = ["### D — Doane et al. (2021), forested hillslopes of southern Indiana (primary data)\n",
         f"Source: [{DOANE['citation']}]({DOANE['url']}); the full text was read. Tree throw accounts for "
         "11 % (west-facing, R = 0.12) to 18 % (east-facing, R = 0.22) of hillslope sediment flux; topographic "
         "variance of 1,910 hillslopes spans 0.001–0.03 m² and increases with slope.\n",
         "**D2/D3 — roughness.** Each simulated hillslope (3000 yr, 40 × 40 m) is block-averaged to 0.75 m and "
         "high-pass filtered with a 3.8 m Gaussian (our reading of the paper's \"length scale of 3.8 m\"), as for "
         "the lidar DEM. Doane's Eq. 9 is then applied with their fitted couplet parameters (μ_A = 0.68, "
         "σ²_A = 0.05, φ = 0.83) to estimate R from roughness, and compared with the model's true R.\n",
         "Eq. 9 assumes every tree falls straight downslope, so its R grows as 1/S on gentle slopes; the model's "
         "fall direction is only weakly biased downslope there (mean cosine of fall azimuth 0.05 at S = 0.05, "
         "0.33 at S = 0.35), so the two diverge below S ≈ 0.2.\n",
         "| gradient | variance mean (m²) | range over seeds (m²) | model R = q_TT/q_c | R from Eq. 9 |",
         "|---|---|---|---|---|"]
    for r in v["rows"]:
        L.append(f"| {r['slope']:.2f} | {r['variance_mean']:.4f} | {r['variance_min']:.4f}–{r['variance_max']:.4f} | "
                 f"{r['R_true']:.2f} | {r['R_doane_eq9']:.2f} |")
    L.append("\n| species (S = 0.35) | variance (m²) | model R | R from Eq. 9 |")
    L.append("|---|---|---|---|")
    for k, d in v["species_at_S0.35"].items():
        L.append(f"| {k} | {d['variance_mean']:.4f} | {d['R_true']:.2f} | {d['R_doane_eq9']:.2f} |")
    L.append("")
    L.append("**D4 — couplet geometry.** Doane's mean fresh couplet (A = 0.68 m², l = 1 + S, w = l/φ) has a mound "
             f"volume of {ref['mound_volume_m3']:.2f} m³ and peak height {ref['mound_height_m']:.2f} m at S = 0.35 "
             "(our conversion of their Eq. 2). Model deciduous mound after "
             + ", ".join(f"{t} yr: {vv:.2f} m³ / {hh:.2f} m" for t, vv, hh in
                         zip(c["years"], c["model_mound_volume_m3"], c["model_mound_height_m"]))
             + f". Couplet length/width: model {c['model_phi']:.2f}, Doane {c['doane_phi']:.2f}, "
             f"Zadrożny pits {c['zadrozny_L_over_W']:.2f}.\n")
    return L
