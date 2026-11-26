"""
forest_papers.py — validation against six further sources (full texts or data read)
===================================================================================
S   Šamonil, Daněk, Adam & Phillips (2017) Breakage or uprooting: how tree death type
    affects hillslope processes in old-growth temperate forests. Geomorphology 299:76-84.
    Four Czech reserves, ~55,000 trees over 33-38 yr (Table 2).
G   Gabet & Mudd (2010) Bedrock erosion by root fracture and tree throw. JGR 115, F04005.
C   Constantine, Schelhaas, Gabet & Mudd (2012) Limits of windthrow-driven hillslope
    sediment flux due to varying storm frequency and intensity. Geomorphology 175-176:66-73.
L   Gallaway, Martin & Johnson (2009) Sediment transport due to tree root throw:
    integrating tree population dynamics, wildfire and geomorphic response. ESPL 34:1255-1269.
    (Only the abstract, introduction and study area were available.)
H   Hoffman & Anderson (2013) Tree root mounds and their role in transporting soil on
    forested landscapes. ESPL, doi:10.1002/esp.3470.
HF  Harvard Forest Data Archive HF320-03 (2016 pit-mound microtopography survey).
"""

from __future__ import annotations

import math

import numpy as np

import TreeThrow as tt

SAMONIL = {
    "citation": "Šamonil P, Daněk P, Adam D, Phillips JD (2017) Geomorphology 299:76-84, doi:10.1016/j.geomorph.2017.09.023",
    # Table 1-2: per site.  stems = living trees >= 10 cm DBH in the 2000s / area.
    "sites": {
        "Ranspurk (lowland floodplain, oak-ash)": dict(slope_deg=1.7, stems=6249 / 22.25, uproot=0.7, broken=1.6,
                                                      share=0.31, deaths=(2.3, 2.5), area=1.5, volume=0.2),
        "Zakova Hora (highland, beech)": dict(slope_deg=8.9, stems=5412 / 17.46, uproot=0.6, broken=1.5,
                                             share=0.25, deaths=(2.3, 2.5), area=2.7, volume=0.6),
        "Zofin (mountain, beech-spruce-fir)": dict(slope_deg=8.6, stems=15221 / 74.20, uproot=1.6, broken=1.9,
                                                  share=0.42, deaths=(3.8, 4.2), area=9.6, volume=4.2),
        "Boubin (mountain, spruce-beech)": dict(slope_deg=14.4, stems=9798 / 46.62, uproot=1.9, broken=1.9,
                                               share=0.50, deaths=(3.8, 4.2), area=10.3, volume=4.3),
    },
    "share_range_unmanaged": (0.23, 0.42),
}
GABET_MUDD = {
    "citation": "Gabet EJ, Mudd SM (2010) J. Geophys. Res. 115, F04005, doi:10.1029/2009JF001526",
    "mortality": 1 / 330, "topple_share": 0.15, "min_topple_age": 80, "pit_volume": (3.0, 1.3),
    "fall_downhill": "P(downhill) rises linearly from 50 % on flat ground to 90 % at 45°",
    "mound_age_michigan_yr": 2420, "mound_slope_deg": 14, "D_creep": 2.5e-3,
}
CONSTANTINE = {
    "citation": "Constantine JA, Schelhaas M-J, Gabet E, Mudd SM (2012) Geomorphology 175-176:66-73, "
                "doi:10.1016/j.geomorph.2012.06.022",
    "rate_present": 1.4, "mean_dbh_present": 41.0, "peak_rate": 8.0, "peak_k_m2_yr": 3.5e-3,
}
GALLAWAY = {
    "citation": "Gallaway JM, Martin YE, Johnson EA (2009) ESPL 34:1255-1269, doi:10.1002/esp.1813",
    "q_prefire_m2_yr": 0.016e-6 / 0.01, "q_postfire_m2_yr": 0.18e-6 / 0.01,   # 0.016 / 0.18 cm3 cm-1 a-1
    "second_pulse_yr": 50, "slopes": "steepest plots; study-plot gradients generally < 30°",
}
HOFFMAN = {
    "citation": "Hoffman BSS, Anderson RS (2013) Earth Surf. Process. Landforms, doi:10.1002/esp.3470",
    "mound_height_cm": (12.2, 6.1), "mound_width_cm": (93.3, 30.7),
}
HARVARD = {
    "citation": "Harvard Forest Data Archive HF320-03, hf320-03-microtopo.csv (2016)",
    "relief_median_cm": 50.0, "relief_mean_cm": 56.266, "relief_max_cm": 238.0, "n": 1763 - 1134,
}


def _verdict(ok, partial=False):
    return "PASS" if ok else ("PARTIAL" if partial else "FAIL")


def _within(x, lo, hi, tol=0.0):
    return lo * (1 - tol) <= x <= hi * (1 + tol)


# ----------------------------------------------------------------------------
# Experiments (top level so they can run in a process pool)
# ----------------------------------------------------------------------------


def run_steep_pine(seed):
    cfg = tt.HillslopeConfig(length=40, width=40, slope=0.5, dx=0.25, years=3000, seed=seed, n_snapshots=2,
                             record_interval=100)
    m = tt.TreeThrowModel(tt.SPECIES["pine"], cfg)
    m.run(progress=False)
    s = m.summary()
    return {"q_tt": s["mean_flux_tree_throw_total_m2_yr"], "q_d": s["mean_flux_diffusion_m2_yr"],
            "rate": s["uprooting_rate_per_ha_yr"]}


def run_fire(seed, t_fire=300.0, years=520):
    """Pine stand burned once (all trees killed); snags topple over the following decades."""
    cfg = tt.HillslopeConfig(length=60, width=60, slope=0.5, dx=0.25, years=years, seed=seed, n_snapshots=2,
                             record_interval=1.0, disturbance="fire", first_disturbance=t_fire,
                             disturbance_severity=1.0)
    m = tt.TreeThrowModel(tt.SPECIES["pine"], cfg)
    m.run(progress=False)
    t = np.array(m.record.time)
    mom = np.diff(np.array(m.record.moment_tt) + np.array(m.record.moment_fresh), prepend=0.0)
    ev = np.array(m.event_log)
    bins = np.arange(0, years + 1)
    counts = np.histogram(ev[:, 0], bins=bins)[0] if len(ev) else np.zeros(years)
    living = np.histogram(ev[ev[:, 9] == 0, 0], bins=bins)[0] if len(ev) else np.zeros(years)
    area = m.L_active * m.W
    return {"t": t.tolist(), "flux": (mom / area).tolist(), "events": counts.tolist(),
            "events_living": living.tolist(), "t_fire": t_fire, "area_ha": area / 1e4}


def run_storm(args):
    """Pine stand with an annual storm of given severity (size-selective windthrow)."""
    severity, seed = args
    cfg = tt.HillslopeConfig(length=60, width=60, slope=math.tan(math.radians(15)), dx=0.5, years=1500,
                             seed=seed, n_snapshots=2, record_interval=100, soil_model=False,
                             disturbance="windstorm" if severity > 0 else "none", disturbance_interval=1.0,
                             first_disturbance=1.0 if severity > 0 else None, disturbance_severity=severity,
                             storm_uproot_fraction=0.8, storm_size_exponent=2.0)
    m = tt.TreeThrowModel(tt.SPECIES["pine"], cfg)
    m.run(progress=False)
    s = m.summary()
    ev = np.array(m.event_log)
    late = ev[ev[:, 0] > 300] if len(ev) else ev
    area_ha = m.L_active * m.W / 1e4
    return {"severity": severity, "seed": seed,
            "rate": len(late) / area_ha / (m.time - 300),
            "q_tt": s["mean_flux_tree_throw_total_m2_yr"],
            "k": s["effective_D_tree_throw_total_m2_yr"],
            "mean_volume": float(late[:, 6].mean()) if len(late) else 0.0,
            "mean_R": float(late[:, 3].mean()) if len(late) else 0.0}


def relief_series(args):
    """Pit-bottom to mound-top relief of one deciduous uprooting through time (S = 0.2)."""
    R, H, slope = args
    sp = tt.SPECIES["deciduous"]
    cfg = dict(length=16.0, width=12.0, slope=slope, dx=0.25, years=1, diffusivity=1e-3, soil_model=False,
               tree_throw=False, seed=0)
    ref = tt.TreeThrowModel(sp, tt.HillslopeConfig(**cfg))
    m = tt.TreeThrowModel(sp, tt.HillslopeConfig(**cfg))
    m.uproot(6.0, 6.0, R, H, 0.0)
    t, ts, rel = 0.0, [0.0], [float((m.z - ref.z).max() - (m.z - ref.z).min())]
    while t < 4000 and rel[-1] > 0.05:
        dt = 1.0 if t < 30 else 10.0
        m.step(dt)
        ref.step(dt)
        t += dt
        d = m.z - ref.z
        ts.append(t)
        rel.append(float(d.max() - d.min()))
    return {"R": R, "H": H, "t": ts, "relief": rel}


def experiments(workers):
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(7)
    sp = tt.SPECIES["deciduous"]
    _, a = tt.draw_ages_at_death(rng, sp, 4000)
    a = a[a >= sp.min_uproot_age][:16]
    r_m, h_m = tt.draw_mature_sizes(rng, sp, a.size)
    r, h = tt.plate_at_age(sp, r_m, h_m, a)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        steep = list(ex.map(run_steep_pine, range(3)))
        fire = list(ex.map(run_fire, range(3)))
        storms = list(ex.map(run_storm, [(sv, sd) for sv in (0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08)
                                         for sd in range(2)]))
        relief = list(ex.map(relief_series, [(float(ri), float(hi), 0.2) for ri, hi in zip(r, h)]))
    return {"steep_pine": steep, "fire": fire, "storms": storms, "relief": relief}


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def _species(res_species, key):
    return [r for r in res_species if r["species"] == key]


def test_samonil(res_species):
    out = {}
    sites = SAMONIL["sites"]
    groups = {"spruce": [k for k in sites if "mountain" in k], "deciduous": [k for k in sites if "mountain" not in k]}
    for key, names in groups.items():
        rs = _species(res_species, key)
        rate = float(np.mean([r["summary"]["uprooting_rate_per_ha_yr"] for r in rs]))
        share = float(np.mean([r["summary"]["uprooting_share_of_deaths"] for r in rs]))
        deaths = float(np.mean([r["summary"]["death_rate_per_ha_yr"] for r in rs]))
        exc = np.concatenate([np.array(r["excavated_volume"]) for r in rs])
        Rs = np.concatenate([np.array(r["event_R"]) for r in rs])
        sp = tt.SPECIES[key]
        # Pit-mound pair footprint ~ pit + mound ellipses (each pi a b = pi R^2).
        area_per_event = float(np.mean(2 * math.pi * Rs ** 2))
        lo_r, hi_r = min(sites[n]["uproot"] for n in names), max(sites[n]["uproot"] for n in names)
        lo_s, hi_s = min(sites[n]["share"] for n in names), max(sites[n]["share"] for n in names)
        mort_obs = [sites[n]["deaths"][1] / sites[n]["stems"] for n in names]
        mort_obs_lo = min(sites[n]["deaths"][0] / sites[n]["stems"] for n in names)
        vol_ev_obs = [sites[n]["volume"] / sites[n]["uproot"] for n in names]
        area_obs = [sites[n]["area"] for n in names]
        vol_obs = [sites[n]["volume"] for n in names]
        r = {
            "sites": names,
            "rate": rate, "rate_obs": [lo_r, hi_r], "verdict_rate": _verdict(_within(rate, lo_r, hi_r, 0.2),
                                                                             partial=_within(rate, lo_r, hi_r, 0.5)),
            "share": share, "share_obs": [lo_s, hi_s],
            "verdict_share": _verdict(_within(share, lo_s, hi_s, 0.1), partial=_within(share, lo_s, hi_s, 0.3)),
            "mortality_per_stem": deaths / sp.density, "mortality_obs": [mort_obs_lo, max(mort_obs)],
            "verdict_mortality": _verdict(_within(deaths / sp.density, mort_obs_lo, max(mort_obs), 0.2),
                                          partial=_within(deaths / sp.density, mort_obs_lo, max(mort_obs), 0.5)),
            "volume_per_event": float(exc.mean()), "volume_per_event_obs": [min(vol_ev_obs), max(vol_ev_obs)],
            "verdict_volume_per_event": _verdict(_within(exc.mean(), min(vol_ev_obs), max(vol_ev_obs), 0.3),
                                                 partial=_within(exc.mean(), min(vol_ev_obs), max(vol_ev_obs), 1.0)),
            "area_flux": rate * area_per_event, "area_flux_obs": [min(area_obs), max(area_obs)],
            "verdict_area_flux": _verdict(_within(rate * area_per_event, min(area_obs), max(area_obs), 0.3),
                                          partial=_within(rate * area_per_event, min(area_obs), max(area_obs), 1.0)),
            "volume_flux": rate * float(exc.mean()), "volume_flux_obs": [min(vol_obs), max(vol_obs)],
            "verdict_volume_flux": _verdict(_within(rate * exc.mean(), min(vol_obs), max(vol_obs), 0.3),
                                            partial=_within(rate * exc.mean(), min(vol_obs), max(vol_obs), 1.0)),
        }
        out[key] = r
    return out


def p_downhill(slope_deg, bias):
    """P(fall has a downhill component) for the model's von Mises fall direction."""
    kappa = bias * math.tan(math.radians(slope_deg))
    if kappa <= 0:
        return 0.5
    phi = np.linspace(-math.pi, math.pi, 20001)
    w = np.exp(kappa * np.cos(phi))
    return float(np.trapezoid(w * (np.abs(phi) < math.pi / 2), phi) / np.trapezoid(w, phi))


def test_gabet_mudd(res_species, v6):
    ang = np.arange(0, 46, 5.0)
    model = np.array([p_downhill(a, tt.SPECIES["pine"].fall_slope_bias) for a in ang])
    rule = 0.5 + 0.4 * ang / 45.0
    pine = _species(res_species, "pine")
    exc = float(np.mean(np.concatenate([np.array(r["excavated_volume"]) for r in pine])))
    share = float(np.mean([r["summary"]["uprooting_share_of_deaths"] for r in pine]))
    m0, s0 = GABET_MUDD["pit_volume"]
    t5 = v6["runs"]["D=0.001"]["time_to_5cm_yr"]
    ratios = {k: tt.SPECIES[k].root_plate_depth / (0.5 * tt.SPECIES[k].root_plate_diameter) for k in tt.SPECIES}
    return {
        "angles_deg": ang.tolist(), "p_downhill_model": model.tolist(), "p_downhill_rule": rule.tolist(),
        "max_abs_diff": float(np.abs(model - rule).max()),
        "verdict_fall": _verdict(np.abs(model - rule).max() <= 0.1, partial=np.abs(model - rule).max() <= 0.2),
        "pine_topple_share": share, "verdict_topple_share": _verdict(abs(share - 0.15) <= 0.03),
        "pine_pit_volume": exc, "verdict_pit_volume": _verdict(abs(exc - m0) <= s0, partial=abs(exc - m0) <= 2 * s0),
        "depth_to_radius": ratios,
        "mound_time_to_5cm_yr": t5,
        "verdict_mound_age": _verdict(t5 is not None and GABET_MUDD["mound_age_michigan_yr"] / 2 <= t5
                                      <= 2 * GABET_MUDD["mound_age_michigan_yr"]),
    }


def test_constantine(res_species, storms):
    # Their transport model: k = rate * V * (2/pi)(r + p) cos(theta) with a half-ellipsoid root plate
    # p = r/2, so V = pi r^3 / 3; V(d) = 5.96 ln d - 17.1 (Douglas fir, d in cm).
    S = 0.35
    th = math.atan(S)
    V = 5.96 * math.log(CONSTANTINE["mean_dbh_present"]) - 17.1
    r = (3 * V / math.pi) ** (1 / 3)
    eff_c = (2 / math.pi) * (1.5 * r) * math.cos(th)               # k / (rate * V), m
    out = {"V_present_m3": V, "r_m": r, "efficiency_constantine_m": eff_c,
           "k_present_m2_yr": CONSTANTINE["rate_present"] / 1e4 * V * eff_c}
    for key in ("pine", "deciduous", "spruce"):
        rs = _species(res_species, key)
        D = float(np.mean([x["summary"]["effective_D_tree_throw_m2_yr"] for x in rs]))
        rate = float(np.mean([x["summary"]["uprooting_rate_per_ha_yr"] for x in rs]))
        exc = float(np.mean(np.concatenate([np.array(x["excavated_volume"]) for x in rs])))
        out[key] = {"efficiency_m": D / (rate / 1e4 * exc), "D_tt": D, "rate": rate, "volume": exc}
        out[key]["ratio"] = out[key]["efficiency_m"] / eff_c
        out[key]["verdict"] = _verdict(0.67 <= out[key]["ratio"] <= 1.5, partial=0.5 <= out[key]["ratio"] <= 2.0)
    # Storm sweep: flux vs. windthrow rate.
    sv = sorted({s["severity"] for s in storms})
    rows = []
    for v in sv:
        ss = [s for s in storms if s["severity"] == v]
        rows.append({"severity": v, "rate": float(np.mean([s["rate"] for s in ss])),
                     "q_tt": float(np.mean([s["q_tt"] for s in ss])), "k": float(np.mean([s["k"] for s in ss])),
                     "mean_volume": float(np.mean([s["mean_volume"] for s in ss]))})
    out["storm_rows"] = rows
    vols = [r_["mean_volume"] for r_ in rows]
    q = [r_["q_tt"] for r_ in rows]
    imax = int(np.argmax(q))
    out["storm_peak_rate"] = rows[imax]["rate"]
    out["verdict_volume_declines"] = _verdict(vols[-1] < 0.8 * max(vols))
    out["storm_peak_k"] = rows[imax]["k"]
    pr = out["storm_peak_rate"] / CONSTANTINE["peak_rate"]
    out["verdict_flux_saturates"] = _verdict(imax < len(q) - 1 and 0.5 <= pr <= 2.0,
                                             partial=imax < len(q) - 1 and 0.25 <= pr <= 4.0)
    return out


def test_gallaway(steep, fire):
    q_pre = float(np.mean([s["q_tt"] for s in steep]))
    t = np.array(fire[0]["t"])
    tf = fire[0]["t_fire"]
    flux = np.mean([np.array(f["flux"]) for f in fire], axis=0)
    ev = np.mean([np.array(f["events"]) for f in fire], axis=0)
    area_ha = fire[0]["area_ha"]
    pre = (t > tf - 200) & (t <= tf)
    post = (t > tf) & (t <= tf + 5)
    q_pre_fire = float(flux[pre].mean())
    q_post_fire = float(flux[post].mean())
    yrs = np.arange(ev.size)
    rate = ev / area_ha
    # Second pulse: uprootings of trees that die in the regrown stand (snag falls excluded) recover to
    # half the pre-fire rate.
    pre_rate = float(rate[(yrs > tf - 200) & (yrs <= tf)].mean())
    living = np.mean([np.array(f["events_living"]) for f in fire], axis=0) / area_ha
    smooth = np.convolve(living, np.ones(10) / 10, mode="same")
    after = np.flatnonzero((yrs > tf + 5) & (smooth >= 0.5 * pre_rate))
    t_second = float(yrs[after[0]] - tf) if after.size else None
    obs = GALLAWAY
    return {
        "q_steep_model": q_pre, "q_prefire_obs": obs["q_prefire_m2_yr"], "q_postfire_obs": obs["q_postfire_m2_yr"],
        "ratio_prefire": q_pre / obs["q_prefire_m2_yr"],
        "verdict_prefire": _verdict(q_pre / obs["q_prefire_m2_yr"] <= 3, partial=q_pre / obs["q_prefire_m2_yr"] <= 10),
        "post_over_pre_model": q_post_fire / q_pre_fire if q_pre_fire > 0 else float("nan"),
        "post_over_pre_obs": obs["q_postfire_m2_yr"] / obs["q_prefire_m2_yr"],
        "verdict_post_over_pre": _verdict(0.5 <= (q_post_fire / q_pre_fire) / (obs["q_postfire_m2_yr"] / obs["q_prefire_m2_yr"])
                                          <= 2.0 if q_pre_fire > 0 else False,
                                          partial=q_post_fire > 3 * q_pre_fire),
        "second_pulse_yr": t_second,
        "verdict_second_pulse": _verdict(t_second is not None and abs(t_second - obs["second_pulse_yr"]) <= 15,
                                         partial=t_second is not None and abs(t_second - obs["second_pulse_yr"]) <= 40),
        "rate_series": rate.tolist(), "flux_series": flux.tolist(), "t_fire": tf, "pre_rate": pre_rate,
    }


def test_hoffman(res_species):
    th, a = math.radians(15), (4.4, 0.45, 400.0)
    betas = np.linspace(0.5, 0.999, 400)
    checks = {
        "max_q_zA_0.1": max(tt.root_cycle_flux(th, *a, b, 0.1) for b in betas),
        "q_zA1_beta0.98": tt.root_cycle_flux(th, *a, 0.98, 1.0),
        "q_zA1_beta0.80": tt.root_cycle_flux(th, *a, 0.80, 1.0),
        "full_over_median": tt.root_cycle_flux(th, *a, 0.95, None) / (a[0] * a[1] / a[2] * math.sin(th)
                                                                     * math.log(0.5) / (100 * math.log(0.95))),
    }
    ok = (checks["max_q_zA_0.1"] < 4e-5 and abs(checks["q_zA1_beta0.98"] - 3.8e-4) < 0.15e-4
          and checks["q_zA1_beta0.80"] < 6e-5 and abs(checks["full_over_median"] - 1.4) < 0.1)
    D_root = {zA: tt.root_cycle_diffusivity(*a, 0.95, zA) for zA in (0.1, 0.2, 0.5, None)}
    dec = _species(res_species, "deciduous")
    D_tt = float(np.mean([r["summary"]["effective_D_tree_throw_total_m2_yr"] for r in dec]))
    curves = {str(zA): [tt.root_cycle_flux(th, *a, b, zA) for b in betas] for zA in (0.1, 0.3, 1.0)}
    return {"checks": checks, "verdict_formula": _verdict(ok), "D_root": {str(k): v for k, v in D_root.items()},
            "D_tt_deciduous": D_tt, "tt_over_root_all_roots": D_tt / D_root[None],
            "tt_over_root_zA0.2": D_tt / D_root[0.2], "betas": betas.tolist(), "curves": curves}


def test_harvard(relief):
    # Steady population of couplets: ages uniform, so each event contributes its relief history
    # (until relief < 5 cm), weighted by time spent at each value.
    vals, w = [], []
    for r in relief:
        t, rel = np.array(r["t"]), np.array(r["relief"])
        dt = np.diff(t)
        vals.append(0.5 * (rel[1:] + rel[:-1]))
        w.append(dt)
    vals, w = np.concatenate(vals), np.concatenate(w)
    order = np.argsort(vals)
    cw = np.cumsum(w[order]) / w.sum()
    median = float(vals[order][np.searchsorted(cw, 0.5)]) * 100
    mean = float(np.sum(vals * w) / w.sum()) * 100
    mx = max(max(r["relief"]) for r in relief) * 100
    obs = HARVARD
    # Couplets younger than 150 yr (ages uniform): the population if old, faint features are not recorded.
    young = np.concatenate([np.interp(np.linspace(0, 150, 151), r["t"], r["relief"]) for r in relief]) * 100
    med_young = float(np.median(young))
    return {"median_cm": median, "mean_cm": mean, "max_cm": mx, "median_young_cm": med_young,
            "verdict_median_young": _verdict(abs(med_young / obs["relief_median_cm"] - 1) <= 0.3,
                                             partial=abs(med_young / obs["relief_median_cm"] - 1) <= 0.6),
            "lifetimes_yr": [r["t"][-1] for r in relief],
            "verdict_median": _verdict(abs(median / obs["relief_median_cm"] - 1) <= 0.3,
                                       partial=abs(median / obs["relief_median_cm"] - 1) <= 0.6),
            "verdict_mean": _verdict(abs(mean / obs["relief_mean_cm"] - 1) <= 0.3,
                                     partial=abs(mean / obs["relief_mean_cm"] - 1) <= 0.6),
            "hist_values": vals.tolist(), "hist_weights": w.tolist()}


def run_all(res_species, v6, exp):
    return {"S_samonil": test_samonil(res_species),
            "G_gabet_mudd": test_gabet_mudd(res_species, v6),
            "C_constantine": test_constantine(res_species, exp["storms"]),
            "L_gallaway": test_gallaway(exp["steep_pine"], exp["fire"]),
            "H_hoffman": test_hoffman(res_species),
            "HF_harvard": test_harvard(exp["relief"])}


def strip_for_json(res):
    res = dict(res)
    res["HF_harvard"] = {k: v for k, v in res["HF_harvard"].items() if not k.startswith("hist_")}
    res["H_hoffman"] = {k: v for k, v in res["H_hoffman"].items() if k not in ("betas", "curves")}
    return res


# ----------------------------------------------------------------------------
# Figure and report
# ----------------------------------------------------------------------------


def plot(res, path):
    plt = tt._import_pyplot()
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Validation against Šamonil 2017, Gabet & Mudd 2010, Constantine 2012, Gallaway 2009, "
                 "Hoffman & Anderson 2013 and Harvard Forest HF320", fontsize=11)

    a = ax[0, 0]
    S = res["S_samonil"]
    labels = ["uprootings\n(/ha/yr)", "share of\ndeaths", "mortality\n(%/yr)", "root-plate soil\n(m³/ha/yr)"]
    x = np.arange(len(labels))
    for i, (key, c) in enumerate((("deciduous", "tab:orange"), ("spruce", "tab:blue"))):
        r = S[key]
        model = [r["rate"], r["share"], 100 * r["mortality_per_stem"], r["volume_flux"]]
        lo = [r["rate_obs"][0], r["share_obs"][0], 100 * r["mortality_obs"][0], r["volume_flux_obs"][0]]
        hi = [r["rate_obs"][1], r["share_obs"][1], 100 * r["mortality_obs"][1], r["volume_flux_obs"][1]]
        a.bar(x + (i - 0.5) * 0.38, model, 0.36, color=c, label=f"model {key}")
        a.vlines(x + (i - 0.5) * 0.38, lo, hi, color="k", lw=4, alpha=0.5)
    a.set_xticks(x, labels, fontsize=8)
    a.set_yscale("log")
    a.set(title="Šamonil 2017: bars = model, black = reserve range\n(deciduous vs. lowland/highland, spruce vs. mountain)")
    a.legend(fontsize=8)

    a = ax[0, 1]
    G = res["G_gabet_mudd"]
    a.plot(G["angles_deg"], G["p_downhill_model"], "o-", label="model (von Mises, bias 2 × gradient)")
    a.plot(G["angles_deg"], G["p_downhill_rule"], "k--", label="Gabet & Mudd 2010 rule")
    a.set(title="Fall direction: probability of a downhill fall", xlabel="slope (°)", ylabel="P(downhill)",
          ylim=(0.45, 1.0))
    a.legend(fontsize=8)

    a = ax[0, 2]
    C = res["C_constantine"]
    rows = C["storm_rows"]
    a.plot([r["rate"] for r in rows], [r["k"] * 1e3 for r in rows], "o-", color="tab:green", label="model k (pine, 15°)")
    a2 = a.twinx()
    a2.plot([r["rate"] for r in rows], [r["mean_volume"] for r in rows], "s--", color="tab:brown",
            label="mean root-plate volume")
    a.axvline(CONSTANTINE["peak_rate"], color="k", ls=":", label="Constantine flux peak ~8 /ha/yr")
    a.axhline(CONSTANTINE["peak_k_m2_yr"] * 1e3, color="tab:green", ls=":", label="Constantine peak k 3.5 m²/kyr")
    a.set(title="Storm frequency sweep (size-selective windthrow)", xlabel="uprootings (/ha/yr)",
          ylabel="tree-throw diffusivity k (m²/kyr)")
    a2.set_ylabel("m³ per event", color="tab:brown")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")

    a = ax[1, 0]
    L = res["L_gallaway"]
    rate = np.array(L["rate_series"])
    yrs = np.arange(rate.size) - L["t_fire"]
    sm = np.convolve(rate, np.ones(5) / 5, mode="same")
    a.plot(yrs, sm, color="tab:red", lw=1)
    a.axvline(0, color="k", lw=0.5)
    a.axvline(GALLAWAY["second_pulse_yr"], color="k", ls=":", label="Gallaway: second pulse ~50 yr")
    a.axhline(L["pre_rate"], color="gray", ls="--", label="pre-fire rate")
    a.set_yscale("symlog", linthresh=0.1)
    a.set(title="Pine stand burned at t = 0 (snags topple at 0.1 /yr)", xlabel="years since fire",
          ylabel="uprootings (/ha/yr, 5-yr mean)", xlim=(-100, 220))
    a.legend(fontsize=8)

    a = ax[1, 1]
    H = res["H_hoffman"]
    for zA, c in (("0.1", "tab:olive"), ("0.3", "tab:green"), ("1.0", "tab:blue")):
        a.plot(H["betas"], np.array(H["curves"][zA]) * 1e4, color=c, label=f"A horizon {zA} m")
    a.axhline(0.4, color="tab:olive", ls=":", label="Hoffman: < 4×10⁻⁵ for 0.1 m")
    a.plot(0.98, 3.8, "k*", ms=10, label="Hoffman: ~3.8×10⁻⁴ (1 m, β 0.98)")
    a.set(title="Root growth-decay flux, 15° slope (Hoffman & Anderson Fig. 17)", xlabel="rooting parameter β",
          ylabel="q (cm²/yr = 10⁻⁴ m²/yr)", ylim=(0, 6))
    a.legend(fontsize=7)

    a = ax[1, 2]
    HF = res["HF_harvard"]
    a.hist(np.array(HF["hist_values"]) * 100, bins=np.arange(0, 160, 5), weights=HF["hist_weights"], density=True,
           color="tab:orange", alpha=0.6, label="model deciduous couplets (age-weighted)")
    a.axvline(HARVARD["relief_median_cm"], color="k", ls="--", label="HF320 median 50 cm")
    a.axvline(HARVARD["relief_mean_cm"], color="k", ls=":", label="HF320 mean 56 cm")
    a.axvline(HF["median_cm"], color="tab:orange", ls="--", label=f"model median {HF['median_cm']:.0f} cm")
    a.set(title="Pit-bottom to mound-top relief", xlabel="relief (cm)", ylabel="density")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def scorecard_rows(res):
    S, G, C, L, H, HF = (res[k] for k in ("S_samonil", "G_gabet_mudd", "C_constantine", "L_gallaway",
                                           "H_hoffman", "HF_harvard"))
    rows = []
    for key, lab in (("deciduous", "lowland/highland"), ("spruce", "mountain")):
        r = S[key]
        rows += [
            f"| S1 uprooting rate, {key} | {r['rate_obs'][0]:.1f}–{r['rate_obs'][1]:.1f} /ha/yr (Šamonil {lab}) | "
            f"{r['rate']:.2f} /ha/yr | {r['verdict_rate']} |",
            f"| S2 uprooting share of deaths, {key} | {100 * r['share_obs'][0]:.0f}–{100 * r['share_obs'][1]:.0f} % | "
            f"{100 * r['share']:.0f} % | {r['verdict_share']} |",
            f"| S3 annual mortality per stem, {key} | {100 * r['mortality_obs'][0]:.1f}–{100 * r['mortality_obs'][1]:.1f} % | "
            f"{100 * r['mortality_per_stem']:.1f} % | {r['verdict_mortality']} |",
            f"| S4 root-plate volume per uprooting, {key} | {r['volume_per_event_obs'][0]:.1f}–{r['volume_per_event_obs'][1]:.1f} m³ | "
            f"{r['volume_per_event']:.1f} m³ | {r['verdict_volume_per_event']} |",
            f"| S5 pit-mound area created, {key} | {r['area_flux_obs'][0]:.1f}–{r['area_flux_obs'][1]:.1f} m²/ha/yr | "
            f"{r['area_flux']:.1f} m²/ha/yr | {r['verdict_area_flux']} |",
            f"| S6 root-plate soil volume, {key} | {r['volume_flux_obs'][0]:.1f}–{r['volume_flux_obs'][1]:.1f} m³/ha/yr | "
            f"{r['volume_flux']:.1f} m³/ha/yr | {r['verdict_volume_flux']} |",
        ]
    rows += [
        f"| G1 fall direction vs. Gabet & Mudd rule (0–45°) | 50 % → 90 % downhill | max difference "
        f"{100 * G['max_abs_diff']:.0f} % | {G['verdict_fall']} |",
        f"| G2 pine topple share of deaths (calibrated) | 15 % (Douglas fir) | {100 * G['pine_topple_share']:.0f} % | "
        f"{G['verdict_topple_share']} |",
        f"| G3 pine pit volume | 3.0 ± 1.3 m³ (Douglas fir) | {G['pine_pit_volume']:.1f} m³ | {G['verdict_pit_volume']} |",
        f"| G4 mound lifetime (to < 5 cm, D = 0.001) | oldest recognisable mound 2420 yr (Michigan) | "
        f"{G['mound_time_to_5cm_yr']:.0f} yr | {G['verdict_mound_age']} |",
    ]
    for key in ("pine", "deciduous", "spruce"):
        r = C[key]
        rows.append(f"| C1 transport per event, {key}: k / (rate × V) | {C['efficiency_constantine_m']:.2f} m "
                    f"(Constantine/Gabet 2003, S = 0.35) | {r['efficiency_m']:.2f} m | {r['verdict']} |")
    t2 = "—" if L["second_pulse_yr"] is None else f"{L['second_pulse_yr']:.0f} yr"
    rows += [
        f"| C2 storms deplete large trees: volume per event falls at high windthrow rates | yes | "
        f"{C['storm_rows'][0]['mean_volume']:.1f} → {C['storm_rows'][-1]['mean_volume']:.1f} m³ | {C['verdict_volume_declines']} |",
        f"| C3 tree-throw flux peaks, then declines, as windthrow rate rises | peak near 8 /ha/yr, k ≈ 3.5 m²/kyr | "
        f"peak at {C['storm_peak_rate']:.1f} /ha/yr, k = {C['storm_peak_k'] * 1e3:.1f} m²/kyr | {C['verdict_flux_saturates']} |",
        f"| L1 pine flux on steep slopes, no fire | {L['q_prefire_obs']:.1e} m²/yr (Gallaway, steepest plots) | "
        f"{L['q_steep_model']:.1e} m²/yr (×{L['ratio_prefire']:.0f}) | {L['verdict_prefire']} |",
        f"| L2 post-fire / pre-fire flux (first 5 yr) | {L['post_over_pre_obs']:.0f}× | "
        f"{L['post_over_pre_model']:.0f}× | {L['verdict_post_over_pre']} |",
        f"| L3 second uprooting pulse after fire | ~50 yr | {t2} | {L['verdict_second_pulse']} |",
        f"| H1 root-cycle flux formula reproduces Hoffman & Anderson | 4 published values | "
        f"{H['checks']['q_zA1_beta0.98'] * 1e4:.2f}×10⁻⁴ at β 0.98 … | {H['verdict_formula']} |",
        f"| HF1 pit-to-mound relief, median, couplets < 150 yr old | 50 cm (Harvard Forest, n = {HARVARD['n']}) | "
        f"{HF['median_young_cm']:.0f} cm | {HF['verdict_median_young']} |",
        f"| HF2 pit-to-mound relief, median, all couplets > 5 cm (steady state) | 50 cm | {HF['median_cm']:.0f} cm | "
        f"{HF['verdict_median']} |",
    ]
    return rows


def detail_lines(res):
    S, G, C, L, H, HF = (res[k] for k in ("S_samonil", "G_gabet_mudd", "C_constantine", "L_gallaway",
                                           "H_hoffman", "HF_harvard"))
    out = ["### S — Šamonil et al. (2017), four Czech old-growth reserves (primary data)\n",
           f"Source: {SAMONIL['citation']}. Table 2 rates per site; mortality per stem is deaths/ha/yr divided "
           "by living stems ≥ 10 cm DBH (Table 1). The model's deciduous stand is compared with the lowland "
           "(oak-ash) and highland (beech) reserves, spruce with the two mountain reserves. Pit-mound area per "
           "event in the model is the pit plus mound ellipse (≈ 2πR²).\n",
           "| site | slope (°) | stems/ha | uprooted /ha/yr | broken /ha/yr | share | area m²/ha/yr | volume m³/ha/yr |",
           "|---|---|---|---|---|---|---|---|"]
    for name, d in SAMONIL["sites"].items():
        out.append(f"| {name} | {d['slope_deg']} | {d['stems']:.0f} | {d['uproot']} | {d['broken']} | "
                   f"{100 * d['share']:.0f} % | {d['area']} | {d['volume']} |")
    out.append("\nThe model's spruce root plates follow Zadrożny's measured pits (≈5–6 m³ excavated), about twice "
               "Šamonil's 2.3–2.6 m³ per uprooting in the mountain reserves, whose volumes come from DBH "
               "regressions over all trees ≥ 10 cm. The two primary sources disagree on spruce root-plate size.\n")
    out.append("### G — Gabet & Mudd (2010)\n")
    out.append(f"Source: {GABET_MUDD['citation']}. Douglas-fir model parameters: annual mortality 1/330; topples "
               "15 % of deaths, only trees > 80 yr; pit volume 3 ± 1.3 m³ (half-ellipsoid, depth ≈ radius/2); "
               "P(downhill) 50 → 90 % from 0 to 45°; soil creep D = 2.5×10⁻³ m²/yr; oldest recognisable "
               "Michigan mound 2420 yr (Schaetzl & Follmer 1990). The model adopts the pine mortality, topple share "
               "and minimum topple age, and the thin-soil root-plate thinning (depth falls to half on bare rock).\n")
    out.append("Root-plate depth / radius in the model: " +
               ", ".join(f"{k} {v:.2f}" for k, v in G["depth_to_radius"].items()) +
               " (Gabet & Mudd and Constantine: ≈ 0.5 for Douglas fir).\n")
    out.append("### C — Constantine et al. (2012)\n")
    out.append(f"Source: {CONSTANTINE['citation']}. Douglas-fir root plates: V = 5.96 ln(d) − 17.1 "
               f"({C['V_present_m3']:.1f} m³ at the 41 cm mean DBH of windthrown trees), half-ellipsoid with p = r/2; "
               "transport distance (2/π)(r + p) sin θ for random fall directions (Gabet et al. 2003). Their k at the "
               f"present windthrow rate (1.4 /ha/yr) is ≈ {C['k_present_m2_yr']:.1e} m²/yr by these formulas at "
               "S = 0.35. The model has no slope-geometry term; its net transport comes from the downslope fall bias "
               "and slope-dependent infill, which together give a similar transport per event (C1).\n")
    out.append("| storm severity (annual) | uprootings /ha/yr | k (m²/yr) | mean root-plate volume (m³) |")
    out.append("|---|---|---|---|")
    for r in C["storm_rows"]:
        out.append(f"| {r['severity']} | {r['rate']:.2f} | {r['k']:.2e} | {r['mean_volume']:.2f} |")
    out.append("")
    out.append("### L — Gallaway et al. (2009), Canadian Rockies (abstract and introduction only)\n")
    out.append(f"Source: {GALLAWAY['citation']}. Root-throw transport on the steepest plots: 0.016 cm³/cm/yr "
               "before the 2003 fire and 0.18 cm³/cm/yr after it; fire-killed trees topple at an exponential rate, "
               "and a second pulse of root throw begins ~50 yr after fire. The model's pine preset is compared at "
               "S = 0.5; the fire scenario kills the whole stand once and lets snags fall at 0.1 /yr with root "
               "plates decaying over 15 yr. The second pulse in the model is set by the 80-yr minimum topple age "
               "(Gabet & Mudd), later than Gallaway's ~50 yr.\n")
    out.append("### H — Hoffman & Anderson (2013), Boulder Creek CZO\n")
    ch = H["checks"]
    out.append(f"Source: {HOFFMAN['citation']}. Root-cycle flux (Gale & Grigal profile, A-horizon roots only), "
               f"15° slope, root mass 4.4 kg/m², turnover 0.45 /yr, root density 400 kg/m³: max over β for a 0.1 m "
               f"A horizon {ch['max_q_zA_0.1']:.2e} (published < 4×10⁻⁵); 1 m A horizon, β 0.98 "
               f"{ch['q_zA1_beta0.98']:.2e} (≈ 3.8×10⁻⁴); β 0.8 {ch['q_zA1_beta0.80']:.2e} (< 6×10⁻⁵); full profile / "
               f"median-depth formula {ch['full_over_median']:.2f} (≈ 40 % larger). Root-cycle diffusivity at β 0.95: "
               + ", ".join(f"A horizon {k} m {v:.1e}" for k, v in H["D_root"].items()) +
               f" m²/yr. Model deciduous tree-throw diffusivity {H['D_tt_deciduous']:.1e} m²/yr is "
               f"{H['tt_over_root_all_roots']:.2f}× the all-roots root-cycle value (Gabet et al. 2003, via Hoffman: "
               "tree throw ≈ 4× root cycling). Root cycling is an optional model process (`root_cycle`), off by "
               "default because the default background diffusivity already lumps it with other creep.\n")
    out.append("### HF — Harvard Forest HF320-03\n")
    out.append(f"Source: {HARVARD['citation']}. Pit-bottom to mound-top relief of {HARVARD['n']} pit-mounds: median "
               "50 cm, mean 56 cm, max 238 cm. Model: 16 deciduous uprootings (sizes drawn from the death "
               "distribution) on a 0.2 gradient, followed until relief < 5 cm; a steady population is age-uniform, "
               f"so each couplet is weighted by time. Model median {HF['median_cm']:.0f} cm, mean {HF['mean_cm']:.0f} cm, "
               f"max {HF['max_cm']:.0f} cm; couplet lifetimes {min(HF['lifetimes_yr']):.0f}–{max(HF['lifetimes_yr']):.0f} yr. "
               f"Couplets younger than 150 yr have a median relief of {HF['median_young_cm']:.0f} cm. In a steady "
               "population of every age, old couplets that linger for centuries at 5–20 cm dominate, so the observed "
               "50 cm median implies that the survey records mainly distinct (younger) pit-mounds, that the stand's "
               "couplets are mostly young, or that long-term creep is faster than 0.001 m²/yr.\n")
    return out
