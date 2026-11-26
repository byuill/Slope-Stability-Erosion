"""QA/QC checks for TreeThrow.py: volume budget, positivity, grid and time-step convergence, determinism,
overlapping disturbances and cost.  Run: python validation/qaqc_checks.py"""
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import TreeThrow as tt


def budget(m, z0):
    """Volume budget over the hillslope rows (channel row excluded)."""
    ca = m.cell_area
    dz = float((m.z[:-1] - z0[:-1]).sum()) * ca
    dil = m.cfg.rock_to_soil_dilation
    gain = (dil - 1) / dil * (m.rock_to_soil_tt + m.rock_to_soil_sp) if m.cfg.soil_model else 0.0
    expected = gain - m.export_tt - m.export_diff + m.clip_error
    return dz, expected


def run_budget(args):
    name, sp_key, kw = args
    cfg = tt.HillslopeConfig(**kw)
    m = tt.TreeThrowModel(tt.SPECIES[sp_key], cfg)
    z0 = m.z.copy()
    t0 = time.time()
    m.run(progress=False)
    dz, exp = budget(m, z0)
    return (name, dz, exp, m.clip_error, m.n_events, time.time() - t0,
            bool(np.isfinite(m.z).all()), float(m.h.min()))


def run_dx(args):
    dx, seed = args
    cfg = tt.HillslopeConfig(length=30, width=20, slope=0.35, dx=dx, years=1500, seed=seed, n_snapshots=2)
    m = tt.TreeThrowModel(tt.SPECIES["deciduous"], cfg)
    m.run(progress=False)
    s = m.summary()
    return dx, seed, s["effective_D_tree_throw_m2_yr"], s["tree_throw_share_of_flux"], s["final_roughness_m"], \
        s["mean_event_volume_m3"]


def run_dt(args):
    dt, seed = args
    cfg = tt.HillslopeConfig(length=30, width=20, slope=0.35, dx=0.25, years=1500, seed=seed, dt=dt,
                             n_snapshots=2, record_interval=max(dt, 10))
    m = tt.TreeThrowModel(tt.SPECIES["spruce"], cfg)
    m.run(progress=False)
    s = m.summary()
    return dt, seed, s["uprooting_rate_per_ha_yr"], s["death_rate_per_ha_yr"], s["tree_throw_share_of_flux"], \
        s["final_roughness_m"]


def single_event_volume(sp_key, dx, R, H):
    cfg = tt.HillslopeConfig(length=20, width=12, slope=0.3, dx=dx, years=1, soil_model=False,
                             infill_fraction=0.0, fresh_diffusivity=0.0, seed=1)
    m = tt.TreeThrowModel(tt.SPECIES[sp_key], cfg)
    ev = m.uproot(10.0, 6.0, R, H, 0.0)
    exact = tt.pit_volume_factor(m.sp.pit_shape_exponent) * math.pi * R * R * H
    return ev["volume"] / exact, ev["moment"] / ev["volume"]


if __name__ == "__main__":
    base = dict(length=40, width=20, slope=0.35, years=1500, seed=11, n_snapshots=2)
    cases = [
        ("deciduous default", "deciduous", base),
        ("spruce default", "spruce", base),
        ("aspen default", "aspen", base),
        ("pine fire", "pine", dict(base, disturbance="fire", first_disturbance=500)),
        ("spruce storms", "spruce", dict(base, disturbance="windstorm", disturbance_interval=100,
                                         disturbance_severity=0.3)),
        ("deciduous nonlinear steep", "deciduous", dict(base, slope=0.9, transport_law="nonlinear")),
        ("deciduous baselevel+wind", "deciduous", dict(base, baselevel_lowering_rate=1e-4, wind_concentration=2,
                                                       wind_direction_deg=90)),
        ("spruce thin soil", "spruce", dict(base, initial_soil_depth=0.15)),
        ("aspen root cycle", "aspen", dict(base, root_cycle=True)),
        ("pine no soil", "pine", dict(base, soil_model=False)),
        ("spruce narrow plot", "spruce", dict(base, width=3.0)),
        ("deciduous dt=5", "deciduous", dict(base, dt=5.0, record_interval=10)),
    ]
    print("== Volume budget (hillslope rows) ==")
    with ProcessPoolExecutor(4) as ex:
        for name, dz, exp, clip, nev, secs, finite, hmin in ex.map(run_budget, cases):
            print(f"{name:28s} dV {dz:10.4f}  expected {exp:10.4f}  residual {dz - exp:+.2e} m3  "
                  f"clip {clip:.2e}  events {nev:5d}  {secs:5.1f}s  finite {finite}  h_min {hmin:.3f}")

    print("\n== Single-event pit volume vs analytic, by dx ==")
    for sp_key in ("aspen", "deciduous", "spruce"):
        sp = tt.SPECIES[sp_key]
        R, H = 0.5 * sp.root_plate_diameter, sp.root_plate_depth
        row = []
        for dx in (0.1, 0.25, 0.5, 1.0):
            r, mom = single_event_volume(sp_key, dx, R, H)
            row.append(f"dx {dx}: vol x{r:.3f} disp {mom:.2f} m")
        a = R / math.sqrt(sp.pit_aspect)
        print(f"{sp_key:10s} R={R:.2f}  " + " | ".join(row) +
              f" | analytic disp {a + tt.pit_centroid_depth_fraction(sp.pit_shape_exponent) * H:.2f}")

    print("\n== Grid convergence (deciduous, 3 seeds) ==")
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(run_dx, [(dx, s) for dx in (0.125, 0.25, 0.5) for s in range(3)]))
    for dx in (0.125, 0.25, 0.5):
        rr = np.array([r[2:] for r in res if r[0] == dx])
        print(f"dx {dx:5.3f}: D_tt {rr[:, 0].mean():.2e}  share {100 * rr[:, 1].mean():.1f}%  "
              f"roughness {100 * rr[:, 2].mean():.1f} cm  event vol {rr[:, 3].mean():.2f}")

    print("\n== Time-step sensitivity (spruce, 3 seeds) ==")
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(run_dt, [(dt, s) for dt in (1.0, 2.0, 5.0) for s in range(3)]))
    for dt in (1.0, 2.0, 5.0):
        rr = np.array([r[2:] for r in res if r[0] == dt])
        print(f"dt {dt:3.0f}: uproot {rr[:, 0].mean():.2f}/ha/yr  deaths {rr[:, 1].mean():.2f}  "
              f"share {100 * rr[:, 2].mean():.1f}%  roughness {100 * rr[:, 3].mean():.1f} cm")

    print("\n== Determinism ==")
    a = tt.TreeThrowModel(tt.SPECIES["aspen"], tt.HillslopeConfig(length=20, width=10, years=200, seed=5))
    b = tt.TreeThrowModel(tt.SPECIES["aspen"], tt.HillslopeConfig(length=20, width=10, years=200, seed=5))
    c = tt.TreeThrowModel(tt.SPECIES["aspen"], tt.HillslopeConfig(length=20, width=10, years=200, seed=6))
    for m in (a, b, c):
        m.run(progress=False)
    print("same seed identical:", np.array_equal(a.z, b.z), "| different seed differs:", not np.array_equal(a.z, c.z))

    print("\n== Overlapping disturbances collapse once (relative pit decay after 3 yr) ==")
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
        print(f"{n_ev} coincident event(s): pit depth {d0:.2f} -> {-(m.z - z0).min():.2f} m "
              f"(ratio {-(m.z - z0).min() / d0:.3f})")

    print("\n== summary() cost ==")
    t0 = time.time()
    a.summary()
    print(f"summary(): {time.time() - t0:.2f} s")
