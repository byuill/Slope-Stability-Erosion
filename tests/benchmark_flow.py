"""Hydraulic-only timing including startup ramp and fixed bed refreshes.

python tests/benchmark_flow.py --baseline /tmp/original-RiverHydraulics.py
The optional baseline is the RiverHydraulics.py from the version being compared.
This measures 500 hydraulic updates, not overall simulation speed.
"""
import argparse
import importlib.util
from pathlib import Path
import statistics
import sys
import time
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import HillslopeSim as model
from RiverHydraulics import RiverHydraulics


def measure(cls, solver):
    backend = types.SimpleNamespace(particle_sizes=[.003]*24, particle_kinds=["circle"]*24,
        particle_b_sizes=[.003]*24, particle_planform_areas=[7e-6]*24)
    cfg = types.SimpleNamespace(**vars(model))
    cfg.DOMAIN_X, cfg.SLOPE_HEIGHT, cfg.FLOW_DEPTH = .4, .02, .1
    cfg.HYDRAULIC_SOLVER, cfg.FLOW_UPDATE_INTERVAL = solver, .05
    cfg.STREAM_STARTUP_TIME, cfg.SIMULATE_TURBULENCE = 1., False
    xs = np.linspace(0., .4, 1000)
    bed = .02+.002*np.sin(2*np.pi*xs/.4)
    exposure, active = np.ones(24), np.ones(24, bool)
    timings = []
    for _ in range(3):
        h = cls(cfg, backend)
        start = time.perf_counter()
        for i in range(500):
            h.update(i*.002, xs, bed, exposure, active)
        timings.append(time.perf_counter()-start)
        assert h.discharge_error < 1e-9
    return statistics.median(timings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location("baseline_hydraulics", args.baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline = module.RiverHydraulics
    for solver in ("simple", "advanced"):
        current = measure(RiverHydraulics, solver)
        comparison = ""
        if baseline:
            before = measure(baseline, solver)
            comparison = f"; baseline {before:.4f} s; speedup {before/current:.2f}x"
        print(f"{solver}: {current:.4f} s / 500 updates{comparison}", flush=True)
