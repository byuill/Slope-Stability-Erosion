"""Reproducible TreeThrow profiles and state fingerprints (no calibration changes).

Run before/after on the same idle machine::

    python validation/profile_tree_throw.py --output ../TreeThrow_review_evidence/before

Each species gets a cProfile run followed by two unprofiled timings.  Fingerprints
cover grids, events and the RNG state; the renderer gets its own frame benchmark.
"""
import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import TreeThrow as tt
import TreeThrowRetro as tr


def fingerprint(m):
    digest = hashlib.sha256()
    for a in (m.h, m.zb, m.fresh_t0, m.age, m.life, np.asarray(m.event_log)):
        digest.update(np.ascontiguousarray(a).tobytes())
    digest.update(json.dumps(m.rng.bit_generator.state, sort_keys=True).encode())
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = tt.HillslopeConfig(length=40, width=40, years=3000, slope=0.35,
                             seed=42, n_snapshots=2, record_interval=50)
    result = {"python": sys.version, "platform": platform.platform(),
              "numpy": np.__version__, "config": tt.asdict(cfg), "species": {}}
    for key, sp in tt.SPECIES.items():
        profiler = cProfile.Profile()
        m = tt.TreeThrowModel(sp, cfg)
        profiler.runcall(m.run, progress=False)
        profiler.dump_stats(str(args.output / f"{key}.prof"))
        with (args.output / f"{key}_profile.txt").open("w", encoding="utf-8") as out:
            stats = pstats.Stats(profiler, stream=out)
            stats.sort_stats("cumulative").print_stats(25)
            stats.sort_stats("tottime").print_stats(15)
        times = []
        for _ in range(args.repeats):
            m = tt.TreeThrowModel(sp, cfg)
            start = time.perf_counter()
            m.run(progress=False)
            times.append(time.perf_counter() - start)
        start = time.perf_counter()
        summary = m.summary()
        cold = time.perf_counter() - start
        start = time.perf_counter()
        m.summary()
        warm = time.perf_counter() - start
        result["species"][key] = {"run_s": times, "best_s": min(times),
                                  "summary_cold_s": cold, "summary_warm_s": warm,
                                  "fingerprint": fingerprint(m), "summary": summary}
        print(f"{key:18s} best {min(times):.3f}s summary {cold:.4f}/{warm:.4f}s", flush=True)
        (args.output / "timings.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    m, frames, meta = tr.run_with_frames(tt.SPECIES["spruce"], tt.HillslopeConfig(
        length=40, width=40, years=300, seed=42), n_frames=40)
    renderer = tr.RetroRenderer(frames, meta)
    start = time.perf_counter()
    pixels = [renderer.render(i) for i in range(len(frames))]
    result["renderer"] = {"frames": len(frames), "seconds": time.perf_counter() - start,
                          "sha256": hashlib.sha256(np.asarray(pixels).tobytes()).hexdigest()}
    (args.output / "timings.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
