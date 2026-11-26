# TreeThrow.py — QA/QC record

**Scope.** A review of the model code (`TreeThrow.py`) with quantitative checks, plus fixes, regression tests and a performance pass.

**How to rerun the checks.**

* `python validation/qaqc_checks.py` — about 1 minute on 4 cores; output in `results/qaqc_report.txt`.
* `python -m pytest tests/` — unit and regression tests.
* `python validation/run_validation.py` — literature validation.

## Checks

| Check | Method | Result |
|---|---|---|
| Global volume budget | For 17 configurations, Δ(surface volume) is compared with bedrock-to-soil dilation gain − channel export + clipping correction. The configurations cover the four generic species, fire, storms, nonlinear creep on a steep slope, base-level fall with wind, thin soil, root cycling, no soil model, a narrow plot and dt = 5 yr, plus the keystone presets: oak snag falls, a lodgepole fire, hemlock outbreaks, Douglas-fir storms and ponderosa on thin soil. | Closes to ≤ 4×10⁻¹² m³ in every case. |
| Soil positivity | Soil volume added by clipping negative soil to zero. | Before: up to 11 % of the budget (1.34 of 12 m³ on a 3 m wide plot); 0.24 m³ on a steep nonlinear slope. **After: zero** (≤ 10⁻¹⁹ m³ round-off). |
| Pit geometry | Single-event pit volume and soil displacement vs. analytic values at dx = 0.1–1 m. | Volume exact at every dx; displacement within 0.06 m of analytic. |
| Grid convergence | dx = 0.125, 0.25, 0.5 m; deciduous, 3 seeds. | Tree-throw D, flux share, roughness and event volume agree to ≤ 1 %. |
| Time-step sensitivity | dt = 1, 2, 5 yr; spruce, 3 seeds. | Rates, flux share and roughness agree within seed noise. |
| Determinism | Same seed / different seed. | Identical / different. |
| Overlapping disturbances | Two coincident uprootings vs. one. | Before: overlapping collapse windows (10 % of collapsing cells) collapsed twice per step. **After: identical relative decay (0.513 vs 0.513).** |
| Periodic edge | A pit straddling the side boundary vs. one mid-plot. | Same decay; volume conserved. |
| Numerical faults | NaN or inf in soil and bedrock, now checked every step. | One fault found and fixed (below). Every validation run is now finite. |
| CLI defaults | Defaults of `TreeThrow.py --help` vs. `HillslopeConfig()`. | Identical. |
| Static analysis | pyflakes on model, GUI, animation, validation and tests. | Clean. |
| Calibration | Full literature validation suite before and after the changes. | Same verdicts (48 pass, 4 partial, 13 fail). Calibrated values unchanged (e.g. Z4 spruce pit 0.63 / 0.53 m, D1 flux share 16 %). |

## Defects found and fixed

1. **Fresh pit-mound collapse counted overlapping disturbances twice.**
   * *Cause:* each uprooting had its own collapse window, applied over its whole bounding box. Cells in two windows collapsed twice per step, and each window had artificial no-flux walls.
   * *Fix:* collapse now uses one diffusivity field built from each cell's own disturbance age. It is split into disjoint, 8-connected patches; each patch is solved on its own box and time step, and patches touching the plot sides are solved periodically.
   * *Reference surface:* relief is measured against the row mean of non-collapsing cells, replacing a least-squares plane fitted per window.
2. **Soil could be driven negative and then clipped, silently creating volume.** Two causes:
   * A pit wider than a narrow plot wrapped onto itself, so each duplicate cell cut up to the full soil depth. Duplicates are now summed before cutting.
   * Explicit creep and collapse could export more soil than a cell held. A positivity-preserving flux limiter now scales each face's outflow by its donor cell's available soil.
3. **The flux limiter could divide 0 by 0** when round-off left a cell at −10⁻²⁰ m of soil. This turned one validation run (spruce, seed 103) into NaN. The limiter now uses non-negative soil, and every step checks that the grids stay finite.
4. **Configuration validation gaps.** Negative diffusivity, a zero record interval, a zero A-horizon depth, a rock/soil density ratio below 1, a negative storm age, a negative first-disturbance year and others are now rejected.
5. **Minor fixes:**
   * A full-width feature mapped its columns incorrectly when registering fresh soil (clip instead of wrap).
   * An unused variable and an unused import.
   * The docs described the old window-based collapse.

## Performance

Fresh-collapse windows were 74 % of run time, dominated by per-window plane fits. The changes:

* disjoint patches, each on its own time step;
* one soil-thickness factor per patch per step;
* a fast path that skips the limiter when it cannot bind;
* the explicit sub-step at 0.24 dx²/D (the 2-D stability limit is 0.25);
* collapse ending at 6τ, when 99.75 % is complete, instead of 8τ.

Single runs, 50 × 20 m, 3000 yr, best of two:

| | aspen | spruce | deciduous | pine |
|---|---|---|---|---|
| before | 9.0 s | 6.5 s | 2.5 s | 2.6 s |
| after | 6.3 s | 5.2 s | 2.4 s | 2.0 s |

## Regression tests added (`tests/test_tree_throw.py::QAQCRegressionTests`)

* the global volume budget closes, with soil never negative, under default settings, storms, and steep nonlinear thin-soil conditions;
* a pit wider than the plot does not over-cut;
* overlapping disturbances collapse once;
* collapse across the periodic edge matches collapse mid-plot;
* the limiter is safe with round-off-negative soil;
* invalid configurations are rejected.
