# TreeThrow second QA/QC review

Date: 2026-11-26. Base: `965cad1`; review branch: `codex/treethrow-qaqc-2026-11-26`.

## Outcome and scope

The full literature suite remains **61 PASS / 5 PARTIAL / 13 FAIL**, and all 79 scorecard rows are identical at their displayed precision. No species preset, calibration coefficient, validation threshold, CLI flag or species-JSON field was retuned or removed. The original model's unrelated files were not changed.

**Scientific outputs are not bit-for-bit identical to the original model.** Joining periodic collapse patches and their interacting transport halos fixes an order-dependent numerical error and slightly changes terrain and transport. The nonlinear stability fix changes steep nonlinear runs. The shortened final step changes runs whose duration is not an integer multiple of `dt`. Exact before/after changes are recorded in [`results/qaqc_2_comparison.json`](results/qaqc_2_comparison.json); all nine ordinary analytic expectations remain bit-for-bit identical.

## Findings

Locations below refer to the reviewed branch. Each fixed issue has a regression in `SecondReviewRegressionTests` or `GuiImportTests`.

| Severity | Location | Failing scenario / implication | Fix or recommendation |
|---|---|---|---|
| High | `TreeThrow.py:1145` | A spruce pit at y=6 m and the same pit shifted by 23 cells across the periodic seam start identically after translation, but differ by 0.03886315 m after 3 yr. Ordinary image labels split one periodic disturbance into two independent solves. | Union lateral labels, including diagonal seam neighbours, before solving. Regression compares the entire translated surface at 1e-12 m tolerance. |
| High | `TreeThrow.py:1145` | Two active cells at row 3, columns 4 and 6 share a calm transport cell. On a seeded random surface, sequential old patch solves differ from a simultaneous band solve by 0.09157857 m after 0.3 yr; the result depends on patch ordering. | Label the one-cell transport halos as well as active cells, so interacting patches are solved together. Tests compare seam, isolated and shared-halo patches with a simultaneous solve. |
| High | `TreeThrow.py:1400` | With D=0.02 m2/yr, S=0.95, Sc=1.25 and dx=0.25 m, nonlinear diffusion amplifies a 1-micrometre alternating perturbation by 10.6381x in one year. The old explicit bound used q/S rather than the derivative dq/dS. | Use the maximum gradient-Jacobian eigenvalue D_eff*(1+r)/(1-r), r=|grad z|^2/Sc^2, below the existing mobility cap. Above that cap mobility is constant in gradient. Noise now decays to about 1.9e-12 m in the interior. The original constitutive law and cap are unchanged. |
| Medium | `TreeThrow.py:1482` | years=2.5 and dt=2 advanced to 4 yr, changing events, production and flux normalisation. Floating-point step counts can also include a redundant zero-length step. | Shorten the final step; validate positive finite durations/step sizes before changing state; retain final records/snapshots. Tests include repeated runs and years=0.1+0.2. |
| Medium | `TreeThrow.py:703` | uproot_fraction=0 with no snag pathway, or a valid snag_fall_rate=0 override, raised ZeroDivisionError in summary/GUI diagnostics. A species with no eligible deaths produced NaN fluxes and warnings. | Return zero event rate/flux and infinite turnover for no events; conditional event means remain NaN. A zero snag toppling rate excludes only snag events. Validate sample count and physical input ranges. |
| Medium | `TreeThrow.py:854` | initial_soil_depth=0 or 1 supplied as an integer created an integer soil array; the first production or diffusion update failed with a NumPy casting error. | Allocate the state as floating point explicitly. Both integer initial-depth cases have regressions. |
| Medium | `TreeThrow.py:119` | Positive infinity passed many sign checks (length, lifespan, dt, etc.); NaN wind azimuth was also accepted, causing late failures/non-finite results. | Reject non-finite numeric parameters before allocating grids or drawing random numbers; reject fractional snapshot counts. |
| Medium | `TreeThrowGUI.py:245` | Stop/close was checked during simulation but ignored by the full animation-render loop, which could continue for every remaining frame. | The worker checks Stop between frames and sends only queue messages. It retains the partial model and all frames, leaves remaining images for lazy display, opens the stopped final frame, and avoids automatic playback. Tests cover Stop before work, during simulation, during rendering, and lazy display. |
| Medium | `validation/qaqc_checks.py:61` | The original QA command printed budget and determinism failures but could still return success; its docstring claimed a report was written although the command did not save it. | Conservation, clipping, finite-state and soil-positivity gates now fail the process; determinism is asserted. Save the report even on failure. A manufactured failing-budget test verifies the gate. Convergence remains a quantitative diagnostic, not a newly relaxed criterion. |
| Low / performance | `TreeThrow.py:747` | Every repeated summary reran the same deterministic 200,000-sample expectation (~0.1 s), including during plotting/saving/GUI summaries. | Bounded cache keyed by every argument, with a fresh returned dictionary. Explicit seed=None bypasses caching. Tests verify reuse, changed arguments, dictionary isolation, unseeded redraws and preservation of the model RNG. |
| Low / portability | `tests/test_retro.py:84` | The existing GIF test left its reader open and failed TemporaryDirectory cleanup with WinError 32 on Windows (54 passed / 1 failed baseline). | Close the reader with a context manager. Both original assertions and all test cases remain; no criterion was weakened or skipped. |
| Medium / deferred | `TreeThrow.py:1322` | Disturbances apply at the first population-step start at or after the due time. A fire requested at 203.5 yr applies at 205 yr with dt=5; stochastic recurrence is redrawn from the applied time, so large dt can bias short return intervals and cannot resolve multiple within-step events. | Documented and quantified below. Keep the established scheduling/RNG convention in this review. A future event-time integrator should split steps and explicitly handle tree ages, new snags and multiple events together; that would need its own stochastic/calibration comparison. |
| Low / recommendation | `TreeThrow.py:650` | The public stand_rates() return is the cached dictionary itself. A caller setting rates['uprootings_per_ha_yr']=0 changes a later species.uproot_rate_per_ha lookup for that species, without changing the frozen species. | Treat this return as read-only. A defensive-copy wrapper around the cached implementation would prevent external cache corruption; no reviewed internal caller mutates the dictionary. Kept outside the current numerical and measured-performance fixes. |

The snag audit found no defect in stored `age_death`, the `killed` cause flag, or `cfg.root_decay_time`: a forced-toppling regression now checks all three, including rejection of a tree that died below the minimum uprooting age. Manufactured limiter tests cover both signs on all faces, periodic/non-periodic sides and a fixed channel row. Existing ridge mirroring, fixed/lowering channel and volume-balance tests remain in force.

## Calibration comparison

| Measure | Before | After |
|---|---:|---:|
| Z4 spruce pit depth, 1 yr (m) | 0.940153134296 | 0.940153134296 |
| Z4 spruce pit depth, 3 yr (m) | 0.629963174312 | 0.629963174312 |
| Z4 spruce pit depth, 5 yr (m) | 0.530735301735 | 0.530735301735 |
| D1 deciduous flux share (%) | 15.9321084889 | 15.9323861506 |
| L1 steep lodgepole flux (m2/yr) | 2.79063408751e-06 | 2.79039597529e-06 |
| L2 post-/pre-fire flux ratio | 15.0003717448 | 15.003491126 |
| L3 second pulse (yr) | 51 | 51 |
| S1 rate (/ha/yr), oak | 0.737847222222 | 0.737847222222 |
| S2 uprooting share, oak | 0.303828016393 | 0.303828016393 |
| S3 mortality (/stem/yr), oak | 0.00867435515873 | 0.00867435515873 |
| S4 volume (m3/event), oak | 0.317240748872 | 0.317240594286 |
| S5 area (m2/ha/yr), oak | 2.12036853133 | 2.12036853133 |
| S6 volume (m3/ha/yr), oak | 0.234075205331 | 0.23407509127 |
| G2 Douglas-fir topple share | 0.146350869217 | 0.146350869217 |
| G3 Douglas-fir mortality (/yr) | 0.00332445090439 | 0.00332445090439 |
| G5 Douglas-fir pit volume (m3) | 2.72677310554 | 2.72677032972 |
| V0 lodgepole_pine simulated/analytic D | 0.892533020234 | 0.89253349337 |
| V0 ponderosa_pine simulated/analytic D | 1.50335526099 | 1.50335589185 |

All 79 displayed scorecard rows and all verdicts are unchanged. The JSON comparison lists all 916 unequal numeric leaves, including round-off changes. Some very small transient Gallaway fluxes change by several percent; near zero, relative changes can be large (e.g. -2.54e-10 to -3.90e-9 m2/yr). These are disclosed, not described as bit-for-bit equivalent to the original solver.

The before run reproduces the committed baseline verdicts; platform-dependent float differences in the raw values are at most about 7.1e-12 relative, found in very small transient fluxes. The same-host fresh before run is used for comparisons here.

The conflicting spruce root-plate literature, omitted Gabet & Mudd slope-geometry term, and previously documented V4/V5/V6/HF2/C3 limitations were left unchanged. No retuning was used to improve a verdict.

## Quantitative QA

All 21 global budgets close within 1e-11 m3. Maximum absolute residual is about 4.1e-12 m3 and maximum clipping is about 5.9e-20 m3. All states are finite, active soil remains nonnegative apart from tolerated round-off, and identical seeds give identical grids and event logs. When `soil_model=False`, `h` is an elevation-change storage field, not physical soil thickness, and may be negative.

The full command output is [`results/qaqc_report.txt`](results/qaqc_report.txt). Budget rows:

```text
deciduous default            dV    10.4018  expected    10.4018  residual -3.42e-12 m3  clip 0.00e+00  events    51    1.3s  finite True  h_min 0.190
spruce default               dV    93.2054  expected    93.2054  residual -8.95e-13 m3  clip 0.00e+00  events   239    2.3s  finite True  h_min 0.138
aspen default                dV    10.9303  expected    10.9303  residual -1.18e-12 m3  clip 0.00e+00  events   450    2.8s  finite True  h_min 0.360
pine fire                    dV    15.6906  expected    15.6906  residual -1.65e-12 m3  clip 0.00e+00  events    40    1.2s  finite True  h_min 0.391
spruce storms                dV    77.5740  expected    77.5740  residual -6.54e-13 m3  clip 0.00e+00  events   249    3.1s  finite True  h_min 0.187
deciduous nonlinear steep    dV   -34.5754  expected   -34.5754  residual -2.70e-12 m3  clip 5.82e-20  events    48   38.4s  finite True  h_min 0.000
deciduous baselevel+wind     dV     5.5761  expected     5.5761  residual -3.64e-12 m3  clip 0.00e+00  events    52    2.1s  finite True  h_min 0.324
spruce thin soil             dV   225.8771  expected   225.8771  residual +1.71e-13 m3  clip 0.00e+00  events   243    7.8s  finite True  h_min 0.000
aspen root cycle             dV     8.2766  expected     8.2766  residual -1.10e-12 m3  clip 0.00e+00  events   450    7.9s  finite True  h_min 0.312
pine no soil                 dV   -11.0912  expected   -11.0912  residual -1.07e-14 m3  clip 0.00e+00  events    46    1.6s  finite True  h_min -0.489
spruce narrow plot           dV    12.1825  expected    12.1825  residual -3.13e-13 m3  clip 1.69e-21  events    38    1.5s  finite True  h_min 0.097
deciduous dt=5               dV     8.6535  expected     8.6535  residual +1.37e-12 m3  clip 0.00e+00  events    51    0.8s  finite True  h_min 0.372
oak snags                    dV    12.3369  expected    12.3369  residual -4.11e-12 m3  clip 0.00e+00  events    79    3.5s  finite True  h_min 0.370
lodgepole fire               dV    12.3174  expected    12.3174  residual -1.62e-12 m3  clip 0.00e+00  events   207    5.2s  finite True  h_min 0.441
hemlock outbreaks            dV    11.6734  expected    11.6734  residual -2.67e-12 m3  clip 0.00e+00  events   162    4.7s  finite True  h_min 0.339
douglas_fir storms           dV    13.6556  expected    13.6556  residual -1.77e-12 m3  clip 0.00e+00  events    98    3.7s  finite True  h_min 0.382
ponderosa thin soil          dV    59.5613  expected    59.5613  residual -1.48e-12 m3  clip 0.00e+00  events     4    2.3s  finite True  h_min 0.083
near-critical very thin      dV    28.7633  expected    28.7633  residual -1.88e-12 m3  clip 3.89e-21  events    49   43.0s  finite True  h_min 0.000
bare rock no plucking        dV     0.0000  expected     0.0000  residual +0.00e+00 m3  clip 0.00e+00  events    44    0.4s  finite True  h_min 0.000
thin narrow storms           dV     3.6422  expected     3.6422  residual -1.28e-13 m3  clip 0.00e+00  events    65    1.1s  finite True  h_min 0.001
lodgepole small plates       dV    12.3902  expected    12.3902  residual -1.81e-12 m3  clip 0.00e+00  events   182    5.0s  finite True  h_min 0.452
```

### Small-plate investigation

On reset planar surfaces, 2,000 matched events per grid use identical R=0.4-0.6 m plates, H=R/2, fall azimuths and subcell phases. This isolates rasterisation from mortality, terrain feedback and soil limitation:

| dx (m) | Signed moment, simulation / analytic | Max relative pit-volume error |
|---|---:|---:|
| 0.125 | 1.00001404 | 6.66e-16 |
| 0.25 | 0.99992606 | 6.66e-16 |
| 0.5 | 1.00360484 | 6.66e-16 |

The three-seed landscape grid check gives mean lodgepole D_tt of 9.925e-6, 9.922e-6 and 9.946e-6 m2/yr at dx=0.125, 0.25 and 0.5 m; SE is about 1e-6. Thus the tested 0.4-0.6 m plates show essentially no grid bias at dx=0.25 m, and the landscape mean changes by only ~0.24% over this grid range. This does not explain or resolve the six-seed V0 mean ~0.893. Finite-domain/terrain/population effects and sampling remain candidates; the user-reported longer ensemble (~0.96) was not replaced by a new 16-run ensemble here. Extremely subgrid decayed plates remain subject to the existing radius/depth floors. At dx=1 m the mature lodgepole test already shows a 2.1% volume inflation and poor displacement resolution; that grid should not be interpreted as converged for small plates. No floors or presets were retuned.

### Disturbance and time-step sensitivity

A lodgepole fire requested at 203.5 yr, three seeds and 600 yr runs, produces:

| dt (yr) | Events (/ha/yr) | Tree-throw flux share (%) | Scheduling delay (yr) |
|---|---:|---:|---:|
| 0.5 | 1.92 | 0.5 | 0.0 |
| 1 | 2.05 | 0.8 | 0.5 |
| 2 | 2.00 | 0.5 | 0.5 |
| 5 | 1.88 | 0.6 | 1.5 |

Every disturbance sensitivity run also passes the budget gate. These are stochastic sensitivity measurements, not a claim of exact time-step invariance. The original spruce dt=1/2/5 yr and deciduous dx=0.125/0.25/0.5 m diagnostics are retained in the QA output.

## Profiling and performance

Host: Windows-11-10.0.26200-SP0; Python 3.14.7; NumPy 2.5.3. Each species was profiled with cProfile on a 40 x 40 m, dx=0.25 m, S=0.35, 3000 yr, dt=1 yr, seed=42 run. The table follows the earlier QA/QC record: best of two separate unprofiled runs (initialisation excluded).

| | pine | deciduous | aspen | spruce | douglas_fir | oak | lodgepole_pine | ponderosa_pine | eastern_hemlock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 4.54 s | 4.33 s | 10.42 s | 24.66 s | 10.03 s | 3.59 s | 5.10 s | 2.13 s | 4.50 s |
| corrected, before halo optimisation | 3.96 s | 4.61 s | 9.10 s | 9.59 s | 2.96 s | 4.93 s | 6.35 s | 2.57 s | 6.53 s |
| after | 4.07 s | 4.98 s | 10.30 s | 8.99 s | 3.24 s | 4.16 s | 6.13 s | 2.40 s | 5.14 s |

These observations include correctness changes and machine-load variability; they are not an isolated speedup claim. The original oak repetitions were 11.04 / 3.59 s and spruce was unusually slow at 25.76 / 24.66 s. The stricter patch coupling can cost time for sparse stands. Nonlinear QA runs intentionally cost more because the stable explicit step is smaller (the steep deciduous budget case rose from ~2.1 to ~38.4 s under the measured concurrent workload).

Measured hotspot example (aspen cProfile cumulative seconds):

| Function | Original | Corrected | Optimised |
|---|---:|---:|---:|
| `_fresh_step` | 8.417 | 7.594 | 8.245 |
| `_fresh_patches` | 1.827 | 2.789 | 2.576 |
| `_collapse_patch` | 4.158 | 2.866 | 3.403 |
| `_diffusion_step` | 2.720 | 1.980 | 2.471 |
| `_throw` | 1.057 | 0.825 | 0.980 |
| `_register_fresh` | 0.247 | 0.195 | 0.228 |

The measured fresh-patch overhead was reduced by replacing padded binary dilation with a separable maximum filter (constant ridge/channel boundary, periodic lateral boundary), and skipping unnecessary seam unions. A 3,000-call 100 x 160 Boolean-band microbenchmark measured 0.505 s vs 0.248 s for halo construction. The corrected and optimised full-run fingerprints **and every summary field match exactly for all nine species**, including grid/event/population-age/RNG fingerprints. Renderer pixel hashes also match for that optimisation. The final full six-seed validation results also match the corrected pre-optimisation results bit-for-bit (elapsed-time metadata excluded). This distinguishes a result-preserving optimisation from the preceding bug fixes.

Repeated summary cost (same model, milliseconds):

| Species | Original repeat | After cold | After cached |
|---|---:|---:|---:|
| pine | 157.445 | 139.840 | 0.513 |
| deciduous | 112.396 | 88.861 | 0.403 |
| aspen | 142.793 | 190.033 | 1.970 |
| spruce | 174.161 | 77.360 | 0.576 |
| douglas_fir | 256.842 | 126.783 | 0.182 |
| oak | 88.004 | 83.791 | 0.347 |
| lodgepole_pine | 130.481 | 142.386 | 0.469 |
| ponderosa_pine | 126.248 | 124.094 | 0.150 |
| eastern_hemlock | 120.746 | 115.505 | 0.417 |

The 200,000-sample calculation, draws and arithmetic are unchanged for ordinary seeded calls; the nine analytic dictionaries match the original bit-for-bit. Cache hits return independent dictionaries. A 64-entry bound limits memory; explicitly unseeded calls are not cached.

Renderer benchmark: 39 frames, 0.216 s before; 39 frames, 0.201 s after. No renderer visual/per-frame optimisation was needed relative to the physics hotspots; GUI cancellation was fixed separately. The original-to-corrected pixel hash may change because terrain changes, whereas corrected-to-optimised rendering is identical.

Full validation elapsed time: 261.8 s before / 503.9 s after, 4 workers. These runs shared the machine with some QA work and are reported as observed wall times, not a controlled throughput speedup. Raw timings and fingerprints are in [`results/qaqc_2_timings.json`](results/qaqc_2_timings.json).

## Verification and reproduction

- Original tests on this Windows host: 54 passed / 1 failed, solely the unclosed GIF-reader cleanup. tkinter is installed, so the GUI import test is not skipped here.
- Final targeted suite: **76 passed**, plus 50 parameter subtests. No original test or literature criterion was loosened, skipped or deleted. The GIF reader is closed after the same two assertions.
- Full literature validation before/after: **61 / 5 / 13** both times; 79 scorecard rows unchanged.
- QA: all 21 budgets, positivity, finite-state checks and determinism pass. Grid/time-step diagnostics and small-plate investigation are recorded above.
- `compileall` and `git diff --check` pass. pyflakes was not installed on this host; no clean static-analysis result is claimed.
- GUI worker paths are exercised with queue/event mocks, without a visible desktop smoke test. The scientific simulation and retro capture/render/GIF tests run normally.

```console
python -m pytest tests/test_tree_throw.py tests/test_retro.py
python validation/qaqc_checks.py --workers 4
python validation/run_validation.py --workers 4
python validation/profile_tree_throw.py --output ../TreeThrow_review_evidence/after
```

Raw before/after validation JSON, all changed numeric leaves, cProfile files/text, timings, test logs and probes were captured in the sibling `TreeThrow_review_evidence` directory. The comparison/timing JSON and QA output are committed with this report; the larger raw evidence bundle remains outside the repository. Local commits are on the review branch; nothing was pushed.
