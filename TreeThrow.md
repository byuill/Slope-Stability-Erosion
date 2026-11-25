# TreeThrow.py — tree-throw hillslope evolution model

A plan-view (x downslope, y across-slope) landscape evolution model for a single
ridge-to-channel hillslope. Soil moves by **background diffusion** and by
**tree throw**. The tree-throw rates come from an individual-based forest whose
species set its lifespan, stand density, uprooting share of mortality and
root-wad (root-plate) size.

```bash
python TreeThrow.py --list-species                       # parameter table
python TreeThrow.py --species pine --length 60 --width 30 --slope 0.4
python TreeThrow.py --species aspen --slope-deg 25 --years 3000
python TreeThrow.py --compare --length 50 --width 20 --slope 0.35   # all four species
python TreeThrow.py --species deciduous --no-tree-throw   # diffusion-only control
python TreeThrow.py --species-file my_species.json        # custom species
```

Requires `numpy` and `matplotlib` (`scipy` is optional). Runs headless (Agg backend).

## Processes

| Process | Formulation |
|---|---|
| Background diffusion | `q = -D grad z` (linear) or `D / (1 - (S/Sc)^2)` (nonlinear, `--transport nonlinear`); scaled by `1 - exp(-h/h*)` when soil is tracked |
| Stand dynamics | Every tree has a position, an age, a lifespan drawn from N(mean, sd), and a mature root-plate diameter and depth drawn from lognormals with negative correlation (`size_depth_correlation`; wide plates are shallow, Zadrożny et al. 2017). Diameter grows as `1 - exp(-3 age / maturity_age)`; depth grows three times faster. Dead trees are replaced after an exponential regeneration delay, so stand density stays constant. |
| Uprooting | A fraction `uproot_fraction` of deaths uproot. The rest snap or die standing and move no soil. |
| Fall direction | von Mises. Its mean vector is `fall_slope_bias * (local gradient over ~2R)` plus an optional wind vector (`--wind-dir`, `--wind-kappa`). |
| Pit | Elliptical paraboloid of equivalent radius R and depth H, `pit_aspect` (default 1.5) times wider across the fall line than along it. It is limited by soil thickness, and a fraction (`bedrock_pluck_fraction`) of root penetration below the soil plucks bedrock. That rock becomes soil, dilated by `rho_r/rho_s`. |
| Mound | The root plate rotates about a hinge at the pit edge. Soil is deposited as an elliptical mound as long as the pit, centred `a + H/3` from the stump along the fall line (a = semi-axis along the fall line). |
| Infill | A fraction of root-plate soil falls back into the pit: `f0 (1 - S_fall / S_crit)`, clipped to 0–0.95. `S_fall` is the ground gradient along the fall direction, `f0 = 0.3` (`--infill`) and `S_crit = tan 47°` (Norman et al. 1995, secondary). Downslope falls shed more soil away from the pit, so displaced volume grows with slope. |
| Pit-mound collapse | Pit, mound and a 0.5 m margin are marked as fresh soil. This soil diffuses with an extra `D_f exp(-age/τ)`, D_f = 0.15 m²/yr and τ = 3 yr (`--fresh-diffusivity`, `--fresh-decay-time`), calibrated to the 1–5 yr spruce chronosequence of Zadrożny et al. (2017). The collapse acts on relief relative to the surrounding surface, inside a window with no-flux borders, so it conserves volume. |
| Soil production | `P = P0 exp(-h/h0)` |
| Boundaries | Ridge (x = 0): no flux; soil thrown over the divide is mirrored back (symmetric divide). Channel (x = L): fixed or lowering elevation, removes all sediment delivered to it. Lateral boundaries are periodic. |

## Species parameter sets

| Parameter | pine | deciduous | aspen | spruce |
|---|---|---|---|---|
| Mean lifespan (yr) | 200 ± 60 | 250 ± 75 | 90 ± 25 | 220 ± 60 |
| Stand density (stems/ha) | 350 | 220 | 1100 | 500 |
| Uprooting share of mortality | 0.35 | 0.55 | 0.30 | 0.60 |
| Mature root-plate diameter (m) | 2.4 | 3.5 | 1.5 | 3.6 |
| Mature root-plate depth (m) | 1.0 | 0.75 | 0.5 | 1.4* |
| Pit width / length | 1.5 | 1.5 | 1.5 | 1.54 |
| Size–depth correlation | −0.6 | −0.6 | −0.6 | −0.8 |
| Maturity age (yr) | 80 | 100 | 40 | 70 |
| Regeneration delay (yr) | 10 | 5 | 2 | 10 |
| → Uprootings (/ha/yr) | 0.58 | 0.47 | 3.59 | 1.30 |
| → Mean displaced volume (m³) | 1.6 | 2.6 | 0.31 | 5.0 |
| → Expected tree-throw D at S = 0.35, uprooting only (m²/yr) | 1.4e-4 | 2.5e-4 | 1.0e-4 | 1.4e-3 |

\*Fresh depth. The model's paraboloid pit collapses to the observed 0.95 m depth within a year.

* **Spruce** is calibrated to 75 uprooted Norway spruce in Babia Góra (Zadrożny et al. 2017). Its pit depth, volume, shape and 1–5 yr degradation, and its size–depth correlation, come from that study.
* **Deciduous** root-plate size is set so displaced volume (2.5 m³) matches published beech and old-growth values (2.4–2.8 m³). Its uprooting share gives an 11–18 % tree-throw share of hillslope flux, as Doane et al. (2021) measured for southern-Indiana hardwoods.
* **Pine** and **aspen** remain literature-informed estimates; no species-specific validation data were available.

These are representative, literature-informed values in round numbers. Calibrate
them against stand inventories and pit-mound surveys before applying them to a
real site. Override any value from the command line (`--density`, `--lifespan`,
`--root-diameter`, `--root-depth`, `--uproot-fraction`). For a completely new
species, supply a JSON file whose keys match the `TreeSpecies` fields.

## Outputs

For each species, in `--output/<species>/`:

* `hillslope_evolution.png`: final hillshade with pit-mound relief, elevation
  change, soil thickness, fall-azimuth rose, profile evolution, the
  tree-throw vs. diffusion flux split, and roughness.
* `events.csv`: every uprooting (time, location, root-plate size, azimuth,
  excavated and displaced volume, volume exported to the channel).
* `summary.json`: configuration, mean fluxes by process (uprooting,
  pit-mound collapse, background diffusion), the emergent tree-throw
  diffusivity (`q_tt / S`) next to its Monte-Carlo analytic expectation,
  sediment yield, and soil produced by root plucking.
* `final_state.npz`: elevation, soil and bedrock grids.

`--compare` also writes `species_comparison.png` and `comparison_summary.json`.

## Checks

`python -m pytest tests/test_tree_throw.py` checks:

* volume conservation per event, including bedrock dilation, channel export and the mirrored divide;
* the diffusion volume balance and the linear flux `D·S`;
* the nonlinear flux enhancement and convex crest formation;
* simulated vs. analytic tree-throw diffusivity and event rate;
* elliptical pits, slope-dependent infill and the size–depth anticorrelation;
* volume conservation, smoothing and switch-off of fresh pit-mound collapse;
* seed reproducibility and the CLI.

## Literature validation

`python validation/run_validation.py` takes about 2 minutes on 4 cores; use `--quick` for a smoke test. It compares the model with:

* pit-mound volumes and slope dependence;
* uprooting rates;
* flux shares and magnitudes;
* pit-mound cover, mound persistence and topographic roughness.

Zadrożny et al. (2017) and Doane et al. (2021) were read in full (`validation/zadrozny.py`, `validation/doane.py`). The other benchmarks come from abstracts and search summaries and are marked *secondary*. Results, figures, the benchmark list and a comparison with the previous model version are in [`validation/results/VALIDATION_REPORT.md`](validation/results/VALIDATION_REPORT.md).
