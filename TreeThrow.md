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
python TreeThrow.py --compare --length 50 --width 20 --slope 0.35
python TreeThrow.py --species deciduous --no-tree-throw   # diffusion-only control
python TreeThrow.py --species-file my_species.json        # custom species
```

Requires `numpy` and `matplotlib` (`scipy` is optional). Runs headless (Agg backend).

## Processes

| Process | Formulation |
|---|---|
| Background diffusion | `q = -D grad z` (linear) or `D / (1 - (S/Sc)^2)` (nonlinear, `--transport nonlinear`); scaled by `1 - exp(-h/h*)` when soil is tracked |
| Stand dynamics | Every tree has a position, an age, a lifespan drawn from N(mean, sd), and a mature root-plate radius/depth drawn from correlated lognormals. Root plates grow as `1 - exp(-3 age / maturity_age)`. Dead trees are replaced after an exponential regeneration delay, so stand density stays constant. |
| Uprooting | A fraction `uproot_fraction` of deaths uproot. The rest snap or die standing and move no soil. |
| Fall direction | von Mises. Its mean vector is `fall_slope_bias * (local gradient over ~2R)` plus an optional wind vector (`--wind-dir`, `--wind-kappa`). |
| Pit | Paraboloid of radius R and depth H. It is limited by soil thickness, and a fraction (`bedrock_pluck_fraction`) of root penetration below the soil plucks bedrock. That rock becomes soil, dilated by `rho_r/rho_s`. |
| Mound | The root plate rotates about a hinge at the pit edge. Soil is deposited as an elliptical mound centred `R + H/3` from the stump along the fall line. A fraction `infill_fraction` falls back into the pit. |
| Soil production | `P = P0 exp(-h/h0)` |
| Boundaries | Ridge (x = 0): no flux; soil thrown over the divide is mirrored back (symmetric divide). Channel (x = L): fixed or lowering elevation, removes all sediment delivered to it. Lateral boundaries are periodic. |

## Species parameter sets

| Parameter | pine | deciduous | aspen |
|---|---|---|---|
| Mean lifespan (yr) | 200 ± 60 | 250 ± 75 | 90 ± 25 |
| Stand density (stems/ha) | 350 | 220 | 1100 |
| Uprooting share of mortality | 0.35 | 0.50 | 0.30 |
| Mature root-plate diameter (m) | 2.4 | 3.2 | 1.5 |
| Mature root-plate depth (m) | 1.0 | 0.75 | 0.5 |
| Maturity age (yr) | 80 | 100 | 40 |
| Regeneration delay (yr) | 10 | 5 | 2 |
| → Uprootings (/ha/yr) | 0.58 | 0.43 | 3.59 |
| → Mean event volume (m³) | 2.6 | 3.6 | 0.50 |
| → Expected tree-throw D at S = 0.35 (m²/yr) | 2.1e-4 | 2.7e-4 | 1.5e-4 |

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
  volume, volume exported to the channel).
* `summary.json`: configuration, mean fluxes by process, the emergent
  tree-throw diffusivity (`q_tt / S`) next to its Monte-Carlo analytic
  expectation, sediment yield, and soil produced by root plucking.
* `final_state.npz`: elevation, soil and bedrock grids.

`--compare` also writes `species_comparison.png` and `comparison_summary.json`.

## Checks

`python -m pytest tests/test_tree_throw.py` checks:

* volume conservation per event, including bedrock dilation, channel export and the mirrored divide;
* the diffusion volume balance and the linear flux `D·S`;
* the nonlinear flux enhancement and convex crest formation;
* simulated vs. analytic tree-throw diffusivity and event rate;
* seed reproducibility and the CLI.

## Literature validation

`python validation/run_validation.py` (about 1 minute on 4 cores; `--quick` for a smoke test)
compares the model against published pit-mound volumes, uprooting rates, flux shares,
flux magnitudes and mound-decay times. Results, figures and the benchmark list are in
[`validation/results/VALIDATION_REPORT.md`](validation/results/VALIDATION_REPORT.md).
The benchmark values were transcribed from abstracts and search summaries, so check
them against the original papers.
