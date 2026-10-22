# Stream flow and particle coupling

Run `python HillslopeSim.py` for the GUI or `python HillslopeSim.py --cli` for
the configured simulation. Use a writable Matplotlib cache and `MPLBACKEND=Agg`
for headless CLI runs. Project Chrono comes from the `projectchrono` conda
channel; the unrelated PyPI package named `pychrono` is unsuitable.

## Flow models

Both solvers construct hydrostatic, locally developed open-channel columns on a
bed-relative, stretched vertical grid. They impose no slip at the bed and a
shear-free fixed water surface. Roughness is `z0 = ks / 30`, with `ks` based on
exposed, projected-area-weighted surface D84.

* **Simple:** a rough-wall log core with an outer-layer correction,
  `u = u*/kappa * [(1 + z0/H) log(1 + y/z0) - y/H]`.
  The correction gives zero surface gradient. Its normalized profiles and
  vertical velocities are cached between geometry refreshes.
* **Advanced:** equilibrium 1DV RANS,
  `(nu + l² |du/dy|) du/dy = u*² (1-y/H)`, with
  `l = kappa (y+z0) sqrt(1-y/H)`.
  A safeguarded Newton solve finds friction velocity using the derivative of
  discrete depth-mean velocity. Geometry and integration weights are reused;
  root-finding does not repeatedly construct velocity profiles.

Velocity forcing sets unit discharge to inlet mean velocity times reference
depth. Steady/harmonic stress sets **reference bed stress at the initial depth**;
the resulting reference discharge is carried through all columns. Consequently,
local stress adjusts over an uneven bed. This replaces the previous inconsistent
combination of uniform stress and differing local discharges.

Mean vertical velocity comes from the streamfunction
`psi = integral_bed^z u dz`: `w = -psi_x|eta + u (1-eta) bed_x`, where
`eta = (z-bed)/H`. This conserves discharge and turns flow up/down over rising/
falling beds. Both mean normal velocity boundaries are zero. Geometry is held
quasi-steady between hydraulic refreshes; this is not a moving-free-surface or
full 2D/3D Navier–Stokes solver. Synthetic turbulence remains a calibrated closure,
not resolved eddies. The simple profile's outer-layer correction changes the
stress inferred from a given velocity relative to the previous pure log law.

## Particle loads and switches

The Particle Physics section inside Discretization & Packing controls contact
friction, restitution, cohesion/bonds, buoyancy, fluid drag/traction, and the
additional lift closure. Switches apply on the next run and persist in JSON.
Disabling a process preserves its material coefficient. Cohesion also requires
a cohesive material; additional lift requires a nonzero lift ratio. Turbulence
controls remain in the Stream tab.

Exposed, submerged caps sample both fluid velocity components. Sampling the
unresolved roughness layer is requested explicitly by the grain adapter; ordinary
field queries respect no-slip and return zero velocity outside the wet domain.
Drag uses fluid velocity minus the **application-point** velocity, including
particle spin. Its backward-Euler impulse includes translational and rotational
mobility. The force and its lever arm reach the particle engine together, so the
same impulse supplies consistent linear and angular momentum. Partial-submergence
buoyancy acts at the particle center; shielding and projected area limit loading.

`frames_fluid_reaction` and `frames_fluid_impulse` record the equal-and-opposite
hydraulic reaction and its time integral (excluding hydrostatic buoyancy).
The prescribed-flow reservoir supplies this momentum: the diagnostic does not
feed reaction forces into a dynamic fluid momentum equation. Pymunk outputs
force per channel width (N/m); Chrono outputs force (N).

## Verification

Run `python -m unittest discover -s tests -v`. Tests cover wall/surface boundaries,
discrete discharge, the laminar RANS limit, RANS stress residuals, differential
continuity, repeatable turbulence, dissipative drag impulses, actual Pymunk and
Chrono linear/angular response, process switches, saved configuration, and four
short complete stream runs (both solvers on both engines). Chrono is required
for the complete integration suite.

For reproducible hydraulic-only timing, save the original `RiverHydraulics.py`
outside the checkout and run
`python tests/benchmark_flow.py --baseline /path/to/original-RiverHydraulics.py`.
The benchmark reports medians of three 500-update runs including the startup
ramp. It does not measure total simulation speed or establish experimental
accuracy; entrainment and turbulence closures still need physical calibration.
