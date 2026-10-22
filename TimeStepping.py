"""Motion-limited DEM timesteps; this is not a contact-error estimator."""
import math
import numpy as np


def validate_controls(s):
    values = (s.DT, s.DT_MIN, s.DT_TRAVEL_FRACTION, s.CHRONO_TOLERANCE,
              s.CHRONO_RECOVERY_SPEED, s.CHRONO_BOUNCE_SPEED, s.CHRONO_ENVELOPE_FACTOR)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Numerical controls must be finite.")
    if not (0 < s.DT_MIN <= s.DT and 0 < s.DT_TRAVEL_FRACTION <= .25
            and s.CHRONO_TOLERANCE > 0 and s.CHRONO_RECOVERY_SPEED >= 0
            and s.CHRONO_BOUNCE_SPEED >= 0 and 0 < s.CHRONO_ENVELOPE_FACTOR <= .1):
        raise ValueError("Invalid adaptive timestep or Chrono tolerance/collision controls.")
    if s.CHRONO_SOLVER not in ("PSOR", "APGD", "BARZILAIBORWEIN"):
        raise ValueError("Unknown Chrono contact solver.")


def computational_dt(s, backend, remaining, forces=None, active=None, time_limit=math.inf,
                     enforce_safety=False, travel_fraction=None):
    """Predict surface travel using v, omega*r and an acceleration bound.

    NSC contact impulses have no elastic contact period. Bound grain travel,
    resolve hydraulic timescales, and fail rather than silently violating dt_min.
    A convergence study is still required for impact/entrainment accuracy.
    """
    maximum = min(s.DT, remaining, time_limit)
    if not s.ADAPTIVE_DT and not enforce_safety:
        return min(s.DT, remaining)
    mask = np.ones(backend.num_particles(), bool) if active is None else active
    ids = np.flatnonzero(mask)
    if not len(ids):
        return maximum
    velocities = backend.get_velocities()[ids]
    diameter = np.asarray(backend.particle_sizes)[ids]
    csize = np.asarray(backend.particle_c_sizes)[ids]
    thickness = np.minimum(diameter, csize)
    radius = np.array([d/2 if backend.particle_kinds[i] == "circle" else math.hypot(d, c)/2
                       for i, d, c in zip(ids, diameter, csize)])
    bodies = [backend.particle_bodies[i] for i in ids]
    if hasattr(backend, "space"):
        spin = np.array([abs(b.angular_velocity) for b in bodies])
        mass = np.array([b.mass for b in bodies])
        inertia = np.array([b.moment/s.PX_PER_M**2 for b in bodies])
    else:
        spin = np.array([b.GetAngVelLocal().Length() for b in bodies])
        mass = np.array([b.GetMass() for b in bodies])
        inertia = np.array([min(b.GetInertiaXX().x, b.GetInertiaXX().y, b.GetInertiaXX().z) for b in bodies])
    # Sphere orientation does not change collision geometry. NSC handles
    # frictional spin implicitly; pure sphere spin is not translational travel.
    rotates_geometry = np.array([backend.particle_kinds[i] != "circle" for i in ids])
    speed = np.linalg.norm(velocities, axis=1)+spin*radius*rotates_geometry
    acceleration = np.full(len(ids), s.GRAVITY_MAG)
    if forces is not None:
        load = np.linalg.norm(forces[ids], axis=1)
        acceleration += load/mass+rotates_geometry*load*radius**2/np.maximum(inertia, 1e-30)
    fraction = s.DT_TRAVEL_FRACTION if s.ADAPTIVE_DT else .25
    if travel_fraction is not None:
        fraction = min(fraction, travel_fraction)
    distance = fraction*thickness
    bound = 2*distance/(speed+np.sqrt(speed*speed+2*acceleration*distance))
    desired = min(s.DT, time_limit, float(bound.min()))
    if desired < s.DT_MIN*(1-1e-10) and remaining > desired*(1+1e-10):
        grain = int(ids[int(np.argmin(bound))])
        limit = "hydraulic timescale" if time_limit < float(bound.min()) else f"grain {grain} travel"
        raise ValueError(f"Adaptive dt requires {desired:.3g} s, below minimum {s.DT_MIN:.3g} s. "
                         f"Limiting criterion: {limit}. "
                         "Use a longer flow startup ramp / implicit drag for abrupt loading; "
                         "reduce minimum DT or forcing if actual grain motion requires this resolution.")
    return min(maximum, desired)
