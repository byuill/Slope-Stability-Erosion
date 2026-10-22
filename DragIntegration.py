"""Backward-Euler quadratic drag impulses at a rigid grain's application point.

Fluid velocity, exposure and drag coefficient are frozen over the substep.
Contact reactions, gravity and optional lift remain in the DEM integrator.
"""
import numpy as np


def implicit_drag_force(relative, coefficient, mass, inertia, lever, dt):
    """Solve J = dt*k*|r_new|*r_new, r_new = r_old - W J.

    W = I/m + a a^T/I_rotation, a=(-lever_z, lever_x), is the
    application-point mobility. Its eigenvectors are parallel/perpendicular
    to the lever. The scalar bracketed solve is monotone and includes the
    angular response rather than stabilizing translation alone.
    """
    if dt <= 0:
        raise ValueError("Implicit drag requires a positive timestep.")
    r = np.asarray(relative, float)
    k = np.asarray(coefficient, float)
    m, moment = np.asarray(mass, float), np.asarray(inertia, float)
    arm = np.asarray(lever, float)
    length = np.linalg.norm(arm, axis=1)
    normal = np.divide(arm, length[:, None], out=np.tile([1., 0.], (len(arm), 1)), where=length[:, None] > 0)
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    rn = np.sum(r*normal, axis=1)
    rt = np.sum(r*tangent, axis=1)
    wn, wt = 1/m, 1/m+length**2/moment
    lo = np.zeros(len(r))
    hi = dt*k*np.linalg.norm(r, axis=1)
    for _ in range(45):
        alpha = .5*(lo+hi)
        speed = np.hypot(rn/(1+alpha*wn), rt/(1+alpha*wt))
        below = alpha < dt*k*speed
        lo, hi = np.where(below, alpha, lo), np.where(below, hi, alpha)
    alpha = .5*(lo+hi)
    updated = rn[:, None]*normal/(1+alpha*wn)[:, None]+rt[:, None]*tangent/(1+alpha*wt)[:, None]
    return alpha[:, None]*updated/dt
