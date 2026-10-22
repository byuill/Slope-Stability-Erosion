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
    magnitude = np.hypot(rn, rt)
    hi = dt*k*magnitude
    # Exact isotropic response (zero lever), also a good first guess for
    # torque-bearing grains. Rationalization is stable for weak drag.
    alpha = 2*hi/(1+np.sqrt(1+4*hi*wn))
    isotropic = wt == wn
    for _ in range(64):
        dn, dtangent = 1+alpha*wn, 1+alpha*wt
        un, ut = rn/dn, rt/dtangent
        speed = np.hypot(un, ut)
        residual = alpha-dt*k*speed
        tolerance = 1e-11*np.maximum(np.maximum(alpha, dt*k*speed), 1e-30)
        done = isotropic | (np.abs(residual) <= tolerance)
        if np.all(done):
            break
        derivative = 1+dt*k*np.divide(un*un*wn/dn+ut*ut*wt/dtangent, speed,
                                      out=np.zeros_like(speed), where=speed > 0)
        lo, hi = np.where(residual < 0, alpha, lo), np.where(residual >= 0, alpha, hi)
        candidate = alpha-residual/derivative
        inside = (candidate > lo) & (candidate < hi)
        alpha = np.where(done, alpha, np.where(inside, candidate, .5*(lo+hi)))
    else:
        raise ValueError("Implicit drag impulse did not converge.")
    updated = rn[:, None]*normal/(1+alpha*wn)[:, None]+rt[:, None]*tangent/(1+alpha*wt)[:, None]
    return alpha[:, None]*updated/dt
