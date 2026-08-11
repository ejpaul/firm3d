"""
Thin Python access to the velocity-space collision physics that ships in
``src/firm3dpp/collisions.h`` (which in turn follows ASCOT5's
``mccc_coefs.h``).

Every number here comes from the compiled module through the ``firm3dpp``
bindings; there is no second implementation of the formulae to drift out of
step with the C++, or to agree with it by sharing the same mistake.  Building
firm3dpp is therefore a prerequisite for the tests that use these helpers.

Species are plain tuples ``(m_b, q_b, n_b, T_b)`` in SI units
(kg, C signed, m^-3, J).  ``ThermalBackground`` carries temperature in eV, so
``_backgrounds`` converts on the way in.
"""

import numpy as np

import firm3dpp as sopp
from firm3d.util.constants import ONE_EV


def _backgrounds(species):
    """
    Wrap ``(m_b, q_b, n_b, T_b)`` tuples as ``firm3dpp.ThermalBackground``.

    These profiles are uniform in s, so two grid points suffice and every
    lookup is exact wherever the interpolation lands.
    """
    out = []
    for m_b, q_b, n_b, T_b in species:
        bg = sopp.ThermalBackground()
        bg.s_grid = [0.0, 1.0]
        bg.n_grid = [float(n_b), float(n_b)]
        bg.T_grid = [float(T_b) / ONE_EV, float(T_b) / ONE_EV]
        bg.mass = float(m_b)
        bg.charge = float(q_b)
        out.append(bg)
    return out


def collision_coefficients(v, m_a, q_a, species):
    """
    Summed GC collision coefficients for test-particle speeds ``v`` (array).

    Returns ``(K, D_par, dD_par_dv, nu_D)`` from
    ``compute_collision_coefficients()`` in collisions.h.  The profiles are
    uniform, so everything is evaluated at s = 0.

    Raises ``RuntimeError`` when the profiles give ln Lambda <= 0 -- the C++
    throws ``std::runtime_error`` there, which pybind surfaces as
    ``RuntimeError``.
    """
    v = np.asarray(v, dtype=float)
    bgs = _backgrounds(species)
    flat = v.ravel()
    K = np.empty(flat.size)
    D_par = np.empty(flat.size)
    dD_par = np.empty(flat.size)
    nu_D = np.empty(flat.size)
    for i, vi in enumerate(flat):
        c = sopp.compute_collision_coefficients(
            float(vi), 0.0, float(m_a), float(q_a), bgs
        )
        K[i], D_par[i], dD_par[i], nu_D[i] = c.K, c.D_par, c.dD_par_dv, c.nu_D
    shape = v.shape
    return (
        K.reshape(shape),
        D_par.reshape(shape),
        dD_par.reshape(shape),
        nu_D.reshape(shape),
    )


def evolve_velocity_ensemble(v, xi, m_a, q_a, species, dt, nsteps, rng):
    """
    Velocity-space-only integration of the collisional SDE for an ensemble.

    Each step draws the noise for the whole ensemble at once, then applies the
    kick one particle at a time, since the binding takes a single (v, xi).

    Returns the final ``(v, xi)`` arrays.
    """
    v = np.array(v, dtype=float, copy=True)
    xi = np.array(xi, dtype=float, copy=True)
    bgs = _backgrounds(species)
    sqdt = np.sqrt(dt)
    for _ in range(nsteps):
        dW_v = rng.normal(0.0, sqdt, v.shape)
        dW_xi = rng.normal(0.0, sqdt, v.shape)
        for i in range(v.size):
            c = sopp.compute_collision_coefficients(
                float(v[i]), 0.0, float(m_a), float(q_a), bgs
            )
            v[i], xi[i] = sopp.milstein_collision_step(
                float(v[i]), float(xi[i]), c, dt, float(dW_v[i]), float(dW_xi[i])
            )
    return v, xi
