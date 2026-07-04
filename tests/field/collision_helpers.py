"""
Pure-Python mirror of the velocity-space collision physics in
``src/firm3dpp/collisions.h`` (which in turn follows ASCOT5's
``mccc_coefs.h``).  No compiled firm3d module is required, so tests built
on these helpers run anywhere numpy/scipy are available.

The source of truth for all formulas is ``collisions.h``; if that file
changes these helpers must be updated to match.

Species are plain tuples ``(m_b, q_b, n_b, T_b)`` in SI units
(kg, C signed, m^-3, J).
"""

import numpy as np
from scipy.special import erf

EPS0 = 8.8541878188e-12  # F/m
HBAR = 1.054571817e-34  # J*s (reduced Planck)


def chandrasekhar_G(x):
    """G(x) = [erf(x) - (2x/sqrt(pi)) exp(-x^2)] / (2x^2); G(0) = 0."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    nz = x != 0
    xs = x[nz]
    out[nz] = (erf(xs) - (2.0 * xs / np.sqrt(np.pi)) * np.exp(-(xs**2))) / (2.0 * xs**2)
    return out


def chandrasekhar_G_deriv(x):
    """G'(x) = (2/sqrt(pi)) exp(-x^2) - 2 G(x)/x; G'(0) = 2/(3 sqrt(pi))."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, 2.0 / (3.0 * np.sqrt(np.pi)))
    nz = x != 0
    xs = x[nz]
    out[nz] = (2.0 / np.sqrt(np.pi)) * np.exp(-(xs**2)) - 2.0 * chandrasekhar_G(xs) / xs
    return out


def debye_length(species):
    """Total Debye length: 1/lambda_D^2 = sum_b n_b q_b^2 / (eps0 T_b)."""
    inv_lD_sq = sum(n * q * q / (EPS0 * T) for (_, q, n, T) in species)
    return 1.0 / np.sqrt(inv_lD_sq)


def coulomb_log(v, m_a, q_a, m_b, q_b, T_b, lambda_D):
    """
    ln Lambda = ln(lambda_D / max(b_cl, b_qm)), ASCOT5 convention, no floor.
    Vectorized over v.
    """
    v_th = np.sqrt(2.0 * T_b / m_b)
    m_r = m_a * m_b / (m_a + m_b)
    v_eff_sq = v * v + v_th * v_th
    b_cl = np.abs(q_a * q_b) / (4.0 * np.pi * EPS0 * m_r * v_eff_sq)
    b_qm = HBAR / (2.0 * m_r * np.sqrt(v_eff_sq))
    return np.log(lambda_D / np.maximum(b_cl, b_qm))


def collision_coefficients(v, m_a, q_a, species):
    """
    Summed GC collision coefficients for test-particle speeds v (array).

    Returns (K, D_par, dD_par_dv, nu_D), mirroring
    compute_collision_coefficients() in collisions.h:

        D_par  = Gamma G(x) / v
        nu_D   = Gamma [erf(x) - G(x)] / v^3
        Q      = -(m_a v / T_b) D_par          (Einstein-relation drag)
        K      = Q + dD_par/dv + 2 D_par / v
    """
    v = np.asarray(v, dtype=float)
    lambda_D = debye_length(species)
    K = np.zeros_like(v)
    D_par = np.zeros_like(v)
    dD_par = np.zeros_like(v)
    nu_D = np.zeros_like(v)
    for m_b, q_b, n_b, T_b in species:
        v_th = np.sqrt(2.0 * T_b / m_b)
        lnL = coulomb_log(v, m_a, q_a, m_b, q_b, T_b, lambda_D)
        Gamma = n_b * q_a**2 * q_b**2 * lnL / (4.0 * np.pi * EPS0**2 * m_a**2)
        x = v / v_th
        G = chandrasekhar_G(x)
        Gp = chandrasekhar_G_deriv(x)
        D_par_b = Gamma * G / v
        dD_par_b = Gamma * (Gp / v_th - G / v) / v
        Q_b = -Gamma * G * m_a / T_b
        D_par += D_par_b
        dD_par += dD_par_b
        nu_D += Gamma * (erf(x) - G) / v**3
        K += Q_b + dD_par_b + 2.0 * D_par_b / v
    return K, D_par, dD_par, nu_D


def speed_cutoff(m_a, species):
    """
    Reflecting speed boundary, ASCOT5 style (MCCC_CUTOFF = 0.1 in mccc.h):
    0.1 * sqrt(T_min / m_a) with T_min the coldest active species.
    Mirrors compute_collision_coefficients() in collisions.h.
    """
    active = [T for (_, _, n, T) in species if n > 0 and T > 0]
    return 0.1 * np.sqrt(min(active) / m_a) if active else 0.0


def evolve_velocity_ensemble(v, xi, m_a, q_a, species, dt, nsteps, rng):
    """
    Velocity-space-only integration of the collisional SDE for an ensemble:
    Euler drift + Milstein noise per step for (v, xi), mirroring the
    operator splitting in trace_particles_boozer_with_collisions (with the
    orbital drifts switched off).

    Boundary conditions follow ASCOT5 (mccc_gc_milstein.c): the speed
    reflects off the thermal cutoff speed_cutoff(), which keeps the
    fixed-step scheme away from v -> 0 where the collision coefficients
    diverge; the pitch mirrors at |xi| = 1.

    Returns the final (v, xi) arrays.
    """
    v = np.array(v, dtype=float, copy=True)
    xi = np.array(xi, dtype=float, copy=True)
    v_cut = speed_cutoff(m_a, species)
    sqdt = np.sqrt(dt)
    for _ in range(nsteps):
        K, D_par, dD_par, nu_D = collision_coefficients(v, m_a, q_a, species)
        dW_v = rng.normal(0.0, sqdt, v.shape)
        dW_xi = rng.normal(0.0, sqdt, v.shape)
        v = v + K * dt + np.sqrt(2.0 * D_par) * dW_v + 0.5 * dD_par * (dW_v**2 - dt)
        xi = (
            xi
            - xi * nu_D * dt
            + np.sqrt(np.maximum(nu_D * (1.0 - xi**2), 0.0)) * dW_xi
            - 0.5 * xi * nu_D * (dW_xi**2 - dt)
        )
        v = np.where(v < v_cut, 2.0 * v_cut - v, v)
        xi = np.where(np.abs(xi) > 1.0, np.sign(xi) * (2.0 - np.abs(xi)), xi)
        xi = np.clip(xi, -1.0, 1.0)
    return v, xi
