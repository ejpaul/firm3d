"""
firm3d side of the collision validation: trace the same alpha ensemble
through trace_particles_boozer_with_collisions and extract per-particle
E(t) and xi(t) on the same grid as the ASCOT5 run.

Usage:
    python run_firm3d.py --outdir /path/to/workdir

Writes firm3d_moments.npz into outdir.
"""

import argparse
import os
import time as walltime

import numpy as np
import params

from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_with_collisions,
)
from firm3d.field.tracing import MaxToroidalFluxStoppingCriterion


def field():
    """Tokamak-like near-axis configuration at the shared (R0, B0) scale.

    G0 = B0 R0, psi0 = B0 a^2 / 2 with a = 2 m (Bbar = B0 so that
    r = sqrt(2 s psi0 / Bbar) reaches a at s = 1), etabar such that
    etabar * a = 0.5 (tokamak-like mirror ratio).
    """
    a = 2.0
    psi0 = params.B0 * a**2 / 2
    return BoozerAnalytic(
        0.25, params.B0, 0, params.B0 * params.R0, psi0, 1.0, Bbar=params.B0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    T = params.TEMPERATURE_EV * params.EV
    electrons = ThermalBackground(
        n_profile=lambda s: params.DENSITY,
        T_profile=lambda s: T,
        mass=params.M_ELECTRON,
        charge=-params.EV,
    )
    hydrogen = ThermalBackground(
        n_profile=lambda s: params.DENSITY,
        T_profile=lambda s: T,
        mass=params.M_PROTON,
        charge=params.EV,
    )

    n = params.N_MARKERS
    v0 = np.sqrt(2 * params.E0_EV * params.EV / params.M_ALPHA)
    rng = np.random.default_rng(params.SEED + 2)
    stz = np.column_stack(
        [
            np.full(n, 0.25),
            rng.uniform(0, 2 * np.pi, n),
            rng.uniform(0, 2 * np.pi, n),
        ]
    )
    vpar = params.initial_pitches() * v0

    t0 = walltime.time()
    res_tys, _ = trace_particles_boozer_with_collisions(
        field(),
        stz,
        vpar,
        backgrounds=[electrons, hydrogen],
        tmax=params.TMAX,
        mass=params.M_ALPHA,
        charge=2 * params.EV,
        Ekin=params.E0_EV * params.EV,
        tol=1e-8,
        dt_save=params.DT_SAVE,
        rng_seed=params.SEED,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    )
    print(f"firm3d tracing: {walltime.time() - t0:.1f} s wall")

    tgrid = params.DT_SAVE * np.arange(1, params.N_SAVE + 1)
    E = np.full((n, tgrid.size), np.nan)
    XI = np.full((n, tgrid.size), np.nan)
    t_end = np.zeros(n)
    e_end = np.zeros(n)
    for k, ty in enumerate(res_tys):
        t_i = ty[:, 0]
        v_i = ty[:, 5]
        xi_i = ty[:, 4] / ty[:, 5]
        e_i = 0.5 * params.M_ALPHA * v_i**2 / params.EV  # eV
        alive = tgrid <= t_i[-1] + params.DT_SAVE
        E[k, alive] = np.interp(tgrid[alive], t_i, e_i)
        XI[k, alive] = np.interp(tgrid[alive], t_i, xi_i)
        t_end[k] = t_i[-1]
        e_end[k] = e_i[-1]

    # Slowing-down time: interpolated crossing of EMIN (firm3d has no
    # energy end condition, so extract it from the saved trajectories)
    t_slow = np.full(n, np.nan)
    for k, ty in enumerate(res_tys):
        e_i = 0.5 * params.M_ALPHA * ty[:, 5] ** 2 / params.EV
        below = np.nonzero(e_i <= params.EMIN_EV)[0]
        if below.size:
            j = below[0]
            if j > 0:
                frac = (params.EMIN_EV - e_i[j - 1]) / (e_i[j] - e_i[j - 1])
                t_slow[k] = ty[j - 1, 0] + frac * (ty[j, 0] - ty[j - 1, 0])
            else:
                t_slow[k] = ty[0, 0]
    slowed = np.isfinite(t_slow)

    out = os.path.join(args.outdir, "firm3d_moments.npz")
    np.savez(
        out,
        tgrid=tgrid,
        E=E,
        XI=XI,
        t_end=t_end,
        e_end=e_end,
        t_slow=t_slow,
        slowed=slowed,
    )
    print(f"saved {out}")
    print(f"slowed to Emin: {slowed.sum()}/{n}")
    if slowed.sum():
        print(f"mean slowing time: {np.nanmean(t_slow) * 1e3:.3f} ms")
    print(f"analytic 0.5 ts ln(E0/Emin): {params.analytic_slowing_time() * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
