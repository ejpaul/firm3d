#!/usr/bin/env python

import os
import logging
import time
import numpy as np
import pandas as pd
from math import sqrt
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
    ALPHA_PARTICLE_CHARGE as CHARGE,
)

import firm3dpp
from firm3d.util.gpu_utils import boozer_interpolant
from firm3d.util.sampling import sample_stz

ALPHA_PARTICLE_SPEED = sqrt(2 * ENERGY / MASS)

# set seed for consistency
np.random.seed(8)

# trace particles
nparticles = 25000

# Maximum tracing time:
t_max = 1e-2


def run_SIMPLE_benchmark(
    equilibrium_filename,
    SIMPLE_results_filename,
    vacuum,
    t_min,
    reltol,
):
    linewidth = 80
    print("\n" + "#" * linewidth)
    print(f"Running SIMPLE benchmark for {os.path.basename(equilibrium_filename)}")
    print("#" * linewidth + "\n")

    # Load SIMPLE benchmark results
    data = np.loadtxt(SIMPLE_results_filename)
    # Drop the first row since it is weird
    data = data[1:, :]
    raw_times = data[:, 0]
    mask = np.logical_and(raw_times >= t_min, raw_times < t_max)
    t = data[mask, 0]
    confined_frac_passing = data[mask, 1]
    confined_frac_trapped = data[mask, 2]
    confined_frac = confined_frac_passing + confined_frac_trapped
    SIMPLE_lost_frac = 1 - confined_frac

    # FIRM3D setup
    bri = BoozerRadialInterpolant(equilibrium_filename, 3, mpol=16, ntor=16, no_K=vacuum)

    nfp = bri.nfp
    degree = 3
    n_metagrid_pts = 15
    srange = (0, 1, n_metagrid_pts)
    thetarange = (0, np.pi, n_metagrid_pts)
    zetarange = (0, 2 * np.pi / nfp, n_metagrid_pts)
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )

    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
        field, nfp, n_metagrid_pts, vacuum=vacuum
    )

    stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
    vpar_inits = ALPHA_PARTICLE_SPEED * np.random.uniform(
        low=-1, high=1, size=nparticles
    )

    print("tracing particles")

    # trace on GPU
    tracing_results = firm3dpp.boozer_gpu_tracing(
        quad_pts=quad_info,
        srange=srange,
        trange=trange,
        zrange=zrange,
        stz_init=stz_inits.copy(),
        m=MASS,
        q=CHARGE,
        vtotal=ALPHA_PARTICLE_SPEED,
        vtang=vpar_inits.copy(),
        tmax=t_max,
        tol=1e-7,
        psi0=field.psi0,
        nparticles=nparticles,
        vacuum=vacuum,
    )

    tracing_results = np.reshape(tracing_results, (nparticles, -1))

    particle_data = pd.DataFrame(
        {
            "s_start": stz_inits[:, 0],
            "t_start": stz_inits[:, 1],
            "z_start": stz_inits[:, 2],
            "vpar_start": vpar_inits,
            "s_end": tracing_results[:, 1],
            "t_end": tracing_results[:, 2],
            "z_end": tracing_results[:, 3],
            "vpar_end": tracing_results[:, 4],
            "last_time": tracing_results[:, 0],
        }
    )
    particle_data.to_csv(f"particle_data_{os.path.basename(equilibrium_filename)[:-3]}.csv")

    did_leave = [t < t_max for t in particle_data["last_time"]]
    loss_frac = sum(did_leave) / len(did_leave)
    print(f"Number of particles= {nparticles}")
    print(f"Loss fraction: {loss_frac:.3f}")

    last_times = particle_data["last_time"].values
    sorted_times = np.sort(last_times)
    firm3d_loss_fraction = np.arange(1, nparticles + 1) / nparticles
    # Interpolate FIRM3D results to SIMPLE times:
    firm3d_loss_fraction_interp = np.interp(
        t, sorted_times, firm3d_loss_fraction
    )

    if False:
        # Plot results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.loglog(t, SIMPLE_lost_frac, label="SIMPLE benchmark results")
        plt.plot(sorted_times, firm3d_loss_fraction, '.', label="FIRM3D GPU results")
        plt.plot(t, firm3d_loss_fraction_interp, label="FIRM3D GPU results interpolated to SIMPLE times")
        plt.legend(loc=0)
        plt.xlabel("Time (s)")
        plt.ylabel("Loss fraction")
        plt.tight_layout()
        plt.show()

    rel_diff = np.abs(
        firm3d_loss_fraction_interp - SIMPLE_lost_frac
    ) / (0.5 * (SIMPLE_lost_frac + firm3d_loss_fraction_interp))
    print(f"Max relative difference between SIMPLE vs FIRM3D: {np.max(rel_diff):.3f}")

    np.testing.assert_allclose(
        firm3d_loss_fraction_interp, SIMPLE_lost_frac, rtol=reltol
    )

if __name__ == "__main__":
    run_SIMPLE_benchmark(
        equilibrium_filename="tests/test_files/wout_n3are_R7.75B5.7.nc",
        SIMPLE_results_filename="tests/test_files/confined_fraction_ariescs.dat",
        vacuum=False,
        t_min=1e-5,
        reltol=0.4,
    )
    run_SIMPLE_benchmark(
        equilibrium_filename="tests/test_files/wout_beta2.5_QA.nc",
        SIMPLE_results_filename="tests/test_files/confined_fraction_beta2.5_QA.dat",
        vacuum=False,
        t_min=2e-5,
        reltol=0.4,
    )
