import os
import time
import numpy as np
import pandas as pd
import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
    ALPHA_PARTICLE_CHARGE as CHARGE,
)
from firm3d.util.gpu_utils import boozer_interpolant
from firm3d.util.sampling import sample_stz

np.random.seed(1800)
ALPHA_PARTICLE_SPEED = np.sqrt(2 * ENERGY / MASS)
nparticles = 25000
t_max = 2e-5


def test_boozer_vacuum_vs_finite_beta():
    """For a vacuum field, trajectories using the vacuum routines should match
    those computed using the full finite-beta routines.
    """
    boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"

    ########################################################
    # Vacuum tracing
    ########################################################

    bri = BoozerRadialInterpolant(boozmn_filename, 3, no_K=True)
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
        field, nfp, 15, vacuum=True
    )

    stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
    vpar_inits = ALPHA_PARTICLE_SPEED * np.random.uniform(
        low=-1, high=1, size=nparticles
    )

    print("tracing particles - vacuum case")

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
        tol=1e-9,
        psi0=field.psi0,
        nparticles=nparticles,
        vacuum=True,
    )

    tracing_results = np.reshape(tracing_results, (nparticles, -1))

    particle_data_vacuum = pd.DataFrame(
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
    particle_data_vacuum.to_csv("particle_data_vacuum.csv")

    ########################################################
    # Finite-beta tracing
    ########################################################

    bri = BoozerRadialInterpolant(boozmn_filename, 3)
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )

    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
        field, nfp, 15, vacuum=False
    )
    # Don't make new samples with the updated maxJ; use exactly the same initial
    # conditions as for the vacuum case.

    print("tracing particles - finite-beta case")

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
        tol=1e-9,
        psi0=field.psi0,
        nparticles=nparticles,
    )

    tracing_results = np.reshape(tracing_results, (nparticles, -1))

    particle_data_finite_beta = pd.DataFrame(
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
    particle_data_finite_beta.to_csv("particle_data_finite_beta.csv")

    # Near the s=0 axis, theta is ill-defined, and the tracing is generally less
    # accurate, so don't try to compare points there:
    mask = np.logical_and(
        particle_data_vacuum["s_end"] > 0.15, particle_data_finite_beta["s_end"] > 0.15
    )

    x_end_vacuum = (
        np.sqrt(particle_data_vacuum["s_end"]) * np.cos(particle_data_vacuum["t_end"])
    )[mask]
    y_end_vacuum = (
        np.sqrt(particle_data_vacuum["s_end"]) * np.sin(particle_data_vacuum["t_end"])
    )[mask]
    x_end_finite_beta = (
        np.sqrt(particle_data_finite_beta["s_end"])
        * np.cos(particle_data_finite_beta["t_end"])
    )[mask]
    y_end_finite_beta = (
        np.sqrt(particle_data_finite_beta["s_end"])
        * np.sin(particle_data_finite_beta["t_end"])
    )[mask]

    np.testing.assert_allclose(x_end_vacuum, x_end_finite_beta, atol=0.002)
    np.testing.assert_allclose(y_end_vacuum, y_end_finite_beta, atol=0.002)
    np.testing.assert_allclose(
        particle_data_vacuum["z_end"][mask],
        particle_data_finite_beta["z_end"][mask],
        atol=0.004,
    )
    np.testing.assert_allclose(
        particle_data_vacuum["vpar_end"][mask],
        particle_data_finite_beta["vpar_end"][mask],
        atol=0.005 * ALPHA_PARTICLE_SPEED,
    )
    np.testing.assert_allclose(
        particle_data_vacuum["last_time"][mask],
        particle_data_finite_beta["last_time"][mask],
        rtol=1e-3,
    )


if __name__ == "__main__":
    test_boozer_vacuum_vs_finite_beta()
