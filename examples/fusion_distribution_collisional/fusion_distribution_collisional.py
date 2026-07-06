"""
Reactor-scale alpha-particle slowing-down with Coulomb collisions.

Traces fusion-born alphas in the Wistell-A configuration scaled to
ARIES-CS size and field strength (as in examples/fusion_distribution,
following Bader et al., Nucl. Fusion 61, 116060 (2021)), with Monte
Carlo Coulomb collisions against a D-T-electron background.  The same
birth ensemble is traced twice -- with and without collisions -- to
separate collisional transport from prompt orbit losses.

Profiles (Bader et al. shapes with reactor absolute scales):
    n_D(s) = n_T(s) = 1e20 (1 - s^5)  [m^-3],  n_e = n_D + n_T
    T(s)   = 11.5 (1 - s) + 0.1       [keV], all species

Outputs (written by rank 0):
    loss_fraction_comparison.png : loss fraction vs time, both runs
    slowing_down_spectrum.png    : confined-alpha energy spectrum vs the
                                   classical slowing-down distribution
    lost_energy_histogram.png    : energy of lost alphas at loss time
    results_collisional.npz      : final states + E(t) histories

Run (Perlmutter): sbatch sample_slurm.sh
Local smoke test: python fusion_distribution_collisional.py --smoke
"""

import argparse
import time

import numpy as np

# Ensure mpi4py is imported and initialized before firm3d modules
from mpi4py import MPI  # noqa: F401

from firm3d.field.boozermagneticfield import InterpolatedBoozerField
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_with_collisions,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    ONE_EV,
    PROTON_MASS,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

ELECTRON_MASS = 9.1093837015e-31  # kg

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke", action="store_true", help="tiny local run to verify the pipeline"
)
parser.add_argument("--nparticles", type=int, default=None)
parser.add_argument("--tmax", type=float, default=None)
args = parser.parse_args()

if args.smoke:
    resolution, nParticles, tmax, tol = 16, 8, 2e-3, 1e-6
    boozmn_filename = "../inputs/boozmn_aten_rescaled_low_res.nc"
else:
    resolution, nParticles, tmax, tol = 48, 1024, 1.5e-1, 1e-8
    boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"
if args.nparticles is not None:
    nParticles = args.nparticles
if args.tmax is not None:
    tmax = args.tmax

dt_save = tmax / 150

setup_logging(f"stdout_collisional_{nParticles}_{comm_size}.txt")

## Field interpolation
field = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=3,
    ns=resolution,
    ntheta=resolution,
    nzeta=resolution,
    comm=comm_world,
)

## Fusion birth distribution (Bader et al. 2021 profile shapes)
N0 = 1e20  # core D (and T) density [m^-3]
T0_KEV = 11.5
T_EDGE_KEV = 0.1  # edge floor: keeps ln Lambda and v_th well-defined

nD = lambda s: 1 - min(s, 1.0) ** 5  # normalized density
nT = nD
T_keV = lambda s: T0_KEV * (1 - min(s, 1.0)) + T_EDGE_KEV


def sigmav(T):
    """D-T reactivity shape (Bader et al. 2021)."""
    return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3)) if T > 0 else 0.0


reactivity = lambda s: nD(s) * nT(s) * sigmav(T_keV(s) - T_EDGE_KEV)

points = initialize_position_profile(field, nParticles, reactivity, comm=comm_world)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
v0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(v0, nParticles, comm=comm_world)

## Thermal backgrounds: D, T, electrons (quasineutral)
T_J = lambda s: T_keV(s) * 1e3 * ONE_EV
deuterium = ThermalBackground(
    n_profile=lambda s: N0 * nD(s),
    T_profile=T_J,
    mass=2 * PROTON_MASS,
    charge=ELEMENTARY_CHARGE,
)
tritium = ThermalBackground(
    n_profile=lambda s: N0 * nT(s),
    T_profile=T_J,
    mass=3 * PROTON_MASS,
    charge=ELEMENTARY_CHARGE,
)
electrons = ThermalBackground(
    n_profile=lambda s: N0 * (nD(s) + nT(s)),
    T_profile=T_J,
    mass=ELECTRON_MASS,
    charge=-ELEMENTARY_CHARGE,
)

stopping = [MaxToroidalFluxStoppingCriterion(1.0)]

## Collisional run (E(t) histories retained)
proc0_print(f"Tracing {nParticles} alphas WITH collisions, tmax = {tmax} s ...")
t1 = time.time()
res_coll, _ = trace_particles_boozer_with_collisions(
    field,
    points,
    vpar_init,
    backgrounds=[deuterium, tritium, electrons],
    tmax=tmax,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    tol=tol,
    dt_save=dt_save,
    comm=comm_world,
    rng_seed=1,
    stopping_criteria=stopping,
)
proc0_print(f"  collisional tracing: {time.time() - t1:.0f} s wall")

## Collisionless companion (same ensemble)
proc0_print("Tracing the same ensemble WITHOUT collisions ...")
t1 = time.time()
res_free, _ = trace_particles_boozer(
    field,
    points,
    vpar_init,
    tmax=tmax,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    comm=comm_world,
    abstol=tol,
    reltol=tol,
    stopping_criteria=stopping,
    forget_exact_path=True,
)
proc0_print(f"  collisionless tracing: {time.time() - t1:.0f} s wall")

## Post-processing (rank 0)
if verbose:
    import matplotlib

    from firm3d.field.trajectory_helpers import compute_loss_fraction

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_end_c = np.array([ty[-1, 0] for ty in res_coll])
    E_end_c = np.array(
        [0.5 * mass * ty[-1, 5] ** 2 / (1e6 * ONE_EV) for ty in res_coll]
    )  # MeV
    lost_c = t_end_c < 0.999 * tmax

    tmin_plot = 1e-5
    times_c, lf_c = compute_loss_fraction(res_coll, tmin=tmin_plot, tmax=tmax)
    times_f, lf_f = compute_loss_fraction(res_free, tmin=tmin_plot, tmax=tmax)

    plt.figure()
    plt.loglog(times_c, lf_c, label="with collisions")
    plt.loglog(times_f, lf_f, "--", label="collisionless")
    plt.xlabel("Time [s]")
    plt.ylabel("Fraction of lost alphas")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_fraction_comparison.png", dpi=150)

    # Confined-alpha energy spectrum vs classical slowing-down
    # distribution f(E) ~ sqrt(E) / (E^{3/2} + Ec^{3/2}), E < E0
    plt.figure()
    E_conf = E_end_c[~lost_c]
    plt.hist(E_conf, bins=40, density=True, alpha=0.6, label="confined alphas (end)")
    # E_c = 14.8 A_alpha T_e [sum_i n_i Z_i^2 / (n_e A_i)]^{2/3}
    # (Stix 1972); 50/50 D-T gives the bracket = 5/12.
    Te_keV = T0_KEV * 0.7  # density-weighted average, roughly
    Ec_MeV = 14.8 * 4.0 * (5.0 / 12.0) ** (2.0 / 3.0) * Te_keV / 1e3
    Egrid = np.linspace(0.05, 3.5, 200)
    fE = np.sqrt(Egrid) / (Egrid**1.5 + Ec_MeV**1.5)
    # normalize over the plotted window
    fE /= np.trapezoid(fE, Egrid)
    plt.plot(Egrid, fE, "k:", label="classical slowing-down (shape)")
    plt.xlabel("E [MeV]")
    plt.ylabel("f(E) [1/MeV]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("slowing_down_spectrum.png", dpi=150)

    # Energy of lost alphas
    plt.figure()
    if lost_c.sum():
        plt.hist(E_end_c[lost_c], bins=40)
    plt.xlabel("E at loss [MeV]")
    plt.ylabel("lost alphas")
    plt.tight_layout()
    plt.savefig("lost_energy_histogram.png", dpi=150)

    np.savez(
        "results_collisional.npz",
        t_end=t_end_c,
        E_end_MeV=E_end_c,
        lost=lost_c,
        loss_times=times_c,
        loss_fraction=lf_c,
        loss_times_free=times_f,
        loss_fraction_free=lf_f,
    )
    proc0_print(
        f"lost (collisional): {lost_c.sum()}/{nParticles} "
        f"({100 * lost_c.mean():.1f} %); "
        f"final loss fraction collisionless: {lf_f[-1]:.3f}"
    )
    proc0_print(
        f"confined mean energy at t = tmax: {E_conf.mean():.2f} MeV (birth 3.52 MeV)"
    )
