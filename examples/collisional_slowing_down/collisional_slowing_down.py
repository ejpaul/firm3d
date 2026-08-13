import time

import numpy as np

# Ensure mpi4py is imported and initialized before firm3d modules
# This ensures the mpi4py C API is available for C++ bindings
from mpi4py import MPI  # noqa: F401

from firm3d.field.boozermagneticfield import (
    InterpolatedBoozerField,
)
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_with_collisions,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    PROTON_MASS,
)
from firm3d.util.functions import in_github_actions, proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

time1 = time.time()

resolution = 10 if in_github_actions else 48  # Resolution for field interpolation
nParticles = 50 if in_github_actions else 5000  # Number of particles to trace
reltol = 1e-4 if in_github_actions else 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-4 if in_github_actions else 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"
tmax = 1e-4 if in_github_actions else 1e-2  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

# Setup logging to redirect output to file
setup_logging(f"stdout_{nParticles}_{resolution}_{comm_size}.txt")

## Setup field interpolation
field = InterpolatedBoozerField.from_booz_xform(
    boozmn_filename,
    degree=order,
    ns=ns_interp,
    ntheta=ntheta_interp,
    nzeta=nzeta_interp,
    comm=comm_world,
)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: 1 - s**5  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV


# D-T cross-section
def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    else:
        return 0


# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

points = initialize_position_profile(field, nParticles, reactivity, comm=comm_world)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world)

# Background plasma the alphas collide with: a 50/50 DT fuel mix and the
# electrons that neutralize it, on the same profiles that set the birth
# distribution above.  n_ref is the on-axis electron density; the ion
# densities halve it so the plasma is quasineutral.  Temperature is in eV
# at this interface, while T(s) above is in keV.
n_ref = 1e20  # m^-3
ne = lambda s: n_ref * nD(s)
Te = lambda s: 1e3 * T(s)

deuterium = ThermalBackground(
    n_profile=lambda s: 0.5 * ne(s),
    T_profile=Te,
    mass=2 * PROTON_MASS,
    charge=ELEMENTARY_CHARGE,
)
tritium = ThermalBackground(
    n_profile=lambda s: 0.5 * ne(s),
    T_profile=Te,
    mass=3 * PROTON_MASS,
    charge=ELEMENTARY_CHARGE,
)
electrons = ThermalBackground(
    n_profile=ne,
    T_profile=Te,
    mass=ELECTRON_MASS,
    charge=-ELEMENTARY_CHARGE,
)

time1 = time.time()
## Trace alpha particles in Boozer coordinates, with collisions, until they
## hit the s = 1 surface.  DP_hmin floors the orbit step: as a particle
## thermalizes the pitch-scattering rate grows as 1/v^3, and without a floor
## the adaptive step grinds down chasing it.
res_tys, res_zeta_hits = trace_particles_boozer_with_collisions(
    field,
    points,
    vpar_init,
    backgrounds=[deuterium, tritium, electrons],
    tmax=tmax,
    mass=mass,
    charge=charge,
    comm=comm_world,
    Ekin=Ekin,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=True,
    abstol=abstol,
    reltol=reltol,
    DP_hmin=1e-10,
    rng_seed=0,
)

time2 = time.time()
proc0_print("Elapsed time for tracing = ", time2 - time1)

## Post-process results.  The collisional output carries the total speed in
## the final column, which the collisionless tracer does not need: without
## collisions the speed is a known invariant, with them it is not.  The
## energy each particle retains is therefore recoverable per particle.
energy_fraction = np.array([(traj[-1, 5] / vpar0) ** 2 for traj in res_tys])
lost = np.array([traj[-1, 0] < tmax for traj in res_tys])
proc0_print(f"Number of particles = {nParticles}")
proc0_print(f"Loss fraction: {np.mean(lost):.3f}")
proc0_print(
    f"Mean energy fraction of confined alphas: {np.mean(energy_fraction[~lost]):.3f}"
)

if verbose and not in_github_actions:
    from firm3d.field.trajectory_helpers import compute_loss_fraction

    times, loss_frac = compute_loss_fraction(res_tys, tmin=1e-5, tmax=tmax)
    import matplotlib

    matplotlib.use("Agg")  # Don't use interactive backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(times, loss_frac)
    axes[0].set_xlim([1e-5, tmax])
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Fraction of lost particles")

    axes[1].hist(energy_fraction[~lost], bins=40)
    axes[1].set_xlabel(r"$E/E_0$ at $t_{max}$ (confined)")
    axes[1].set_ylabel("Number of particles")
    fig.tight_layout()
    fig.savefig("collisional_slowing_down.png")
