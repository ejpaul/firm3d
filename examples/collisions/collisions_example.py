"""
Slowing-down of 3.52 MeV fusion alpha particles in a deuterium–tritium plasma.

Demonstrates ``trace_particles_boozer_with_collisions`` using the analytic
Boozer field (BoozerAnalytic) so no equilibrium file is required.

Background species: 50/50 D–T ions plus electrons (quasi-neutrality:
n_e = n_D + n_T = 2 × n_ref/2 = n_ref).

Expected physics:
  * Mean kinetic energy decreases monotonically due to drag.
  * Electrons dominate the drag for MeV-range particles (v_alpha ≪ v_th,e).
  * Pitch-angle distribution isotropises as pitch-angle scattering acts.
"""

import time

import numpy as np

from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_with_collisions,
)
from firm3d.field.tracing import MaxToroidalFluxStoppingCriterion
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    ONE_EV,
    PROTON_MASS,
)
from firm3d.util.functions import proc0_print

ELECTRON_MASS = 9.1093837015e-31  # kg

# ---------------------------------------------------------------------------
# Field setup (analytic near-axis stellarator geometry)
# ---------------------------------------------------------------------------
etabar = 1.0
B0 = 5.0  # T
G0 = 40.0  # m·T / (2π)
psi0 = 0.5  # Wb / rad  (ψ_edge = psi0)
iota0 = 0.4

field = BoozerAnalytic(etabar, B0, 0, G0, psi0, iota0)

# ---------------------------------------------------------------------------
# Thermal-background species: 50/50 D–T at n_D = n_T = 10²⁰ m⁻³, T_peak = 10 keV
# Electrons satisfy quasi-neutrality: n_e = n_D + n_T = 2×10²⁰ m⁻³
# ---------------------------------------------------------------------------
n_ion = 1e20  # m^-3 per ion species
T_peak = 10e3 * ONE_EV  # 10 keV in J

deuterium = ThermalBackground(
    n_profile=lambda s: n_ion * (1 - s**2),
    T_profile=lambda s: T_peak * (1 - 0.8 * s),
    mass=2 * PROTON_MASS,
    charge=ELEMENTARY_CHARGE,
)

tritium = ThermalBackground(
    n_profile=lambda s: n_ion * (1 - s**2),
    T_profile=lambda s: T_peak * (1 - 0.8 * s),
    mass=3 * PROTON_MASS,
    charge=ELEMENTARY_CHARGE,
)

electrons = ThermalBackground(
    n_profile=lambda s: 2 * n_ion * (1 - s**2),  # quasi-neutrality: n_e = n_D + n_T
    T_profile=lambda s: T_peak * (1 - 0.8 * s),
    mass=ELECTRON_MASS,
    charge=-ELEMENTARY_CHARGE,
)

# ---------------------------------------------------------------------------
# Initial conditions: 10 alpha particles near s = 0.3, uniformly isotropic ξ
# ---------------------------------------------------------------------------
nParticles = 10
rng = np.random.default_rng(42)

stz_inits = np.zeros((nParticles, 3))
stz_inits[:, 0] = 0.3  # s
stz_inits[:, 1] = rng.uniform(0, 2 * np.pi, nParticles)  # theta
stz_inits[:, 2] = rng.uniform(0, 2 * np.pi, nParticles)  # zeta

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
vtotal = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
xi_inits = rng.uniform(-1, 1, nParticles)  # isotropic pitch
vpar_inits = xi_inits * vtotal

# ---------------------------------------------------------------------------
# Trace with collisions
# ---------------------------------------------------------------------------
tmax = 2e-3  # 2 ms  (alpha slowing-down time ~ 0.1–1 s in a reactor, but
# this short run demonstrates energy loss and pitch scattering)
dt_save = 1e-5

t0 = time.time()
res_tys, res_hits = trace_particles_boozer_with_collisions(
    field,
    stz_inits,
    vpar_inits,
    backgrounds=[deuterium, tritium, electrons],
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=Ekin,
    tol=1e-8,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    dt_save=dt_save,
    rng_seed=0,
)
proc0_print(f"Tracing time: {time.time() - t0:.2f} s")


# ---------------------------------------------------------------------------
# Post-process: track mean energy and pitch
# ---------------------------------------------------------------------------
def particle_energy(traj, mass):
    """Return (times, kinetic-energy) from a 6-column [t,s,θ,ζ,v∥,v] array."""
    v = traj[:, 5]
    return traj[:, 0], 0.5 * mass * v**2


E0 = FUSION_ALPHA_PARTICLE_ENERGY
for i, traj in enumerate(res_tys):
    times, Ekin_t = particle_energy(traj, ALPHA_PARTICLE_MASS)
    xi_t = traj[:, 4] / traj[:, 5]  # v∥ / v
    dE = (Ekin_t[-1] - E0) / E0
    proc0_print(
        f"Particle {i:2d}: ΔE/E₀ = {dE:.4f},  "
        f"ξ_init = {xi_inits[i]:+.3f},  ξ_final = {xi_t[-1]:+.3f}"
    )

proc0_print("\nDone.")
