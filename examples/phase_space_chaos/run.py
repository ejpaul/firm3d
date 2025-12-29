import numpy as np
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
    ALPHA_PARTICLE_CHARGE,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer_perturbed,
)
from firm3d.field.trajectory_helpers import compute_peta, WBAParticles, compute_loss_fraction
from firm3d.field.boozermagneticfield import ShearAlfvenHarmonic, ShearAlfvenWave, ShearAlfvenWavesSuperposition
import time 
from firm3d.field.tracing_helpers import initialize_position_profile, initialize_position_uniform_vol, initialize_velocity_uniform
from matplotlib import pyplot as plt
import matplotlib as mpl

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    verbose = comm.rank == 0
except ImportError:
    comm = None
    comm_size = 1
    verbose = True

resolution = 48  # Resolution for field interpolation
fusion_nParticles = 5000  # Number of fusion born particles to trace
uniform_nParticles = 8000  # Number of uniformly sampled particles to trace
reltol = 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
boozmn_filename = "boozmn_betaQH.nc"
tmax = 1e-2  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

helicity_M = 1
helicity_N = -4
helicity_Mp = 0
helicity_Np = -1

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
vpar0 = np.sqrt(2 * Ekin / mass)

# generate perfect field 

## Setup perfect radial interpolation
bri_perfect = BoozerRadialInterpolant(boozmn_filename, order, helicity_M=helicity_M, helicity_N=helicity_N, no_K=True, comm=comm)

## Setup 3d interpolation
field_perfect = InterpolatedBoozerField(
    bri_perfect,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

# generate 2.5 beta QH field 
bri_beta_QH = BoozerRadialInterpolant(boozmn_filename, order, helicity_M=helicity_M, helicity_N=helicity_N, no_K=True, comm=comm)

## Setup 3d interpolation
field_beta_QH = InterpolatedBoozerField(
    bri_beta_QH,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: (1 - s**5)  # Normalized density
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

fusionborn_points = initialize_position_profile(field_perfect, fusion_nParticles, reactivity, comm=comm)
fusion_vpar_init = initialize_velocity_uniform(vpar0, fusion_nParticles, comm=comm)

field_perfect.set_points(fusionborn_points)
# Initialize uniformly distributed parallel velocities
fusion_perfect_mu_init = (vpar0**2 - fusion_vpar_init**2) / (2 * field_perfect.modB()[:, 0])

field_beta_QH.set_points(fusionborn_points)
# Initialize uniformly distributed parallel velocities
fusion_beta_mu_init = (vpar0**2 - fusion_vpar_init**2) / (2 * field_beta_QH.modB()[:, 0])

# generate uniformly sampled particles
uniform_points = initialize_position_uniform_vol(
            field_perfect,
            uniform_nParticles,
            comm=comm,
            seed=None,
        )
uniform_vpar_init = initialize_velocity_uniform(vpar0, uniform_nParticles, comm=comm)

field_perfect.set_points(uniform_points)
# Initialize uniformly distributed parallel velocities
uniform_perfect_mu_init = (vpar0**2 - uniform_vpar_init**2) / (2 * field_perfect.modB()[:, 0])

field_beta_QH.set_points(uniform_points)
# Initialize uniformly distributed parallel velocities
uniform_beta_mu_init = (vpar0**2 - uniform_vpar_init**2) / (2 * field_beta_QH.modB()[:, 0])

saw_filepaths = ['10h/QH_10harmonics_scale0_001.npy', '10h/QH_10harmonics_scale0_00215443.npy', '10h/QH_10harmonics_scale0_00215443.npy', '10h/QH_10harmonics_scale0_00464159.npy', '10h/QH_10harmonics_scale0_01.npy']

strengths = [0, 0.001, 0.00215443, 0.00464159, 0.01]
uniform_perfect_fraction = []
uniform_beta_fraction = []
fusion_perfect_fraction = []
fusion_beta_fraction = []

# zero pertubation case

AE = AE3DEigenvector.load_from_numpy(saw_filepaths[0])
saw_perfect = ShearAlfvenWavesSuperposition.from_ae3d(AE,
            B0 = field_perfect,
            max_dB_normal_by_B0 = 0,
            minor_radius_meters = 1.7,
            phase=0.0
        )
saw_beta = ShearAlfvenWavesSuperposition.from_ae3d(AE,
            B0 = field_beta_QH,
            max_dB_normal_by_B0 = 0,
            minor_radius_meters = 1.7,
            phase=0.0
        )

res_tys_perfect, res_zeta_hits_perfect = trace_particles_boozer_perturbed(
        saw_perfect,
        fusionborn_points,
        fusion_vpar_init,
        fusion_perfect_mu_init,
        mass=mass,
        charge=charge,
        comm=comm,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True,
        abstol=abstol,
        reltol=reltol,
        tmax=tmax,
    )
res_tys_beta, res_zeta_hits_beta = trace_particles_boozer_perturbed(
    saw_beta,
    fusionborn_points,
    fusion_vpar_init,
    fusion_beta_mu_init,
    mass=mass,
    charge=charge,
    comm=comm,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=True,
    abstol=abstol,
    reltol=reltol,
    tmax=tmax,
)

fusion_perfect_loss_fraction = compute_loss_fraction(res_tys_perfect, tmin=5e-6, tmax=1e-2)

fusion_beta_loss_fraction = compute_loss_fraction(res_tys_beta, tmin=5e-6, tmax=1e-2)

WBAParticles_equilibrium_perfect = WBAParticles(
    saw_perfect,
    uniform_points,
    uniform_vpar_init,
    uniform_perfect_mu_init,
    mass,
    charge,
    Ekin,
    helicity_N,
    helicity_M,
    helicity_Mp,
    helicity_Np,
    savedata=(True, f'DATA/0_perfect_'),
    comm=comm,
    skipped_particles = [],
)

WBAParticles_equilibrium_beta = WBAParticles(
    saw_beta,
    uniform_points,
    uniform_vpar_init,
    uniform_beta_mu_init,
    mass,
    charge,
    Ekin,
    helicity_N,
    helicity_M,
    helicity_Mp,
    helicity_Np,
    savedata=(True, f'DATA/0_beta_'),
    comm=comm,
    skipped_particles = [],
)

fusion_perfect_fraction.append(fusion_perfect_loss_fraction[-1])
fusion_beta_fraction.append(fusion_beta_loss_fraction[-1])
uniform_beta_fraction.append(WBAParticles_equilibrium_beta.compute_fractions())
uniform_perfect_fraction.append(WBAParticles_equilibrium_perfect.compute_fractions())

for saw in saw_filepaths:
    AE = AE3DEigenvector.load_from_numpy(saw)

    saw_perfect = ShearAlfvenWavesSuperposition.from_ae3d(AE,
                B0 = field_perfect,
                max_dB_normal_by_B0 = None,
                minor_radius_meters = 1.7,
                phase=0.0
            )
    saw_beta = ShearAlfvenWavesSuperposition.from_ae3d(AE,
                B0 = field_beta_QH,
                max_dB_normal_by_B0 = None,
                minor_radius_meters = 1.7,
                phase=0.0
            )
    
    res_tys_perfect, res_zeta_hits_perfect = trace_particles_boozer_perturbed(
        saw_perfect,
        fusionborn_points,
        fusion_vpar_init,
        fusion_perfect_mu_init,
        mass=mass,
        charge=charge,
        comm=comm,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True,
        abstol=abstol,
        reltol=reltol,
        tmax=tmax,
    )
    res_tys_beta, res_zeta_hits_beta = trace_particles_boozer_perturbed(
        saw_beta,
        fusionborn_points,
        fusion_vpar_init,
        fusion_beta_mu_init,
        mass=mass,
        charge=charge,
        comm=comm,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True,
        abstol=abstol,
        reltol=reltol,
        tmax=tmax,
    )

    fusion_perfect_loss_fraction = compute_loss_fraction(res_tys_perfect, tmin=5e-6, tmax=1e-2)

    fusion_beta_loss_fraction = compute_loss_fraction(res_tys_beta, tmin=5e-6, tmax=1e-2)

    WBAParticles_perfect = WBAParticles(
        saw_perfect,
        uniform_points,
        uniform_vpar_init,
        uniform_perfect_mu_init,
        mass,
        charge,
        Ekin,
        helicity_N,
        helicity_M,
        helicity_Mp,
        helicity_Np,
        savedata=(True, f'DATA/{saw[:-4]}_perfect'),
        comm=comm,
        skipped_particles = WBAParticles_equilibrium_perfect.wall_lost_indicies,
    )

    WBAParticles_beta = WBAParticles(
        saw_beta,
        uniform_points,
        uniform_vpar_init,
        uniform_beta_mu_init,
        mass,
        charge,
        helicity_N,
        helicity_M,
        helicity_Mp,
        helicity_Np,
        savedata=(True, f'DATA/{saw[:-4]}_beta'),
        comm=comm,
        skipped_particles = WBAParticles_equilibrium_beta.wall_lost_indicies,
    )
    fusion_perfect_fraction.append(fusion_perfect_loss_fraction[-1])
    fusion_beta_fraction.append(fusion_beta_loss_fraction[-1])
    uniform_beta_fraction.append(WBAParticles_equilibrium_beta.compute_fractions())
    uniform_perfect_fraction.append(WBAParticles_equilibrium_perfect.compute_fractions())

np.savetxt('out.txt', np.column_stack([strengths, uniform_perfect_fraction, uniform_beta_fraction, fusion_perfect_fraction, fusion_beta_fraction]))

colors = ['#0a0f18', '#9ba9b2'] 
from matplotlib.lines import Line2D

fig, ax1 = plt.subplots()


ax1.plot(strengths, uniform_perfect_fraction,  color=colors[0])
ax1.plot(strengths, uniform_beta_fraction,color=colors[1])
ax2 = ax1.twinx()
ax2.plot(strengths, fusion_perfect_fraction,  color=colors[0], linestyle='--')
ax2.plot(strengths, fusion_beta_fraction,  color=colors[1], linestyle='--')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\sqrt{\frac{\delta U}{U_0}}$')

custom_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(colors))]

custom_lines2 = [
    Line2D([0], [0], color='k', lw=2, linestyle='-'),
    Line2D([0], [0], color='k', lw=2, linestyle='--')
]
legend1 = ax1.legend(
    handles=custom_lines,
    labels=[r"2.5 % \beta", r"2.5 % \beta perfect QH"],
    title='QH Equilibrium',
    loc='upper left',
    frameon=False
)
legend2 = ax1.legend(
    handles=custom_lines2,
    labels=[r"% Phase Space DA < 3", r"% Fusion Crossed s=1"],
    frameon=False,
    loc='lower right'
)

np.savetxt('out.txt', np.column_stack([strengths, uniform_perfect_fraction, uniform_beta_fraction, fusion_perfect_fraction, fusion_beta_fraction]))