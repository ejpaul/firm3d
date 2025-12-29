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
from firm3d.util.functions import proc0_print
from firm3d.field.trajectory_helpers import compute_peta, WBAParticles
from firm3d.field.boozermagneticfield import ShearAlfvenHarmonic, ShearAlfvenWave, ShearAlfvenWavesSuperposition
import time 
from firm3d.field.tracing_helpers import initialize_position_profile, initialize_position_uniform_surf, initialize_velocity_uniform
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
boozmn_filename = "boozmn_beta_QH.nc"
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

# generate for fusionborn and for 
field_perfect.setpoints(fusionborn_points)
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, fusion_nParticles, comm=comm)
mu_init = (vpar0**2 - vpar_init**2) / (2 * perfect_field.modB()[:, 0])

# generate uniformly sampled particles



