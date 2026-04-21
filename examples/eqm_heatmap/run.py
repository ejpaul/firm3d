import time

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
    ShearAlfvenWave,
    ShearAlfvenWavesSuperposition,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_position_uniform_surf,
)
from firm3d.field.trajectory_helpers import (
    MapEquilibrium,
    PassingPerturbedPoincare,
    PassingPoincare,
    compute_peta,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print
from os import makedirs
from firm3d.util.functions import proc0_print, setup_logging
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    verbose = comm.rank == 0
except ImportError:
    comm = None
    comm_size = 1
    verbose = True



boozmn_filename = "../inputs/boozmn_beta2.5_QA.nc"

mpl.rcParams["font.size"] = 14  # base font size
mpl.rcParams["axes.labelsize"] = 14  # x/y labels
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["figure.titlesize"] = 14

order = 3
degree = 3
resolution = 10
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation

helicity_M = 1  # field strength helicity (QA)
helicity_N = 0  # field strength helicity (QA)
helicity_Mp = 0
helicity_Np = -1

bri = BoozerRadialInterpolant(
    boozmn_filename,
    order,
    no_K=True,
    #helicity_M=helicity_M,
    #helicity_N=helicity_N,
    comm=comm,
)  
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

# saw = ShearAlfvenHarmonic(Phihat, Phim=Phim, Phin=Phin,omega=omega, B0=field, phase=0)
sign_vpar = 1  # 1 for co-passing, -1 for counter-passing
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
Ekin = FUSION_ALPHA_PARTICLE_ENERGY


heat_map = MapEquilibrium(
    field,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    plot_s=True,
    helicity_N=helicity_N,
    helicity_M=helicity_M,
    helicity_Mp=helicity_Mp,
    helicity_Np=helicity_Np,
    ns_points=50,
    particles_per_surface=15,
    nlambda_points=50,
    comm=comm,
    savedata=(True, ""),
)

if verbose:
    heat_map.plot_surfaces(savepath="full_field_QS.png")
    plt.clf()
