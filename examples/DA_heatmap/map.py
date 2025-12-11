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
from firm3d.field.trajectory_helpers import PassingPerturbedPoincare, PassingPoincare, compute_peta, MapPhaseSpace
from firm3d.field.boozermagneticfield import ShearAlfvenHarmonic, ShearAlfvenWave, ShearAlfvenWavesSuperposition
import time 
from firm3d.field.tracing_helpers import initialize_position_profile, initialize_position_uniform_surf
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


boozmn_filename = "boozmn_betaQH.nc"
AE_filename = "QH_10harmonics_scale0_00464159.npy"
folder = 'figs'
harmonic = 7

mpl.rcParams['font.size'] = 16          # base font size
mpl.rcParams['axes.labelsize'] = 20     # x/y labels
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['figure.titlesize'] = 18

order = 3
degree = 3
resolution = 50
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation

helicity_M = 1
helicity_N = -4
helicity_Mp = 0
helicity_Np = -1

bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, helicity_M=helicity_M, helicity_N=helicity_N, comm=comm)
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,

)


AE_temp = AE3DEigenvector.load_from_numpy(AE_filename)
omega =  np.sqrt(AE_temp.eigenvalue)*1000
Phihat = (AE_temp.s_coords, AE_temp.harmonics[harmonic].amplitudes)
Phihat = max(np.abs(Phihat[1]))# * 50
Phin = AE_temp.harmonics[harmonic].n
Phim = AE_temp.harmonics[harmonic].m

saw = ShearAlfvenHarmonic(Phihat, Phim=Phim, Phin=Phin,omega=omega, B0=field, phase=0)
sign_vpar = 1  # 1 for co-passing, -1 for counter-passing
p0_int = 0.0
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
vtotal = np.sqrt(
            2 * Ekin / mass
        ) 
lam = 0.0  
nchi_poinc = 5
ns_poinc = 200
Nmaps = 1500
p0 = np.zeros((1, 3))
p0[0, 0] = p0_int # s

map = PassingPerturbedPoincare(
    saw,
    sign_vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
    Ekin=Ekin,
    p0=p0,
    lam=lam,
    ns_poinc=ns_poinc,
    nchi_poinc=nchi_poinc,
    Nmaps=Nmaps,
    DA_poinc=True,
    nconvergence_points=5,
    comm=comm
)


heat_map = MapPhaseSpace(
    saw,
    Phin,
    Phim,
    omega,
    sign_vpar,
    mass,
    charge,
    Ekin,
    helicity_N,
    helicity_M,
    helicity_Mp,
    helicity_Np,
    randomize_particles = True,
    number_of_particles = 5000,
    comm=comm
)

def compute_rotational_profile(field, pitch, sgn, mass, charge, Ekin, comm):
    # return omega_theta, omega_zeta, radial_position
    poinc = PassingPoincare(field,
            np.abs(pitch),
            sgn,
            mass,
            charge,
            Ekin,
            ns_poinc=50,
            ntheta_poinc=3,
            Nmaps=10,
            comm=comm,
            tmax=1e-2,
            solver_options={'axis':0}
        )
    data = poinc.compute_frequencies()
    data = np.column_stack([data[2], data[0], data[1], [data[0][i]/data[1][i] for i in range(len(data[0]))]])
    profiles = data[data[:, 0].argsort()]
    # return radial_position, omega_theta, omega_zeta, orbit_helicity
    return profiles
def calculate_crossings(drift_helicity, h_res, radial_position):
    diff = drift_helicity - h_res
    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    crossings = []
    for i in sign_changes:
        s = radial_position[i]
        crossings.append(s)
    return crossings


def calculate_QS_resonance(Phim,Phin,M,N,omega,drift_omega_zeta, ell):
    return (Phin - N*Phim - omega / drift_omega_zeta) / (Phim+ell) + N

profile = compute_rotational_profile(field, lam, sign_vpar, mass, charge, Ekin, comm)
drift_omega_zeta = np.mean(profile[:,2])
if verbose: 
    lines = []
    for ell in range(-5, 6):
        h_res = calculate_QS_resonance(Phim, Phin, helicity_M, helicity_N,
                                       omega, drift_omega_zeta, ell)
        r_res = calculate_crossings(profile[:,3], h_res, profile[:,0])
        for elem in r_res:
            lines.append((ell, elem))

    fig, (ax_left,ax_center, ax_right) = plt.subplots(1, 3,
    sharey=True,
    gridspec_kw={"width_ratios": [1, 4, 3]},
    figsize=(25,12)) 
    ax_left.set_xlabel(r"$\hat{\phi}$")
    ax_left.set_ylabel(r"$s$") 

    
    if isinstance(Phihat, tuple):
        filename = f"{folder}/{harmonic}_{lam}_0.004"
        ax_left.plot(Phihat[1][:-1], Phihat[0])
        a, lines_colors = map.plot_poincare(ax = ax_center, filename=filename, lines=lines,s_axis_label=False)
        for elem in lines_colors:
            ell, arr, color = elem
            ax_left.plot([min(Phihat[1]), max(Phihat[1])], [arr, arr], lw=10, color=color)
        heat_map.plot_surfaces(ax = ax_right)
        a.get_figure()
        fig.savefig(filename + ".png", dpi = 400)
    else:
        filename = f"{folder}/flat_{harmonic}_{lam}_{round(Phihat,0)}"
        ax_left.plot([Phihat, Phihat],[0,1])
        a, lines_colors = map.plot_poincare(ax = ax_center, filename=filename, lines=lines,s_axis_label=False)
        a.get_figure()
        for elem in lines_colors:
            ell, arr, color = elem
            ax_left.axhline(y=arr, color=color,  linewidth=10)
        heat_map.plot_surfaces(ax = ax_right)
        fig.savefig(filename + ".png", dpi = 400)