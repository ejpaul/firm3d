import numpy as np
import time
import sys
import pickle
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
from firm3d.field.tracing import trace_particles_boozer_perturbed, MaxToroidalFluxStoppingCriterion
from firm3d.field.tracing_helpers import initialize_position_profile
from firm3d.field.trajectory_helpers import PassingPerturbedPetaPoincare, PassingPoincare, compute_peta, vpar_func_perturbed
from firm3d.field.boozermagneticfield import ShearAlfvenHarmonic, ShearAlfvenWave, ShearAlfvenWavesSuperposition
import time 
import os
#from helpers import compute_rotational_profile, calculate_QS_resonance, calculate_crossing
from matplotlib import pyplot as plt

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    verbose = comm.rank == 0
except ImportError:
    comm = None
    comm_size = 1
    verbose = True

import numpy as np

perfect = True

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY


boozmn_filename = "boozmn_betaQH.nc"
AE_filename = "/Users/alexandralachmann/Downloads/all_sorts_poinc/QH/5h/QH_5harmonics_scale0_00215443.npy"
path = AE_filename[:-4]+ "_save"



resolution = 50  # Resolution for field interpolation
sign_vpar = 1.0  # sign(vpar). should be +/- 1.
lam = 0.0  # lambda = v_perp^2/(v^2 B) = const. along trajectory
nchi_poinc = 10  # Number of chi initial conditions for poincare
ns_poinc = 150  # Number of s initial conditions for poincare
Nmaps = 1500  # Number of Poincare return maps to compute
p0_int = 0.5
sign = 1  # Sign for parallel velocity when computing initial conditions
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-9  # Tolerance for ODE solver
helicity_M = 1  # helicity of field strength contours
helicity_N = -4
helicity_Mp = 0  # helicity of field strength contours
helicity_Np = -1
degree = 3  # Degree for Lagrange interpolation
tmax = 1e-2
Nparticles = 20000

time1 = time.time()

if perfect: bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, helicity_M=helicity_M, helicity_N=helicity_N, comm=comm)
else: bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm)
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,

)


if perfect:
    bri_p = bri
    field_p = field
else:
    bri_p = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, helicity_M=helicity_M, helicity_N=helicity_N, comm=comm)
    field_p = InterpolatedBoozerField(
        bri_p,
        degree,
        ns_interp=ns_interp,
        ntheta_interp=ntheta_interp,
        nzeta_interp=nzeta_interp,
    )
#profile = compute_rotational_profile(field, lam, sgn, mass, charge, Ekin, comm)
#if verbose:
#    np.savetxt(f'{folder}rotational_profile_QH_{lam * sgn}.txt', profile)

# SAW parameters
#Phihat = 1000
#Phim = -15
#Phin = 13
#omega = 133e3

#mode = 0
#AE = AE3DEigenvector.load_from_numpy(AE_filename)

#scaling_factor = max(np.abs(AE.harmonics[mode].amplitudes)) / amplitude

#Phihat = (AE.s_coords,scaling_factor * AE.harmonics[mode].amplitudes)
#Phihat = amplitude
#Phin = AE.harmonics[mode].n
#Phim = AE.harmonics[mode].m
#omega = np.sqrt(AE.eigenvalue)*1000


Phihat = 1e4
Phin = -2
Phim = -1
omega = 133e3



saw = ShearAlfvenHarmonic(Phihat, Phim=Phim, Phin=Phin,omega=omega, B0=field, phase=0)


print(f'{Phin=}\t{Phim=} \t omega={omega}')

if isinstance(Phihat, tuple):
    protypical_fname = f"m={Phim}_n={Phin}_w={round(omega/1000, 2)}_phiprofile={round(np.abs(max(Phihat[1])),2)}_QH"
else:
    protypical_fname = f"m={Phim}_n={Phin}_w={round(omega/1000, 2)}_phiflat={round(Phihat,2)}_QH"

p0 = np.zeros((1, 3))
p0[0, 0] = p0_int # s
v0 = np.sqrt(2 * Ekin / mass)  # Total velocity from kinetic energy
mu = 0.5 * lam * v0**2  # mu = vperp^2/(2 B)
saw.B0.set_points(p0)
modB = saw.B0.modB()[0, 0]
if 1 - lam * modB < 0:
    raise ValueError(
        "Invalid parameter p0: 1 - lambda * modB must be non-negative."
    )
vpar = sign * v0 * np.sqrt(1 - lam * modB)  # Parallel velocity
Peta0 = compute_peta(
    saw.B0,
    p0,
    vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
)
nprime = (Phim * helicity_N - Phin * helicity_M) / (
            helicity_Np * helicity_M - helicity_N * helicity_Mp
        )
Eprime = nprime * Ekin - omega * Peta0
Eprime = Eprime[0]


convergence_test_indicies = 5
poinc_pert = PassingPerturbedPetaPoincare(
    saw,
    sign_vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
    Ekin=Ekin,
    Eprime=Eprime,
    mu=mu,
    ns_poinc=ns_poinc,
    nchi_poinc=nchi_poinc,
    Nmaps=Nmaps,
    DA_poinc=True,
    nconvergence_points=convergence_test_indicies,
    comm=comm,
    solver_options={'axis':0}
)

linear = lambda s: 2 * (1 - s)

points = initialize_position_profile(
    field_p,
    Nparticles,
    linear,
    comm=comm,
    seed=None,
)

s = points[:, 0]
theta = points[:, 1]
zeta = points[:, 2]

vtot = np.sqrt(2 * Ekin / mass)

s_init = []
theta_init = []
zeta_init = []
vpars_init = []
for i in range(points.shape[0]):
    try:
        vpar = vpar_func_perturbed(
            saw, 
            s[i], 
            theta[i], 
            zeta[i],
            mass,
            helicity_M,
            helicity_N,
            helicity_Mp,
            helicity_Np,
            Eprime,
            mu,
            charge,
            nprime,
            omega,
            sign_vpar=1,)
        s_init.append(s[i])
        theta_init.append(theta[i])
        zeta_init.append(zeta[i])
        vpars_init.append(vpar)
    except RuntimeError:
        continue


mus = np.ones_like(vpars_init) * mu

points_t = np.hstack((np.array([s_init, theta_init, zeta_init]).T, np.zeros((len(s_init), 1))))

peta_init = compute_peta(
    saw,
    points_t,
    np.array(vpars_init),
    mass,
    charge,
    helicity_M,
    helicity_N,
    helicity_Mp=helicity_Mp,
    helicity_Np=helicity_Np,
)

E_init = 0.5 * mass * np.array(vpars_init)**2 + mass * mu * modBs + charge * saw.Phi()[:, 0]


gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
    perturbed_field=saw,
    stz_inits=points_t,
    parallel_speeds=np.array(vpars_init),
    mus=mus,
    tmax=tmax,
    mass=mass,
    charge=charge,
    Ekin=Ekin,
    abstol=1e-9,
    reltol=1e-9,
    forget_exact_path=True,
    comm=comm,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    mode="gc_noK",
    ODE_solver="dormand_prince"
)

petas_final = []
Es_final = []
for i, elem in enumerate(gc_tys):
    elem = elem[-1,:]
    # [t, s, theta, zeta, v_par]

    point = np.zeros((1,4))
    point[0,0] = elem[1]  # s
    point[0,1] = elem[2]  # theta
    point[0,2] = elem[3]  # zeta
    point[0,3] = elem[0]  # t

    vpar = np.array([elem[4]])  # v_par

    saw.set_points(point)
    modBs = saw.B0.modB()[:,0]
    

    peta = compute_peta(
        saw,
        point,
        vpar,
        mass,
        charge,
        helicity_M,
        helicity_N,
        helicity_Mp=helicity_Mp,
        helicity_Np=helicity_Np,
    )
    E_f = 0.5 * mass * vpar**2 + mass * mus[i] * modBs + charge * saw.Phi()[:, 0]
    petas_final.append(peta)
    Es_final.append(E_f)


if verbose: 
    np.savetxt(f'peta_final.txt', np.array(petas_final))
    np.savetxt(f'peta_init.txt', peta_init)
    np.savetxt(f'Es_final.txt', np.array(Es_final))
    np.savetxt(f'E_init.txt', np.array(E_init))


if verbose:
    poinc_pert.plot_poincare(filename=f'peta_poinc.png')
    #omega_theta_prof, omega_zeta_prof, s_prof, omega_zeta_exp_prof, ds_all, diff_eprime, std_eprime = poinc_pert.compute_frequencies()

    plt.clf()
    plt.hist(peta_init, alpha=0.5, bins=50, label=r"initial $p_\eta$")
    plt.hist(petas_final, alpha=0.5, bins=50, label=r"final $p_\eta$")
    plt.legend()
    plt.xlabel(r"$p_\eta$")
    #plt.ylabel("Frequency")
    plt.savefig(f'peta_hist.png')

    plt.clf()
    plt.hist(E_init,alpha=0.5, bins=50, label=r"initial $E$")
    plt.hist(Es_final, alpha=0.5, bins=50, label=r"final $E$")
    plt.legend()
    plt.xlabel(r"$E$")
    #plt.ylabel("Frequency")
    plt.savefig(f'E_hist.png')

"""
    drift_freq = profile[:,1]/ profile[:,2]
    perturbed_drift_freq = omega_theta_prof / omega_zeta_prof
    perturbed_drift_freq_exp = omega_theta_prof / omega_zeta_exp_prof

    print('perturbed_drift_freq_exp:', perturbed_drift_freq_exp)
"""


