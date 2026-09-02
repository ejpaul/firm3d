#!/usr/bin/env python


import numpy as np
import pandas as pd
import time

from firm3d.catapult.tracing import trace_particles_boozer_gpu
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import in_github_actions

resolution = 5 if in_github_actions else 15  # Resolution for field interpolation
nparticles = 100 if in_github_actions else 1000  # Number of particles to trace
tol = 1e-4 if in_github_actions else 1e-8  # Tolerance for ODE solver


### CREATE A FIELD FOR TRACING
boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"
bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=True)

field = InterpolatedBoozerField(
    bri,
    3,
    ns_interp=resolution,
    ntheta_interp=resolution,
    nzeta_interp=resolution,
)
# set seed for consistency
np.random.seed(8)

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


np.random.seed(0)
# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))
stz_inits = initialize_position_profile(field, nparticles, reactivity, seed=1)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles, seed=1)


tmax = 1e-4
print(stz_inits)
last_time_dbl = trace_particles_boozer_gpu(
    bri,
    stz_inits,
    vpar_inits,
    tmax=tmax,
    mass=mass,
    charge=charge,
    vtotal=vpar0,
    tol=tol,
    ns=resolution,
    ntheta=resolution,
    nzeta=resolution,
)

last_time_flt = trace_particles_boozer_gpu(
    bri,
    stz_inits.astype(np.float32),
    vpar_inits.astype(np.float32),
    tmax=tmax,
    mass=mass,
    charge=charge,
    vtotal=vpar0,
    tol=tol,
    ns=resolution,
    ntheta=resolution,
    nzeta=resolution,
)
particle_data = pd.DataFrame(
    {
        "s_start": stz_inits[:, 0],
        "t_start": stz_inits[:, 1],
        "z_start": stz_inits[:, 2],
        "vpar_start": vpar_inits,
        "s_end_dbl": last_time_dbl[:, 1],
        "t_end_dbl": last_time_dbl[:, 2],
        "z_end_dbl": last_time_dbl[:, 3],
        "vpar_end_dbl": last_time_dbl[:, 4],
        "last_time_dbl": last_time_dbl[:, 0],
        "dt_end_dbl": last_time_dbl[:, 5],
        "s_end_flt": last_time_flt[:, 1],
        "t_end_flt": last_time_flt[:, 2],
        "z_end_flt": last_time_flt[:, 3],
        "vpar_end_flt": last_time_flt[:, 4],
        "last_time_flt": last_time_flt[:, 0],
        "dt_end_flt": last_time_flt[:, 5]
    }
)
particle_data.to_csv("./particle_data.csv")


print(f"Number of particles= {nparticles}")
did_leave = [t < tmax for t in particle_data["last_time_flt"]]
loss_frac = sum(did_leave) / len(did_leave)
print(f"Flt. Loss fraction: {loss_frac:.3f}")
did_leave = [t < tmax for t in particle_data["last_time_dbl"]]
loss_frac = sum(did_leave) / len(did_leave)
print(f"Dbl. Loss fraction: {loss_frac:.3f}")

# tmax = 1e-3
# ## intialize data collection
# loss_frac = []
# precision = []
# wall_clock = []
# n = []
# for nparticles in [100*2**k for k in range(0,15)]:
#     stz_inits = initialize_position_profile(field, nparticles, reactivity, seed=1)
#     vpar_inits = initialize_velocity_uniform(vpar0, nparticles, seed=1)

#     start = time.time()
#     last_time_dbl = firm3dpp.boozer_gpu_tracing(
#         quad_pts=quad_info.astype(np.float64),
#         srange=srange,
#         trange=trange,
#         zrange=zrange,
#         stz_init=stz_inits.astype(np.float64).copy(),
#         m=mass,
#         q=charge,
#         vtotal=vpar0,
#         vtang=vpar_inits.astype(np.float64).copy(),
#         tmax=tmax,
#         tol=tol,
#         dt_in=-np.ones(nparticles).astype(np.float64).astype(stz_inits.dtype),
#         psi0=field.psi0,
#         nparticles=nparticles,
#         vacuum=vacuum,
#     )
#     dbl_time = time.time() - start
#     last_time_dbl = np.reshape(last_time_dbl, (nparticles, 6))
#     loss_frac.append(np.mean(last_time_dbl[:, 0] < tmax))
#     precision.append("double")
#     wall_clock.append(dbl_time)
#     n.append(nparticles)


#     start = time.time()
#     last_time_flt = firm3dpp.boozer_gpu_tracing(
#         quad_pts=quad_info.astype(np.float32),
#         srange=srange,
#         trange=trange,
#         zrange=zrange,
#         stz_init=stz_inits.astype(np.float32).copy(),
#         m=mass,
#         q=charge,
#         vtotal=vpar0,
#         vtang=vpar_inits.astype(np.float32).copy(),
#         tmax=tmax,
#         tol=tol,
#         dt_in=-np.ones(nparticles).astype(np.float32),
#         psi0=field.psi0,
#         nparticles=nparticles,
#         vacuum=vacuum,
#     )
#     flt_time = time.time() - start
#     last_time_flt = np.reshape(last_time_flt, (nparticles, 6))
#     loss_frac.append(np.mean(last_time_flt[:, 0] < tmax))
#     precision.append("single")
#     wall_clock.append(flt_time)
#     n.append(nparticles)

#     df = pd.DataFrame({"nparticles": n,
#                         "loss_frac":loss_frac,
#                         "precision":precision,
#                         "wallclock":wall_clock})
#     df.to_csv("precision_comparison.csv")

# # particle_data = pd.DataFrame(
# #     {
# #         "s_start": stz_inits[:, 0],
# #         "t_start": stz_inits[:, 1],
# #         "z_start": stz_inits[:, 2],
# #         "vpar_start": vpar_inits,
# #         "s_end_dbl": last_time_dbl[:, 1],
# #         "t_end_dbl": last_time_dbl[:, 2],
# #         "z_end_dbl": last_time_dbl[:, 3],
# #         "vpar_end_dbl": last_time_dbl[:, 4],
# #         "last_time_dbl": last_time_dbl[:, 0],
# #         "dt_end_dbl": last_time_dbl[:, 5],
# #         "s_end_flt": last_time_flt[:, 1],
# #         "t_end_flt": last_time_flt[:, 2],
# #         "z_end_flt": last_time_flt[:, 3],
# #         "vpar_end_flt": last_time_flt[:, 4],
# #         "last_time_flt": last_time_flt[:, 0],
# #         "dt_end_flt": last_time_flt[:, 5]
# #     }
# # )
# # particle_data.to_csv("./particle_data.csv")


# # print(f"Number of particles= {nparticles}")
# # did_leave = [t < tmax for t in particle_data["last_time_flt"]]
# # loss_frac = sum(did_leave) / len(did_leave)
# # print(f"Flt. Loss fraction: {loss_frac:.3f}")
# # did_leave = [t < tmax for t in particle_data["last_time_dbl"]]
# # loss_frac = sum(did_leave) / len(did_leave)
# # print(f"Dbl. Loss fraction: {loss_frac:.3f}")