#!/usr/bin/env python
from math import sqrt

import numpy as np
import pandas as pd

import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.util.constants import ALPHA_PARTICLE_CHARGE as CHARGE
from firm3d.util.constants import ALPHA_PARTICLE_MASS as MASS
from firm3d.util.constants import FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
from firm3d.util.gpu_utils import boozer_interpolant
from firm3d.util.sampling import sample_stz

### CREATE A FIELD FOR TRACING
boozmn_filename = "examples/inputs/boozmn_aten_rescaled.nc"
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

srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, nfp, 15)


# set seed for consistency
np.random.seed(8)

# trace particles
nparticles = 25000

stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
vpar = np.sqrt(2 * ENERGY / MASS)

vpar_inits = vpar * np.random.uniform(low=-1, high=1, size=nparticles)

print("tracing particles")

# trace on GPU
last_time = firm3dpp.boozer_gpu_tracing(
    quad_pts=quad_info,
    srange=srange,
    trange=trange,
    zrange=zrange,
    stz_init=stz_inits,
    m=MASS,
    q=CHARGE,
    vtotal=sqrt(2 * ENERGY / MASS),
    vtang=vpar_inits,
    tmax=1e-4,
    tol=1e-9,
    psi0=field.psi0,
    nparticles=nparticles,
)

last_time = np.reshape(last_time, (nparticles, 5))


particle_data = pd.DataFrame(
    {
        "s_start": stz_inits[:, 0],
        "t_start": stz_inits[:, 1],
        "z_start": stz_inits[:, 2],
        "vpar_start": vpar_inits,
        "s_end": last_time[:, 1],
        "t_end": last_time[:, 2],
        "z_end": last_time[:, 3],
        "vpar_end": last_time[:, 4],
        "last_time": last_time[:, 0],
    }
)
particle_data.to_csv("./examples/gpu_boozer_tracing/particle_data.csv")


did_leave = [t < 1e-4 for t in particle_data["last_time"]]
loss_frac = sum(did_leave) / len(did_leave)
print(f"Number of particles= {nparticles}")
print(f"Loss fraction: {loss_frac:.3f}")
