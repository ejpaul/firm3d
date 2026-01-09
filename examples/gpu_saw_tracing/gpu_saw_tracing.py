import numpy as np

import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)

# for SAW wave
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import ALPHA_PARTICLE_CHARGE as CHARGE
from firm3d.util.constants import ALPHA_PARTICLE_MASS as MASS
from firm3d.util.constants import FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
from firm3d.util.gpu_utils import boozer_saw_interpolant
from firm3d.util.sampling import sample_stz

np.random.seed(1800)

### tracing parameters
nparticles = 25000
tmax = 1e-3


### CREATE A FIELD FOR TRACING
boozmn_filename = "examples/inputs/boozmn_aten_rescaled.nc"
bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=True)

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

### SET UP A PERTURBED B FIELD
saw_filename = "./examples/tracing_with_AE/ae.npy"

# generate saw object
saw = ShearAlfvenWavesSuperposition.from_ae3d(
    eigenvector=AE3DEigenvector.load_from_numpy(
        filename=saw_filename,
    ),
    B0=field,
    max_dB_normal_by_B0=5e-3,
    minor_radius_meters=1.7,
)


# set up interpolant data
srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
    field, nfp, n_metagrid_pts
)

stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
VELOCITY = np.sqrt(2 * ENERGY / MASS)
vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (nparticles,))
stz = np.ascontiguousarray(stz_inits)

# get bfield info and SAW data
psi0 = field.psi0
saw_nharmonics = 5

## load saw data as arrays
saw_data = np.load(saw_filename, allow_pickle=True)
saw_data = saw_data[()]
saw_omega = saw.get_wave(0).omega

s = saw.get_wave(0).phihat.get_s_basis()
saw_srange = (s[0], s[-1], len(s))

saw_m = [saw.get_wave(i).Phim for i in range(saw_nharmonics)]
saw_n = [saw.get_wave(i).Phin for i in range(saw_nharmonics)]
saw_phihats = np.ascontiguousarray(
    np.column_stack(
        [
            np.array([saw.get_wave(i).phihat(s_val) for s_val in s])
            for i in range(saw_nharmonics)
        ]
    )
)

last_time = firm3dpp.boozer_saw_gpu_tracing(
    quad_pts=quad_info,
    srange=srange,
    trange=trange,
    zrange=zrange,
    saw_omega=saw_omega,
    saw_srange=saw_srange,
    saw_m=saw_m,
    saw_n=saw_n,
    saw_phihats=saw_phihats,
    saw_nharmonics=saw_nharmonics,
    stz_init=stz,
    m=MASS,
    q=CHARGE,
    vtotal=np.sqrt(2 * ENERGY / MASS),
    vtang=vpar_init,
    tmax=tmax,
    tol=1e-9,
    psi0=psi0,
    nparticles=nparticles,
)


last_time = np.reshape(last_time, (nparticles, 5))
loss_times = last_time[:, 0]
print("loss times: ", loss_times)
print("loss frac: ", np.mean(loss_times < tmax))
