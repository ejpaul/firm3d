import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from simsopt.field import (
    BiotSavart,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
    load_coils_from_makegrid_file,
)
from simsopt.field.sampling import draw_uniform_on_surface
from simsopt.geo import SurfaceRZFourier
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)

from firm3d.catapult.tracing import trace_particles_cartesian_gpu
from firm3d.field.tracing_helpers import (
    initialize_velocity_uniform,
)

degree = 3  # degree of interpolant
n = 16  # resolution of interpolant
order = 12  # order of coil curves

filename = "../inputs/coils.curves_22_7_21"
wout_filename = "../inputs/wout_aten_rescaled.nc"


def assert_coil_field_matches_equilibrium(bs, surface, wout_filename, rtol=0.03):
    """
    Abort unless the coil field reproduces the equilibrium it is paired with.

    Averaging |B| over a grid spanning whole periods in both angles leaves only
    the (m, n) = (0, 0) harmonic, so the equilibrium's mean |B| on the LCFS is
    bmnc(0, 0) carried to s = 1; bmnc lives on the half mesh, hence the
    extrapolation from the last two surfaces.

    The failure this catches is a coil set replicated too many times.  A coil
    file may hold one field period, or the stellarator-symmetric half of the
    torus, or the whole torus, and over-applying the symmetries scales |B| by
    an integer factor while leaving a field that looks perfectly well formed.
    """
    with netcdf_file(wout_filename, mmap=False) as f:
        xm_nyq = np.asarray(f.variables["xm_nyq"][:])
        xn_nyq = np.asarray(f.variables["xn_nyq"][:])
        bmnc = np.asarray(f.variables["bmnc"][:])
    (i00,) = np.where((xm_nyq == 0) & (xn_nyq == 0))[0]
    modB_equil = 1.5 * bmnc[-1, i00] - 0.5 * bmnc[-2, i00]

    bs.set_points(surface.gamma().reshape(-1, 3))
    modB_coils = np.linalg.norm(bs.B(), axis=1).mean()

    ratio = modB_coils / modB_equil
    print(
        f"coil field check: mean |B| on LCFS = {modB_coils:.4f} T "
        f"(equilibrium {modB_equil:.4f} T, ratio {ratio:.4f})"
    )
    if abs(ratio - 1) > rtol:
        raise SystemExit(
            f"coil field is {ratio:.3f}x the equilibrium -- check the symmetry "
            "arguments to coils_via_symmetries against the coil file"
        )


surf = SurfaceRZFourier.from_wout(wout_filename)

coils = load_coils_from_makegrid_file(filename, order, ppp=20, group_names=None)

curves = []
currents = []
for _i, coil in enumerate(coils):
    curves.append(coil.curve)
    currents.append(coil.current)

# coils.curves_22_7_21 holds 20 coils: the stellarator-symmetric half of a
# full-torus 40-coil set.  Only stellsym is left to apply -- passing surf.nfp
# here would replicate an already complete set nfp times and make |B| about
# nfp times too strong.  The check below is what pins this down.
coils_full = coils_via_symmetries(curves, currents, 1, True)
bs = BiotSavart(coils_full)

assert_coil_field_matches_equilibrium(bs, surf, wout_filename)

surf_launch = SurfaceRZFourier.from_wout(wout_filename, s=0.3)

sc_particle = SurfaceClassifier(surf, h=0.1, p=2)
rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
zs = surf.gamma()[:, :, 2]

rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / surf.nfp, n * 2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), n // 2)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=surf.nfp, stellsym=True
)

# sample particles from surface
# The loss fraction is what a mis-scaled field would show up in, so it needs
# enough events that Poisson noise does not hide a factor-of-several error.
nparticles = 10000
xyz, _ = draw_uniform_on_surface(surf_launch, nparticles, safetyfactor=10)

vpar0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles)

tmax = 1e-5
last_time = trace_particles_cartesian_gpu(
    bsh,
    sc_particle,
    xyz,
    vpar_inits,
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    vtotal=vpar0,
    tol=1e-8,
)
particle_data = pd.DataFrame(
    {
        "x_start": xyz[:, 0],
        "y_start": xyz[:, 1],
        "z_start": xyz[:, 2],
        "vpar_start": vpar_inits,
        "x_end": last_time[:, 1],
        "y_end": last_time[:, 2],
        "z_end": last_time[:, 3],
        "vpar_end": last_time[:, 4],
        "last_time": last_time[:, 0],
        "dt_end": last_time[:, 5],
    }
)
particle_data.to_csv("./particle_data.csv")


did_leave = [t < tmax for t in particle_data["last_time"]]
n_lost = sum(did_leave)
loss_frac = n_lost / len(did_leave)
# Printed with its counting uncertainty: the loss fraction is the figure a
# mis-scaled field would move, and it is only meaningful once it is many events.
rel_err = 1.0 / np.sqrt(n_lost) if n_lost else np.inf
print(f"Number of particles= {nparticles}")
print(f"Particles lost: {n_lost} ({loss_frac:.3e} +/- {loss_frac * rel_err:.1e})")
