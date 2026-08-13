import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from simsopt.field import (
    BiotSavart,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
    load_coils_from_makegrid_file,
)
from simsopt.field.sampling import draw_uniform_on_surface
from simsopt.geo import SurfaceRZFourier

from firm3d.catapult.tracing import trace_particles_cartesian_with_collisions_gpu
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.collisions import ThermalBackground
from firm3d.field.coordinates import boozer_to_cylindrical
from firm3d.field.tracing_helpers import (
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    PROTON_MASS,
)

degree = 3  # degree of interpolant
n = 16  # resolution of interpolant
order = 12  # order of coil curves

filename = "../inputs/coils.curves_22_7_21"
wout_filename = "../inputs/wout_aten_rescaled.nc"

surf = SurfaceRZFourier.from_wout(wout_filename)

coils = load_coils_from_makegrid_file(filename, order, ppp=20, group_names=None)

curves = []
currents = []
for _i, coil in enumerate(coils):
    curves.append(coil.curve)
    currents.append(coil.current)

coils_full = coils_via_symmetries(curves, currents, surf.nfp, True)
bs = BiotSavart(coils_full)

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

# The thermal profiles are functions of the normalized flux s, which the
# Cartesian state does not carry, so the tracer needs it as a scalar field
# s(r, phi, z) to interpolate alongside the magnetic field.  Build it from
# the equilibrium that the coils were designed for: map a dense grid of
# Boozer coordinates forward to cylindrical, then answer queries with the
# s of the nearest mapped point.  Inverting the map with root finding per
# query point would cost minutes; the label error here is set by the sample
# spacing, well below what the collision rates resolve.  Points outside the
# plasma inherit s ~ 1 from the nearest boundary sample, which is harmless:
# the profile lookup clamps to its grid, and such particles are terminated
# by the boundary check anyway.
bri = BoozerRadialInterpolant(wout_filename, 3, enforce_vacuum=True)
bfield = InterpolatedBoozerField(bri, 3, ns_interp=n, ntheta_interp=n, nzeta_interp=n)

n_s, n_ang = 48, 48
s_grid = np.linspace(0.02, 1.0, n_s)
theta_grid = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
zeta_grid = np.linspace(0, 2 * np.pi, bri.nfp * n_ang, endpoint=False)
stz_samples = (
    np.array(np.meshgrid(s_grid, theta_grid, zeta_grid, indexing="ij")).reshape(3, -1).T
)
cyl_samples = boozer_to_cylindrical(bfield, stz_samples)
tree = cKDTree(
    np.column_stack(
        [
            cyl_samples[:, 0] * np.cos(cyl_samples[:, 1]),
            cyl_samples[:, 0] * np.sin(cyl_samples[:, 1]),
            cyl_samples[:, 2],
        ]
    )
)
s_samples = stz_samples[:, 0]


def flux_label(points_rphiz):
    """Normalized toroidal flux at cylindrical points (r, phi, z)."""
    p = np.asarray(points_rphiz)
    _, idx = tree.query(
        np.column_stack([p[:, 0] * np.cos(p[:, 1]), p[:, 0] * np.sin(p[:, 1]), p[:, 2]])
    )
    return s_samples[idx]


# Background plasma the alphas collide with: a 50/50 DT fuel mix and the
# electrons that neutralize it.  Temperature is in eV at this interface.
n_ref = 1e20  # m^-3
ne = lambda s: n_ref * (1 - s**5)
Te = lambda s: 11.5e3 * (1 - s)

backgrounds = [
    ThermalBackground(
        n_profile=lambda s: 0.5 * ne(s),
        T_profile=Te,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    ),
    ThermalBackground(
        n_profile=lambda s: 0.5 * ne(s),
        T_profile=Te,
        mass=3 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    ),
    ThermalBackground(
        n_profile=ne,
        T_profile=Te,
        mass=ELECTRON_MASS,
        charge=-ELEMENTARY_CHARGE,
    ),
]

# sample particles from surface
nparticles = 1000
xyz, _ = draw_uniform_on_surface(surf_launch, nparticles, safetyfactor=10)

vpar0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
vpar_inits = initialize_velocity_uniform(vpar0, nparticles)

# The alpha slowing-down time in this background is of order 0.1 s, so
# the collisionless examples' 1e-5 s window would show no slowing at
# all.  1e-2 s costs a few tens of seconds on one GPU and takes about
# a tenth of the birth energy off the confined population.
tmax = 1e-2
last_time = trace_particles_cartesian_with_collisions_gpu(
    bsh,
    sc_particle,
    flux_label,
    xyz,
    vpar_inits,
    backgrounds=backgrounds,
    tmax=tmax,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    vtotal=vpar0,
    tol=1e-8,
    rng_seed=0,
)
# The collisional output has seven columns rather than six: the total speed
# is reported before the final step size, because collisions change it and
# it is no longer recoverable from the launch energy.
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
        "v_end": last_time[:, 5],
        "last_time": last_time[:, 0],
        "dt_end": last_time[:, 6],
    }
)
particle_data.to_csv("./particle_data.csv")


did_leave = np.array([t < tmax for t in particle_data["last_time"]])
loss_frac = did_leave.sum() / len(did_leave)
# Averaged over confined particles only: a lost particle's speed is frozen
# at the moment it left, so mixing them in would average over different
# elapsed times.
energy_fraction = ((particle_data["v_end"] / vpar0) ** 2)[~did_leave]
print(f"Number of particles= {nparticles}")
print(f"Loss fraction: {loss_frac:.3f}")
print(f"Mean energy fraction of confined particles: {energy_fraction.mean():.4f}")
