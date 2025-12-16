import numpy as np
from simsopt.configs import get_ncsx_data
from simsopt.field import (
    BiotSavart,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
)
from simsopt.geo import SurfaceRZFourier

import firm3dpp as sopp
from firm3d.util.gpu_utils import cartesian_interpolant


def test_interpolant_bfield(field, sc_particle, nfp, n_metagrid_pts, n_test_pts):
    r_range = field.r_range
    phi_range = field.phi_range
    z_range = field.z_range

    # generate test points
    r = np.random.uniform(low=r_range[0], high=r_range[1], size=(n_test_pts, 1))
    phi = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=-z_range[1], high=z_range[1], size=(n_test_pts, 1))
    rphiz = np.hstack((r, phi, z))
    # rphiz = np.array([[ 1.16373528,  0.04877549, -0.13471403]])
    rphiz = np.ascontiguousarray(rphiz)

    # SIMSOPT INTERPOLANT
    field.set_points_cyl(rphiz)
    B = field.B_cyl()
    GradAbsB = field.GradAbsB_cyl()
    signed_dist_vals = sc_particle.evaluate_rphiz(rphiz)
    simsopt_interpolation = np.hstack((B, GradAbsB, signed_dist_vals))

    ## NEW INTERPOLANT
    r_range, phi_range, z_range, quad_info = cartesian_interpolant(
        field, sc_particle, nfp, n_metagrid_pts
    )

    # Calculate interpolation
    new_interpolation = sopp.test_gpu_interpolation(
        quad_info,
        r_range,
        phi_range,
        z_range,
        rphiz,
        "cartesian_vacuum",
        rphiz.shape[0],
    )
    new_interpolation = np.reshape(new_interpolation, (rphiz.shape[0], 7))

    # print(np.abs(simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    rel_err = np.abs(
        (simsopt_interpolation - new_interpolation) / simsopt_interpolation
    )
    dist_fn = simsopt_interpolation[:, 6]
    rel_err = rel_err[dist_fn > 0, :-1]  # don't test boundary distance for now
    diff = np.max(rel_err)
    print(
        f"Maximum relative error in interpolation values on {rel_err.shape[0]} points: {diff}"
    )
    if diff > 1e-8:
        print("INTERPOLANT TEST FAILED")
        print("culprit particle:")
        row_index = np.argmax(rel_err) // rel_err.shape[1]
        print(rphiz[row_index, :])
        print("simsopt", simsopt_interpolation[row_index, :])
        print("new", new_interpolation[row_index, :])
        print(rel_err[row_index, :])
    else:
        print("INTERPOLANT TEST SUCCESS")


if __name__ == "__main__":
    ### CREATE A FIELD FOR TRACING
    nfp = 3
    curves, currents, ma = get_ncsx_data()
    coils = coils_via_symmetries(curves, currents, nfp, True)
    curves = [c.curve for c in coils]
    bs = BiotSavart(coils)
    # proc0_print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
    # proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))

    mpol = 5
    ntor = 5
    stellsym = True
    s = SurfaceRZFourier.from_nphi_ntheta(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        range="full torus",
        nphi=64,
        ntheta=24,
    )
    s.fit_to_curve(ma, 0.20, flip_theta=False)

    n_metagrid_pts = 30
    degree = 3
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    sc_particle = SurfaceClassifier(s, h=0.1, p=2)

    rrange = (np.min(rs), np.max(rs), n_metagrid_pts)
    phirange = (0, 2 * np.pi / nfp, n_metagrid_pts)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, np.max(zs), n_metagrid_pts)
    # print(rrange, phirange, zrange)
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
    )
    np.random.seed(1800)
    test_interpolant_bfield(bsh, sc_particle, nfp, n_metagrid_pts, 100000)
