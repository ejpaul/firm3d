import time
import numpy as np

import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.util.gpu_utils import boozer_interpolant

np.random.seed(1865)

def test_interpolant_bfield(field, nfp, n_metagrid_pts, n_test_pts, vacuum):
    # generate test points
    s = np.random.uniform(low=0, high=1, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    stz = np.hstack((s, t, z))

    # SIMSOPT INTERPOLANT
    field.set_points(stz)
    G = field.G()
    iota = field.iota()
    modB = field.modB()
    modB_derivs = field.modB_derivs()
    if vacuum:
        simsopt_interpolation = np.hstack((modB, modB_derivs, G, iota))
        interpolation_type = "boozer_vacuum"
        n_quantities = 6
    else:
        dGds = field.dGds()
        I = field.I()
        dIds = field.dIds()
        K = field.K()
        K_derivs = field.K_derivs()
        simsopt_interpolation = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, K, K_derivs))
        interpolation_type = "boozer"
        n_quantities = 12

    ## NEW INTERPOLANT
    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
        field, nfp, n_metagrid_pts, vacuum=vacuum
    )
    stz = np.ascontiguousarray(stz)
    new_interpolation = firm3dpp.test_gpu_interpolation(
        quad_info, srange, trange, zrange, stz, interpolation_type, stz.shape[0]
    )
    new_interpolation = np.reshape(new_interpolation, (stz.shape[0], n_quantities))

    rel_err = np.abs(
        (simsopt_interpolation - new_interpolation) / simsopt_interpolation
    )
    diff = np.max(rel_err)
    print(
        f"Maximum relative error in interpolation values on {n_test_pts} points: {diff}"
    )
    beta_string = "VACUUM" if vacuum else "FINITE_BETA"
    if diff > 1e-8:
        print(f"BOOZER INTERPOLANT {beta_string} TEST FAILED")
        print("culprit particle:")
        row_index = np.argmax(rel_err) // rel_err.shape[1]
        print(stz[row_index, :])
        print("simsopt", simsopt_interpolation[row_index, :])
        print("new    ", new_interpolation[row_index, :])
        print(rel_err[row_index, :])
    else:
        print(f"BOOZER INTERPOLANT {beta_string} TEST SUCCESS")


def get_field(boozmn_filename, n_metagrid_pts, vacuum):
    start_time = time.time()
    bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=vacuum)
    print(f"Time to initialize BoozerRadialInterpolant: {time.time() - start_time} seconds")
    nfp = bri.nfp
    degree = 3
    start_time = time.time()
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )
    print(f"Time to initialize InterpolatedBoozerField: {time.time() - start_time} seconds")
    # Even though bri isn't used further in this script, we need to return it,
    # or else it is garbage-collected, resuliting in an error.
    return bri, field, nfp


if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    n_metagrid_pts = 15

    ### Vacuum case
    linewidth = 80
    print("\n" + "=" * linewidth)
    print("TESTING VACUUM CASE")
    print("=" * linewidth)
    boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
    vacuum = True
    bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 10000, vacuum)

    ### Finite-beta case
    print("\n" + "=" * linewidth)
    print("TESTING FINITE-BETA CASE")
    print("=" * linewidth)
    boozmn_filename = "examples/inputs/boozmn_ariescs_low_res.nc"
    vacuum = False
    bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 10000, vacuum)
