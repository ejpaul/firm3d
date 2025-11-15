import numpy as np
import unittest
from numpy.testing import assert_raises
import numpy as np
import os

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
np.random.seed(1865)

from firm3d.util.gpu_utils import boozer_interpolant
import firm3dpp

def test_interpolant_bfield_vacuum(field, nfp, n_metagrid_pts, n_test_pts):

    # generate test points
    s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
    t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    stz = np.hstack((s,t,z))

    # SIMSOPT INTERPOLANT
    field.set_points(stz)
    G = field.G()
    iota = field.iota()
    modB = field.modB()
    modB_derivs = field.modB_derivs()
    simsopt_interpolation = np.hstack((modB, modB_derivs, G, iota))

    ## NEW INTERPOLANT
    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, nfp, n_metagrid_pts, vacuum=True)
    stz = np.ascontiguousarray(stz)

    # print("stz", stz)

    # Calculate interpolation
    # print(zrange)
    # exit()
    new_interpolation = firm3dpp.test_gpu_interpolation(quad_info, srange, trange, zrange, stz, "boozer_vacuum", stz.shape[0])
    new_interpolation = np.reshape(new_interpolation, (stz.shape[0], 6))

    # print(np.abs(simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    rel_err = np.abs((simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    diff = np.max(rel_err)
    print("Maximum relative error in interpolation values on {} points: {}".format(n_test_pts, diff))

    if diff > 1e-8:
        print("BOOZER INTERPOLANT TEST FAILED")
        print("culprit particle:")
        row_index = np.argmax(rel_err) // rel_err.shape[1]
        print(stz[row_index, :])
        print("simsopt", simsopt_interpolation[row_index, :])
        print("new    ", new_interpolation[row_index, :])
        print(rel_err[row_index, :])
    else:
        print("BOOZER INTERPOLANT TEST SUCCESS (vacuum, 6 fields)")


def test_interpolant_bfield_finite_beta(field, nfp, n_metagrid_pts, n_test_pts):

    # generate test points
    s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
    t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    stz = np.hstack((s,t,z))

    # SIMSOPT INTERPOLANT
    field.set_points(stz)

    G = field.G()
    iota = field.iota()
    modB = field.modB()
    modB_derivs = field.modB_derivs()
    dGds = field.dGds()
    I = field.I()
    dIds = field.dIds()
    diotads = field.diotads()
    K = field.K()
    K_derivs = field.K_derivs()

    simsopt_interpolation_gc = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, diotads, K, K_derivs))

    # GPU interpolant for full GC
    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, nfp, n_metagrid_pts, vacuum=False)
    new_interpolation_gc = firm3dpp.test_gpu_interpolation(quad_info, srange, trange, zrange, stz, "boozer", stz.shape[0])
    new_interpolation_gc = np.reshape(new_interpolation_gc, (stz.shape[0], 13))

    rel_err = np.abs((simsopt_interpolation_gc - new_interpolation_gc) / simsopt_interpolation_gc)
    max_rel_err = np.max(rel_err)
    print(f"Maximum relative error in interpolation values (GC, 13 fields) on all points: {max_rel_err}")
    abs_diff = np.max(np.abs(simsopt_interpolation_gc - new_interpolation_gc))
    print(f"Maximum abs error in interpolation values (GC, 13 fields) on all points: {abs_diff}")

    if max_rel_err > 1e-8:
        print("BOOZER INTERPOLANT TEST FAILED (general GC)")
        row_index = np.argmax(rel_err) // rel_err.shape[1]
        print(stz[row_index, :])
        print("simsopt", simsopt_interpolation_gc[row_index, :])
        print("new    ", new_interpolation_gc[row_index, :])
        print(rel_err[row_index, :])
    else:
        print("BOOZER INTERPOLANT TEST SUCCESS (general GC, 13 fields)")



if __name__ == "__main__":
    np.set_printoptions(linewidth=300)

    ### Vacuum case
    boozmn_filename = "examples/inputs/boozmn_aten_rescaled.nc"
    bri = BoozerRadialInterpolant(boozmn_filename, 3, no_K=True)

    nfp = bri.nfp
    degree = 3
    n_metagrid_pts = 15
    srange = (0, 1, n_metagrid_pts)
    thetarange = (0, np.pi, n_metagrid_pts)
    zetarange = (0, 2*np.pi/nfp, n_metagrid_pts)
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )
    test_interpolant_bfield_vacuum(field, nfp, n_metagrid_pts, 10000)

    ### Finite-beta case
    boozmn_filename = "examples/inputs/boozmn_ariescs.nc"
    bri = BoozerRadialInterpolant(boozmn_filename, 3, no_K=False)

    nfp = bri.nfp
    degree = 3
    n_metagrid_pts = 15
    srange = (0, 1, n_metagrid_pts)
    thetarange = (0, np.pi, n_metagrid_pts)
    zetarange = (0, 2*np.pi/nfp, n_metagrid_pts)
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )
    test_interpolant_bfield_finite_beta(field, nfp, n_metagrid_pts, 10000)