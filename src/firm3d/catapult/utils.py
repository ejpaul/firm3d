__all__ = []


import numpy as np

__all__ = ["boozer_interpolant", "cartesian_interpolant"]


def boozer_interpolant(field, nfp, ns, ntheta, nzeta, vacuum=False):
    r"""
    Set up a Boozer vacuum interpolant for tracing.

    Args:
        field: BoozerMagneticField object
        nfp: Integer, number of field periods in the device
        n_meta_grid_pts: Integer, number of cells in s, theta, zeta
            to use for interpolation

    Returns:
        srange : (s_start, s_end, number of grid points in s for interpolation)
        trange : same as srange, but for theta
        zrange : same as srange, but for zeta
        cell_quad_pts : The interpolant data. Each row is a point in the grid,
            data is stored in columns modB, dmodBds, dmodBdtheta, dmodBdzeta, G, iota
        maximum observed J at grid points, useful for rejection sampling
    """
    srange = (0, 1.0, 3 * ns + 1)
    trange = (0, np.pi, 3 * ntheta + 1)
    zrange = (0, 2 * np.pi / nfp, 3 * nzeta + 1)

    s_grid = np.linspace(srange[0], srange[1], srange[2])
    theta_grid = np.linspace(trange[0], trange[1], trange[2])
    zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

    quad_pts = np.empty((srange[2] * trange[2] * zrange[2], 3))
    for i in range(srange[2]):
        for j in range(trange[2]):
            for k in range(zrange[2]):
                quad_pts[trange[2] * zrange[2] * i + zrange[2] * j + k, :] = [
                    s_grid[i],
                    theta_grid[j],
                    zeta_grid[k],
                ]

    field.set_points(quad_pts)

    # Quantities to interpolate
    G = field.G()
    I = field.I()
    iota = field.iota()
    modB = field.modB()
    modB_derivs = field.modB_derivs()

    if vacuum:
        # Vacuum approximation: G=const, I=0, K=0
        quad_info = np.hstack((modB, modB_derivs, G, iota))
    else:
        # Full guiding center equations: include I and K
        dGds = field.dGds()
        dIds = field.dIds()
        K = field.K()
        K_derivs = field.K_derivs()
        quad_info = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, K, K_derivs))

    # calculate max J for sampling
    J = (G + iota * I) / (modB**2)

    # reorder for device memory acceesses
    # print("reordering interpolant data from device accesses")
    s_ncells = int((srange[2] - 1) / 3)
    t_ncells = int((trange[2] - 1) / 3)
    z_ncells = int((zrange[2] - 1) / 3)
    cell_quad_pts = np.empty((s_ncells * t_ncells * z_ncells * 64, quad_info.shape[1]))

    for cell_s in range(s_ncells):
        for cell_t in range(t_ncells):
            for cell_z in range(z_ncells):
                row_start = 64 * (
                    cell_s * t_ncells * z_ncells + cell_t * z_ncells + cell_z
                )

                # iterate over spline locations for this cell
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            row_idx = row_start + 16 * i + 4 * j + k
                            cell_quad_pts[row_idx, :] = quad_info[
                                trange[2] * zrange[2] * (3 * cell_s + i)
                                + zrange[2] * (3 * cell_t + j)
                                + 3 * cell_z
                                + k,
                                :,
                            ]
    cell_quad_pts = np.ascontiguousarray(cell_quad_pts)
    return srange, trange, zrange, cell_quad_pts, np.max(J)


def boozer_saw_interpolant(field, nfp, ns, ntheta, nzeta):
    r"""
    Set up a Boozer vacuum interpolant for tracing.

    Args:
        field: BoozerMagneticField object
        nfp: Integer, number of field periods in the device
        n_meta_grid_pts: Integer, number of cells in s, theta, zeta
            to use for interpolation

    Returns:
        srange : (s_start, s_end, number of grid points in s for interpolation)
        trange : same as srange, but for theta
        zrange : same as srange, but for zeta
        cell_quad_pts : The interpolant data. Each row is a point in the grid,
            data is stored in columns
            modB, dmodBds, dmodBdtheta, dmodBdzeta, G, dGds, I, dIds, iota, diotads
        maximum observed J at grid points, useful for rejection sampling
    """

    srange = (0, 1.0, 3 * ns + 1)
    trange = (0, np.pi, 3 * ntheta + 1)
    zrange = (0, 2 * np.pi / nfp, 3 * nzeta + 1)

    s_grid = np.linspace(srange[0], srange[1], srange[2])
    theta_grid = np.linspace(trange[0], trange[1], trange[2])
    zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

    quad_pts = np.empty((srange[2] * trange[2] * zrange[2], 3))
    for i in range(srange[2]):
        for j in range(trange[2]):
            for k in range(zrange[2]):
                quad_pts[trange[2] * zrange[2] * i + zrange[2] * j + k, :] = [
                    s_grid[i],
                    theta_grid[j],
                    zeta_grid[k],
                ]

    field.set_points(quad_pts)

    # Quantities to interpolate
    G = field.G()
    iota = field.iota()
    modB = field.modB()
    modB_derivs = field.modB_derivs()
    dGds = field.dGds()
    I = field.I()
    dIds = field.dIds()
    diotads = field.diotads()
    quad_info = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, diotads))

    # calculate max J for sampling
    I = field.I()
    J = (G + iota * I) / (modB**2)

    # reorder for device memory acceesses
    # print("reordering interpolant data from device accesses")
    s_ncells = int((srange[2] - 1) / 3)
    t_ncells = int((trange[2] - 1) / 3)
    z_ncells = int((zrange[2] - 1) / 3)
    cell_quad_pts = np.empty((s_ncells * t_ncells * z_ncells * 64, quad_info.shape[1]))

    for cell_s in range(s_ncells):
        for cell_t in range(t_ncells):
            for cell_z in range(z_ncells):
                row_start = 64 * (
                    cell_s * t_ncells * z_ncells + cell_t * z_ncells + cell_z
                )

                # iterate over spline locations for this cell
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            row_idx = row_start + 16 * i + 4 * j + k
                            cell_quad_pts[row_idx, :] = quad_info[
                                trange[2] * zrange[2] * (3 * cell_s + i)
                                + zrange[2] * (3 * cell_t + j)
                                + 3 * cell_z
                                + k,
                                :,
                            ]
    cell_quad_pts = np.ascontiguousarray(cell_quad_pts)
    return srange, trange, zrange, cell_quad_pts, np.max(J)


def cartesian_interpolant(field, sc_particle, nfp, n_metagrid_pts):
    r"""
    Set up a Boozer vacuum interpolant for tracing.

    Args:
        field: MagneticField object
        nfp: Integer, number of field periods in the device
        n_meta_grid_pts: Integer, number of cells in r, phi, zeta
            to use for interpolation

    Returns:
        r_range : (r_start, r_end, number of grid points in r for interpolation)
        phi_range : same as r_range, but for phi
        z_range : same as r_range, but for zeta
        cell_quad_pts : The interpolant data. Each row is a point in the grid,
         data is stored in columns B, GradAbsB, signed distance function
    """

    r_range = (field.r_range[0], field.r_range[1], 3 * field.r_range[2] + 1)
    phi_range = (field.phi_range[0], field.phi_range[1], 3 * field.phi_range[2] + 1)
    z_range = (field.z_range[0], field.z_range[1], 3 * field.z_range[2] + 1)

    r_grid = np.linspace(r_range[0], r_range[1], r_range[2])
    phi_grid = np.linspace(phi_range[0], phi_range[1], phi_range[2])
    z_grid = np.linspace(z_range[0], z_range[1], z_range[2])

    quad_pts = np.empty((r_range[2] * phi_range[2] * z_range[2], 3))
    for i in range(r_range[2]):
        for j in range(phi_range[2]):
            for k in range(z_range[2]):
                quad_pts[phi_range[2] * z_range[2] * i + z_range[2] * j + k, :] = [
                    r_grid[i],
                    phi_grid[j],
                    z_grid[k],
                ]

    field.set_points_cyl(quad_pts)

    # Quantities to interpolate
    B = field.B_cyl()
    GradAbsB = field.GradAbsB_cyl()

    signed_dist_vals = sc_particle.evaluate_rphiz(quad_pts)

    quad_info = np.hstack((B, GradAbsB, signed_dist_vals))

    # reorder for device memory accesses
    # print("reordering interpolant data form device accesses")
    cell_quad_pts = np.empty(
        (
            field.r_range[2] * field.z_range[2] * field.phi_range[2] * 64,
            quad_info.shape[1],
        )
    )
    for cell_r in range(field.r_range[2]):
        for cell_phi in range(field.phi_range[2]):
            for cell_z in range(field.z_range[2]):
                row_start = 64 * (
                    cell_r * field.phi_range[2] * field.z_range[2]
                    + cell_phi * field.z_range[2]
                    + cell_z
                )

                # if cell_r == 24 and cell_phi == 22 and cell_z == 20:
                #     print(row_start)

                assert 3 * cell_r + i < r_range[2]
                # iterate over spline locations for this cell
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            row_idx = row_start + 16 * i + 4 * j + k

                            cell_quad_pts[row_idx, :] = quad_info[
                                phi_range[2] * z_range[2] * (3 * cell_r + i)
                                + z_range[2] * (3 * cell_phi + j)
                                + 3 * cell_z
                                + k,
                                :,
                            ]

    cell_quad_pts = np.ascontiguousarray(cell_quad_pts)

    return r_range, phi_range, z_range, cell_quad_pts
