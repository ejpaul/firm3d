import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.io import netcdf_file
from scipy.optimize import root
from scipy.spatial import KDTree

__all__ = [
    "boozer_to_cylindrical",
    "cylindrical_to_boozer",
    "boozer_to_vmec",
    "vmec_to_boozer",
    "vmec_to_cylindrical",
    "cylindrical_to_vmec",
    "BoozerCoordinateTransformer",
    "VMECCoordinateTransformer",
]


class BoozerCoordinateTransformer:
    """
    A class for efficient coordinate transformations between Boozer and cylindrical
    coordinates with reusable grid-based initialization.

    This class builds a coordinate grid once and reuses it for multiple transformations.

    Args:
        field: The BoozerMagneticField instance used for field evaluation
        grid_resolution: Tuple of (n_s, n_theta, n_zeta) for grid resolution

    Example:
        transformer = BoozerCoordinateTransformer(field, grid_resolution=(15, 30, 30))
        points_boozer = transformer.cylindrical_to_boozer(points_cyl)
        # Grid is reused for subsequent calls
        points_boozer2 = transformer.cylindrical_to_boozer(points_cyl2)
    """

    def __init__(self, field, grid_resolution=(50, 50, 50)):
        self.field = field
        self.grid_resolution = grid_resolution
        self._grid_coords = None
        self._grid_cylindrical = None
        self._grid_built = False

    def _build_coordinate_grid(self, n_s, n_theta, n_zeta):
        """
        Build a grid of Boozer coordinates and their corresponding cylindrical
        coordinates.
        Takes advantage of nfp symmetry to reduce grid size.

        Args:
            n_s: Number of s grid points
            n_theta: Number of theta grid points
            n_zeta: Number of zeta grid points

        Returns:
            boozer_coords: Array of shape (n_points, 3) with (s, theta, zeta)
                coordinates
            cylindrical_coords: Array of shape (n_points, 3) with (R, phi, Z)
                coordinates
        """
        # Get nfp from field if available
        nfp = getattr(self.field, "nfp", 1)

        # Create coordinate grids
        s_grid = np.linspace(0, 1, n_s)
        theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        # Take advantage of nfp symmetry - only need one field period
        zeta_grid = np.linspace(0, 2 * np.pi / nfp, n_zeta, endpoint=False)

        # Create meshgrid
        s_mesh, theta_mesh, zeta_mesh = np.meshgrid(
            s_grid, theta_grid, zeta_grid, indexing="ij"
        )

        s_mesh = s_mesh.flatten()
        theta_mesh = theta_mesh.flatten()
        zeta_mesh = zeta_mesh.flatten()

        # Remove duplicative points on the magnetic axis
        bool_mask = (s_mesh == 0) * (theta_mesh > 0)
        s_mesh = np.delete(s_mesh, bool_mask)
        theta_mesh = np.delete(theta_mesh, bool_mask)
        zeta_mesh = np.delete(zeta_mesh, bool_mask)

        # Flatten to create coordinate arrays
        n_points = len(s_mesh)
        boozer_coords = np.zeros((n_points, 3))
        boozer_coords[:, 0] = s_mesh
        boozer_coords[:, 1] = theta_mesh
        boozer_coords[:, 2] = zeta_mesh

        # Convert to cylindrical coordinates
        self.field.set_points(boozer_coords)
        R = self.field.R()[:, 0]
        Z = self.field.Z()[:, 0]
        nu = self.field.nu()[:, 0]
        phi = zeta_mesh.flatten() - nu

        cylindrical_coords = np.zeros((n_points, 3))
        cylindrical_coords[:, 0] = R
        cylindrical_coords[:, 1] = phi
        cylindrical_coords[:, 2] = Z

        return boozer_coords, cylindrical_coords

    def _ensure_grid_built(self):
        """Build the coordinate grid if not already built."""
        if not self._grid_built:
            try:
                n_s, n_theta, n_zeta = self.grid_resolution
                self._grid_coords, self._grid_cylindrical = self._build_coordinate_grid(
                    n_s, n_theta, n_zeta
                )
                self._grid_built = True
            except Exception as e:
                raise RuntimeError(f"Failed to build coordinate grid: {e}") from e

    def cylindrical_to_boozer(self, points_cyl, n_guesses=10, ftol=1e-6):
        """
        Convert from cylindrical coordinates to Boozer coordinates.
        All initial guesses are generated from the coordinate grid.

        Args:
            points_cyl: A numpy array of shape (npoints, 3) containing the
                cylindrical coordinates (R, phi, Z).
            n_guesses: Number of grid-based initial guesses to try per point
            ftol: Tolerance for root finding convergence

        Returns:
            points_boozer: A numpy array of shape (npoints, 3) containing the
                Boozer coordinates (s, theta, zeta).
        """
        # Ensure grid is built
        self._ensure_grid_built()

        # Validate input shape
        if len(points_cyl.shape) != 2 or points_cyl.shape[1] != 3:
            raise ValueError("points_cyl must have shape (npoints, 3)")

        npoints = points_cyl.shape[0]
        if npoints == 0:
            raise ValueError("Input arrays cannot be empty")

        points_boozer = np.zeros((npoints, 3))

        def objective_function(x, points_cyl_target):
            """Objective function for root finding."""
            s_val, theta_val, zeta_val = x
            s_val = np.clip(s_val, 0.0, 1.0)

            points = np.zeros((1, 3))
            points[0, 0] = s_val
            points[0, 1] = theta_val
            points[0, 2] = zeta_val

            self.field.set_points(points)

            R_computed = self.field.R()[0, 0]
            Z_computed = self.field.Z()[0, 0]
            nu_computed = self.field.nu()[0, 0]
            phi_computed = zeta_val - nu_computed

            return [
                R_computed - points_cyl_target[0],
                np.arctan2(
                    np.sin(phi_computed - points_cyl_target[1]),
                    np.cos(phi_computed - points_cyl_target[1]),
                ),
                Z_computed - points_cyl_target[2],
            ]

        def get_grid_guesses(target_point, n_guesses):
            """Get multiple grid-based initial guesses using k-nearest neighbors."""
            # Build KDTree for efficient nearest neighbor search
            tree = KDTree(self._grid_cylindrical)

            # Map target phi to fundamental domain for KDTree search only
            # Record how many field periods (integer multiples of 2π/nfp) to add back
            nfp = getattr(self.field, "nfp", 1)
            phi_period = 2 * np.pi / nfp

            # Calculate number of complete field periods in target phi
            n_field_periods = int(np.floor(target_point[1] / phi_period))

            # Map to fundamental domain [0, 2π/nfp)
            target_phi_mapped = target_point[1] - n_field_periods * phi_period
            target_mapped = target_point.copy()
            target_mapped[1] = target_phi_mapped

            # Find k nearest neighbors (more than n_guesses to have options)
            n_guesses = min(n_guesses * 2, len(self._grid_coords))
            distances, indices = tree.query(target_mapped, k=n_guesses)

            # Add back the field periods that were subtracted
            # Grid coords are in Boozer: (s, theta, zeta)
            # We need to adjust zeta to account for the field periods
            zeta_offset = n_field_periods * phi_period

            selected_guesses = []
            for idx in indices:
                guess = self._grid_coords[idx].copy()
                # Add back the n_field_periods * (2π/nfp) to zeta coordinate
                guess[2] = guess[2] + zeta_offset
                selected_guesses.append(guess)

            return selected_guesses

        for i in range(npoints):
            success = False

            # Get multiple grid-based guesses
            target_point = points_cyl[i, :]
            initial_guesses = get_grid_guesses(target_point, n_guesses)

            for x0 in initial_guesses:
                sol = root(
                    objective_function,
                    x0,
                    args=(points_cyl[i, :]),
                    method="hybr",
                    tol=ftol,
                )
                if sol.success:
                    points_boozer[i, 0] = np.clip(sol.x[0], 0.0, 1.0)
                    points_boozer[i, 1] = sol.x[1]
                    points_boozer[i, 2] = sol.x[2]
                    success = True
                    break

            if not success:
                raise RuntimeError(
                    f"Root finding failed for point {i} with coordinates "
                    f"R={points_cyl[i, 0]}, phi={points_cyl[i, 1]}, "
                    f"Z={points_cyl[i, 2]}"
                )

        return points_boozer

    def boozer_to_cylindrical(self, points_boozer):
        """
        Convert from Boozer coordinates to cylindrical coordinates.

        Args:
            points_boozer: A numpy array of shape (npoints, 3) containing the
                Boozer coordinates (s, theta, zeta).

        Returns:
            points_cyl: A numpy array of shape (npoints, 3) containing the
                cylindrical coordinates (R, phi, Z).
        """
        return boozer_to_cylindrical(self.field, points_boozer)


class VMECCoordinateTransformer:
    """
    A class for efficient coordinate transformations between VMEC and cylindrical
    coordinates with reusable grid-based initialization.

    This class builds a coordinate grid once and reuses it for multiple transformations,
    providing better performance and robustness than the standalone functions.

    Args:
        wout_filename: Path to VMEC wout file
        grid_resolution: Tuple of (n_s, n_theta, n_phi) for grid resolution

    Example:
        transformer = VMECCoordinateTransformer("wout.nc", grid_resolution=(15, 30, 30))
        points_vmec = transformer.cylindrical_to_vmec(points_cyl)
        # Grid is reused for subsequent calls
        points_vmec2 = transformer.cylindrical_to_vmec(points_cyl2)
    """

    def __init__(self, wout_filename, grid_resolution=(50, 50, 50)):
        self.wout_filename = wout_filename
        self.grid_resolution = grid_resolution
        self._grid_coords = None
        self._grid_cylindrical = None
        self._grid_built = False
        self._nfp = None

    def _build_coordinate_grid(self, n_s, n_theta, n_phi):
        """
        Build a grid of VMEC coordinates and their corresponding cylindrical
        coordinates.
        Takes advantage of nfp symmetry to reduce grid size.

        Args:
            n_s: Number of s grid points
            n_theta: Number of theta grid points
            n_phi: Number of phi grid points

        Returns:
            vmec_coords: Array of shape (n_points, 3) with (s, theta, phi) coordinates
            cylindrical_coords: Array of shape (n_points, 3) with (R, phi_cyl, Z)
                coordinates
        """
        # Get nfp from VMEC file
        with netcdf_file(self.wout_filename, "r") as f:
            self._nfp = int(f.variables["nfp"][()])

        # Create coordinate grids
        s_grid = np.linspace(0, 1, n_s)
        theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        # Take advantage of nfp symmetry - only need one field period
        phi_grid = np.linspace(0, 2 * np.pi / self._nfp, n_phi, endpoint=False)

        # Create meshgrid
        s_mesh, theta_mesh, phi_mesh = np.meshgrid(
            s_grid, theta_grid, phi_grid, indexing="ij"
        )
        # Flatten to create coordinate arrays
        s_mesh = s_mesh.flatten()
        theta_mesh = theta_mesh.flatten()
        phi_mesh = phi_mesh.flatten()

        # Remove duplicative points on the magnetic axis
        bool_mask = (s_mesh == 0) * (theta_mesh > 0)
        s_mesh = np.delete(s_mesh, bool_mask)
        theta_mesh = np.delete(theta_mesh, bool_mask)
        phi_mesh = np.delete(phi_mesh, bool_mask)

        n_points = len(s_mesh)
        vmec_coords = np.zeros((n_points, 3))
        vmec_coords[:, 0] = s_mesh
        vmec_coords[:, 1] = theta_mesh
        vmec_coords[:, 2] = phi_mesh

        # Convert to cylindrical coordinates using vmec_to_cylindrical
        points_cyl = vmec_to_cylindrical(self.wout_filename, vmec_coords)

        return vmec_coords, points_cyl

    def _ensure_grid_built(self):
        """Build the coordinate grid if not already built."""
        if not self._grid_built:
            try:
                n_s, n_theta, n_phi = self.grid_resolution
                self._grid_coords, self._grid_cylindrical = self._build_coordinate_grid(
                    n_s, n_theta, n_phi
                )
                self._grid_built = True
            except Exception as e:
                raise RuntimeError(f"Failed to build coordinate grid: {e}") from e

    def cylindrical_to_vmec(self, points_cyl, n_guesses=10, ftol=1e-6):
        """
        Convert from cylindrical coordinates to VMEC coordinates using robust
        pseudo-Cartesian coordinates x = sqrt(s)*cos(theta), y = sqrt(s)*sin(theta).
        All initial guesses are generated from the coordinate grid.

        Args:
            points_cyl: A numpy array of shape (npoints, 3) containing the
                cylindrical coordinates (R, phi, Z).
            n_guesses: Number of grid-based initial guesses to try per point
            ftol: Tolerance for root finding convergence

        Returns:
            points_vmec: A numpy array of shape (npoints, 3) containing the
                VMEC coordinates (s_vmec, theta_vmec, phi_vmec).
        """
        # Ensure grid is built
        self._ensure_grid_built()

        # Validate input shape
        if len(points_cyl.shape) != 2 or points_cyl.shape[1] != 3:
            raise ValueError("points_cyl must have shape (npoints, 3)")

        npoints = points_cyl.shape[0]
        if npoints == 0:
            raise ValueError("Input arrays cannot be empty")

        # Load VMEC data for objective function
        with netcdf_file(self.wout_filename, "r") as f:
            rmnc = f.variables["rmnc"][:]
            zmns = f.variables["zmns"][:]
            xm = f.variables["xm"][:]
            xn = f.variables["xn"][:]
            ns = int(f.variables["ns"][()])
            s_full = np.linspace(0, 1, ns)

        points_vmec = np.zeros((npoints, 3))

        def objective_function(x_norm, points_cyl_target):
            """
            Objective function using normalized coordinates
            x = sqrt(s)*cos(theta), y = sqrt(s)*sin(theta).
            This avoids singularity issues at s=0.
            """
            x_coord, y_coord = x_norm

            # Convert normalized coordinates back to s, theta
            s_i = x_coord**2 + y_coord**2
            s_i = np.clip(s_i, 0, 1)

            theta_i = np.arctan2(y_coord, x_coord)

            # Interpolate harmonics
            rmnc_s = np.zeros_like(rmnc[0, :])
            zmns_s = np.zeros_like(zmns[0, :])

            for j in range(rmnc.shape[1]):
                rmnc_s[j] = np.interp(s_i, s_full, rmnc[:, j])
                zmns_s[j] = np.interp(s_i, s_full, zmns[:, j])

            # Compute R and Z
            R_computed = 0.0
            Z_computed = 0.0
            for j in range(len(xm)):
                angle = xm[j] * theta_i - xn[j] * points_cyl_target[1]
                R_computed += rmnc_s[j] * np.cos(angle)
                Z_computed += zmns_s[j] * np.sin(angle)

            return [
                R_computed - points_cyl_target[0],
                Z_computed - points_cyl_target[2],
            ]

        def convert_to_normalized(s, theta):
            """Convert (s, theta, phi) to normalized (x, y, phi) coordinates."""
            sqrt_s = np.sqrt(max(s, 0))
            x = sqrt_s * np.cos(theta)
            y = sqrt_s * np.sin(theta)
            return [x, y]

        def convert_from_normalized(x, y):
            """Convert normalized (x, y, phi) to (s, theta, phi) coordinates."""
            s = x**2 + y**2
            s = np.clip(s, 0, 1)
            theta = np.arctan2(y, x)
            return s, theta

        def get_grid_guesses(target_point, n_guesses):
            """Get multiple grid-based initial guesses using k-nearest neighbors."""
            # Build KDTree for efficient nearest neighbor search
            tree = KDTree(self._grid_cylindrical)

            # Map target phi to fundamental domain for KDTree search only
            # Record how many field periods (integer multiples of 2π/nfp) to add back
            phi_period = 2 * np.pi / self._nfp

            # Calculate number of complete field periods in target phi
            n_field_periods = int(np.floor(target_point[1] / phi_period))

            # Map to fundamental domain [0, 2π/nfp)
            target_phi_mapped = target_point[1] - n_field_periods * phi_period
            target_mapped = target_point.copy()
            target_mapped[1] = target_phi_mapped

            # Find k nearest neighbors (more than n_guesses to have options)
            distances, indices = tree.query(target_mapped, k=n_guesses)

            # Convert to list if single neighbor
            if n_guesses == 1:
                indices = [indices]

            # Add back the field periods that were subtracted
            # Grid coords are in VMEC: (s, theta, phi)
            # We need to adjust phi to account for the field periods
            phi_offset = n_field_periods * phi_period

            selected_guesses = []
            for idx in indices:
                grid_coords = self._grid_coords[idx].copy()
                # Add back the n_field_periods * (2π/nfp) to phi coordinate
                grid_coords[2] = grid_coords[2] + phi_offset
                guess_norm = convert_to_normalized(grid_coords[0], grid_coords[1])
                selected_guesses.append(guess_norm)

            return selected_guesses

        for i in range(npoints):
            success = False

            # Get multiple grid-based guesses
            initial_guesses_normalized = get_grid_guesses(points_cyl[i, :], n_guesses)
            for x0_norm in initial_guesses_normalized:
                try:
                    sol = root(
                        objective_function,
                        x0_norm,
                        args=(points_cyl[i, :]),
                        method="hybr",
                        tol=ftol,
                    )

                    if sol.success:
                        # Convert solution back to (s, theta, phi)
                        s_result, theta_result = convert_from_normalized(
                            sol.x[0], sol.x[1]
                        )
                        points_vmec[i, 0] = s_result
                        points_vmec[i, 1] = theta_result
                        points_vmec[i, 2] = points_cyl[i, 1]
                        success = True
                        break
                except Exception:
                    continue

            if not success:
                raise RuntimeError(
                    f"Root finding failed for point {i} with coordinates "
                    f"R={points_cyl[i, 0]}, phi={points_cyl[i, 1]}, "
                    f"Z={points_cyl[i, 2]}"
                )

        return points_vmec

    def vmec_to_cylindrical(self, points_vmec):
        """
        Convert from VMEC coordinates to cylindrical coordinates.

        Args:
            points_vmec: A numpy array of shape (npoints, 3) containing the
                VMEC coordinates (s_vmec, theta_vmec, phi_vmec).

        Returns:
            points_cyl: A numpy array of shape (npoints, 3) containing the
                cylindrical coordinates (R, phi_cyl, Z).
        """
        return vmec_to_cylindrical(self.wout_filename, points_vmec)


def boozer_to_cylindrical(field, points):
    r"""
    Convert from Boozer coordinates to cylindrical coordinates.

    Args:
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        points : A numpy array of shape (npoints, 3) containing the
            Boozer coordinates (s, theta, zeta).

    Returns:
        points_cylindrical : A numpy array of shape (npoints, 3) containing the
            cylindrical coordinates (R, phi, Z).
    """
    # Validate input shape
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (npoints, 3)")

    npoints = points.shape[0]
    if npoints == 0:
        raise ValueError("Input arrays cannot be empty")

    field.set_points(points)

    points_cyl = np.zeros((npoints, 3))
    points_cyl[:, 0] = field.R()[:, 0]
    points_cyl[:, 1] = points[:, 2] - field.nu()[:, 0]
    points_cyl[:, 2] = field.Z()[:, 0]

    return points_cyl


def cylindrical_to_boozer(
    field,
    points_cyl,
    n_guesses=10,
    ftol=1e-6,
    grid_resolution=(50, 50, 50),
):
    r"""
    Convert from cylindrical coordinates to Boozer coordinates using root finding.

    Args:
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        points_cyl : A numpy array of shape (npoints, 3) containing the
            cylindrical coordinates (R, phi, Z).
        n_guesses : int, optional
            Number of grid-based initial guesses to try for each point (default: 10).
            Must be a positive integer.
        ftol : float, optional
            Tolerance for root finding convergence (default: 1e-6).
        grid_resolution : tuple of int, optional
            Grid resolution as (n_s, n_theta, n_zeta) (default: (50, 50, 50)).

    Returns:
        points_boozer : A numpy array of shape (npoints, 3) containing the
            Boozer coordinates (s, theta, zeta).
    """
    # Validate input shape
    if len(points_cyl.shape) != 2 or points_cyl.shape[1] != 3:
        raise ValueError("points_cyl must have shape (npoints, 3)")

    npoints = points_cyl.shape[0]
    if npoints == 0:
        raise ValueError("Input arrays cannot be empty")

    transformer = BoozerCoordinateTransformer(field, grid_resolution)
    return transformer.cylindrical_to_boozer(points_cyl, n_guesses=n_guesses, ftol=ftol)


def vmec_to_boozer(wout_filename, field, points_vmec, ftol=1e-6):
    r"""
    Convert from VMEC coordinates to Boozer coordinates.

    Args:
        wout_filename : str
            The name of the VMEC wout file.
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        points_vmec : A numpy array of shape (npoints, 3) containing the
            VMEC coordinates (s_vmec, theta_vmec, phi_vmec).
        ftol : float, optional
            Tolerance for root finding convergence (default: 1e-6).

    Returns:
        points_boozer : A numpy array of shape (npoints, 3) containing the
            Boozer coordinates (s, theta, zeta).
    """
    # Validate input shape
    if len(points_vmec.shape) != 2 or points_vmec.shape[1] != 3:
        raise ValueError("points_vmec must have shape (npoints, 3)")

    npoints = points_vmec.shape[0]
    if npoints == 0:
        raise ValueError("Input arrays cannot be empty")

    # Load VMEC and booz_xform data
    f = netcdf_file(wout_filename, mmap=False)
    lmns = f.variables["lmns"][()]
    mnmax = f.variables["mnmax"][()]
    ns = f.variables["ns"][()]
    xm = f.variables["xm"][()]
    xn = f.variables["xn"][()]
    f.close()

    s_full_grid = np.linspace(0, 1, ns)
    s_half_grid = (s_full_grid[0:-1] + s_full_grid[1::]) / 2.0

    # Create splines for lmns
    lmns_splines = []
    for jmn in range(mnmax):
        lmns_splines.append(InterpolatedUnivariateSpline(s_half_grid, lmns[1::, jmn]))

    def vartheta_vmec(s, theta_vmec, phi_vmec):
        """Compute vartheta from VMEC data."""
        lmns = np.zeros((1, mnmax))
        for jmn in range(mnmax):
            lmns[:, jmn] = lmns_splines[jmn](s)

        angle = xm * theta_vmec - xn * phi_vmec
        sinangle = np.sin(angle)

        lambd = np.sum(lmns * sinangle)
        vartheta = theta_vmec + lambd
        return vartheta

    def vartheta_phi_vmec(s, theta_b, zeta_b):
        """Compute PEST angles from Boozer coordinates."""
        points = np.zeros((1, 3))
        points[:, 0] = s
        points[:, 1] = theta_b
        points[:, 2] = zeta_b
        field.set_points(points)
        nu = field.nu()[0, 0]
        iota = field.iota()[0, 0]
        vartheta = theta_b - iota * nu
        phi = zeta_b - nu
        return vartheta, phi

    def func_root(x, s, vartheta_target, phi_target):
        """Root finding function."""
        theta_b = x[0]
        zeta_b = x[1]
        vartheta, phi = vartheta_phi_vmec(s, theta_b, zeta_b)
        vartheta_diff = np.arctan2(
            np.sin(vartheta - vartheta_target), np.cos(vartheta - vartheta_target)
        )
        phi_diff = np.arctan2(np.sin(phi - phi_target), np.cos(phi - phi_target))
        return [vartheta_diff, phi_diff]

    points_boozer = np.zeros((npoints, 3))
    for i in range(npoints):
        s_vmec = points_vmec[i, 0]
        theta_vmec = points_vmec[i, 1]
        phi_vmec = points_vmec[i, 2]

        vartheta = vartheta_vmec(s_vmec, theta_vmec, phi_vmec)
        sol = root(
            func_root,
            [vartheta, phi_vmec],
            args=(s_vmec, vartheta, phi_vmec),
            method="hybr",
            tol=ftol,
        )
        if sol.success:
            points_boozer[i, 0] = s_vmec
            points_boozer[i, 1] = sol.x[0]
            points_boozer[i, 2] = sol.x[1]
        else:
            raise RuntimeError(
                f"Root finding failed for point {i} with coordinates "
                f"s={s_vmec}, theta_vmec={theta_vmec}, phi_vmec={phi_vmec}. "
            )

    return points_boozer


def boozer_to_vmec(wout_filename, field, points_boozer, ftol=1e-6):
    r"""
    Convert from Boozer coordinates to VMEC coordinates.

    Args:
        wout_filename : str
            The name of the VMEC wout file.
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        points_boozer : A numpy array of shape (npoints, 3) containing the
            Boozer coordinates (s, theta, zeta).
        ftol : float, optional
            Tolerance for root finding convergence (default: 1e-6).

    Returns:
        points_vmec : A numpy array of shape (npoints, 3) containing the
            VMEC coordinates (s_vmec, theta_vmec, phi_vmec).
    """
    # Validate input shape
    if len(points_boozer.shape) != 2 or points_boozer.shape[1] != 3:
        raise ValueError("points_boozer must have shape (npoints, 3)")

    npoints = points_boozer.shape[0]
    if npoints == 0:
        raise ValueError("Input arrays cannot be empty")

    # Load VMEC and booz_xform data
    f = netcdf_file(wout_filename, mmap=False)
    lmns = f.variables["lmns"][()]
    mnmax = f.variables["mnmax"][()]
    ns = f.variables["ns"][()]
    xm = f.variables["xm"][()]
    xn = f.variables["xn"][()]
    f.close()

    s_full_grid = np.linspace(0, 1, ns)
    s_half_grid = (s_full_grid[0:-1] + s_full_grid[1::]) / 2.0

    # Create splines for lmns
    lmns_splines = []
    for jmn in range(mnmax):
        lmns_splines.append(InterpolatedUnivariateSpline(s_half_grid, lmns[1::, jmn]))

    def vartheta_vmec(s, theta_vmec, phi_vmec):
        """Compute vartheta from VMEC data."""
        lmns = np.zeros((1, mnmax))
        for jmn in range(mnmax):
            lmns[:, jmn] = lmns_splines[jmn](s)

        angle = xm * theta_vmec - xn * phi_vmec
        sinangle = np.sin(angle)

        lambd = np.sum(lmns * sinangle)
        vartheta = theta_vmec + lambd
        return vartheta

    def vartheta_phi_vmec(s, theta_b, zeta_b):
        """Compute PEST angles from Boozer coordinates."""
        points = np.zeros((1, 3))
        points[:, 0] = s
        points[:, 1] = theta_b
        points[:, 2] = zeta_b
        field.set_points(points)
        nu = field.nu()[0, 0]
        iota = field.iota()[0, 0]
        vartheta = theta_b - iota * nu
        phi = zeta_b - nu
        return vartheta, phi

    def func_root(x, s, vartheta_boozer, zeta_boozer):
        """Root finding function."""
        theta_vmec = x[0]
        # Compute PEST angles from desired Boozer coordinates
        vartheta_target, phi_target = vartheta_phi_vmec(s, vartheta_boozer, zeta_boozer)
        # Compute PEST angles from VMEC coordinates
        vartheta = vartheta_vmec(s, theta_vmec, phi_target)
        vartheta_diff = np.arctan2(
            np.sin(vartheta - vartheta_target), np.cos(vartheta - vartheta_target)
        )
        return [vartheta_diff]

    points_vmec = np.zeros((npoints, 3))
    for i in range(npoints):
        s = points_boozer[i, 0]
        theta_b = points_boozer[i, 1]
        zeta_b = points_boozer[i, 2]

        sol = root(
            func_root,
            [theta_b],
            args=(s, theta_b, zeta_b),
            method="hybr",
            tol=ftol,
        )
        if sol.success:
            points_vmec[i, 0] = s
            points_vmec[i, 1] = sol.x[0]
            vartheta, points_vmec[i, 2] = vartheta_phi_vmec(s, theta_b, zeta_b)
        else:
            raise RuntimeError(
                f"Root finding failed for point {i} with coordinates "
                f"s={s}, theta_b={theta_b}, zeta_b={zeta_b}. "
            )

    return points_vmec


def vmec_to_cylindrical(wout_filename, points_vmec):
    r"""
    Convert from VMEC coordinates to cylindrical coordinates.

    Args:
        wout_filename : str
            The name of the VMEC wout file.
        points_vmec : A numpy array of shape (npoints, 3) containing the
            VMEC coordinates (s_vmec, theta_vmec, phi_vmec).

    Returns:
        points_cyl : A numpy array of shape (npoints, 3) containing the
            cylindrical coordinates (R, phi_cyl, Z).
    """
    # Validate input shape
    if len(points_vmec.shape) != 2 or points_vmec.shape[1] != 3:
        raise ValueError("points_vmec must have shape (npoints, 3)")

    npoints = points_vmec.shape[0]
    if npoints == 0:
        raise ValueError("Input arrays cannot be empty")

    # Load VMEC data
    with netcdf_file(wout_filename, "r") as f:
        rmnc = f.variables["rmnc"][:]  # R harmonics (cos)
        zmns = f.variables["zmns"][:]  # Z harmonics (sin)
        xm = f.variables["xm"][:]  # poloidal mode numbers
        xn = f.variables["xn"][:]  # toroidal mode numbers
        ns = int(f.variables["ns"][()])  # number of radial surfaces (scalar)
        s_full = np.linspace(0, 1, ns)  # full radial grid

    points_cyl = np.zeros((npoints, 3))
    # For each point, compute R and Z using VMEC Fourier harmonics
    for i in range(npoints):
        s_i = points_vmec[i, 0]
        theta_i = points_vmec[i, 1]
        phi_i = points_vmec[i, 2]

        # Interpolate harmonics to the desired s value
        rmnc_s = np.zeros_like(rmnc[0, :])
        zmns_s = np.zeros_like(zmns[0, :])

        for j in range(rmnc.shape[1]):  # Loop over mode numbers
            # Interpolate rmnc and zmns to s_i
            rmnc_s[j] = np.interp(s_i, s_full, rmnc[:, j])
            zmns_s[j] = np.interp(s_i, s_full, zmns[:, j])

        # Compute R and Z using Fourier series
        for j in range(len(xm)):
            angle = xm[j] * theta_i - xn[j] * phi_i
            points_cyl[i, 0] += rmnc_s[j] * np.cos(angle)
            points_cyl[i, 2] += zmns_s[j] * np.sin(angle)

    # phi_cyl is the same as phi_vmec
    points_cyl[:, 1] = points_vmec[:, 2]

    return points_cyl


def cylindrical_to_vmec(
    wout_filename,
    points_cyl,
    n_guesses=10,
    ftol=1e-6,
    grid_resolution=(50, 50, 50),
):
    r"""
    Convert from cylindrical coordinates to VMEC coordinates using robust
    pseudo-Cartesian coordinates x = sqrt(s)*cos(theta), y = sqrt(s)*sin(theta).

    Args:
        wout_filename : str
            The name of the VMEC wout file.
        points_cyl : A numpy array of shape (npoints, 3) containing the
            cylindrical coordinates (R, phi_cyl, Z).
        n_guesses : int, optional
            Number of grid-based initial guesses to try for each point (default: 10).
            Must be a positive integer.
        ftol : float, optional
            Tolerance for root finding convergence (default: 1e-6).
        grid_resolution : tuple of int, optional
            Grid resolution as (n_s, n_theta, n_phi) (default: (50, 50, 50)).

    Returns:
        points_vmec : A numpy array of shape (npoints, 3) containing the
            VMEC coordinates (s_vmec, theta_vmec, phi_vmec).
    """
    # Validate input shape
    if len(points_cyl.shape) != 2 or points_cyl.shape[1] != 3:
        raise ValueError("points_cyl must have shape (npoints, 3)")

    npoints = points_cyl.shape[0]
    if npoints == 0:
        raise ValueError("Input arrays cannot be empty")

    transformer = VMECCoordinateTransformer(wout_filename, grid_resolution)
    return transformer.cylindrical_to_vmec(points_cyl, n_guesses=n_guesses, ftol=ftol)
