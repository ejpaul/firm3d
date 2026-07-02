import numpy as np

__all__ = ["OrbitClassification"]


def _isotonic_regression(y, increasing=True):
    r"""
    Isotonic regression (pool-adjacent-violators) on uniformly spaced samples.

    Returns the closest (in least squares) monotonic sequence to ``y``.
    Use ``increasing=False`` on the branch approaching ``|B|`` minimum and
    ``increasing=True`` on the branch leaving it.
    """
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        return y
    if not increasing:
        return _isotonic_regression(y[::-1], increasing=True)[::-1]

    block_sums = [float(y[0])]
    block_weights = [1.0]
    block_counts = [1]
    for value in y[1:]:
        block_sums.append(float(value))
        block_weights.append(1.0)
        block_counts.append(1)
        while (
            len(block_sums) >= 2
            and block_sums[-2] / block_weights[-2] > block_sums[-1] / block_weights[-1]
        ):
            block_sums[-2] += block_sums[-1]
            block_weights[-2] += block_weights[-1]
            block_counts[-2] += block_counts[-1]
            del block_sums[-1], block_weights[-1], block_counts[-1]

    y_mono = np.empty(y.size)
    index = 0
    for block_sum, block_weight, block_count in zip(
        block_sums, block_weights, block_counts
    ):
        y_mono[index : index + block_count] = block_sum / block_weight
        index += block_count
    return y_mono


def _monotonic_fit(x, y, x_eval=None, increasing=True):
    r"""
    Force monotonicity on non-monotonic samples.

    Applies isotonic regression on the sample points in array order (left/right
    branch direction). If ``x_eval`` differs from ``x``, evaluates a
    shape-preserving cubic (PCHIP) through the isotonic values on sorted ``x``;
    PCHIP preserves the monotonicity of its input data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_eval is None:
        x_eval = x
    x_eval = np.asarray(x_eval, dtype=float)
    if y.size < 2:
        return np.interp(x_eval, x, y)

    if x_eval is x or (x_eval.shape == x.shape and np.array_equal(x_eval, x)):
        return _isotonic_regression(y, increasing=increasing)

    y_mono = _isotonic_regression(y, increasing=increasing)
    sort_idx = np.argsort(x)
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(x[sort_idx], y_mono[sort_idx], extrapolate=False)(
        x_eval
    )


def _chi_branch_shift(chi_ref, chi_target, period=2 * np.pi):
    """Shift ``chi_ref`` by an integer multiple of ``period`` to overlap ``chi_target``."""
    chi_ref = np.asarray(chi_ref, dtype=float)
    chi_target = np.atleast_1d(np.asarray(chi_target, dtype=float))
    if chi_target.size == 0:
        return 0.0
    chi_center = np.median(chi_target)
    k0 = int(np.round((chi_center - np.median(chi_ref)) / period))
    best_k = k0
    best_score = np.inf
    for k in range(k0 - 2, k0 + 3):
        shifted = chi_ref + k * period
        score = np.max(np.min(np.abs(chi_target[:, None] - shifted[None, :]), axis=1))
        if score < best_score:
            best_score = score
            best_k = k
    return best_k * period


def _chi_on_branch(chi, chi_center, period=2 * np.pi):
    """Express ``chi`` on the same 2pi branch as ``chi_center``."""
    chi = np.asarray(chi, dtype=float)
    return chi_center + ((chi - chi_center + period / 2) % period) - period / 2


class OrbitClassification:
    r"""
    A class to classify the trapping state and other diagnostics of a particle based on
    its trajectory and mirror points.

    This class analyzes charged particle orbits in a magnetic field and classifies them
    into different trapping regimes:

    - **Banana trapped (0)**: Particles trapped in the lowest-order
      toroidal magnetic well, executing large banana-shaped orbits that
      close poloidally.
    - **Barely trapped (1)**: Particles near the trapped-passing boundary
      with large excursions in helical angle chi
      (dchi > barely_trapped_crit), transitioning between banana trapped
      and passing states.
    - **Ripple trapped (2)**: Particles trapped in higher-order magnetic
      ripples, executing small bounce motions
      (dchi < ripple_trapped_crit * dchi_predicted).

    The classification is based on analyzing bounce segments between
    mirror points, computing the change in helical angle (dchi), and
    comparing it to critical thresholds. Additional diagnostics include
    parallel action variable (Jpar), gamma_c parameter, and transition
    statistics between trapping states.
    """

    def __init__(
        self,
        field,
        Ekin,
        mass,
        charge,
        helicity_M,
        helicity_N,
        barely_trapped_crit=2 * np.pi * 1.25,
        ripple_trapped_crit=0.5,
        ripple_trapped_crit_min=0.05,
    ):
        r"""
        Initialize the OrbitClassification class.

        Args:
            field: BoozerMagneticField object.
            Ekin (float): Kinetic energy of the particle [SI units, Joules].
            mass (float): Mass of the particle [SI units, kg].
            charge (float): Charge of the particle [SI units, Coulombs].
            helicity_M (int): Poloidal mode number of the helicity being
                analyzed (M=0 for axisymmetric, M≠0 for helical).
            helicity_N (int): Toroidal mode number of the helicity being
                analyzed. The helical angle is defined as chi = M*theta - N*zeta.
            barely_trapped_crit (float, optional): Critical angle [radians]
                for barely trapped classification. Particles with
                dchi > barely_trapped_crit are classified as barely trapped.
                Default is 2π*1.25.
            ripple_trapped_crit (float, optional): Critical ratio for ripple
                trapped classification. Particles with
                dchi < ripple_trapped_crit * dchi_predicted are classified
                as ripple trapped. Default is 0.5.

        Notes:
            - The helicity defines the direction in which modB contours close.
            - For M=0 (axisymmetric), modB contours close poloidally, so
              theta is used as the mapping coordinate (Mp=1, Np=0).
            - For M≠0 (helical), modB contours close toroidally, so zeta
              is used as the mapping coordinate (Mp=0, Np=-1).
        """
        self.field = field
        self.Ekin = Ekin
        self.mass = mass
        self.charge = charge
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self.nfp = field.nfp
        # Critical angle for barely trapped particle classification
        self.barely_trapped_crit = barely_trapped_crit
        # Critical value of dchi/dchi_predicted for ripple trapped
        self.ripple_trapped_crit = ripple_trapped_crit

        # If modB contours close poloidally, then use theta as mapping coordinate
        if helicity_M == 0:
            self.helicity_Mp = 1
            self.helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate
        else:
            self.helicity_Mp = 0
            self.helicity_Np = -1

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to Boozer angles (theta, zeta).

        The helical coordinate system is defined by:
        - chi = M*theta - N*zeta (helical angle along which modB varies)
        - eta = Mp*theta - Np*zeta (mapping angle perpendicular to chi)

        This method inverts the transformation to obtain (theta, zeta) from (chi, eta).

        Args:
            chi (float or array): Helical angle chi [radians].
            eta (float or array): Mapping angle eta [radians].

        Returns:
            tuple: (theta, zeta) where
                - theta (float or array): Poloidal Boozer angle [radians].
                - zeta (float or array): Toroidal Boozer angle [radians].

        Notes:
            The transformation is:
            theta = (Np*chi - N*eta) / (Np*M - N*Mp)
            zeta = (Mp*chi - M*eta) / (Np*M - N*Mp)
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def classify_orbit(self, res_ty, res_hit):
        r"""
        Classify the orbit of a particle based on its trajectory and mirror points.

        This method analyzes the particle trajectory, identifies bounce
        segments between mirror points (where vpar=0), and classifies each
        segment as banana trapped, barely trapped, or ripple trapped based
        on the change in helical angle dchi.

        Args:
            res_ty (ndarray): Trajectory array with shape (ntimes, ncols) containing:
                - Column 0: time [SI units, seconds]
                - Column 1: s (radial flux coordinate)
                - Column 2: theta (poloidal Boozer angle) [radians]
                - Column 3: zeta (toroidal Boozer angle) [radians]
                - Column 4: vpar (parallel velocity) [SI units, m/s]

            res_hit (ndarray): Hit points array with shape (nhits, ncols)
                containing:
                - Column 0: time of hit [seconds]
                - Column 1: hit type (0=vpar plane, other values for
                  walls/boundaries)
                - Additional columns may contain hit location information

        Returns:
            dict: Dictionary containing classification results and
                diagnostics with keys:

                **Basic trajectory information:**
                - 'losttime' (float): Total integration time before
                  particle was lost to the wall [seconds]
                - 'nbounce' (int): Number of bounce segments (times
                  particle hit vpar=0 plane)
                - 'bounce_times' (list): Times when bounces occur
                  [seconds], length nbounce
                - 'lam' (float): Trapping parameter
                  λ = v_perp^2 / (v^2 * B), [1/Tesla]
                - 'point0' (ndarray): Initial position [s, theta, zeta]
                - 'vpar0' (float): Initial parallel velocity [m/s]

                **Per-bounce-segment arrays:**
                These arrays have length nbounce-1 if pre-first-bounce
                segment is not classified, otherwise length is nbounce.
                - 'status' (ndarray): Trapping state classification for
                  each segment: 0 = banana trapped, 1 = barely trapped,
                  2 = ripple trapped
                - 'dss' (list): Change in s over each bounce segment
                - 'dalphas' (list): Change in field line label
                  alpha = theta - iota*zeta [radians]
                - 'dchis' (ndarray): Change in helical angle chi over
                  each segment [radians]
                - 'dchis_predicted' (ndarray): Expected dchi based on
                  mirror point locations [radians]
                - 'gammacs' (list): Gamma_c parameter =
                  (2/π)*arctan(|ds|/|dalpha|) for each segment
                - 'Jpars' (list): Parallel action J_|| for each
                  half-bounce segment
                - 's_means' (list): Mean radial position during each
                  bounce segment

                **Cumulative statistics:**
                - 'banana_frac' (float): Fraction of bounces classified
                  as banana trapped
                - 'barely_trapped_frac' (float): Fraction of bounces
                  classified as barely trapped
                - 'ripple_trapped_frac' (float): Fraction of bounces
                  classified as ripple trapped
                - 'ntransitions' (int): Number of transitions between
                  trapping states
                - 'Jpar_var' (float): Normalized standard deviation of
                  J_|| over full bounce periods (computed as
                  std(Jpar_full)/mean(Jpar_full) where
                  Jpar_full = Jpar[i] + Jpar[i+1])
                - 'gammac_mean' (float): Mean value of gamma_c over all
                  bounce segments

        Notes:
            - Bounce segments are identified by finding times when
              res_hit[:,1]==0 (vpar plane hits)
            - The classification criteria are:
                * Barely trapped: dchi > barely_trapped_crit
                * Ripple trapped:
                  dchi < ripple_trapped_crit * dchi_predicted
                * Banana trapped: otherwise
            - J_|| is computed using the trapezoidal rule as:
              ∫ v_|| dζ / (b·∇ζ)
            - Mirror segments requires at least 2 bounces for classification;
              If no bounces, particle is classified as passing;
              If 1 bounce, classified as barely trapped if
              dchi_total > barely_trapped_crit.
              When no classification is possible, returns zeros.
            - Requires at least 4 bounces for Jpar_var computation
        """

        # Compute all of the times when the particle bounces off vpar plane
        # These are mirror points where particle reverses direction
        bounce_times = []
        nhits = len(res_hit)

        for j in range(nhits):
            if res_hit[j, 1] == 0:  # vpar plane was hit
                bounce_times.append(res_hit[j, 0])

        nbounce = len(bounce_times)

        # Unwrap theta to handle periodic boundary crossings (prevents jumps at ±π)
        thetas = np.unwrap(res_ty[:, 2])
        zetas = res_ty[:, 3]

        # Extract initial conditions
        point = np.zeros((1, 3))
        point[0, :] = res_ty[0, 1:4]  # Initial position [s, theta, zeta]
        vpar_init = res_ty[0, 4]  # Initial parallel velocity [m/s]

        # Compute trapping parameter λ = v_perp^2 / (v^2 * B)
        # From energy conservation: v^2 = vpar^2 + vperp^2 = 2*Ekin/mass
        # At mirror point: vpar=0, so vperp^2 = v^2, giving B_mirror = v^2/(λ*v^2) = 1/λ
        self.field.set_points(point)
        modB_0 = self.field.modB()[0, 0]
        lam = (2 * self.Ekin / self.mass - vpar_init**2) / (
            modB_0 * 2 * self.Ekin / self.mass
        )
        modB_crit = 1 / lam  # Critical |B| at mirror points

        # Initialize arrays to store per-bounce-segment diagnostics
        Jpars = []  # Parallel action variable for each half-bounce
        s_means = []  # Mean radial position during each bounce
        alpha_means = []  # Mean field line label alpha during each bounce segment
        dchis = []  # Change in helical angle for each bounce segment
        dchis_predicted = []  # Expected dchi based on mirror point locations
        gammacs = []  # Orbit width parameter gamma_c for each segment
        dss = []  # Change in radial coordinate for each bounce
        dalphas = []  # Change in field line label alpha for each bounce
        debug_data = []  # per-segment data for plot_bounce_segment()
        # Iterate over bounce segments (from mirror point j to mirror point j+1)
        for j in range(nbounce - 1):
            # Find trajectory indices corresponding to this bounce segment
            index_start = np.argmin(np.abs(bounce_times[j] - res_ty[:, 0]))
            index_end = np.argmin(np.abs(bounce_times[j + 1] - res_ty[:, 0]))

            res_ty_seg = res_ty[index_start : index_end + 1, :]
            # --- Trajectory arrays (field evaluation on trajectory points only) ---
            point = np.zeros((len(res_ty_seg), 3))
            point[:, 0] = res_ty_seg[:, 1]   # s
            point[:, 1] = res_ty_seg[:, 2]   # theta
            point[:, 2] = res_ty_seg[:, 3]   # zeta
            self.field.set_points(point)
            modB_traj     = self.field.modB()[:, 0]
            vpar_traj = res_ty[index_start : index_end + 1, 4]

            lam_traj = (2 * self.Ekin / self.mass - vpar_traj**2) / (
                modB_traj * 2 * self.Ekin / self.mass
            )
            modB_crit_traj = np.mean(1 / lam_traj)

            # Compute radial excursion during this bounce
            ds = res_ty[index_end, 1] - res_ty[index_start, 1]
            dss.append(ds)

            # Compute change in helical angle chi = M*theta - N*zeta
            # This is the primary quantity used for classification
            dtheta = thetas[index_end] - thetas[index_start]
            dzeta = zetas[index_end] - zetas[index_start]
            thetas_unwrap = np.unwrap(thetas[index_start : index_end + 1])
            zetas_unwrap = np.unwrap(zetas[index_start : index_end + 1])
            chis_unwrap = np.unwrap(self.helicity_M * thetas_unwrap - self.helicity_N * zetas_unwrap)
            dchi = np.abs(chis_unwrap[-1] - chis_unwrap[0])
            dchis.append(np.abs(dchi))
 
            # Compute mean radial position during this bounce segment
            mean_s = np.mean(res_ty[index_start : index_end + 1, 1])
            s_means.append(mean_s)

            # Compute mean rotational transform during this bounce segment
            point = np.zeros((1, 3))
            point[0, 0] = mean_s
            self.field.set_points(point)
            iota_s = self.field.iota()[0, 0]

            # Compute mean field-line label alpha during this bounce segment
            mean_alpha = np.mean(np.unwrap(res_ty[index_start : index_end + 1, 2] - iota_s * res_ty[index_start : index_end + 1, 3]))
            alpha_means.append(mean_alpha)

            # Trajectory chi center: used to center the field-line sweep and to
            # locate the bounding peaks.
            chi_traj_center = 0.5 * (np.max(chis_unwrap) + np.min(chis_unwrap))

            # Sample modB along the mean field line sweeping ±2π in χ at fixed α.
            # Using χ as the sweep variable gives a grid uniform in χ and centred
            # on the trajectory.  Inverting χ = M*θ − N*ζ and α = θ − ι*ζ gives:
            #   θ = (N*α − ι*χ) / (N − ι*M)
            #   ζ = (M*α − χ)   / (N − ι*M)
            _denom = self.helicity_N - iota_s * self.helicity_M
            chi_grid_mean = np.linspace(
                chi_traj_center - 2 * np.pi,
                chi_traj_center + 2 * np.pi,
                1000,
            )
            zeta_grid = (self.helicity_M * mean_alpha - chi_grid_mean) / _denom
            theta_fieldline = (self.helicity_N * mean_alpha - iota_s * chi_grid_mean) / _denom
            points = np.zeros((1000, 3))
            points[:, 0] = mean_s
            points[:, 1] = theta_fieldline
            points[:, 2] = zeta_grid
            self.field.set_points(points)
            modB = self.field.modB()[:, 0]

            # Locate the well the particle is bouncing in by finding the nearest
            # |B| peaks on either side of the trajectory center, then taking the
            # global Bmin between those two peaks.  Bounding the branches to
            # [peak, minimum, peak] ensures both sides are monotone.
            idx_center = len(chi_grid_mean) // 2

            # Nearest |B| peak on the low-chi side (left half) and high-chi side
            # (right half).  chi_grid_mean is increasing so lower indices = lower chi.
            idx_l_peak = int(np.argmax(modB[: idx_center]))
            idx_r_peak = idx_center + int(np.argmax(modB[idx_center :])) 
            # Identified maxima are more than a field period apart
            if np.abs(chi_grid_mean[idx_l_peak] - chi_grid_mean[idx_r_peak]) > 1.25 * 2*np.pi:
                # Find min between max l and center
                idx_min_l = int(np.argmin(modB[idx_l_peak : idx_center + 1]))
                # Find min between center and max r
                idx_min_r = int(np.argmin(modB[idx_center : idx_r_peak + 1]))
                # Find max between min l and center 
                idx_max_l = int(np.argmax(modB[(idx_min_l + idx_l_peak) : idx_center + 1]))
                # Find max between center and min r
                idx_max_r = int(np.argmax(modB[idx_center : (idx_center +idx_min_r) + 1]))
                # Choose larger of the two maxima 
                if modB[idx_min_l + idx_l_peak + idx_max_l] > modB[idx_center + idx_max_r]:
                    idx_l_peak = idx_min_l + idx_l_peak + idx_max_l
                else:
                    idx_r_peak = idx_center + idx_max_r

            # Global Bmin between the two flanking peaks.
            min_loc = idx_l_peak + int(np.argmin(modB[idx_l_peak : idx_r_peak + 1]))
            chi_min = chi_grid_mean[min_loc]
            # Split field line into left (low-chi) and right (high-chi) branches,
            # bounded by the peaks.
            chi_left = chi_grid_mean[idx_l_peak : min_loc + 1]
            chi_right = chi_grid_mean[min_loc : idx_r_peak + 1]
            modB_left = modB[idx_l_peak : min_loc + 1]
            modB_right = modB[min_loc : idx_r_peak + 1]

            # Force monotonic |B| on each side of the well minimum.
            modB_left_mon = _monotonic_fit(chi_left, modB_left, chi_left, increasing=False)
            modB_right_mon = _monotonic_fit(chi_right, modB_right, chi_right, increasing=True)
            chi_mirror_left = chi_left[np.argmin(np.abs(modB_left_mon - modB_crit_traj))]
            chi_mirror_right = chi_right[np.argmin(np.abs(modB_right_mon - modB_crit_traj))]

            # Classification quantities derived from already-available variables.
            dalpha = dtheta - iota_s * dzeta
            dalphas.append(dalpha)
            gammac = (2 / np.pi) * np.arctan(np.abs(ds) / np.abs(dalpha))
            gammacs.append(gammac)

            # Predicted dchi: 2× distance from chi_min to the nearer mirror point.
            dchi_predicted = np.min([
                np.abs(2 * (chi_mirror_left  - chi_min)),
                np.abs(2 * (chi_mirror_right - chi_min)),
                2 * np.pi,
            ])
            dchis_predicted.append(dchi_predicted)

            # Per-segment data for plot_bounce_segment.  Field-line sweep and
            # monotonic-fit arrays are stored here so plot_bounce_segment only
            # needs to unpack and plot (no re-evaluation of the field).
            debug_data.append({
                'dchi': dchi,
                'dchi_predicted': dchi_predicted,
                'modB_crit': modB_crit,
                'modB_crit_traj': modB_crit_traj,
                'chi_min': chi_min,
                'chi_mirror_left': chi_mirror_left,
                'chi_mirror_right': chi_mirror_right,
                'mean_s': mean_s,
                'mean_alpha': mean_alpha,
                'iota_s': iota_s,
                'chi_traj_center': chi_traj_center,
                'res_ty_segment': res_ty[index_start:index_end + 1, :].copy(),
                # Field-line sweep and monotonic well (precomputed for plotting)
                'chi_grid_mean': chi_grid_mean.copy(),
                'modB_grid_mean': modB.copy(),
                'chi_left_mon': chi_left.copy(),
                'chi_right_mon': chi_right.copy(),
                'modB_left_mon': modB_left_mon.copy(),
                'modB_right_mon': modB_right_mon.copy(),
            })

            # Compute parallel action variable J_|| = ∮ v_|| dℓ_|| / (2π)
            # Using the canonical form: J_|| = ∫ v_|| dζ / (b·∇ζ)
            # where b·∇ζ = B / (G + ιI) in Boozer coordinates
            points = np.zeros((index_end - index_start + 1, 3))
            points[:, 0] = res_ty[index_start : index_end + 1, 1]
            points[:, 1] = res_ty[index_start : index_end + 1, 2]
            points[:, 2] = res_ty[index_start : index_end + 1, 3]
            self.field.set_points(points)
            bdotgradzeta = self.field.modB()[:, 0] / (
                self.field.G()[:, 0] + self.field.iota()[:, 0] * self.field.I()[:, 0]
            )
            vpar = res_ty[index_start : index_end + 1, 4]

            # Integrate using trapezoidal rule
            vpar_center = 0.5 * (vpar[1::] + vpar[0:-1])
            bdotgradzeta_center = 0.5 * (bdotgradzeta[1::] + bdotgradzeta[0:-1])
            delta_zeta = points[1::, 2] - points[0:-1, 2]
            Jpar = np.sum(vpar_center * delta_zeta / bdotgradzeta_center)
            Jpars.append(Jpar)

        dchis = np.array(dchis)
        dchis_predicted = np.array(dchis_predicted)

        chi = self.helicity_M * thetas - self.helicity_N * zetas

        # Classify the trapping state based on dchi for each bounce segment
        if nbounce == 0:
            # Particle never bounced, no segments to classify
            status = np.array([])
            banana_frac = 0.0
            barely_trapped_frac = 0.0
            ripple_trapped_frac = 0.0
            Jpar_var = 0.0
            gammac_mean = 0.0
            ntransitions = 0
        elif nbounce == 1:
            # Particle only bounced once - cannot classify trapping state unless
            # it is barely trapped
            dchi_total = np.abs(chi[-1] - chi[0])
            if dchi_total > self.barely_trapped_crit:
                # Only one mirror point — no full bounce segment exists.
                # Treat the entire trajectory as a single pre-first-bounce segment:
                # classify as barely trapped if the total chi excursion exceeds the
                # threshold. All other per-segment quantities are undefined; insert
                # zero placeholders so all arrays have consistent length (1 or 0).
                status = np.array([1])
                dchis = np.array([dchi_total])
                dchis_predicted = np.insert(dchis_predicted, 0, 0)
                gammacs = np.insert(gammacs, 0, 0)
                Jpars = np.insert(Jpars, 0, 0)
                s_means = np.insert(s_means, 0, 0)
                dalphas = np.insert(dalphas, 0, 0)
                dss = np.insert(dss, 0, 0)
                banana_frac = 0.0
                barely_trapped_frac = 1.0
                ripple_trapped_frac = 0.0
            else:
                status = np.array([])
                banana_frac = 0.0
                barely_trapped_frac = 0.0
                ripple_trapped_frac = 0.0
            Jpar_var = 0.0
            gammac_mean = 0.0
            ntransitions = 0
        else:
            # Classification logic:
            # - Start with all segments as banana trapped (status=0)
            # - Overwrite to barely trapped (status=1) if
            #   dchi > barely_trapped_crit
            # - Overwrite to ripple trapped (status=2) if
            #   dchi < ripple_trapped_crit * dchi_predicted
            status = np.zeros_like(dchis)
            status[dchis > self.barely_trapped_crit] = 1
            status[dchis < self.ripple_trapped_crit * dchis_predicted] = 2

            # Compute mean gamma_c over all bounce segments
            gammac_mean = np.mean(gammacs)
            # Compute variation in J_|| over full bounce periods
            # Full bounce = half-bounce up + half-bounce down
            # Low variation indicates good adiabatic invariant
            if nbounce > 3:
                # Sum consecutive half-bounces
                Jpars_arr = np.array(Jpars)
                Jpar_full = Jpars_arr[0:-1] + Jpars_arr[1::]
                # Normalized std deviation
                Jpar_var = np.std(Jpar_full) / np.mean(Jpar_full)
            else:
                Jpar_var = 0.0

            # Consider pre-first-bounce as incomplete bounce segment,
            # try to classify as barely trapped segment.
            end_index = np.argmin(np.abs(bounce_times[0] - res_ty[:, 0]))
            dchi_init = np.abs(chi[end_index] - chi[0])
            if dchi_init > self.barely_trapped_crit:
                status = np.insert(status, 0, 1)
                dchis = np.insert(dchis, 0, dchi_init)
                # Below quantities not well-defined for pre-first-bounce segment,
                # insert zeros as placeholders:
                dchis_predicted = np.insert(dchis_predicted, 0, 0)
                gammacs = np.insert(gammacs, 0, 0)
                Jpars = np.insert(Jpars, 0, 0)
                s_means = np.insert(s_means, 0, 0)
                dalphas = np.insert(dalphas, 0, 0)
                dss = np.insert(dss, 0, 0)
                # Keep debug_data aligned with status/dchis.
                debug_data.insert(0, None)

            # Compute fraction of time in each trapping state
            barely_trapped_frac = np.count_nonzero(
                dchis > self.barely_trapped_crit
            ) / len(dchis)
            ripple_trapped_frac = np.count_nonzero(
                (dchis < self.ripple_trapped_crit * dchis_predicted) & (np.abs(dchis_predicted) > self.ripple_trapped_crit_min*2*np.pi)
            ) / len(dchis)
            banana_frac = np.count_nonzero(
                (dchis <= self.barely_trapped_crit)
                * (dchis >= self.ripple_trapped_crit * dchis_predicted)
                * (np.abs(dchis_predicted) > self.ripple_trapped_crit_min*2*np.pi)
            ) / len(dchis)

            # Count transitions between different trapping states
            # Fewer transitions = more stable classification
            ntransitions = np.count_nonzero(status[0:-1] != status[1::])

        particle_dict = {
            "losttime": res_ty[-1, 0],
            "nbounce": nbounce,
            "bounce_times": bounce_times,
            "lam": lam,
            "point0": point,
            "vpar0": vpar_init,
            "status": status,
            "dss": dss,
            "dalphas": dalphas,
            "dchis": dchis,
            "dchis_predicted": dchis_predicted,
            "gammacs": gammacs,
            "Jpars": Jpars,
            "s_means": s_means,
            # Cumulative statistics
            "banana_frac": banana_frac,
            "barely_trapped_frac": barely_trapped_frac,
            "ripple_trapped_frac": ripple_trapped_frac,
            "ntransitions": ntransitions,
            "Jpar_var": Jpar_var,
            "gammac_mean": gammac_mean,
            # Per-segment data for diagnostic plotting (see plot_bounce_segment)
            "debug_data": debug_data,
        }
        return particle_dict

    def plot_bounce_segment(self, data, show=True):
        r"""
        Generate diagnostic plots for a single bounce segment.

        Produces two figures:

        1. **Full field-line well** – |B| vs χ for the mean field line, the
           α / s envelope field lines, and the monotonic-fit well shape over
           the full ±2π χ window.
        2. **Zoom plot** – same curves restricted to the trajectory χ range,
           with mirror-point / well-minimum markers.

        Args:
            data (dict): One entry from ``particle_dict['debug_data']``, as
                returned by :meth:`classify_orbit`.
            show (bool, optional): Call ``plt.show()`` after each figure.
                Set ``False`` when saving figures programmatically.
                Default ``True``.
        """
        import matplotlib.pyplot as plt

        # --- Unpack precomputed data from classify_orbit ---
        dchi             = data['dchi']
        dchi_predicted   = data['dchi_predicted']
        modB_crit        = data['modB_crit']
        modB_crit_traj   = data['modB_crit_traj']
        chi_min          = data['chi_min']
        chi_mirror_left  = data['chi_mirror_left']
        chi_mirror_right = data['chi_mirror_right']
        res_ty_seg       = data['res_ty_segment']
        chi_traj_center  = data['chi_traj_center']
        chi_grid_mean    = data['chi_grid_mean']
        modB_grid_mean   = data['modB_grid_mean']
        chi_left_mon     = data['chi_left_mon']
        chi_right_mon    = data['chi_right_mon']
        modB_left_mon    = data['modB_left_mon']
        modB_right_mon   = data['modB_right_mon']

        # --- Trajectory arrays (field evaluation on trajectory points only) ---
        point = np.zeros((len(res_ty_seg), 3))
        point[:, 0] = res_ty_seg[:, 1]   # s
        point[:, 1] = res_ty_seg[:, 2]   # theta
        point[:, 2] = res_ty_seg[:, 3]   # zeta
        self.field.set_points(point)
        modB_traj     = self.field.modB()[:, 0]
        chi_traj_plot = np.unwrap(self.helicity_M * point[:, 1] - self.helicity_N * point[:, 2])

        title_str = 'dchi={:.2f}, dchi_predicted={:.2f}'.format(dchi, dchi_predicted)

        # Align precomputed field-line χ grid to the same branch as the trajectory.
        chi_shift             = _chi_branch_shift(chi_grid_mean, chi_traj_plot)
        chi_field             = chi_grid_mean    + chi_shift
        chi_min_plot          = chi_min          + chi_shift
        chi_mirror_left_plot  = chi_mirror_left  + chi_shift
        chi_mirror_right_plot = chi_mirror_right + chi_shift
        chi_traj_center_plot  = chi_traj_center  + chi_shift

        chi_left_mon_plot  = chi_left_mon  + chi_shift
        chi_right_mon_plot = chi_right_mon + chi_shift

        # --- Figure 1: Full field-line well ---
        fig1 = plt.figure()
        plt.plot(chi_field, modB_grid_mean,
                 color="blue",      label=r"$\alpha_{\rm mean}$",   zorder=1)
        plt.plot(chi_left_mon_plot,  modB_left_mon,
                 color="magenta", linewidth=2, linestyle="-", label=r"monotonic well", zorder=3)
        plt.plot(chi_right_mon_plot, modB_right_mon,
                 color="magenta", linewidth=2, linestyle="-", zorder=3)
        plt.plot(chi_traj_plot, modB_traj,
                 color="black",     label="trajectory",             linestyle="--", zorder=5)
        plt.axhline(modB_crit, color="red", label=r"$B_{\rm crit}$")
        plt.axhline(modB_crit_traj, color="green", label=r"$B_{\rm crit, traj}$")
        plt.axvline(chi_traj_center_plot, color="gray", linestyle="-.", linewidth=1.5, label=r"$\chi_{\rm center}$")
        plt.ylim(modB_grid_mean.min(), modB_grid_mean.max())
        plt.xlabel(r"$\chi$"); plt.ylabel(r"$|B|$")
        plt.title(title_str); plt.legend()
        if show:
            plt.show()

        # --- Figure 2: Zoom on trajectory χ window ---
        chi_ptp    = np.ptp(chi_traj_plot)
        chi_margin = max(0.01, 0.1 * chi_ptp) if chi_ptp > 0 else 0.01
        chi_xlim   = (chi_traj_plot.min() - chi_margin,
                      chi_traj_plot.max() + chi_margin)

        def _in_win(chi_arr):
            return (chi_arr >= chi_xlim[0]) & (chi_arr <= chi_xlim[1])

        field_mask     = _in_win(chi_field)

        mon_left_mask  = _in_win(chi_left_mon_plot)
        mon_right_mask = _in_win(chi_right_mon_plot)
        b_parts = [modB_traj, np.array([modB_crit])]
        if field_mask.any():
            b_parts.append(modB_grid_mean[field_mask])
        if mon_left_mask.any():
            b_parts.append(modB_left_mon[mon_left_mask])
        if mon_right_mask.any():
            b_parts.append(modB_right_mon[mon_right_mask])
        b_vals = np.concatenate(b_parts)
        b_lo, b_hi = b_vals.min(), b_vals.max()
        b_pad = max(0.05 * (b_hi - b_lo), 1e-4 * modB_crit)

        fig2 = plt.figure()
        if np.any(field_mask):
            plt.plot(chi_field[field_mask], modB_grid_mean[field_mask],
                     color="blue",      label=r"$\alpha_{\rm mean}$", linestyle="--", linewidth=1.5, zorder=2)
        plt.plot(chi_left_mon_plot,  modB_left_mon,
                 color="magenta", linewidth=2, linestyle="-", label="monotonic well", zorder=4)
        plt.plot(chi_right_mon_plot, modB_right_mon,
                 color="magenta", linewidth=2, linestyle="-", zorder=4)
        plt.plot(chi_traj_plot, modB_traj,
                 color="black", linestyle="--", marker="o", markersize=4, label="trajectory", zorder=3)
        plt.axhline(modB_crit, color="red", label=r"$B_{\rm crit}$")
        plt.axhline(modB_crit_traj, color="green", label=r"$B_{\rm crit, traj}$")
        for chi_val, col, lbl in ((chi_min_plot,          "green",  "chi_min"),
                                   (chi_mirror_left_plot,  "orange", "chi_mirror_left"),
                                   (chi_mirror_right_plot, "purple", "chi_mirror_right"),
                                   (chi_traj_center_plot,  "gray",   r"$\chi_{\rm center}$")):
            if chi_xlim[0] <= chi_val <= chi_xlim[1]:
                plt.axvline(chi_val, color=col, linestyle=":", label=lbl)
        plt.xlim(chi_xlim); plt.ylim(b_lo - b_pad, b_hi + b_pad)
        plt.xlabel(r"$\chi$"); plt.ylabel(r"$|B|$")
        plt.title(title_str); plt.legend()
        if show:
            plt.show()

        # --- Figure 3: v_par vs χ ---
        vpar_traj = res_ty_seg[:, 4]
        fig3 = plt.figure()
        plt.plot(chi_traj_plot, vpar_traj,
                 color="black", linestyle="--", marker="o", markersize=4, zorder=2)
        plt.axhline(0, color="red", linewidth=1, linestyle="-", label=r"$v_{\parallel}=0$")
        for chi_val, col, lbl in ((chi_traj_center_plot, "gray",   r"$\chi_{\rm center}$"),):
            if chi_xlim[0] <= chi_val <= chi_xlim[1]:
                plt.axvline(chi_val, color=col, linestyle=":", label=lbl)
        plt.xlabel(r"$\chi$"); plt.ylabel(r"$v_{\parallel}$")
        plt.title(title_str); plt.legend()
        if show:
            plt.show()

        return [fig1, fig2, fig3]
