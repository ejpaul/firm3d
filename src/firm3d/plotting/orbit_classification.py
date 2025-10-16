import numpy as np

__all__ = ["OrbitClassification"]

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
                analyzed (in units of nfp). The helical angle is defined as
                chi = M*theta - N*zeta.
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
              is used as the mapping coordinate (Mp=0, Np=nfp).
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
            self.helicity_Np = self.nfp

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
                  λ = v_perp^2 / (v^2 * B), dimensionless
                - 'point0' (ndarray): Initial position [s, theta, zeta]
                - 'vpar0' (float): Initial parallel velocity [m/s]

                **Per-bounce-segment arrays (all have length nbounce-1):**
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
              ∫ v_|| dζ / (B·∇ζ)
            - Requires at least 2 bounces for classification; returns
              zeros if nbounce < 2
            - Requires at least 4 bounces for Jpar_var computation
        """

        # Compute all of the times when the particle bounces off vpar plane
        # These are mirror points where particle reverses direction
        bounce_times = []
        nhits = len(res_hit[:, 0])
        for j in range(nhits):
            if res_hit[j, 1] == 0:  # vpar plane was hit
                bounce_times.append(res_hit[j, 0])

        nbounce = len(bounce_times)

        # Unwrap theta to handle periodic boundary crossings (prevents jumps at ±π)
        thetas = np.unwrap(res_ty[:, 2])

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
        dchis = []  # Change in helical angle for each bounce segment
        dchis_predicted = []  # Expected dchi based on mirror point locations
        gammacs = []  # Orbit width parameter gamma_c for each segment
        dss = []  # Change in radial coordinate for each bounce
        dalphas = []  # Change in field line label alpha for each bounce

        # Iterate over bounce segments (from mirror point j to mirror point j+1)
        for j in range(nbounce - 1):
            # Find trajectory indices corresponding to this bounce segment
            index_start = np.argmin(np.abs(bounce_times[j] - res_ty[:, 0]))
            index_end = np.argmin(np.abs(bounce_times[j + 1] - res_ty[:, 0]))

            # Compute radial excursion during this bounce
            ds = res_ty[index_end, 1] - res_ty[index_start, 1]
            dss.append(ds)

            # Compute change in helical angle chi = M*theta - N*zeta
            # This is the primary quantity used for classification
            dtheta = thetas[index_end] - thetas[index_start]
            dzeta = res_ty[index_end, 3] - res_ty[index_start, 3]
            dchi = self.helicity_M * dtheta - self.helicity_N * dzeta
            dchis.append(np.abs(dchi))

            # Compute mean radial position during this bounce segment
            mean_s = np.mean(res_ty[index_start : index_end + 1, 1])

            # Predict dchi based on mirror point locations on constant-s
            # Sample modB on a chi grid at fixed s and eta=0
            chi_grid = np.linspace(0, 2 * np.pi, 100)
            theta, zeta = self.chi_eta_to_theta_zeta(chi_grid, np.zeros_like(chi_grid))
            points = np.zeros((len(chi_grid.flatten()), 3))
            points[:, 0] = mean_s
            points[:, 1] = theta
            points[:, 2] = zeta
            self.field.set_points(points)
            iota_s = self.field.iota()[0, 0]

            # Compute change in field line label alpha = theta - iota*zeta
            # This characterizes motion across flux surfaces
            dalpha = dtheta - iota_s * dzeta
            dalphas.append(dalpha)

            # Compute gamma_c parameter that characterizes orbit width
            # gamma_c → 0 for thin orbits, gamma_c → 1 for wide orbits
            gammac = (2 / np.pi) * np.arctan(np.abs(ds) / np.abs(dalpha))
            gammacs.append(gammac)

            # Find mirror point (where |B| = B_critical) and min |B|
            modB = self.field.modB()[:, 0]
            mirror_loc = np.argmin(np.abs(modB - modB_crit))
            chi_mirror = chi_grid[mirror_loc]
            min_loc = np.argmin(modB)
            chi_min = chi_grid[min_loc]

            # Predicted dchi = 2 * (chi_mirror - chi_min)
            # Factor of 2 from bouncing between mirror points
            dchi_predicted = np.min(
                [
                    np.abs(2 * (chi_mirror - chi_min)),
                    np.abs(2 * (chi_mirror - (chi_min + 2 * np.pi))),
                    np.abs(2 * (chi_mirror - (chi_min - 2 * np.pi))),
                ]
            )
            dchis_predicted.append(dchi_predicted)

            # Compute parallel action variable J_|| = ∮ v_|| dℓ_|| / (2π)
            # Using the canonical form: J_|| = ∫ v_|| dζ / (B·∇ζ)
            # where B·∇ζ = B / (G + ιI) in Boozer coordinates
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

        # Classify the trapping state based on dchi for each bounce segment
        if nbounce < 2:
            # Particle never mirrored - cannot classify trapping state
            Jpar_var = 0.0
            gammac_mean = 0.0
            ntransitions = 0
            # Initialize empty arrays for consistency
            status = np.array([])
            banana_frac = 0.0
            barely_trapped_frac = 0.0
            ripple_trapped_frac = 0.0
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

            # Compute fraction of time in each trapping state
            barely_trapped_frac = np.count_nonzero(
                dchis > self.barely_trapped_crit
            ) / len(dchis)
            ripple_trapped_frac = np.count_nonzero(
                dchis < self.ripple_trapped_crit * dchis_predicted
            ) / len(dchis)
            banana_frac = np.count_nonzero(
                (dchis <= self.barely_trapped_crit)
                * (dchis >= self.ripple_trapped_crit)
            ) / len(dchis)

            # Count transitions between different trapping states
            # Fewer transitions = more stable classification
            ntransitions = np.count_nonzero(status[0:-1] != status[1::])

            # Compute mean gamma_c over all bounce segments
            gammac_mean = np.mean(gammacs) if nbounce > 1 else 0.0

            # Compute variation in J_|| over full bounce periods
            # Full bounce = half-bounce up + half-bounce down
            # Low variation indicates good adiabatic invariant
            if nbounce > 3:
                # Sum consecutive half-bounces
                Jpar_full = Jpars[0:-1] + Jpars[1::]
                # Normalized std deviation
                Jpar_var = np.std(Jpar_full) / np.mean(Jpar_full)
            else:
                Jpar_var = 0.0

        particle_dict = {
            "losttime": res_ty[-1, 0],
            "nbounce": nbounce,
            "bounce_times": bounce_times,
            "lam": lam,
            "point0": point,
            "vpar0": vpar_init,
            # All of these quantities have length nbounce-1
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
        }
        return particle_dict
