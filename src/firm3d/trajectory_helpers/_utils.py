from os.path import exists

import numpy as np
from scipy import integrate

from ..field.boozermagneticfield import (
    InterpolatedShearAlfvenWave,
    ShearAlfvenHarmonic,
    ShearAlfvenWave,
    ShearAlfvenWavesSuperposition,
)


def compute_loss_fraction(res_tys, tmin=1e-7, tmax=1e-2, ntime=1000):
    r"""
    Compute the fraction of particles lost as a function of time.

    Args:
        res_tys : List of particle trajectories, where each trajectory is a 2D
                  array with shape (nsteps, 5) containing time and coordinates
                  (t, s, theta, zeta, vpar).
        tmin : Minimum time to consider for loss fraction (default: 1e-7)
        tmax : Maximum time to consider for loss fraction (default: 1e-2)
        ntime : Number of time points to evaluate the loss fraction (default: 1000)
    Returns:
        times : A numpy array of shape (ntime,) containing the time points at
                which the loss fraction is evaluated.
        loss_frac : A numpy array of shape (ntime,) containing the fraction of
                    particles lost at each time point.
    """
    nparticles = len(res_tys)

    timelost = np.zeros((nparticles,))
    for ip in range(nparticles):
        timelost[ip] = res_tys[ip][-1, 0]

    times = np.logspace(np.log10(tmin), np.log10(tmax), ntime)

    loss_frac = np.zeros_like(times)
    for it in range(ntime):
        loss_frac[it] = np.count_nonzero(timelost < times[it] - 1e-15) / nparticles

    return times, loss_frac


def compute_trajectory_cylindrical(res_ty, field):
    r"""
    Compute the cylindrical coordinates (R, Z, phi) in a given
    BoozerMagneticField for each particle trajectory.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the
                 trajectory of a single particle in Boozer coordinates
                 (s, theta, zeta, vpar). Tracing should be performed with
                 forget_exact_path=False to save the trajectory information.
        field : The :class:`BoozerMagneticField` instance used to set the
                points for the field.

    Returns:
        R_traj : A numpy array with shape (nsteps,) containing the radial
                 coordinate R for the particle trajectory.
        Z_traj : A numpy array with shape (nsteps,) containing the vertical
                 coordinate Z for the particle trajectory.
        phi_traj : A numpy array with shape (nsteps,) containing the
                   azimuthal angle phi for the particle trajectory.
    """
    nsteps = len(res_ty[:, 0])
    points = np.zeros((nsteps, 3))
    points[:, 0] = res_ty[:, 1]
    points[:, 1] = res_ty[:, 2]
    points[:, 2] = res_ty[:, 3]
    field.set_points(points)

    R_traj = field.R()[:, 0]
    Z_traj = field.Z()[:, 0]
    nu = field.nu()[:, 0]
    phi_traj = res_ty[:, 3] - nu

    return R_traj, phi_traj, Z_traj


def min_volumemodB(B0, NFP=None):
    r"""
    Estimate minimum magnetic-field magnitude over sampled surfaces
    by evaluating |B| on a uniform grid of flux surfaces.

    Args:
        B0  : The :class:`BoozerMagneticField` instance to evaluate.
        NFP : Number of field periods. If None, defaults to 1.

    Returns:
        min_modB : Approximate minimum value of |B| in the sampled volume.
    """
    if NFP is None:
        NFP = 1

    s_grid = np.linspace(0, 1, 100)
    theta_grid = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    zeta_grid = np.linspace(0, 2 * np.pi / NFP, 100, endpoint=False)
    [zeta_grid, theta_grid, s_grid] = np.meshgrid(zeta_grid, theta_grid, s_grid)
    points = np.zeros((len(theta_grid.flatten()), 3))
    points[:, 0] = s_grid.flatten()
    points[:, 1] = theta_grid.flatten()
    points[:, 2] = zeta_grid.flatten()

    B0.set_points(points)
    modB = B0.modB()[:, 0]
    return np.min(modB)


def chi(theta, zeta, helicity_M, helicity_N):
    r"""
    Compute the helical angle chi = M*theta - N*zeta.

    Args:
        theta : Poloidal angle.
        zeta : Toroidal angle.
        helicity_M : Poloidal helicity number.
        helicity_N : Toroidal helicity number.
    Returns:
        chi : The helical angle.
    """
    return helicity_M * theta - helicity_N * zeta


def eta(theta, zeta, helicity_Mp, helicity_Np):
    r"""
    Compute the mapping angle eta = Mp*theta - Np*zeta.

    Args:
        theta : Poloidal angle.
        zeta : Toroidal angle.
        helicity_Mp : Poloidal helicity number.
        helicity_Np : Toroidal helicity number.
    Returns:
        eta : The mapping angle.
    """
    return helicity_Mp * theta - helicity_Np * zeta


def chi_eta_to_theta_zeta(chi, eta, helicity_M, helicity_N, helicity_Mp, helicity_Np):
    r"""
    Convert helical angles (chi, eta) to (theta, zeta).

    Args:
        chi : Helical angle chi.
        eta : Mapping angle eta.
        helicity_M : Poloidal helicity number defining chi = M*theta - N*zeta.
        helicity_N : Toroidal helicity number defining chi = M*theta - N*zeta.
        helicity_Mp : Poloidal helicity number defining
                      eta = Mp*theta - Np*zeta.
        helicity_Np : Toroidal helicity number defining
                      eta = Mp*theta - Np*zeta.
    Returns:
        theta : Poloidal angle.
        zeta : Toroidal angle.
    """
    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp
    theta = (helicity_Np * chi - helicity_N * eta) / denom
    zeta = (helicity_Mp * chi - helicity_M * eta) / denom

    return theta, zeta


def compute_peta(
    field_or_saw,
    points,
    vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
    helicity_Mp=None,
    helicity_Np=None,
):
    r"""
    Given a ShearAlfvenWave or BoozerMagneticField instance, a point in
    Boozer coordinates, and particle properties, compute the value of the
    canonical momentum, :math:`p_{\eta}`. This quantity is conserved under
    the unperturbed guiding center equations if the field strength is exactly
    quasisymmetric with helicity (M,N) and :math:`\alpha = 0`.

    :math:`p_{\eta} = (M G + N I) \left(\frac{m v_{\|\|}}{B} + q \alpha \right) +
    q (M \psi - M \psi')`

    If field_or_saw is a BoozerMagneticField instance, then alpha = 0.

    Args:
        field_or_saw : The BoozerMagneticField or ShearAlfvenWave instance.
        points : A numpy array of shape (npoints,4) containing the coordinates
                 (s,theta,zeta,t).
            If field_or_saw is a ShearAlfvenWave, then t is the time coordinate.
            If field_or_saw is a BoozerMagneticField, then t is ignored, and
            points is allowed to have shape (npoints,3) for (s,theta,zeta).
        vpar : A numpy array of shape (npoints,) containing the parallel velocity.
        mass : Mass of the particle.
        charge : Charge of the particle.
        helicity_M : Poloidal helicity of the magnetic field.
        helicity_N : Toroidal helicity of the magnetic field.
        helicity_Mp : Poloidal helicity of the mapping coordinate eta.
            If None, then eta is chosen based on the helicity of the field strength.
        helicity_Np : Toroidal helicity of the mapping coordinate eta.
            If None, then eta is chosen based on the helicity of the field strength.

    Returns:
        peta : A numpy array of shape (npoints,) containing the value of the canonical
            momentum :math:`p_{\eta}` at each point.
    """
    if points.shape[1] not in [3, 4]:
        raise ValueError(
            "Points must have shape (npoints, 4) for (s, theta, zeta, t) or "
            "(npoints, 3) for (s, theta, zeta)"
        )
    if isinstance(vpar, float):
        vpar = np.array([vpar])
    if isinstance(vpar, list):
        vpar = np.array(vpar)
    assert vpar.shape[0] == points.shape[0], (
        "vpar must have the same number of points as points"
    )

    if isinstance(
        field_or_saw,
        (ShearAlfvenWave, ShearAlfvenWavesSuperposition, InterpolatedShearAlfvenWave),
    ):
        field = field_or_saw.B0
        field_or_saw.set_points(points)
        alpha = field_or_saw.alpha()[:, 0]
    else:
        field = field_or_saw
        alpha = 0.0
        if points.shape[1] == 4:
            points = points[:, :3]
        field.set_points(points)

    modB = field.modB()[:, 0]
    G = field.G()[:, 0]
    I = field.I()[:, 0]
    psi = field.psi0 * points[:, 0]
    psip = field.psip()[:, 0]

    if helicity_Mp is None and helicity_Np is None:
        # If modB contours close poloidally, then use theta as mapping coordinate
        if helicity_M == 0:
            helicity_Mp = 1
            helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate
        else:
            helicity_Mp = 0
            helicity_Np = -1
    else:
        if (helicity_Mp * helicity_N) == (helicity_Np * helicity_M):
            raise ValueError(
                "Chosen helicities (N, M, N', M') do not create a well "
                "defined Jacobian."
            )
    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp
    peta = (
        -(
            (helicity_M * G + helicity_N * I) * (mass * vpar / modB + charge * alpha)
            + charge * (helicity_N * psi - helicity_M * psip)
        )
        / denom
    )
    return peta


def compute_Eprime(saw, points, vpar, mu, mass, charge, helicity_M, helicity_N):
    r"""
    Compute the invariant Eprime for a ShearAlfvenHarmonic instance given
    points in Boozer coordinates.

    Args:
        saw : An instance of ShearAlfvenHarmonic.
        points : A numpy array of shape (npoints,4) containing the coordinates
                 (s,theta,zeta,t).
        vpar : A numpy array of shape (npoints,) containing the parallel velocity.
        mu : Magnetic moment of the particle, vperp^2/(2 B).
        mass : Mass of the particle.
        charge : Charge of the particle.
        helicity_M : Poloidal helicity of the magnetic field strength.
        helicity_N : Toroidal helicity of the magnetic field strength.

    Returns:
        Eprime : A numpy array of shape (npoints,) containing the shifted energy
            invariant :math:`E' = n' E - \omega p_\eta` at each point.
    """
    if points.shape[1] != 4:
        raise ValueError("Points must have shape (npoints, 4) for (s, theta, zeta, t)")
    if isinstance(vpar, float):
        vpar = np.array([vpar])
    if isinstance(vpar, list):
        vpar = np.array(vpar)
    assert vpar.shape[0] == points.shape[0], (
        "vpar must have the same number of points as points"
    )
    if vpar.shape[0] != points.shape[0]:
        raise ValueError("vpar must have the same number of points as points")
    if isinstance(saw, ShearAlfvenHarmonic) is False:
        raise TypeError("Expected saw to be an instance of ShearAlfvenHarmonic")

    # If modB contours close poloidally, then use theta as mapping coordinate
    if helicity_M == 0:
        helicity_Mp = 1
        helicity_Np = 0
    # Otherwise, use zeta as mapping coordinate
    else:
        helicity_Mp = 0
        helicity_Np = -1

    Phim = saw.Phim
    Phin = saw.Phin
    omega = saw.omega

    # Compute the canonical momentum p_eta
    p_eta = compute_peta(saw, points, vpar, mass, charge, helicity_M, helicity_N)

    # Compute the energy E
    modB = saw.B0.modB()[:, 0]
    E = 0.5 * mass * vpar**2 + mass * mu * modB + charge * saw.Phi()[:, 0]

    # Compute the invariant Eprime
    nprime = (Phim * helicity_N - Phin * helicity_M) / (
        helicity_Np * helicity_M - helicity_N * helicity_Mp
    )
    Eprime = nprime * E - omega * p_eta
    return Eprime


def g(t, T):
    """
    Smooth bump weight on (0, T) with g=0 at t<=0 or t>=T.
    """
    t = np.asarray(t, dtype=float)
    T = float(T)

    s = t / T

    # denom = s*(1-s); positive only for s in (0,1)
    w = np.zeros_like(s)

    interior = (s > 0.0) & (s < 1.0)
    denom = s[interior] * (1.0 - s[interior])
    w[interior] = np.exp(-1.0 / denom)

    return w


def return_DA(array):
    """
    Compute the DA metric for a given momentum 2D array.
    Computes the normalized digit difference between the
    WBA at the first half of the trajectory and the final
    given point.
    Args:
        array : A numpy array of shape (npoints, 2) containing the time and
                momentum values.
    Returns:
        T : Final computed time of the trajectory.
        da_c : The computed DA metric value.
    """
    time_m = array[:, 0]
    momentum = array[:, 1]

    if len(time_m) < 4:
        return 0, np.nan

    # full trajectory
    T_idx = len(time_m)
    T_time = time_m
    T_mom = momentum
    T = float(T_time[-1])

    t_idx = int(T_idx / 2)
    t_time = T_time[:t_idx]
    t_mom = T_mom[:t_idx]
    t = float(t_time[-1])

    # weight arrays
    g_t = g(t_time, t)
    g_T = g(T_time, T)

    g_t_int = integrate.trapezoid(g_t, t_time)
    g_T_int = integrate.trapezoid(g_T, T_time)

    if np.isclose(g_t_int, 0.0) and np.isclose(g_T_int, 0.0):
        return T, np.nan

    t_wavg = integrate.trapezoid(g_t * t_mom, t_time) / g_t_int
    T_wavg = integrate.trapezoid(g_T * T_mom, T_time) / g_T_int

    # DA metric
    diff = np.abs(t_wavg - T_wavg)
    denom = 0.5 * (np.abs(t_wavg) + np.abs(T_wavg))

    if diff == 0.0:
        return T, 16

    ratio = diff / denom
    da_absolute = -np.log10(ratio)
    da_relative = -np.log10(diff / (np.nanmax(np.abs(T_mom))))

    return T, max(da_relative, da_absolute)


def _solve_vpar_energy(B0, point, mass, Ekin, mu, sgn):
    r"""
    Solve for the parallel velocity at the given (s, theta, zeta) position(s)
    from energy conservation, Ekin = 0.5 * mass * vpar**2 + mu * modB. Works
    for a single point or an array of points.

    Args:
        B0 : Magnetic field instance used to evaluate modB at the given
             points.
        point : Position(s) at which to evaluate vpar, as (s, theta, zeta)
                coordinates. Array of shape (3,) for a single point, or
                (npoints, 3) for multiple points.
        mass : Particle mass.
        Ekin : Total kinetic energy.
        mu : Magnetic moment (scalar, or array-like matching point).
        sgn : Desired sign of the parallel velocity (+1 or -1).

    Returns:
        vpar : Parallel velocity consistent with Ekin, or NaN where the
               perpendicular energy exceeds the total energy. Scalar if
               point is 1D, otherwise an array of length npoints.
    """
    scalar_input = point.ndim == 1
    points = point[None, :] if scalar_input else point
    B0.set_points(points)
    modB = B0.modB()[:, 0]

    energy_par = Ekin - mu * modB
    vpar = sgn * np.sqrt(np.maximum(2 * energy_par / mass, 0))
    vpar = np.where(energy_par > 0, vpar, np.nan)
    return vpar[0] if scalar_input else vpar


def _solve_vpar_perturbed(
    B0,
    saw,
    point,
    helicity_M,
    helicity_N,
    helicity_Np,
    helicity_Mp,
    mass,
    nprime,
    omega,
    charge,
    Eprime,
    mu,
    sgn,
):
    r"""
    Solve the perturbed-orbit quadratic for vpar at the given (s, theta, zeta)
    position(s) such that the shifted energy invariant equals Eprime. Works
    for a single point or an array of points.

    Args:
        B0 : Unperturbed magnetic field instance used to evaluate modB, G, I,
             psi0, and psip at the given points.
        saw : Perturbed field instance (wrapping B0) used to evaluate Phi and
              alpha at the given points, and to set the evaluation points
              shared with B0.
        point : Position(s) at which to evaluate vpar, as (s, theta, zeta)
                coordinates. Array of shape (3,) for a single point, or
                (npoints, 3) for multiple points.
        helicity_M : Poloidal helicity number defining chi = M*theta - N*zeta.
        helicity_N : Toroidal helicity number defining chi = M*theta - N*zeta.
        helicity_Np : Toroidal helicity number defining
                      eta = Mp*theta - Np*zeta.
        helicity_Mp : Poloidal helicity number defining
                      eta = Mp*theta - Np*zeta.
        mass : Particle mass.
        nprime : Coefficient n' in the shifted-energy invariant
                 Eprime = n' * E - omega * p_eta.
        omega : Coefficient omega in the shifted-energy invariant
                Eprime = n' * E - omega * p_eta.
        charge : Particle charge.
        Eprime : Prescribed value of the shifted-energy invariant.
        mu : Magnetic moment divided by mass (scalar, or array-like matching
             point).
        sgn : Desired sign of the parallel velocity (+1 or -1).

    Returns:
        vpar : Solution(s) for vpar, or NaN where no real root exists.
               Scalar if point is 1D, otherwise an array of length npoints.
    """
    scalar_input = point.ndim == 1
    points = point[None, :] if scalar_input else point
    s = points[:, 0]
    points4 = np.zeros((points.shape[0], 4))  # 4th column initialized as t = 0
    points4[:, :3] = points
    saw.set_points(points4)
    modB = B0.modB()[:, 0]
    G = B0.G()[:, 0]
    I = B0.I()[:, 0]
    psi = B0.psi0 * s
    psip = B0.psip()[:, 0]
    Phi = saw.Phi()[:, 0]
    alpha = saw.alpha()[:, 0]

    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp  # - 1 in QA
    d_peta_d_vpar = (
        -((helicity_M * G + helicity_N * I) * (mass / modB)) / denom
    )  # G m/ modB in QA
    d_E_d_vpar2 = 0.5 * mass
    a = nprime * d_E_d_vpar2  # Coefficient of vpar^2
    b = -omega * d_peta_d_vpar  # Coefficient of vpar
    # Constant term
    c = (
        nprime * (mass * mu * modB + charge * Phi)
        + omega
        * (
            (helicity_M * G + helicity_N * I) * charge * alpha
            + charge * (helicity_N * psi - helicity_M * psip)
        )
        / denom
        - Eprime
    )
    discriminant = b**2 - 4 * a * c
    valid = discriminant >= 0

    if a != 0:
        # mask negatives before sqrt so we don't get warnings
        safe_disc = np.where(valid, discriminant, 0.0)
        result = (-b + sgn * np.sqrt(safe_disc)) / (2 * a)
    else:
        # discriminant = b**2 >= 0 always
        result = (-c / b) * sgn

    vpar = np.where(valid, result, np.nan)
    return vpar[0] if scalar_input else vpar


def _check_filepaths(filepaths):
    r"""
    Check whether all provided output file paths exist.

    Args:
        filepaths : Dictionary of file labels to filesystem paths.
    Returns:
        exists_all : True if every path exists, otherwise False.
    """
    return all(exists(fp) for fp in filepaths.values())


def return_bounces_and_passes(vpar_path, zeta_path):
    r"""
    Count guiding-center bounces and toroidal transits along a trajectory.

    Bounces are detected as sign changes of vpar. Transits are detected as
    negative jumps in zeta mod 2*pi; a candidate transit is rejected if a
    bounce occurs between it and the next wrap.

    Args:
        vpar_path : Array of parallel velocity samples.
        zeta_path : Array of zeta samples (radians).

    Returns:
        bounce_indices : Trajectory indices where vpar changes sign.
        true_passes : Trajectory indices of confirmed toroidal transits.
    """
    v_par_signs = np.sign(vpar_path)

    mask = v_par_signs != 0
    v = v_par_signs[mask]
    # Keep track of original indices after removing zeros
    orig_idx = np.where(mask)[0]

    # find vpar sign changes
    bounce_local = np.where(v[1:] * v[:-1] < 0)[0]
    # map back to original trajectory indexing, pre zero removal
    bounce_indices = orig_idx[bounce_local + 1]

    zeta_path = np.mod(zeta_path, 2 * np.pi)
    dzeta = np.diff(zeta_path)

    # find large negative jump, this is where mod
    # brings factors of 2pi back to zero, and pass
    wrap_idx = np.where(dzeta < -np.pi)[0]

    # isolate transits across zeta of 2pi
    true_passes = []
    for passing_index in range(len(wrap_idx) - 1):
        pass1 = wrap_idx[passing_index]
        pass2 = wrap_idx[passing_index + 1]

        # ensure no bounce between these two toroidal passes
        if not np.any((bounce_indices > pass1) & (bounce_indices < pass2)):
            true_passes.append(wrap_idx[passing_index])

    return bounce_indices, true_passes


def return_chaotic_boolean_array(DA_at_tfinal, cutoff=3):
    r"""
    Return a boolean array classifying particles as chaotic or regular based
    on their final WBA digit accuracy.

    Args:
        DA_at_tfinal : Array-like of final WBA digit accuracy values.
        cutoff : Digit accuracy threshold for classifying chaos (default: 3).

    Returns:
        chaotic_indices : Boolean array of shape (N,) where True indicates
            a chaotic particle.
    """
    return np.array(DA_at_tfinal) < cutoff


def return_chaotic_percentage(DA_at_tfinal, cutoff=3):
    r"""
    Return the percentage of particles classified as chaotic based on their
    final WBA digit accuracy.

    Args:
        DA_at_tfinal : Array-like of final WBA digit accuracy values.
        cutoff : Digit accuracy threshold for classifying chaos (default: 3).

    Returns:
        chaotic_percentage : Percentage of particles classified as chaotic.
    """
    chaotic_indices = return_chaotic_boolean_array(DA_at_tfinal, cutoff)
    return np.mean(chaotic_indices) * 100


def return_chaotic_initial_conditions(DA_at_tfinal, s0, theta0, zeta0, vpar0, mus):
    r"""
    Return the initial conditions of particles classified as chaotic based on
    their final WBA digit accuracy.

    Args:
        DA_at_tfinal : Array-like of final WBA digit accuracy values.
        s0 : Array of initial flux-surface labels.
        theta0 : Array of initial poloidal angles.
        zeta0 : Array of initial toroidal angles.
        vpar0 : Array of initial parallel velocities.
        mus : Array of initial magnetic moments divided by mass.

    Returns:
        chaotic_points : Array of shape (N_chaotic, 3) of (s, theta, zeta)
            initial conditions for chaotic particles.
        chaotic_vpars : Array of shape (N_chaotic,) of initial parallel
            velocities for chaotic particles.
        chaotic_mus : Array of shape (N_chaotic,) of initial magnetic moments
            divided by mass for chaotic particles.
    """
    chaotic_indices = return_chaotic_boolean_array(DA_at_tfinal)

    chaotic_s0s = s0[chaotic_indices]
    chaotic_theta0s = theta0[chaotic_indices]
    chaotic_zeta0s = zeta0[chaotic_indices]
    chaotic_points = np.stack((chaotic_s0s, chaotic_theta0s, chaotic_zeta0s), axis=-1)
    chaotic_vpars = vpar0[chaotic_indices]
    chaotic_mus = mus[chaotic_indices]
    return chaotic_points, chaotic_vpars, chaotic_mus


def trajectory_to_vtk(res_ty, field, filename="trajectory"):
    r"""
    Save a single particle trajectory in Cartesian coordinates to a VTK file.
    Requires the pyevtk package to be installed.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the trajectory of a
                 single particle in Boozer coordinates.
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        filename : The name of the output VTK file.
    """
    from pyevtk.hl import polyLinesToVTK

    R_traj, phi_traj, Z_traj = compute_trajectory_cylindrical(res_ty, field)

    X_traj = R_traj * np.cos(phi_traj)
    Y_traj = R_traj * np.sin(phi_traj)

    ppl = np.asarray([len(R_traj)])  # Number of points along trajectory
    polyLinesToVTK(filename, X_traj, Y_traj, Z_traj, pointsPerLine=ppl)
