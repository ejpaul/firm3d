import warnings

import numpy as np

import firm3dpp as sopp

from .._core.types import RealArray
from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import BoozerMagneticField
from ..util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
    REDUCED_PLANCK_CONSTANT,
    VACUUM_PERMITTIVITY,
)

__all__ = [
    "ThermalBackground",
    "trace_particles_boozer_with_collisions",
]


class ThermalBackground:
    r"""
    Maxwellian background species for Monte Carlo Coulomb collision calculations.

    Density and temperature profiles are specified as callables of the
    normalised toroidal flux ``s``, following the same convention used by
    :func:`~firm3d.field.tracing_helpers.initialize_position_profile`.

    The Coulomb logarithm is computed locally from the profiles as the ratio
    of the Debye length to the minimum impact parameter:

    .. math::

        \ln\Lambda_{ab} = \ln\!\left(\frac{\lambda_D}{b_\mathrm{min}}\right),
        \qquad
        b_\mathrm{min} = \max(b_\mathrm{cl},\, b_\mathrm{qm}),

    .. math::

        b_\mathrm{cl} = \frac{|q_a q_b|}
                             {4\pi\varepsilon_0\,m_r\,v_\mathrm{eff}^2},
        \qquad
        b_\mathrm{qm} = \frac{\hbar}{2\,m_r\,v_\mathrm{eff}},
        \qquad
        v_\mathrm{eff}^2 = v^2 + v_{\mathrm{th},b}^2

    where :math:`\lambda_D` is the total Debye length from all background
    species and :math:`m_r = m_a m_b/(m_a+m_b)` is the reduced mass.
    :math:`b_\mathrm{cl}` is the classical 90-degree deflection radius and
    :math:`b_\mathrm{qm}` the de Broglie wavelength; taking the larger covers
    the quantum regime, which fast EPs reach against electrons.  The
    :math:`v_\mathrm{eff}^2` form handles both velocity limits: fast EP
    (:math:`v \gg v_{\mathrm{th},b}`, relevant for ions) and slow EP against
    electrons (:math:`v \ll v_{\mathrm{th},e}`).  This matches ASCOT5's
    ``mccc_coefs_clog`` term for term.

    No floor is applied to :math:`\ln\Lambda`.  Instead
    :func:`trace_particles_boozer_with_collisions` checks the profiles up
    front: :math:`\ln\Lambda \le 0` raises, since the binary-collision model
    is undefined there, and :math:`0 < \ln\Lambda < 2` warns that the
    coefficients are not quantitatively trustworthy.

    Args:
        n_profile: Callable ``n(s)`` returning number density in m\ :sup:`-3`.
        T_profile: Callable ``T(s)`` returning temperature in J.
            To convert from keV use ``T_profile = lambda s: T_keV(s) * 1e3 * ONE_EV``.
        mass: Species mass in kg.
        charge: Species charge in C (signed).
        n_grid_points: Number of uniformly-spaced points in ``s`` on which
            the profiles are pre-evaluated before passing to C++.  The C++
            layer uses linear interpolation between these points.

    Example::

        from firm3d.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
        n_ref = 1e20          # m^-3
        T_ref = 10e3 * ONE_EV # 10 keV in J

        background = ThermalBackground(
            n_profile=lambda s: n_ref * (1 - s**5),
            T_profile=lambda s: T_ref * (1 - s),
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
    """

    def __init__(
        self,
        n_profile,
        T_profile,
        mass,
        charge,
        n_grid_points=512,
    ):
        if not callable(n_profile):
            raise ValueError("n_profile must be callable.")
        if not callable(T_profile):
            raise ValueError("T_profile must be callable.")

        self.n_profile = n_profile
        self.T_profile = T_profile
        self.mass = float(mass)
        self.charge = float(charge)
        self._n_grid_points = n_grid_points

    def _to_cpp(self):
        """Return a sopp.ThermalBackground struct with pre-evaluated grids."""
        s_grid = np.linspace(0.0, 1.0, self._n_grid_points)
        n_vals = np.array([self.n_profile(s) for s in s_grid], dtype=float)
        T_vals = np.array([self.T_profile(s) for s in s_grid], dtype=float)

        bg = sopp.ThermalBackground()
        bg.s_grid = s_grid.tolist()
        bg.n_grid = n_vals.tolist()
        bg.T_grid = T_vals.tolist()
        bg.mass = self.mass
        bg.charge = self.charge
        return bg


def _validate_coulomb_log(cpp_backgrounds, mass, charge):
    r"""
    Check the Coulomb logarithm implied by the background profiles.

    Raises :class:`ValueError` if :math:`\ln\Lambda \le 0` anywhere, and warns
    once if :math:`0 < \ln\Lambda < 2`, where the binary-collision
    approximation is marginal.  Both diagnostics are reported once per call
    for the worst point in the domain; the C++ layer cannot do this, since it
    sees one (v, s) at a time inside the collision kick and would have to emit
    the warning on every sub-step.

    ``compute_collision_coefficients`` in ``collisions.h`` throws when
    :math:`\ln\Lambda \le 0`, which happens when the temperature falls to
    zero at finite density.  Left to the C++ layer that throw fires from
    part-way through a trace, on whichever rank happens to own the offending
    particle -- and under MPI the healthy
    ranks then block forever in the ``allgather`` below.  Checking here
    turns it into a collective failure: every rank holds identical profiles,
    so every rank raises before any tracing starts.

    :math:`\ln\Lambda = \ln(\lambda_D / \max(b_\mathrm{cl}, b_\mathrm{qm}))`
    and both impact parameters decrease monotonically with
    :math:`v_\mathrm{eff}^2 = v^2 + v_{\mathrm{th},b}^2`, so
    :math:`\ln\Lambda` is smallest at :math:`v = 0`.  Evaluating there gives
    a rigorous lower bound over all particle speeds, and the check operates
    on the grids actually handed to C++ rather than on the profile
    callables, so it sees exactly what the solver will see.
    """
    grids = [
        (
            np.asarray(bg.s_grid, dtype=float),
            np.asarray(bg.n_grid, dtype=float),
            np.asarray(bg.T_grid, dtype=float),
            bg.mass,
            bg.charge,
        )
        for bg in cpp_backgrounds
    ]
    if not grids:
        return

    # Common grid: the finest of the per-species grids.  C++ interpolates
    # each species linearly, which np.interp reproduces exactly.
    s = max((g[0] for g in grids), key=len)
    species = [
        (np.interp(s, g[0], g[1]), np.interp(s, g[0], g[2]), g[3], g[4]) for g in grids
    ]

    # Debye length from every active species, matching the C++ pre-pass.
    inv_lD_sq = np.zeros_like(s)
    for n_b, T_b, _, q_b in species:
        active = (n_b > 0.0) & (T_b > 0.0)
        # np.where evaluates both branches, so keep the divisor finite.
        T_safe = np.where(active, T_b, 1.0)
        inv_lD_sq += np.where(
            active, n_b * q_b**2 / (VACUUM_PERMITTIVITY * T_safe), 0.0
        )

    shielded = inv_lD_sq > 0.0
    if not np.any(shielded):
        return  # no active species anywhere; C++ skips them all
    lambda_D = np.where(
        shielded, 1.0 / np.sqrt(np.where(shielded, inv_lD_sq, 1.0)), 0.0
    )

    worst = None  # (ln_Lambda, s, m_b, q_b, n_b, T_b) at the global minimum
    for n_b, T_b, m_b, q_b in species:
        active = shielded & (n_b > 0.0) & (T_b > 0.0)
        if not np.any(active):
            continue
        # v_eff^2 at v = 0 is the species thermal speed squared.
        v_eff_sq = np.where(active, 2.0 * T_b / m_b, 1.0)
        m_r = mass * m_b / (mass + m_b)
        b_cl = abs(charge * q_b) / (4.0 * np.pi * VACUUM_PERMITTIVITY * m_r * v_eff_sq)
        b_qm = REDUCED_PLANCK_CONSTANT / (2.0 * m_r * np.sqrt(v_eff_sq))
        b_min = np.maximum(b_cl, b_qm)
        lD_safe = np.where(active, lambda_D, 1.0)
        ln_lambda = np.where(active, np.log(lD_safe / b_min), np.inf)

        k = int(np.argmin(ln_lambda))
        if worst is None or ln_lambda[k] < worst[0]:
            worst = (ln_lambda[k], s[k], m_b, q_b, n_b[k], T_b[k])

    if worst is None:
        return

    ln_min, s_min, m_b, q_b, n_min, T_min = worst
    where = (
        f"as v -> 0 at s = {s_min:.4f}, for the species with mass {m_b:.3e} kg, "
        f"charge {q_b:.3e} C (n = {n_min:.3e} m^-3, T = {T_min:.3e} J)"
    )
    bound = (
        "This bound is taken at v -> 0, where ln_Lambda is smallest: a fast "
        "particle that never slows into the thermal range would not reach it, "
        "but a slowing-down one will."
    )

    if ln_min <= 0.0:
        raise ValueError(
            f"background profiles give ln_Lambda = {ln_min:.3f} <= 0 {where}.  "
            f"The binary-collision model is undefined there and the C++ layer "
            f"would abort mid-trace.  {bound}  Raise T wherever n > 0, or "
            f"restrict the profiles to the region you actually trace."
        )

    if ln_min < 2.0:
        warnings.warn(
            f"background profiles give ln_Lambda = {ln_min:.3f} < 2 {where}.  "
            f"The binary-collision approximation is marginal there, so the "
            f"collision coefficients should not be trusted quantitatively.  "
            f"{bound}",
            RuntimeWarning,
            stacklevel=3,
        )


def trace_particles_boozer_with_collisions(
    field: BoozerMagneticField,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    backgrounds,
    tmax=1e-2,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    abstol=None,
    reltol=None,
    comm=None,
    stopping_criteria=None,
    dt_save=1e-6,
    forget_exact_path=False,
    axis=2,
    ode_solver="dormand_prince",
    DP_hmin=0.0,
    rng_seed=0,
    validate_profiles=True,
    mode=None,
):
    r"""
    Trace guiding-centre particles including Monte Carlo Coulomb collisions
    with one or more Maxwellian background species.

    The orbit is advanced by adaptive Dormand-Prince in
    :math:`(s, \theta, \zeta, v_\parallel)` at fixed :math:`\mu`, and the
    collision operator is applied as a kick to :math:`(v, \xi)` at each
    accepted step -- drift by explicit Euler plus the Milstein noise term,
    following ASCOT5's ``mccc_gc_milstein.c``.  The kick is sub-cycled when
    the collision rates are fast compared with the orbit step, which the
    scheme depends on for accuracy in the thermal regime rather than being an
    implementation detail.  Because :math:`\mu` is
    constant within a step it is a parameter of the orbit equations rather
    than a state variable, so the vacuum, ``noK`` and full guiding-centre
    equations are all supported; which one is used follows
    ``field.field_type``, as in
    :func:`~firm3d.field.tracing.trace_particles_boozer`.

    Shear-Alfven-wave (perturbed) fields are *not* supported: they do work on
    the particle, so :math:`\mu` is not conserved across an orbit step and the
    splitting above does not hold.  This function takes a
    :class:`~firm3d.field.boozermagneticfield.BoozerMagneticField`, so a
    perturbed field cannot be passed in the first place; the perturbed
    right-hand sides additionally leave ``set_mu`` unimplemented, so a future
    caller that reached them would raise rather than silently integrate the
    wrong equations.

    See Hirvijoki et al., *Phys. Plasmas* **20**, 092505 (2013) and Boozer &
    Kuo-Petravic, *Phys. Fluids* **24**, 851 (1981) for the theoretical basis.

    Args:
        field: The :class:`~firm3d.field.boozermagneticfield.BoozerMagneticField`
            instance.
        stz_inits: ``(nparticles, 3)`` array of initial positions
            :math:`(s, \theta, \zeta)` in Boozer coordinates.
        parallel_speeds: ``(nparticles,)`` array of initial :math:`v_\parallel`
            in m/s.
        backgrounds: A :class:`ThermalBackground` or a list thereof.  When
            multiple species are provided their collision coefficients are
            summed.
        tmax: Integration time in seconds.
        mass: EP mass in kg.
        charge: EP charge in C.
        Ekin: Initial kinetic energy in J.  Either a scalar applied to all
            particles or a ``(nparticles,)`` array.
        tol: Default tolerance when solver-specific tolerances are not set.
        abstol: Absolute tolerance for the DP adaptive step control.
        reltol: Relative tolerance for the DP adaptive step control.
        comm: MPI communicator; particles are distributed across ranks.
        stopping_criteria: List of stopping criteria (same as
            :func:`~firm3d.field.tracing.trace_particles_boozer`).
        dt_save: Time interval at which trajectory snapshots are saved (s).
        forget_exact_path: If ``True``, return only the first and last state
            of each particle.
        axis: Coordinate singularity handling (0, 1, or 2; default 2).
        ode_solver: ``"dormand_prince"`` (recommended) or ``"boost"``.
        DP_hmin: Minimum step size for the Dormand-Prince solver, in
            seconds.  When the adaptive step falls below this value the
            step is accepted anyway.  Prevents the solver from grinding
            when a particle diffuses deep below the background thermal
            speed, where the pitch-scattering rate diverges as 1/v^3.
        rng_seed: Seed for the per-particle Wiener process.  Each particle
            uses ``rng_seed + particle_index`` so that MPI runs are
            reproducible.
        validate_profiles: If ``True`` (default), check up front that the
            background profiles give :math:`\ln\Lambda > 0` everywhere and
            raise :class:`ValueError` if not, rather than letting the C++
            layer abort part-way through a trace.  The check is evaluated at
            :math:`v \to 0`, where :math:`\ln\Lambda` is smallest, so it is
            conservative: it rejects profiles that a fast particle which
            never slows into the thermal range would survive.  Set to
            ``False`` to trace such a case anyway.
        mode: Which guiding-centre equations to use: ``"gc"``, ``"gc_vac"``
            or ``"gc_nok"``.  Defaults to ``"gc_" + field.field_type``;
            passing a value inconsistent with the field warns and proceeds
            with the value given.

    Returns:
        Tuple ``(res_tys, res_hits)`` where each element of ``res_tys`` is a
        numpy array of shape ``(ntimesteps, 6)`` with columns
        ``[t, s, θ, ζ, v_par, v]``.  The extra column ``v`` (total speed)
        allows the kinetic energy :math:`E = \tfrac{1}{2} m v^2` and
        magnetic moment :math:`\mu = (v^2 - v_\parallel^2) / (2B)` to be
        reconstructed at each saved point.
    """
    if stopping_criteria is None:
        stopping_criteria = []
    if abstol is None:
        abstol = tol
    if reltol is None:
        reltol = tol
    if dt_save <= 0:
        raise ValueError("dt_save must be positive.")

    # Accept a single background or a list
    if isinstance(backgrounds, ThermalBackground):
        backgrounds = [backgrounds]
    cpp_backgrounds = [b._to_cpp() for b in backgrounds]
    if validate_profiles:
        _validate_coulomb_log(cpp_backgrounds, float(mass), float(charge))

    # Select the orbit equations from the field, exactly as
    # trace_particles_boozer does.  Before this existed the collisional tracer
    # always used the vacuum equations, so a finite-beta field was silently
    # traced with the wrong orbits.
    if mode is not None:
        mode = mode.lower()
        assert mode in ["gc", "gc_vac", "gc_nok"]
        if "gc_" + field.field_type != mode:
            warnings.warn(
                f"Prescribed mode is inconsistent with field_type. "
                f"Proceeding with mode={mode}.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        mode = "gc_" + field.field_type

    nparticles = stz_inits.shape[0]
    assert len(parallel_speeds) == nparticles

    if np.isscalar(Ekin):
        Ekin = Ekin * np.ones(nparticles)
    assert len(Ekin) == nparticles

    vtotal = np.sqrt(2.0 * np.asarray(Ekin) / mass)

    res_tys = []
    res_hits = []
    first, last = parallel_loop_bounds(comm, nparticles)
    # Any exception escaping this loop must not skip the collectives below:
    # on the ranks that did not fail, allgather would block forever waiting
    # for a rank that has already unwound.  Capture instead, agree on the
    # outcome collectively, then raise everywhere.
    failure = None
    failure_exc = None
    for i in range(first, last):
        seed_i = int(rng_seed) + i
        try:
            res_ty, res_hit = sopp.particle_guiding_center_boozer_collision_tracing(
                field,
                stz_inits[i, :].tolist(),
                float(mass),
                float(charge),
                float(vtotal[i]),
                float(parallel_speeds[i]),
                float(tmax),
                cpp_backgrounds,
                vacuum=(mode == "gc_vac"),
                noK=(mode == "gc_nok"),
                stopping_criteria=stopping_criteria,
                dt_save=float(dt_save),
                forget_exact_path=bool(forget_exact_path),
                axis=int(axis),
                abstol=float(abstol),
                reltol=float(reltol),
                ode_solver=str(ode_solver),
                DP_hmin=float(DP_hmin),
                rng_seed=seed_i,
            )
        except Exception as exc:
            # Recorded rather than raised, so that every rank reaches the
            # allgather below; re-raised with `from` on all ranks afterwards.
            failure = f"particle {i}: {type(exc).__name__}: {exc}"
            failure_exc = exc
            break
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_hits.append(np.asarray(res_hit))

    if comm is not None:
        failures = comm.allgather(failure)
        failed = [(r, m) for r, m in enumerate(failures) if m is not None]
        if failed:
            rank, msg = failed[0]
            raise RuntimeError(
                f"collision tracing failed on {len(failed)} of {comm.size} "
                f"rank(s); first failure on rank {rank} -- {msg}"
            )
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_hits = [i for o in comm.allgather(res_hits) for i in o]
    elif failure is not None:
        raise RuntimeError(f"collision tracing failed on {failure}") from failure_exc

    return res_tys, res_hits
