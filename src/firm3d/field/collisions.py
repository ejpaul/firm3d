import numpy as np

import firm3dpp as sopp

from .._core.types import RealArray
from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import BoozerMagneticField
from ..util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
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

    The Coulomb logarithm is computed locally from the profiles using the
    classical Debye-shielding formula:

    .. math::

        \ln\Lambda_{ab} = \ln\!\left(
            \frac{4\pi\varepsilon_0\,\lambda_D\,m_r\,v_\mathrm{eff}^2}{|q_a q_b|}
        \right), \qquad
        v_\mathrm{eff}^2 = v^2 + v_{\mathrm{th},b}^2

    where :math:`\lambda_D` is the total Debye length from all background
    species and :math:`m_r = m_a m_b/(m_a+m_b)` is the reduced mass.
    The :math:`v_\mathrm{eff}^2` form handles both limits: fast EP
    (:math:`v \gg v_{\mathrm{th},b}`, relevant for ions) and slow EP against
    electrons (:math:`v \ll v_{\mathrm{th},e}`).  A floor of 2 is applied.

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
):
    r"""
    Trace guiding-centre particles including Monte Carlo Coulomb collisions
    with one or more Maxwellian background species.

    Uses adaptive Dormand-Prince for the full deterministic drift in
    :math:`(s, \theta, \zeta, v, \xi)` coordinates, followed by a Milstein
    noise step for :math:`(v, \xi)` at each accepted step.  See Hirvijoki
    et al., *Phys. Plasmas* **20**, 092505 (2013) and Boozer & Kuo-Petravic,
    *Phys. Fluids* **24**, 851 (1981) for the theoretical basis.

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

    nparticles = stz_inits.shape[0]
    assert len(parallel_speeds) == nparticles

    if np.isscalar(Ekin):
        Ekin = Ekin * np.ones(nparticles)
    assert len(Ekin) == nparticles

    vtotal = np.sqrt(2.0 * np.asarray(Ekin) / mass)

    res_tys = []
    res_hits = []
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        seed_i = int(rng_seed) + i
        res_ty, res_hit = sopp.particle_guiding_center_boozer_collision_tracing(
            field,
            stz_inits[i, :].tolist(),
            float(mass),
            float(charge),
            float(vtotal[i]),
            float(parallel_speeds[i]),
            float(tmax),
            cpp_backgrounds,
            stopping_criteria,
            float(dt_save),
            bool(forget_exact_path),
            int(axis),
            float(abstol),
            float(reltol),
            str(ode_solver),
            float(DP_hmin),
            seed_i,
        )
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_hits.append(np.asarray(res_hit))

    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_hits = [i for o in comm.allgather(res_hits) for i in o]

    return res_tys, res_hits
