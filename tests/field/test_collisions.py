"""
Tests for Monte Carlo Coulomb collision tracing.

All tests use BoozerAnalytic (no file I/O) and short integration times.

Quantitative drag-rate test — why electrons, not ions
------------------------------------------------------
For 3.52 MeV fusion alphas (v₀ ≈ 1.3×10⁷ m/s) the signal-to-noise ratio for
measuring the mean drag depends on the background species through the
Chandrasekhar G function.  Values below are computed from the formulas in
``collisions.h`` at n = 10²⁰ m⁻³, T = 10 keV:

  D-ion background:  x_D = v₀/v_th,D ≈ 13.3  →  G ≈ 2.8×10⁻³
    K_D ≈ −1.7×10⁶ m/s²

  Electron background: x_e = v₀/v_th,e ≈ 0.22  →  G ≈ 0.080
    K_e ≈ −3.7×10⁷ m/s²

Electron drag therefore dominates by a factor ≈ 21, not by the mass ratio:
G(x_e) is 37% of its maximum (0.214 at x ≈ 0.968), and the 1/v² factors do
most of the rest.  TestCollisionCoefficients.test_K_electron_larger_than_ion
pins this ratio to [10, 50].

Note the drag signal is small against the diffusive scatter: at tmax = 5 μs,
K_e·tmax ≈ −180 m/s while √(2 D_par tmax) ≈ 2.6×10³ m/s per particle, so
σ(mean) ≈ 370 m/s at N = 50 — an SNR below 1.  The drag tests below are
therefore weak detectors of magnitude; they check sign and order, and
TestCollisionCoefficients is where the coefficients themselves are pinned.

Test structure
--------------
  TestThermalBackground       — Python-layer ThermalBackground class
  TestTrajectoryShape         — output format, monotone time, column order
  TestCollisionlessLimit      — exact recovery of collisionless dynamics (n = 0)
  TestCollisionCoefficients   — analytical K, ν_D, Chandrasekhar G, scalings
  TestCollisionPhysics        — qualitative + quantitative simulation checks
  TestPitchIsotropization     — qualitative broadening (ξ₀=0) + quantitative
                                exponential decay ⟨ξ(t)⟩=ξ₀exp(−ν_D t) (ASCOT5)
"""

import unittest
import warnings

import numpy as np
import scipy.special
import scipy.stats

import firm3dpp as sopp
from firm3d.field.boozermagneticfield import (
    BoozerAnalytic,
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
)
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_perturbed_with_collisions,
    trace_particles_boozer_with_collisions,
)
from firm3d.field.tracing import (
    IterationStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    ONE_EV,
    PROTON_MASS,
    VACUUM_PERMITTIVITY,
)


def _mpi_comm():
    """MPI communicator when launched under an MPI launcher, else None.

    The tracer distributes particles over ranks and allgathers the
    results, so every rank sees identical data and runs identical
    assertions.  Per-particle RNG seeds are rank-independent, making the
    results bit-identical to a serial run.  Launch e.g. with
    srun -n 64 python -m pytest tests/field/test_collisions.py
    (see tests/perlmutter/run_slow_tests.sh).  mpi4py is only imported
    when an MPI launcher is detected, so serial runs never touch MPI.
    """
    import os

    if (
        int(os.environ.get("SLURM_NTASKS", "1")) > 1
        or int(os.environ.get("PMI_SIZE", "1")) > 1
        or int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1
    ):
        try:
            from mpi4py import MPI

            return MPI.COMM_WORLD
        except ImportError:
            return None
    return None


_COMM = _mpi_comm()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _field():
    """BoozerAnalytic near-axis field (axisymmetric, vacuum)."""
    return BoozerAnalytic(1.0, 5.0, 0, 40.0, 0.5, 0.4)


def _zero_background():
    """Zero-density background: all collision coefficients identically zero."""
    return ThermalBackground(
        n_profile=lambda s: 0.0,
        T_profile=lambda s: 10e3 * ONE_EV,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    )


def _hot_background(n=1e20, T_keV=10.0):
    """Dense, hot deuterium background."""
    T = T_keV * 1e3 * ONE_EV
    return ThermalBackground(
        n_profile=lambda s: n,
        T_profile=lambda s: T,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    )


def _cold_background(n=1e20, T_keV=0.01):
    """Cold background: large drag on 3.5 MeV alphas."""
    T = T_keV * 1e3 * ONE_EV
    return ThermalBackground(
        n_profile=lambda s: n,
        T_profile=lambda s: T,
        mass=2 * PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    )


def _coll_trace(
    field,
    backgrounds,
    tmax=1e-6,
    vpar_fraction=0.5,
    tol=1e-9,
    dt_save_factor=20,
    **kwargs,
):
    """Single-particle collision trace; returns (ntimesteps, 6) array."""
    Ekin = FUSION_ALPHA_PARTICLE_ENERGY
    v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
    stz = np.array([[0.3, 0.0, 0.0]])
    vpar = np.array([vpar_fraction * v0])
    res_tys, _ = trace_particles_boozer_with_collisions(
        field,
        stz,
        vpar,
        backgrounds=backgrounds,
        tmax=tmax,
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        Ekin=Ekin,
        tol=tol,
        dt_save=tmax / dt_save_factor,
        **kwargs,
    )
    return res_tys[0]


# ---------------------------------------------------------------------------
# Analytical helpers (mirror collisions.h in Python)
# ---------------------------------------------------------------------------


def _chandrasekhar_G(x):
    """G(x) = [erf(x) - (2x/√π)exp(-x²)] / (2x²)."""
    if x == 0.0:
        return 0.0
    return (scipy.special.erf(x) - (2.0 * x / np.sqrt(np.pi)) * np.exp(-(x**2))) / (
        2.0 * x**2
    )


def _chandrasekhar_G_deriv(x):
    """G'(x) = (2/√π) exp(-x²) - 2 G(x)/x."""
    if x == 0.0:
        return 2.0 / (3.0 * np.sqrt(np.pi))
    return (2.0 / np.sqrt(np.pi)) * np.exp(-(x**2)) - 2.0 * _chandrasekhar_G(x) / x


_HBAR = 1.054571817e-34  # J·s (reduced Planck)


def _coulomb_log(v, m_a, q_a, m_b, q_b, T_b, lambda_D):
    """
    ln Λ = ln(λ_D / b_min), b_min = max(b_cl, b_qm).
    Mirrors compute_collision_coefficients() in collisions.h (ASCOT5
    convention, no floor).
    """
    v_th_b = np.sqrt(2.0 * T_b / m_b)
    m_r = m_a * m_b / (m_a + m_b)
    v_eff_sq = v**2 + v_th_b**2
    b_cl = abs(q_a * q_b) / (4.0 * np.pi * VACUUM_PERMITTIVITY * m_r * v_eff_sq)
    b_qm = _HBAR / (2.0 * m_r * np.sqrt(v_eff_sq))
    return np.log(lambda_D / max(b_cl, b_qm))


def _debye_length(s, backgrounds):
    """Total Debye length from all background species at flux label s."""
    inv_lD_sq = 0.0
    for bg in backgrounds:
        n_b = bg.n_profile(s)
        T_b = bg.T_profile(s)
        if n_b > 0 and T_b > 0:
            inv_lD_sq += n_b * bg.charge**2 / (VACUUM_PERMITTIVITY * T_b)
    return 1.0 / np.sqrt(inv_lD_sq) if inv_lD_sq > 0 else 0.0


def _analytical_K(v, s, m_a, q_a, backgrounds):
    """
    Deterministic drag coefficient K [m/s²].
    K = Q + dD_par/dv + 2*D_par/v  summed over all background species.
    Mirrors compute_collision_coefficients() in collisions.h.
    """
    lambda_D = _debye_length(s, backgrounds)
    K = 0.0
    for bg in backgrounds:
        n_b = bg.n_profile(s)
        T_b = bg.T_profile(s)
        if n_b <= 0 or T_b <= 0:
            continue
        m_b, q_b = bg.mass, bg.charge
        lnL = _coulomb_log(v, m_a, q_a, m_b, q_b, T_b, lambda_D)
        v_th = np.sqrt(2.0 * T_b / m_b)
        x = v / v_th
        G = _chandrasekhar_G(x)
        Gp = _chandrasekhar_G_deriv(x)
        Gamma = (
            n_b
            * q_a**2
            * q_b**2
            * lnL
            / (4.0 * np.pi * VACUUM_PERMITTIVITY**2 * m_a**2)
        )
        D_par = Gamma * G / v
        dD_par_dv = Gamma * (Gp / v_th - G / v) / v
        # Einstein-relation drag (matches ASCOT5 mccc_coefs_Q)
        Q = -Gamma * G * m_a / T_b
        K += Q + dD_par_dv + 2.0 * D_par / v
    return K


def _analytical_nu_D(v, s, m_a, q_a, backgrounds):
    """Pitch-angle scattering frequency ν_D [s⁻¹] summed over species."""
    lambda_D = _debye_length(s, backgrounds)
    nu_D = 0.0
    for bg in backgrounds:
        n_b = bg.n_profile(s)
        T_b = bg.T_profile(s)
        if n_b <= 0 or T_b <= 0:
            continue
        m_b, q_b = bg.mass, bg.charge
        lnL = _coulomb_log(v, m_a, q_a, m_b, q_b, T_b, lambda_D)
        v_th = np.sqrt(2.0 * T_b / m_b)
        x = v / v_th
        G = _chandrasekhar_G(x)
        Gamma = (
            n_b
            * q_a**2
            * q_b**2
            * lnL
            / (4.0 * np.pi * VACUUM_PERMITTIVITY**2 * m_a**2)
        )
        nu_D += Gamma * (scipy.special.erf(x) - G) / v**3
    return nu_D


# ===========================================================================
# Test classes
# ===========================================================================


class TestThermalBackground(unittest.TestCase):
    def test_non_callable_n_raises(self):
        with self.assertRaises(ValueError):
            ThermalBackground(
                n_profile=1e20,
                T_profile=lambda s: 1.0,
                mass=PROTON_MASS,
                charge=ELEMENTARY_CHARGE,
            )

    def test_non_callable_T_raises(self):
        with self.assertRaises(ValueError):
            ThermalBackground(
                n_profile=lambda s: 1e20,
                T_profile=5.0,
                mass=PROTON_MASS,
                charge=ELEMENTARY_CHARGE,
            )

    def test_to_cpp_grid_length(self):
        bg = ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 1.0,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
            n_grid_points=64,
        )
        cpp = bg._to_cpp()
        self.assertEqual(len(cpp.s_grid), 64)
        self.assertEqual(len(cpp.n_grid), 64)
        self.assertEqual(len(cpp.T_grid), 64)

    def test_to_cpp_endpoint_values(self):
        """Profile values at s = 0 and s = 1 survive the grid round-trip."""
        bg = ThermalBackground(
            n_profile=lambda s: 1e20 * (1 - s),
            T_profile=lambda s: 1e3 * ONE_EV * s,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
            n_grid_points=100,
        )
        cpp = bg._to_cpp()
        self.assertAlmostEqual(cpp.n_grid[0], 1e20, delta=1e15)
        self.assertAlmostEqual(cpp.n_grid[-1], 0.0, delta=1e15)
        self.assertAlmostEqual(cpp.T_grid[0], 0.0, delta=1.0)
        self.assertAlmostEqual(cpp.T_grid[-1], 1e3 * ONE_EV, delta=1e-16 * ONE_EV)

    def test_single_background_accepted(self):
        """A bare ThermalBackground (not in a list) is accepted."""
        field = _field()
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
        res_tys, _ = trace_particles_boozer_with_collisions(
            field,
            np.array([[0.3, 0.0, 0.0]]),
            np.array([0.5 * v0]),
            backgrounds=_hot_background(),
            tmax=1e-7,
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            Ekin=Ekin,
            dt_save=5e-8,
        )
        self.assertEqual(len(res_tys), 1)


class TestTrajectoryShape(unittest.TestCase):
    def test_output_shape(self):
        """Each trajectory has shape (ntimesteps, 6): [t, s, θ, ζ, v∥, v]."""
        traj = _coll_trace(_field(), _hot_background(), tmax=1e-6)
        self.assertEqual(traj.ndim, 2)
        self.assertEqual(traj.shape[1], 6)

    def test_time_monotone(self):
        """Time (column 0) must be strictly increasing."""
        traj = _coll_trace(_field(), _hot_background(), tmax=1e-6)
        self.assertTrue(np.all(np.diff(traj[:, 0]) > 0))

    def test_vpar_leq_v(self):
        """|v∥| ≤ v at every saved point."""
        traj = _coll_trace(_field(), _hot_background(), tmax=1e-6)
        vpar, v = traj[:, 4], traj[:, 5]
        self.assertTrue(np.all(np.abs(vpar) <= v * (1 + 1e-10)))

    def test_forget_exact_path_returns_two_rows(self):
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
        res_tys, _ = trace_particles_boozer_with_collisions(
            _field(),
            np.array([[0.3, 0.0, 0.0]]),
            np.array([0.5 * v0]),
            backgrounds=_hot_background(),
            tmax=1e-6,
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            Ekin=Ekin,
            dt_save=1e-7,
            forget_exact_path=True,
        )
        self.assertEqual(res_tys[0].shape[0], 2)


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------


class TestIterationStoppingCriterion(unittest.TestCase):
    """
    The `iter` argument passed to stopping criteria must count accepted
    solver steps, as it does in the collisionless tracer.

    solve_sde() originally synthesised this argument from the elapsed
    normalised time, ``(int)(tau / (dtau_max * 1e-3))``, which is a
    proxy for simulated time rather than a step count.  Two consequences
    are asserted against here:

      * the stop time was independent of the solver tolerance, since it
        depended only on tau (pre-fix, max_iter = 5 stopped at exactly
        t = 3.7532e-09 s at both 1e-6 and 1e-11);
      * the stop time saturated in max_iter, so distinct limits produced
        identical traces (pre-fix, max_iter = 10 and 20 both stopped at
        t = 1.9392e-08 s).

    ToroidalTransitStoppingCriterion is affected by the same argument --
    it keys its `zeta_init` initialisation on ``iter == 1`` -- but needs
    a confined multi-transit orbit to exercise, so the step counter is
    tested directly here instead.
    """

    _TMAX = 1e-6

    def _stop_time(self, max_iter, tol):
        """Final time of a single zero-density trace under an iteration cap."""
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
        res_tys, res_hits = trace_particles_boozer_with_collisions(
            _field(),
            np.array([[0.3, 0.0, 0.0]]),
            np.array([0.5 * v0]),
            backgrounds=_zero_background(),
            tmax=self._TMAX,
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            Ekin=Ekin,
            dt_save=self._TMAX / 20,
            tol=tol,
            stopping_criteria=[IterationStoppingCriterion(max_iter)],
        )
        return res_tys[0][-1, 0], np.asarray(res_hits[0])

    def test_criterion_fires_and_is_recorded(self):
        """A low cap stops the trace early and records a hit at index -1."""
        t_end, hits = self._stop_time(5, tol=1e-9)
        self.assertLess(t_end, self._TMAX)
        self.assertEqual(hits.shape[0], 1)
        self.assertEqual(hits[0, 1], -1.0)

    def test_no_stop_when_cap_is_unreachable(self):
        """A cap above the step count runs to tmax with no hit recorded."""
        t_end, hits = self._stop_time(10**6, tol=1e-9)
        self.assertAlmostEqual(t_end, self._TMAX, delta=1e-15)
        self.assertEqual(hits.size, 0)

    def test_stop_time_depends_on_tolerance(self):
        """Tighter tolerance means smaller steps, so N steps cover less time."""
        t_loose, _ = self._stop_time(5, tol=1e-6)
        t_tight, _ = self._stop_time(5, tol=1e-11)
        self.assertGreater(
            t_loose,
            2 * t_tight,
            f"stop time barely moved with tolerance (loose {t_loose:.4e}, "
            f"tight {t_tight:.4e}); iter is not counting solver steps",
        )

    def test_stop_time_increases_with_cap(self):
        """Doubling the cap must let the trace run measurably further.

        Run at 1e-11 so that 20 steps still fall well short of tmax; at
        looser tolerances the steps are large enough that the cap is
        never reached and both traces simply clamp to tmax.
        """
        t_10, hits_10 = self._stop_time(10, tol=1e-11)
        t_20, hits_20 = self._stop_time(20, tol=1e-11)
        # Both traces must have stopped on the criterion, or the
        # comparison below is between two values clamped at tmax.
        self.assertEqual(hits_10.shape[0], 1, "max_iter = 10 ran to tmax")
        self.assertEqual(hits_20.shape[0], 1, "max_iter = 20 ran to tmax")
        self.assertGreater(
            t_20,
            1.5 * t_10,
            f"stop time saturated in max_iter (10 -> {t_10:.4e}, 20 -> {t_20:.4e})",
        )


# ---------------------------------------------------------------------------
# Collisionless limit
# ---------------------------------------------------------------------------


class TestCollisionlessLimit(unittest.TestCase):
    """
    When n = 0 all collision coefficients Γ are identically zero.

    Expected consequences:
      K  = 0  →  v̇ = 0  →  v(t) is bitwise identical to v(0) at every
                             saved point (the ODE RHS v-component is exactly
                             zero, so Dormand-Prince never perturbs it).
      g_v = √(2 D_par) = 0  →  Milstein noise step is exactly zero.

    The (s, θ, ζ) orbit must therefore satisfy the same vacuum GC equations
    as the collisionless tracer, recovering collisionless dynamics exactly.
    """

    _tmax = 5e-7
    _tol = 1e-9
    _stz = np.array([[0.3, 0.1, 0.2]])
    _vpfrac = 0.6

    def _traces(self):
        """Return (traj_coll, traj_nocoll) for the same particle and settings."""
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        vpar = np.array([self._vpfrac * v0])
        field = _field()
        kwargs = {
            "tmax": self._tmax,
            "mass": m,
            "charge": q,
            "Ekin": Ekin,
            "abstol": self._tol,
            "reltol": self._tol,
            "dt_save": self._tmax / 20,
        }

        res_c, _ = trace_particles_boozer_with_collisions(
            field,
            self._stz,
            vpar,
            backgrounds=_zero_background(),
            ode_solver="dormand_prince",
            axis=2,
            **kwargs,
        )

        res_nc, _ = trace_particles_boozer(
            field,
            self._stz,
            vpar,
            ODE_solver="dormand_prince",
            axis=2,
            **kwargs,
        )
        return res_c[0], res_nc[0]

    # ------------------------------------------------------------------
    # 1. Speed exactly preserved
    # ------------------------------------------------------------------
    def test_speed_constant_to_solver_tolerance(self):
        """
        v(t) must equal v(0) to integration accuracy when n = 0.

        The integrated state is (s, theta, zeta, v_par) at fixed mu, and the
        speed is reconstructed as v^2 = v_par^2 + 2 mu |B|.  Energy
        conservation therefore rests on the mirror force in v_par balancing
        the change in |B| along the orbit, which holds to solver tolerance
        rather than exactly.  This matches the collisionless tracer, which
        integrates the same variables.

        (Before mu became a parameter of the orbit equations, v was itself a
        state variable with dv/dt identically zero here, so it was bitwise
        constant.  That was a property of the old state layout, not of the
        physics.)

        The observed drift tracks the tolerance closely -- about 0.15 * tol,
        measured as 1.6e-10 at tol = 1e-9 and 1.2e-12 at tol = 1e-11 -- so the
        bound below sits ~3 orders above the real level while staying far
        tighter than any genuine mu-handling bug, which would show up at O(1).
        """
        traj = _coll_trace(
            _field(),
            _zero_background(),
            tmax=self._tmax,
            vpar_fraction=self._vpfrac,
            tol=self._tol,
            ode_solver="dormand_prince",
            axis=2,
        )
        v0 = traj[0, 5]
        drift = np.abs(traj[:, 5] / v0 - 1.0).max()
        self.assertLess(
            drift,
            100 * self._tol,
            f"speed drifted by {drift:.3e} relative with n = 0, above "
            f"{100 * self._tol:.1e}; energy is not being conserved by the "
            f"orbit integration",
        )

    # ------------------------------------------------------------------
    # 2. Magnetic moment conserved
    # ------------------------------------------------------------------
    def test_magnetic_moment_matches_initial_conditions(self):
        """
        μ along the trajectory must equal the value implied by the caller's
        own (Ekin, v∥, position).

        μ conservation itself is no longer a numerical property worth
        asserting: μ is a parameter of the orbit equations, held fixed
        between collision kicks by construction, and the trace reports
        v = √(v∥² + 2μ|B|).  Recovering μ from those columns therefore
        returns it by definition — an assertion on that drift would hold
        even if the orbit equations were wrong.  Energy conservation, which
        is a real numerical property here, is covered by
        test_speed_constant_to_solver_tolerance.

        What remains falsifiable is the *value*: if C++ derived μ_init
        wrongly from vtotal, v∥ and |B| at the launch point, every
        reconstructed speed is wrong by that factor.  This pins it against
        an independent Python computation from the inputs alone.
        """
        field = _field()
        traj = _coll_trace(
            field,
            _zero_background(),
            tmax=self._tmax,
            vpar_fraction=self._vpfrac,
            tol=self._tol,
            ode_solver="dormand_prince",
            axis=2,
        )
        # mu_init from the caller's inputs, touching no trajectory output.
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        vpar0 = self._vpfrac * v0
        field.set_points(np.array([[0.3, 0.0, 0.0]]))
        mu_expected = (v0**2 - vpar0**2) / (2.0 * np.array(field.modB()).flatten()[0])

        pts = np.column_stack([traj[:, 1], traj[:, 2], traj[:, 3]])
        field.set_points(pts)
        B = np.array(field.modB()).flatten()
        mu = (traj[:, 5] ** 2 - traj[:, 4] ** 2) / (2.0 * B)

        rel_err = np.abs(mu - mu_expected) / mu_expected
        self.assertLess(
            np.max(rel_err),
            1e-5,
            f"μ = {mu[0]:.6e} disagrees with the value implied by the inputs "
            f"({mu_expected:.6e}) by {np.max(rel_err):.2e} relative",
        )

    # ------------------------------------------------------------------
    # 3. Orbit agrees with collisionless tracer
    # ------------------------------------------------------------------
    def test_orbit_matches_collisionless_tracer(self):
        """
        (s, θ, ζ, v∥) from the collision tracer with n = 0 must agree with
        the standard collisionless tracer to within ~1e-4 relative.

        Both use Dormand-Prince, the same tolerances, and axis = 2.
        Any discrepancy is purely due to the (v, ξ) vs (v∥) state conditioning.
        """
        traj_c, traj_nc = self._traces()
        n = min(traj_c.shape[0], traj_nc.shape[0])

        # Saved time stamps must align
        np.testing.assert_allclose(
            traj_c[:n, 0],
            traj_nc[:n, 0],
            rtol=1e-6,
            atol=0,
            err_msg=(
                "Saved times disagree between collision (n=0) and collisionless tracers"
            ),
        )

        # (s, θ, ζ)
        for col, name in [(1, "s"), (2, "θ"), (3, "ζ")]:
            np.testing.assert_allclose(
                traj_c[:n, col],
                traj_nc[:n, col],
                rtol=1e-4,
                atol=1e-6,
                err_msg=(
                    f"{name}(t) differs between n=0 collision and collisionless traces"
                ),
            )

        # v∥ = ξ·v (collision) vs v∥ (collisionless)
        np.testing.assert_allclose(
            traj_c[:n, 4],
            traj_nc[:n, 4],
            rtol=1e-4,
            atol=1.0,  # 1 m/s floor for numerical noise
            err_msg="v∥ differs between n=0 collision and collisionless traces",
        )


# ---------------------------------------------------------------------------
# Orbit equations for non-vacuum fields
# ---------------------------------------------------------------------------


class TestNonVacuumOrbitEquations(unittest.TestCase):
    """
    Collisional tracing must use the orbit equations the field calls for.

    trace_particles_boozer picks vacuum / noK / full from field.field_type.
    The collisional tracer used to build the vacuum right-hand side
    unconditionally, so a finite-beta field was traced with the wrong orbit
    equations and nothing warned.  mu is now a parameter of the orbit
    equations rather than a state variable, so every static-field variant can
    be driven by the collision operator and the selection is shared.
    """

    _tmax = 5e-7
    _tol = 1e-10
    _stz = np.array([[0.3, 0.0, 0.0]])

    # I0 alone gives field_type 'nok'; adding K1 gives '' (full GC).  Both are
    # orbit models the collisional tracer could not reach before.
    _FIELDS = {
        "nok": {"I0": 0.5},
        "": {"I0": 0.5, "K1": 0.3},
    }

    @classmethod
    def _nonvacuum_field(cls, field_type=""):
        return BoozerAnalytic(1.0, 5.0, 0, 40.0, 0.5, 0.4, **cls._FIELDS[field_type])

    def _endpoint(self, collisional, mode=None, field_type=""):
        """Final [s, theta, zeta, v_par] for one particle."""
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        vpar = np.array([0.6 * v0])
        kwargs = {
            "tmax": self._tmax,
            "mass": m,
            "charge": q,
            "Ekin": Ekin,
            "abstol": self._tol,
            "reltol": self._tol,
            "dt_save": self._tmax,
        }
        # Pin both to the same stepper.  The two functions default
        # differently (dormand_prince vs boost), and leaving that alone
        # would make the agreement margin below a measure of stepper
        # difference rather than of the orbit equations.
        if collisional:
            res, _ = trace_particles_boozer_with_collisions(
                self._nonvacuum_field(field_type),
                self._stz,
                vpar,
                backgrounds=_zero_background(),
                mode=mode,
                ode_solver="dormand_prince",
                **kwargs,
            )
        else:
            res, _ = trace_particles_boozer(
                self._nonvacuum_field(field_type),
                self._stz,
                vpar,
                mode=mode,
                ODE_solver="dormand_prince",
                **kwargs,
            )
        return np.asarray(res[0])[-1, 1:5]

    def test_field_type_is_as_expected(self):
        for ft in self._FIELDS:
            with self.subTest(field_type=ft):
                self.assertEqual(self._nonvacuum_field(ft).field_type, ft)

    def test_collisionless_limit_matches_collisionless_tracer(self):
        """
        With n = 0 the collisional tracer must reproduce trace_particles_boozer
        for every non-vacuum orbit model.  This is what fails if the vacuum
        equations are used for a field that needs another set.
        """
        for ft in self._FIELDS:
            with self.subTest(field_type=ft):
                coll = self._endpoint(collisional=True, field_type=ft)
                nocoll = self._endpoint(collisional=False, field_type=ft)
                np.testing.assert_allclose(
                    coll,
                    nocoll,
                    rtol=1e-6,
                    atol=1e-9,
                    err_msg=(
                        f"collisional trace at n=0 disagrees with the "
                        f"collisionless tracer for field_type={ft!r}: the "
                        f"orbit equations differ"
                    ),
                )

    def test_vacuum_equations_are_measurably_wrong_here(self):
        """
        Forcing gc_vac on this field must give a different orbit.

        Without this the test above could pass simply because the two
        formulations happen to agree for this equilibrium, which would make it
        blind to the bug it is meant to catch.
        """
        full = self._endpoint(collisional=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vac = self._endpoint(collisional=True, mode="gc_vac")
        rel = np.abs(full - vac) / np.maximum(np.abs(full), 1e-30)
        self.assertGreater(
            rel.max(),
            1e-3,
            f"vacuum and full GC equations agree to {rel.max():.2e} on this "
            f"field, so this configuration cannot detect the wrong choice",
        )


# ---------------------------------------------------------------------------
# Perturbed (shear-Alfven-wave) collisional tracing
# ---------------------------------------------------------------------------


class TestPerturbedCollisions(unittest.TestCase):
    """
    Collisions in a shear-Alfven-wave field.

    The wave does work on the particle, so the speed is not conserved between
    kicks.  The operator splitting survives that because it needs mu to be
    invariant across an orbit step, not v: at SAW frequencies
    (omega << Omega_c) mu remains an adiabatic invariant, which is why the
    perturbed right-hand sides carry it as a constructor parameter in the
    first place.  The kick reconstructs v from v_par^2 + 2 mu |B0| each time,
    so a changing energy is handled.
    """

    _tmax = 2e-7
    _tol = 1e-11
    _stz = np.array([[0.3, 0.0, 0.0]])

    def _setup(self, Phihat, field_kwargs=None):
        # The field must stay referenced: ShearAlfvenWave.B0 only returns the
        # Python subclass (and hence field_type) while a Python reference to
        # it survives.  Returned alongside the wave for that reason.
        field = BoozerAnalytic(1.0, 5.0, 0, 40.0, 0.5, 0.4, **(field_kwargs or {}))
        self._field = field
        saw = ShearAlfvenHarmonic(Phihat, 2, 1, 1e5, 0.0, field)
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        field.set_points(self._stz)
        B0 = field.modB()[0, 0]
        vpar0 = 0.6 * v0
        mu0 = (v0**2 - vpar0**2) / (2.0 * B0)
        return saw, v0, np.array([vpar0]), np.array([mu0])

    def _kw(self):
        return {
            "tmax": self._tmax,
            "mass": ALPHA_PARTICLE_MASS,
            "charge": ALPHA_PARTICLE_CHARGE,
            "abstol": self._tol,
            "reltol": self._tol,
            "dt_save": self._tmax,
        }

    def test_collisionless_limit_matches_perturbed_tracer(self):
        """
        At zero density the collisional path must reproduce
        trace_particles_boozer_perturbed, including when the wave is strong
        enough to change the energy substantially.
        """
        for Phihat in (1e-3, 1e6):
            with self.subTest(Phihat=Phihat):
                saw, v0, vpar, mus = self._setup(Phihat)
                c, _ = trace_particles_boozer_perturbed_with_collisions(
                    saw,
                    self._stz,
                    vpar,
                    mus,
                    backgrounds=_zero_background(),
                    ode_solver="dormand_prince",
                    **self._kw(),
                )
                n, _ = trace_particles_boozer_perturbed(
                    saw,
                    self._stz,
                    vpar,
                    mus,
                    ODE_solver="dormand_prince",
                    **self._kw(),
                )
                np.testing.assert_allclose(
                    np.asarray(c[0])[-1, 1:5],
                    np.asarray(n[0])[-1, 1:5],
                    rtol=1e-9,
                    err_msg=f"perturbed collisional path diverges at Phihat={Phihat}",
                )

    def test_wave_does_work_on_the_particle(self):
        """
        The premise the splitting is built on: v is *not* conserved here.

        Without this the test above could pass on a wave too weak to matter,
        which would say nothing about whether a changing energy is handled.
        """
        saw, v0, vpar, mus = self._setup(1e6)
        res, _ = trace_particles_boozer_perturbed_with_collisions(
            saw,
            self._stz,
            vpar,
            mus,
            backgrounds=_zero_background(),
            ode_solver="dormand_prince",
            **self._kw(),
        )
        dv = np.asarray(res[0])[-1, 5] / v0 - 1.0
        self.assertGreater(
            abs(dv),
            1e-2,
            f"wave changed the speed by only {dv:.2e}; too weak "
            f"to exercise a non-conserved energy",
        )

    def test_collisions_change_mu(self):
        """
        With a background present mu must evolve, since the kick rewrites it
        from the post-kick (v, xi).  Collisionlessly it is held fixed.
        """
        saw, v0, vpar, mus = self._setup(1e-3)
        kw = dict(self._kw())
        kw["dt_save"] = self._tmax / 20
        out = {}
        for label, bg in [
            ("collisionless", _zero_background()),
            ("collisional", _hot_background()),
        ]:
            res, _ = trace_particles_boozer_perturbed_with_collisions(
                saw,
                self._stz,
                vpar,
                mus,
                backgrounds=bg,
                ode_solver="dormand_prince",
                rng_seed=3,
                **kw,
            )
            t = np.asarray(res[0])
            saw.B0.set_points(np.column_stack([t[:, 1], t[:, 2], t[:, 3]]))
            B = np.array(saw.B0.modB()).flatten()
            out[label] = (t[:, 5] ** 2 - t[:, 4] ** 2) / (2.0 * B)
        drift_free = np.abs(out["collisionless"] / mus[0] - 1).max()
        drift_coll = np.abs(out["collisional"] / mus[0] - 1).max()
        self.assertLess(drift_free, 1e-6, "mu moved without collisions")
        self.assertGreater(
            drift_coll,
            10 * max(drift_free, 1e-12),
            f"mu barely moved with collisions ({drift_coll:.2e}); the kick is "
            f"not updating it",
        )

    def test_zero_amplitude_matches_static_collisional_path(self):
        """
        At zero wave amplitude the perturbed path must reproduce the static
        collisional path, collisions included.

        This is the test that observes the ``set_mu`` overrides this feature
        adds to the two perturbed right-hand sides.  Nothing else does: the
        speed column is written as ``sqrt(v_par^2 + 2 mu |B|)`` from the
        integrator's own ``mu``, so any assertion that recovers ``mu`` from
        the output returns it by construction and holds even if the orbit
        integrated a stale value.

        Here the kick's ``mu`` reaches the orbit only through ``set_mu``, and
        the static path -- whose ``set_mu`` is exercised by the rest of the
        suite -- provides an independent reference for the same physics, since
        the perturbed equations reduce to the vacuum ones when Phihat = 0.
        A background is essential: with no collisions ``mu`` never changes and
        ``set_mu`` cannot matter.

        Measured: 2.0e-11 as shipped, against 2.0e-04 with both perturbed
        ``set_mu`` overrides replaced by no-ops -- seven orders of separation,
        so the bound below is not delicately placed.
        """
        field = BoozerAnalytic(1.0, 5.0, 0, 40.0, 0.5, 0.4)
        saw = ShearAlfvenHarmonic(0.0, 2, 1, 1e5, 0.0, field)
        bg = ThermalBackground(
            n_profile=lambda s: 1e21,
            T_profile=lambda s: 1e3 * ONE_EV,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        v0 = np.sqrt(2 * Ekin / ALPHA_PARTICLE_MASS)
        vpar0 = 0.6 * v0
        field.set_points(self._stz)
        mu0 = (v0**2 - vpar0**2) / (2.0 * field.modB()[0, 0])

        kw = {
            "tmax": 3e-7,
            "mass": ALPHA_PARTICLE_MASS,
            "charge": ALPHA_PARTICLE_CHARGE,
            "abstol": 1e-11,
            "reltol": 1e-11,
            "dt_save": 3e-7,
            "rng_seed": 7,
            "backgrounds": bg,
        }
        static, _ = trace_particles_boozer_with_collisions(
            field, self._stz, np.array([vpar0]), Ekin=Ekin, **kw
        )
        perturbed, _ = trace_particles_boozer_perturbed_with_collisions(
            saw, self._stz, np.array([vpar0]), np.array([mu0]), **kw
        )
        np.testing.assert_allclose(
            np.asarray(perturbed[0])[-1, 1:],
            np.asarray(static[0])[-1, 1:],
            rtol=1e-8,
            atol=1e-12,
            err_msg=(
                "zero-amplitude perturbed collisional trace disagrees with the "
                "static one: the kick's mu is not reaching the perturbed orbit "
                "equations"
            ),
        )

    def test_negative_mu_is_refused(self):
        """
        A negative or NaN mu must be refused rather than traced.

        Without the check the trace runs to completion and returns finite,
        plausible-looking numbers, because the two consumers of mu disagree:
        the orbit right-hand side integrates the negative value while the
        kick's speed reconstruction clamps v_perp^2 up to zero.  Measured on
        the unguarded code with vpar0 = 0.5 v0 and mu = -1e12: the same speed
        as mu = 0 (6.514e+06) but a different theta (0.01513 against
        0.01542) -- a trajectory belonging to no particle at all, with
        nothing in the output marking it as such.  This fixture launches at
        0.6 v0 and negates its own mu0, so its numbers differ from those;
        the failure mode is the same.

        NaN is checked because the natural spelling of the guard misses it:
        NaN < 0 is False, so a "< 0" test admits exactly the input the check
        exists to reject.  mu = 0 is legal (a strictly passing particle) and
        must still trace, so the boundary is checked in both directions.
        """
        saw, _, vpar0, mu0 = self._setup(1e-5)
        kw = {
            "tmax": 1e-8,
            "mass": ALPHA_PARTICLE_MASS,
            "charge": ALPHA_PARTICLE_CHARGE,
            "dt_save": 1e-8,
            "rng_seed": 1,
            "backgrounds": _hot_background(),
        }
        for label, bad_mu in (
            ("negative", -abs(mu0[0])),
            ("nan", np.nan),
        ):
            with self.subTest(mu=label):
                with self.assertRaises(ValueError) as cm:
                    trace_particles_boozer_perturbed_with_collisions(
                        saw, self._stz, vpar0, np.array([bad_mu]), **kw
                    )
                msg = str(cm.exception)
                self.assertIn("non-negative", msg)
                self.assertIn("index 0", msg)

        # The boundary itself is a legal input, not an error.
        res, _ = trace_particles_boozer_perturbed_with_collisions(
            saw, self._stz, vpar0, np.array([0.0]), **kw
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(res[0]))))

    def test_dead_b0_reference_is_refused_with_the_cause(self):
        """
        A ShearAlfvenWave whose BoozerMagneticField reference has been dropped
        must be refused, whether or not mode is supplied.

        B0 returns the Python subclass only while a Python reference to it
        survives.  Once it does not, the field cannot be evaluated at all --
        iota, G and modB are all implemented on the subclass -- so tracing
        fails inside the first right-hand-side evaluation with
        "_iota_impl was not implemented", which names nothing about the cause.
        Passing mode explicitly does not help: it gets past the field_type
        lookup and then fails exactly the same way, deeper.
        """

        def wave_with_dropped_field():
            field = BoozerAnalytic(1.0, 5.0, 0, 40.0, 0.5, 0.4)
            return ShearAlfvenHarmonic(1e-4, 2, 1, 1e5, 0.0, field)

        kw = {
            "tmax": 1e-8,
            "mass": ALPHA_PARTICLE_MASS,
            "charge": ALPHA_PARTICLE_CHARGE,
            "dt_save": 1e-8,
            "backgrounds": _zero_background(),
        }
        for mode in (None, "gc_vac"):
            with self.subTest(mode=mode):
                saw = wave_with_dropped_field()
                self.assertIsNone(getattr(saw.B0, "field_type", None))
                with self.assertRaises(ValueError) as cm:
                    trace_particles_boozer_perturbed_with_collisions(
                        saw,
                        self._stz,
                        np.array([1e6]),
                        np.array([1e6]),
                        mode=mode,
                        **kw,
                    )
                # Pin the diagnosis and the remedy, not the exact phrasing:
                # the message must name field_type as the observable symptom
                # and point at the object's lifetime as the usual cause.  It
                # deliberately does not assert the subclass was ever present
                # -- a bare C++ BoozerMagneticField never has field_type, and
                # is refused here for the same reason.
                msg = str(cm.exception)
                self.assertIn("field_type", msg)
                self.assertIn("outlives the call", msg)

    def test_stopping_criterion_fires_and_pins_the_hit_layout(self):
        """
        Stopping criteria on the perturbed collisional path.

        The hit row is the only place the post-kick state is exposed to the
        caller, and it was previously untested here.  It also pins the
        ``res_hits`` layout, which is a trap for anyone migrating from
        :func:`~firm3d.field.tracing.trace_particles_boozer_perturbed`:
        ``solve_sde`` emits ``[t, index, s, theta, zeta, v_par, v]`` while the
        collisionless perturbed tracer's seventh column is ``t``.  Both are
        seven wide, so reading the wrong one yields a speed where a time was
        expected with no shape error to catch it -- here 1.3e7 against 2e-7.
        """
        saw, _, vpar, mus = self._setup(1e-3)
        # s runs 0.30 -> 0.42 over _tmax on this fixture, so this is crossed
        # partway rather than at either end.
        s_stop = 0.35
        kw = self._kw()
        kw["dt_save"] = 2e-8
        res, hits = trace_particles_boozer_perturbed_with_collisions(
            saw,
            self._stz,
            vpar,
            mus,
            backgrounds=_zero_background(),
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(s_stop)],
            **kw,
        )
        traj, hit = np.asarray(res[0]), np.asarray(hits[0])

        self.assertEqual(
            hit.shape, (1, 7), f"expected one 7-column hit, got {hit.shape}"
        )
        self.assertEqual(hit[0, 1], -1.0, "hit should index the first criterion")
        self.assertGreaterEqual(hit[0, 2], s_stop, "hit recorded below the threshold")
        self.assertLess(
            traj[-1, 0], self._tmax, "trace ran to tmax despite the criterion firing"
        )
        self.assertEqual(
            traj.shape[1], 6, "res_tys must stay [t, s, theta, zeta, v_par, v]"
        )

        # Column 6 is a speed, not the time.  With no background mu is fixed,
        # so it is reconstructible from the hit's own position and v_par.
        field = self._field
        field.set_points(np.array([hit[0, 2:5]]))
        v_expected = np.sqrt(hit[0, 5] ** 2 + 2.0 * mus[0] * field.modB()[0, 0])
        self.assertAlmostEqual(
            hit[0, 6] / v_expected,
            1.0,
            places=9,
            msg=(
                f"hit column 6 is {hit[0, 6]:.6e}; expected the speed "
                f"{v_expected:.6e}, not the time {hit[0, 0]:.3e}"
            ),
        )

    def test_full_k_field_is_rejected(self):
        """There is no full-K perturbed right-hand side; say so, don't guess."""
        saw, _, vpar, mus = self._setup(1e-3, {"I0": 0.5, "K1": 0.3})
        self.assertEqual(saw.B0.field_type, "")
        with self.assertRaises(ValueError) as cm:
            trace_particles_boozer_perturbed_with_collisions(
                saw,
                self._stz,
                vpar,
                mus,
                backgrounds=_zero_background(),
                **self._kw(),
            )
        self.assertIn("full-K", str(cm.exception))


# ---------------------------------------------------------------------------
# Shipped C++ coefficients vs the Python transcription
# ---------------------------------------------------------------------------


class TestCppCoefficientsMatchMirror(unittest.TestCase):
    """
    Pin the Python helpers in this file to the C++ that actually ships.

    _analytical_K / _analytical_nu_D / _chandrasekhar_G are hand-written
    transcriptions of collisions.h, and TestCollisionCoefficients asserts
    against them.  Until compute_collision_coefficients was bound, nothing
    connected the two: a sign or factor error in the C++ was invisible to
    every one of those assertions, since both sides would have had to be
    wrong in the same way to agree.  This class is that connection, so the
    rest of the suite inherits its authority from the shipped code.

    It also covers the constants: the C++ carries its own COLL_HBAR and
    COLL_EPSILON0, and the mirror uses firm3d.util.constants.  The
    coefficients depend on both, so any divergence shows up here.
    """

    def _cases(self):
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        multi = [
            _hot_background(),
            ThermalBackground(
                n_profile=lambda s: 1e20,
                T_profile=lambda s: 10e3 * ONE_EV,
                mass=ELECTRON_MASS,
                charge=-ELEMENTARY_CHARGE,
            ),
        ]
        for label, bgs in [
            ("single ion", [_hot_background()]),
            ("cold ion", [_cold_background()]),
            ("ion + electron", multi),
        ]:
            for vf in (0.05, 0.2, 0.5, 1.0, 3.0):
                for s in (0.0, 0.3, 1.0):
                    yield label, bgs, vf * v0, s

    def test_matches_python_transcription(self):
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        for label, bgs, v, s in self._cases():
            with self.subTest(case=label, v=v, s=s):
                c = sopp.compute_collision_coefficients(
                    v, s, float(m), float(q), [b._to_cpp() for b in bgs]
                )
                np.testing.assert_allclose(
                    [c.K, c.nu_D],
                    [
                        _analytical_K(v, s, m, q, bgs),
                        _analytical_nu_D(v, s, m, q, bgs),
                    ],
                    rtol=1e-12,
                    err_msg=f"C++ and Python disagree for {label}",
                )

    def test_mirror_is_discriminating(self):
        """The comparison must be able to see a wrong coefficient at all."""
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        bgs = [_hot_background()]
        c = sopp.compute_collision_coefficients(
            v0,
            0.3,
            float(ALPHA_PARTICLE_MASS),
            float(ALPHA_PARTICLE_CHARGE),
            [b._to_cpp() for b in bgs],
        )
        # A 1e-6 relative perturbation must break the rtol=1e-12 assertion,
        # otherwise the test above proves nothing.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                c.K * (1 + 1e-6),
                _analytical_K(v0, 0.3, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, bgs),
                rtol=1e-12,
            )


# ---------------------------------------------------------------------------
# Collision-kick sub-cycling
# ---------------------------------------------------------------------------


class TestInterpolatedField(unittest.TestCase):
    """
    Collisional tracing on an InterpolatedBoozerField.

    Every other test in this file traces BoozerAnalytic, which evaluates the
    field from closed-form expressions.  Real use is an interpolant built from
    a boozmn file, and that is also what the GPU entry point takes, so the
    combination needs at least one test that reaches it -- the kick calls
    modB_ref() on whatever field it is handed.
    """

    _FILE = "examples/inputs/boozmn_aten_rescaled_low_res.nc"

    @classmethod
    def setUpClass(cls):
        # bri must stay referenced: InterpolatedBoozerField does not keep it
        # alive, and evaluating a field whose source has been collected fails
        # inside the first right-hand-side call.
        cls.bri = BoozerRadialInterpolant(cls._FILE, 3, enforce_vacuum=True)
        cls.field = InterpolatedBoozerField(
            cls.bri, 3, ns_interp=15, ntheta_interp=15, nzeta_interp=15
        )

    def test_kick_reaches_the_interpolated_field(self):
        """
        The collisional trace must differ from the collisionless one.

        Not a physics check -- the drag is checked against closed forms
        elsewhere.  This asserts that the kick actually ran when the field is
        an interpolant, which finiteness alone would not: a kick that silently
        did nothing here would leave a perfectly plausible trace behind.

        The separation is small on purpose.  An alpha's slowing-down time in
        this background is of order 0.1 s, so over 1e-7 s the speed moves by
        ~1e-6 relative -- far above the 1e-8 orbit tolerance, but the drag
        does not dominate, and a single realization can gain speed as easily
        as lose it.  Asserting a decrease here would be wrong: measured, this
        seed gains 6.6e-07.
        """
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        stz = np.array([[0.3, 0.0, 0.0], [0.3, 1.0, 0.5]])
        vpar0 = 0.5 * v0 * np.ones(2)
        shared = {
            "tmax": 1e-7,
            "mass": ALPHA_PARTICLE_MASS,
            "charge": ALPHA_PARTICLE_CHARGE,
            "tol": 1e-8,
            "dt_save": 1e-7,
        }
        res, _ = trace_particles_boozer_with_collisions(
            self.field,
            stz,
            vpar0,
            Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
            rng_seed=0,
            backgrounds=_cold_background(),
            **shared,
        )
        free, _ = trace_particles_boozer(
            self.field,
            stz,
            vpar0,
            Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
            **shared,
        )

        self.assertEqual(len(res), 2)
        for i in range(2):
            with self.subTest(particle=i):
                a = np.asarray(res[i])
                self.assertTrue(np.all(np.isfinite(a)), "non-finite output")
                self.assertAlmostEqual(a[-1, 0], 1e-7, places=12)
                self.assertGreater(a[-1, 5], 0.0)

                collisionless_vpar = np.asarray(free[i])[-1, 4]
                self.assertNotAlmostEqual(
                    a[-1, 4] / collisionless_vpar,
                    1.0,
                    places=8,
                    msg=(
                        "collisional and collisionless traces agree to the "
                        "orbit tolerance; the kick is not reaching an "
                        "interpolated field"
                    ),
                )


class TestProfileGridValidation(unittest.TestCase):
    """
    The profile grid must have at least two nodes.

    Linear interpolation needs two nodes and a non-zero spacing.  With one the
    spacing is 0/0 and the lookup indexes off the end of the sample array;
    measured before the check, that produced nu_D = K = D_par = nan from the
    shipped coefficient routine rather than any error.  The same lookup runs
    on the GPU against a raw device pointer, where the equivalent read is
    unchecked.
    """

    def _bg(self, n_grid_points):
        return ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 10e3 * ONE_EV,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
            n_grid_points=n_grid_points,
        )

    def test_degenerate_grid_is_refused(self):
        for npts in (1, 0, -3):
            with self.subTest(n_grid_points=npts):
                with self.assertRaises(ValueError) as cm:
                    self._bg(npts)
                self.assertIn("at least 2", str(cm.exception))

    def test_two_points_is_accepted_and_finite(self):
        """Two nodes is the smallest legal grid and must still work."""
        c = sopp.compute_collision_coefficients(
            1.0e7,
            0.5,
            float(ALPHA_PARTICLE_MASS),
            float(ALPHA_PARTICLE_CHARGE),
            [self._bg(2)._to_cpp()],
        )
        for name in ("nu_D", "K", "D_par"):
            with self.subTest(coefficient=name):
                self.assertTrue(np.isfinite(getattr(c, name)))
        self.assertGreater(c.nu_D, 0.0)


class TestCollisionSubstepping(unittest.TestCase):
    """
    The orbit stepper sizes its step from orbit dynamics alone, so the
    collision terms get no say in it.  Applying them as a single explicit
    Euler kick over that step is badly wrong once the collision rates are
    fast compared with it, which is the thermal regime.  The kick is
    therefore sub-cycled.
    """

    def _coef(self, v, bgs, s=0.3):
        return sopp.compute_collision_coefficients(
            v,
            s,
            float(PROTON_MASS),
            float(ELEMENTARY_CHARGE),
            [b._to_cpp() for b in bgs],
        )

    def test_slow_rates_need_no_subcycling(self):
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        c = self._coef(v0, [_hot_background()])
        self.assertEqual(sopp.collision_substeps(v0, c, 1e-9), 1)

    def test_fast_rates_subcycle(self):
        """A thermal proton over a long step must split the kick."""
        bgs = [
            ThermalBackground(
                n_profile=lambda s: 1e21,
                T_profile=lambda s: 1e3 * ONE_EV,
                mass=PROTON_MASS,
                charge=ELEMENTARY_CHARGE,
            )
        ]
        v_th = np.sqrt(2 * 1e3 * ONE_EV / PROTON_MASS)
        c = self._coef(v_th, bgs)
        n = sopp.collision_substeps(v_th, c, 1e-4)
        self.assertGreater(n, 1, f"expected sub-cycling, got nsub = {n}")

    def test_degenerate_inputs_are_defined(self):
        """
        Non-finite and huge rates must not reach the double->int cast.

        Converting an out-of-range or NaN double to int is undefined
        behaviour; on x86-64 it yields INT_MIN, which a `< 1` clamp would
        then turn into a single sub-step -- the runaway guard failing open
        in exactly the regime it exists for.
        """
        v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
        c = self._coef(v0, [_hot_background()])
        for h in (0.0, -1.0, float("nan"), float("inf"), 1e300):
            with self.subTest(h=h):
                n = sopp.collision_substeps(v0, c, h)
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 10000)
        for v in (0.0, -1.0, float("nan")):
            with self.subTest(v=v):
                self.assertEqual(sopp.collision_substeps(v, c, 1e-6), 1)

    def test_equilibrium_is_independent_of_orbit_tolerance(self):
        """
        Regression test for the bug sub-cycling fixes.

        With one kick per orbit step the equilibrium tracked the orbit
        tolerance instead of the physics: <E>/T_b came out at 7.8, 3.1, 3.0
        and 1.3 for tol = 1e-6, 1e-8, 1e-10, 1e-12, against an expected 1.5.
        Sub-cycling decouples them.  The loose tolerance is the discriminating
        one -- it is where h is longest relative to the collision time -- so
        this fails against the pre-fix code.
        """
        T_b, n_b, nP = 1e3 * ONE_EV, 1e21, 24
        tmax = 8 * 3.5e-5  # 8 collisional relaxation times
        bg = ThermalBackground(
            n_profile=lambda s: n_b,
            T_profile=lambda s: T_b,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        E0 = 4.5 * T_b
        v0 = np.sqrt(2 * E0 / PROTON_MASS)
        rng = np.random.default_rng(42)
        stz = np.column_stack(
            [
                np.full(nP, 0.3),
                rng.uniform(0, 2 * np.pi, nP),
                rng.uniform(0, 2 * np.pi, nP),
            ]
        )
        res, _ = trace_particles_boozer_with_collisions(
            BoozerAnalytic(0.25, 5.0, 0, 40.0, 2.0, 0.8),
            stz,
            0.7 * v0 * np.ones(nP),
            backgrounds=[bg],
            tmax=tmax,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
            Ekin=E0,
            tol=1e-6,
            dt_save=tmax,
            forget_exact_path=True,
            rng_seed=42,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        v_end = np.array([t[-1, 5] for t in res])
        E_mean = 0.5 * PROTON_MASS * np.mean(v_end**2) / T_b
        self.assertGreater(
            E_mean, 0.8, f"<E>/T_b = {E_mean:.2f} at tol=1e-6; expected ~1.5"
        )
        self.assertLess(
            E_mean,
            3.0,
            f"<E>/T_b = {E_mean:.2f} at tol=1e-6; expected ~1.5. Pre-sub-cycling "
            f"this read 7.8, i.e. the equilibrium tracked the orbit tolerance",
        )


# ---------------------------------------------------------------------------
# Collision coefficients (analytical)
# ---------------------------------------------------------------------------


class TestUnphysicalCoulombLog(unittest.TestCase):
    """
    ln Lambda <= 0 (Debye length below the minimum impact parameter,
    e.g. T -> 0 at finite density) makes the binary-collision model
    undefined; the tracer must raise rather than integrate garbage.
    ASCOT5 handles the equivalent situation by aborting markers through
    its input-evaluation errors.
    """

    @staticmethod
    def _bad_background():
        return ThermalBackground(
            n_profile=lambda s: 1e30,
            T_profile=lambda s: 1e-3 * ONE_EV,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )

    def test_rejected_before_tracing(self):
        """The up-front profile check rejects it without starting a trace.

        Catching this in Python matters under MPI: the C++ throw fires
        from inside the RHS on whichever rank owns the offending
        particle, and the ranks that did not fail then block forever in
        the allgather that collects results.
        """
        with self.assertRaises(ValueError):
            _coll_trace(_field(), self._bad_background(), tmax=1e-8)

    def test_cpp_still_refuses_when_validation_bypassed(self):
        """With validate_profiles=False the C++ layer is the backstop.

        The failure must surface as an exception from the tracer call
        rather than being swallowed by the per-particle error capture.
        """
        with self.assertRaises(RuntimeError):
            _coll_trace(
                _field(),
                self._bad_background(),
                tmax=1e-8,
                validate_profiles=False,
            )

    def test_healthy_profiles_are_not_rejected(self):
        """The v -> 0 bound must not reject an ordinary reactor profile."""
        traj = _coll_trace(_field(), _hot_background(), tmax=1e-8)
        self.assertEqual(traj.shape[1], 6)

    @staticmethod
    def _marginal_background():
        """0 < ln_Lambda < 2 as v -> 0: usable, but not quantitative."""
        return ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 0.025 * ONE_EV,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

    def test_marginal_warns_exactly_once(self):
        """A marginal ln_Lambda warns once per call, not once per evaluation.

        The C++ layer sees one (v, s) at a time from inside the ODE
        right-hand side, so warning there fired on every evaluation --
        roughly seven times per accepted step, per particle.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _coll_trace(_field(), self._marginal_background(), tmax=1e-7)
        hits = [w for w in caught if "ln_Lambda" in str(w.message)]
        self.assertEqual(len(hits), 1, f"expected 1 warning, got {len(hits)}")
        self.assertTrue(issubclass(hits[0].category, RuntimeWarning))

    def test_healthy_profiles_do_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _coll_trace(_field(), _hot_background(), tmax=1e-8)
        hits = [w for w in caught if "ln_Lambda" in str(w.message)]
        self.assertEqual(hits, [], f"unexpected warning: {hits}")


class TestCollisionCoefficients(unittest.TestCase):
    """
    Unit tests for the analytical collision coefficient formulae that mirror
    compute_collision_coefficients() in collisions.h.

    These tests are deterministic and instantaneous — no simulation needed.
    They verify that the Chandrasekhar G function, the drag K, and the
    pitch-angle scattering rate ν_D have the correct signs, limits, and
    scalings.  Any discrepancy here would be reflected identically in the C++
    code since both implementations follow the same formulae.
    """

    _v0 = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)
    _m, _q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
    _s = 0.3

    # ------------------------------------------------------------------
    # Chandrasekhar G function
    # ------------------------------------------------------------------
    def test_G_positive(self):
        """G(x) > 0 for all x > 0."""
        for x in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            self.assertGreater(_chandrasekhar_G(x), 0, f"G({x}) ≤ 0")

    def test_G_zero_at_zero(self):
        """G(0) = 0 (G → 0 as x → 0)."""
        self.assertAlmostEqual(_chandrasekhar_G(0.0), 0.0, places=12)

    def test_G_small_x_limit(self):
        """For x ≪ 1: G(x) ≈ 2x / (3√π)."""
        for x in [0.001, 0.01, 0.05]:
            G_approx = 2.0 * x / (3.0 * np.sqrt(np.pi))
            self.assertAlmostEqual(
                _chandrasekhar_G(x),
                G_approx,
                delta=0.01 * G_approx,
                msg=f"G({x}) deviates from small-x limit 2x/(3√π)",
            )

    def test_G_large_x_limit(self):
        """For x ≫ 1: G(x) ≈ 1 / (2x²)."""
        for x in [5.0, 10.0, 20.0]:
            G_approx = 1.0 / (2.0 * x**2)
            self.assertAlmostEqual(
                _chandrasekhar_G(x),
                G_approx,
                delta=0.05 * G_approx,
                msg=f"G({x}) deviates from large-x limit 1/(2x²)",
            )

    def test_G_deriv_positive_small_x(self):
        """G'(x) > 0 for small x (G is increasing near 0)."""
        for x in [0.1, 0.3, 0.5]:
            self.assertGreater(_chandrasekhar_G_deriv(x), 0, f"G'({x}) ≤ 0")

    def test_G_deriv_sign_change(self):
        """G has a maximum: G'(x) changes sign from + to − around x ≈ 0.92."""
        self.assertGreater(_chandrasekhar_G_deriv(0.5), 0)
        self.assertLess(_chandrasekhar_G_deriv(2.0), 0)

    # ------------------------------------------------------------------
    # Drag coefficient K
    # ------------------------------------------------------------------
    def test_K_zero_when_n_zero(self):
        """K = 0 when n = 0."""
        K = _analytical_K(self._v0, self._s, self._m, self._q, [_zero_background()])
        self.assertEqual(K, 0.0)

    def test_K_negative(self):
        """K < 0 for any finite background (drag always decelerates)."""
        for bg in [_hot_background(), _cold_background(), _hot_background(T_keV=100.0)]:
            K = _analytical_K(self._v0, self._s, self._m, self._q, [bg])
            self.assertLess(K, 0, "K must be negative (decelerating drag)")

    def test_K_linear_in_density(self):
        """K nearly proportional to n; small deviation via ln Λ ∝ ln(λ_D)."""
        K1 = _analytical_K(
            self._v0, self._s, self._m, self._q, [_hot_background(n=1e20)]
        )
        K2 = _analytical_K(
            self._v0, self._s, self._m, self._q, [_hot_background(n=2e20)]
        )
        self.assertAlmostEqual(K2 / K1, 2.0, delta=0.05)

    def test_K_quartic_in_EP_charge(self):
        """K ∝ q_a²; <15% deviation because ln Λ also depends on |q_a q_b|."""
        bg = _hot_background()
        K_alpha = _analytical_K(
            self._v0, self._s, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, [bg]
        )
        # Same mass but half the charge
        K_half = _analytical_K(
            self._v0, self._s, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE / 2.0, [bg]
        )
        self.assertAlmostEqual(
            K_alpha / K_half,
            4.0,
            delta=0.5,
            msg="K must scale approximately as q_a² (factor ~4 for half charge)",
        )

    def test_K_additive_over_species(self):
        """
        K is approximately additive when two identical backgrounds are combined:
        the Debye length from two identical species is 1/√2 of a single species,
        changing ln Λ by only −½ ln 2 ≈ −0.35, so K[bg+bg] ≈ 2 K[bg] to ~5%.
        """
        bg = _hot_background(n=1e20)
        K_one = _analytical_K(self._v0, self._s, self._m, self._q, [bg])
        K_both = _analytical_K(self._v0, self._s, self._m, self._q, [bg, bg])
        self.assertAlmostEqual(K_both / K_one, 2.0, delta=0.1)

    def test_K_electron_larger_than_ion(self):
        """
        For fusion alphas in a 10 keV plasma, electron drag dominates over
        deuteron drag.

        Physical basis: the alpha birth energy (3.5 MeV) is far above the
        critical energy E_c ≈ 0.4 MeV where electron and ion drag are equal,
        so electron drag dominates by roughly (E/E_c)^{3/2} ≈ 20-30.
        """
        electron_bg = ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 10e3 * ONE_EV,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )
        K_e = _analytical_K(self._v0, self._s, self._m, self._q, [electron_bg])
        K_D = _analytical_K(self._v0, self._s, self._m, self._q, [_hot_background()])
        ratio = abs(K_e) / abs(K_D)
        self.assertGreater(ratio, 10, f"|K_e|/|K_D| = {ratio:.1f}, expected ≈ 20")
        self.assertLess(ratio, 50, f"|K_e|/|K_D| = {ratio:.1f}, expected ≈ 20")

    # ------------------------------------------------------------------
    # Pitch-angle scattering rate ν_D
    # ------------------------------------------------------------------
    def test_nu_D_positive(self):
        """ν_D > 0 for any finite, positive background."""
        for bg in [_hot_background(), _cold_background()]:
            nu_D = _analytical_nu_D(self._v0, self._s, self._m, self._q, [bg])
            self.assertGreater(nu_D, 0)

    def test_nu_D_zero_when_n_zero(self):
        """ν_D = 0 when n = 0."""
        nu_D = _analytical_nu_D(
            self._v0, self._s, self._m, self._q, [_zero_background()]
        )
        self.assertEqual(nu_D, 0.0)

    def test_nu_D_linear_in_density(self):
        """ν_D nearly proportional to n; small deviation via ln Λ ∝ ln(λ_D)."""
        nu_1 = _analytical_nu_D(
            self._v0, self._s, self._m, self._q, [_hot_background(n=1e20)]
        )
        nu_2 = _analytical_nu_D(
            self._v0, self._s, self._m, self._q, [_hot_background(n=2e20)]
        )
        self.assertAlmostEqual(nu_2 / nu_1, 2.0, delta=0.05)

    def test_nu_D_large_x_limit(self):
        """
        For x ≫ 1 (EP much faster than background thermal speed):
        ν_D ≈ Γ × (erf(x) − G(x)) / v³ → Γ / v³  (since erf(x)→1, G(x)→0).
        """
        bg = _hot_background()
        v0, s = self._v0, self._s
        m_b, q_b = bg.mass, bg.charge
        T_b, n_b = bg.T_profile(s), bg.n_profile(s)
        v_th = np.sqrt(2 * T_b / m_b)
        x = v0 / v_th  # ≈ 13 for fusion alphas in 10 keV D plasma

        lambda_D = _debye_length(s, [bg])
        lnL = _coulomb_log(v0, self._m, self._q, m_b, q_b, T_b, lambda_D)
        Gamma = (
            n_b
            * self._q**2
            * q_b**2
            * lnL
            / (4 * np.pi * VACUUM_PERMITTIVITY**2 * self._m**2)
        )
        nu_D_approx = Gamma / v0**3  # large-x limit
        nu_D_exact = _analytical_nu_D(v0, s, self._m, self._q, [bg])

        self.assertAlmostEqual(
            nu_D_exact / nu_D_approx,
            1.0,
            delta=0.02,
            msg=f"ν_D deviates from Γ/v³ limit at x={x:.1f}",
        )


# ---------------------------------------------------------------------------
# End-to-end thermalization through the production tracer (fast, local)
# ---------------------------------------------------------------------------


class TestMaxwellianEquilibration(unittest.TestCase):
    """
    End-to-end thermalization through trace_particles_boozer_with_collisions
    with a BoozerAnalytic field -- the full production path (DP stepper +
    Milstein noise + orbit dynamics), runnable locally in a few seconds.

    Skew: all collision rates scale linearly with background density, so
    n_b = 1e21 m^-3 makes the collisional relaxation time of a thermal
    proton in a 1 keV proton background ~35 us, while its orbital transit
    time (~14 us at v_th) remains shorter -- the adaptive DP stepper still
    resolves the orbits and the Milstein noise stays accurate (nu h << 1).

    The reflecting thermal-cutoff speed boundary (ASCOT5's MCCC_CUTOFF)
    keeps particles out of the v -> 0 region where the collision
    coefficients diverge.

    The field differs from _field() used elsewhere: collisional v_par
    kicks displace the canonical angular momentum, giving radial steps
    ~ R0/(iota psi0) in normalized flux, and in the small-psi0 _field()
    configuration thermal protons random-walk out through s = 1 within a
    few relaxation times.  A fatter, higher-iota configuration (psi0 = 2,
    iota0 = 0.8, aspect ratio 4) keeps all particles confined over the
    test duration.  MaxToroidalFluxStoppingCriterion(1.0) guards the
    s <= 1 domain: without it, escaped particles are silently integrated
    in the unphysical analytic continuation of the near-axis field until
    the adaptive stepper grinds to a halt chasing the runaway trajectory.

    This is the direct end-to-end regression test for the Einstein-relation
    drag Q = -(m_a v / T_b) D_par: with a wrong drag the stationary energy
    is wrong by O(1) (the pre-fix drag equilibrated to <E> = 3.7 T_b).
    """

    _T_B = 1e3 * ONE_EV
    _N_B = 1e21  # m^-3
    _TAU = 3.5e-5  # collisional relaxation time at (_N_B, _T_B)
    _N_PART = 40

    @staticmethod
    def _confining_field():
        """Near-axis field with slow collisional radial transport."""
        return BoozerAnalytic(0.25, 5.0, 0, 40.0, 2.0, 0.8)

    def _equilibrate(self, E0_over_T, seed=42):
        """Trace N_PART protons from a monoenergetic start; return final v."""
        bg = ThermalBackground(
            n_profile=lambda s: self._N_B,
            T_profile=lambda s: self._T_B,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        E0 = E0_over_T * self._T_B
        v0 = np.sqrt(2 * E0 / PROTON_MASS)
        rng = np.random.default_rng(seed)
        stz = np.column_stack(
            [
                np.full(self._N_PART, 0.3),
                rng.uniform(0, 2 * np.pi, self._N_PART),
                rng.uniform(0, 2 * np.pi, self._N_PART),
            ]
        )
        vpar = 0.7 * v0 * np.ones(self._N_PART)
        tmax = 8 * self._TAU
        res_tys, _ = trace_particles_boozer_with_collisions(
            self._confining_field(),
            stz,
            vpar,
            backgrounds=[bg],
            tmax=tmax,
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
            Ekin=E0,
            tol=1e-8,
            dt_save=tmax,
            forget_exact_path=True,
            rng_seed=seed,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        v_end = np.array([ty[-1, 5] for ty in res_tys])
        vpar_end = np.array([ty[-1, 4] for ty in res_tys])
        t_end = np.array([ty[-1, 0] for ty in res_tys])

        # Every particle must stay confined to tmax with a finite, nonzero
        # speed: catches losses, frozen v = 0 particles, and NaN states.
        self.assertTrue(np.all(t_end >= 0.999 * tmax), "particles lost early")
        self.assertTrue(np.all(np.isfinite(v_end)), "non-finite final v")
        self.assertTrue(np.all(np.isfinite(vpar_end)), "non-finite final v_par")
        self.assertGreater(v_end.min(), 0.0, "dead particle at v = 0")
        return v_end, vpar_end

    def _check_maxwellian(self, v_end, lo, hi):
        E_mean = 0.5 * PROTON_MASS * np.mean(v_end**2) / self._T_B
        self.assertGreater(
            E_mean, lo, f"<E>/T_b = {E_mean:.2f}, expected ~1.5 at equilibrium"
        )
        self.assertLess(
            E_mean, hi, f"<E>/T_b = {E_mean:.2f}, expected ~1.5 at equilibrium"
        )
        p = scipy.stats.kstest(
            v_end,
            scipy.stats.maxwell(scale=np.sqrt(self._T_B / PROTON_MASS)).cdf,
        ).pvalue
        self.assertGreater(p, 0.005, f"KS vs Maxwell speed law: p = {p:.2e}")

    def test_equilibration_from_hot_start(self):
        """Monoenergetic protons at E0 = 4.5 T_b cool to the Maxwellian."""
        v_end, _ = self._equilibrate(E0_over_T=4.5)
        self._check_maxwellian(v_end, lo=1.0, hi=2.3)

    def test_equilibration_from_cold_start(self):
        """Monoenergetic protons at E0 = 0.45 T_b heat to the Maxwellian."""
        v_end, _ = self._equilibrate(E0_over_T=0.45)
        self._check_maxwellian(v_end, lo=0.9, hi=2.1)

    def test_pitch_isotropization(self):
        """Initial xi = 0.7 must isotropize: <xi> ~ 0, <xi^2> ~ 1/3."""
        v_end, vpar_end = self._equilibrate(E0_over_T=1.5)
        xi = vpar_end / v_end
        self.assertLess(abs(np.mean(xi)), 0.3, f"<xi> = {np.mean(xi):.3f}")
        self.assertGreater(np.mean(xi**2), 0.18, f"<xi^2> = {np.mean(xi**2):.3f}")
        self.assertLess(np.mean(xi**2), 0.50, f"<xi^2> = {np.mean(xi**2):.3f}")


# ---------------------------------------------------------------------------
# Qualitative collision physics (simulation)
# ---------------------------------------------------------------------------


class TestCollisionPhysics(unittest.TestCase):
    """
    Quantitative drag checks against the analytic coefficients.

    The three drag tests guard the domain with
    MaxToroidalFluxStoppingCriterion(1.0), for the reason spelled out in
    TestMaxwellianEquilibration: _field() does not confine 3.52 MeV alphas,
    and past s = 1 BoozerAnalytic is an unphysical analytic continuation.
    (test_speed_non_negative and test_rng_seed_reproducibility deliberately
    do not, since they assert format invariants that hold regardless of
    where the particle is; note they do run out to s of order 100.)

    These three tests originally ran unguarded.  Measured here, every
    particle crosses s = 1 at t = 0.107 * tmax, so ~89% of each trajectory
    was integrated outside the plasma, where the collision coefficients
    clamp s into [0, 1] and so keep returning finite numbers.  The runaway
    was found because the speed is now reconstructed as
    v^2 = v_par^2 + 2 mu |B| rather than integrated as a state variable, so
    a degenerate field evaluation reaches the output instead of being
    absorbed.

    With the guard the drag is measured over the window the field can
    actually hold, and the suite drops from ~105 s for a single test to
    ~30 s for all five.
    """

    def test_energy_decreases_cold_plasma(self):
        """
        Mean speed must decrease under electron drag (reliable SNR with N=20 particles).

        Cold D background gives x_D ≈ 133, G ≈ 1/(2x²) ≈ 2.8×10⁻⁵: negligible
        mean drift vs. noise for a single particle.  Electron background at
        T=10 keV gives x_e ≈ 0.22, G ≈ 0.046 (near maximum), K_e ≈ −3.5×10⁸ m/s²:
        SNR ≈ 3 with N=20 in tmax=5 μs.

        Intended to run on Perlmutter or equivalent HPC resources.
        """
        electron_bg = ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 10e3 * ONE_EV,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        nP = 20
        stz = np.tile([0.3, 0.0, 0.0], (nP, 1))
        vpar = np.full(nP, 0.5 * v0)
        res_tys, _ = trace_particles_boozer_with_collisions(
            _field(),
            stz,
            vpar,
            backgrounds=electron_bg,
            tmax=5e-6,
            mass=m,
            charge=q,
            Ekin=Ekin,
            tol=1e-9,
            dt_save=5e-6,
            forget_exact_path=True,
            rng_seed=0,
            comm=_COMM,
            DP_hmin=1e-10,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        v_final = np.array([t[-1, 5] for t in res_tys])
        self.assertLess(
            np.mean(v_final), v0, "Mean speed must decrease under electron drag"
        )

    def test_speed_non_negative(self):
        """v ≥ 0 must hold under extreme drag."""
        traj = _coll_trace(_field(), _cold_background(n=1e22, T_keV=0.1), tmax=2e-6)
        self.assertTrue(np.all(traj[:, 5] >= 0))

    def test_two_backgrounds_more_drag_than_one(self):
        """
        Adding a second identical electron background doubles drag → lower mean final v.

        Cold D background is unsuitable here: for 3.52 MeV alphas, x_D ≈ 133,
        G(x_D) ≈ 3×10⁻⁵, and drag is negligible compared to diffusion noise for
        a single particle.  Electron background at 10 keV gives G(x_e) ≈ 0.046
        (near maximum) and detectable drag for N = 20 particles.
        """
        electron_bg = ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 10e3 * ONE_EV,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        nP = 20
        tmax = 5e-6
        stz = np.tile([0.3, 0.0, 0.0], (nP, 1))
        vpar = np.full(nP, 0.5 * v0)

        res_one, _ = trace_particles_boozer_with_collisions(
            _field(),
            stz,
            vpar,
            backgrounds=[electron_bg],
            tmax=tmax,
            mass=m,
            charge=q,
            Ekin=Ekin,
            tol=1e-9,
            dt_save=tmax,
            forget_exact_path=True,
            rng_seed=0,
            comm=_COMM,
            DP_hmin=1e-10,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        res_two, _ = trace_particles_boozer_with_collisions(
            _field(),
            stz,
            vpar,
            backgrounds=[electron_bg, electron_bg],
            tmax=tmax,
            mass=m,
            charge=q,
            Ekin=Ekin,
            tol=1e-9,
            dt_save=tmax,
            forget_exact_path=True,
            rng_seed=0,
            comm=_COMM,
            DP_hmin=1e-10,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        v_one = np.mean([t[-1, 5] for t in res_one])
        v_two = np.mean([t[-1, 5] for t in res_two])
        self.assertLess(
            v_two, v_one, "Two electron backgrounds must give more drag than one"
        )

    def test_rng_seed_reproducibility(self):
        """Same seed → bitwise-identical trajectories; different seed → different."""
        field, bg = _field(), _hot_background()
        t1 = _coll_trace(field, bg, tmax=1e-6, rng_seed=42)
        t2 = _coll_trace(field, bg, tmax=1e-6, rng_seed=42)
        t3 = _coll_trace(field, bg, tmax=1e-6, rng_seed=99)
        np.testing.assert_array_equal(t1, t2)
        self.assertFalse(np.array_equal(t1, t3))

    def test_electron_drag_mean_velocity(self):
        """
        Electron background gives detectable mean energy loss with N=50 alphas
        in tmax = 5 μs.

        For 3.52 MeV alphas (v₀ ≈ 1.3×10⁷ m/s) against a 10 keV electron
        plasma at n = 10²⁰ m⁻³:
          x_e = v₀/v_th,e ≈ 0.22  →  G(x_e) ≈ 0.046  (near maximum)
          K_e ≈ −3.5×10⁸ m/s²     (drag ~70 000× larger than D-ion drag)

        SNR analysis: drift = K × tmax ≈ −1770 m/s;
          σ(noise per particle) ≈ √(2 D_par × tmax) ≈ 2500 m/s;
          σ(mean) = σ/√N ≈ 354 m/s  →  SNR ≈ 5 with N = 50.

        The test asserts (1) mean Δv < 0 (correct sign) and (2) the ratio
        mean_dv / (K_theory × tmax) ∈ [0.3, 3.0], which fails with probability
        < 0.003 for any statistically correct implementation.
        """
        electron_bg = ThermalBackground(
            n_profile=lambda s: 1e20,
            T_profile=lambda s: 10e3 * ONE_EV,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        s0 = 0.3

        K_theory = _analytical_K(v0, s0, m, q, [electron_bg])
        self.assertLess(K_theory, 0, "Electron drag K must be negative")

        tmax = 5e-6
        nP = 50
        stz = np.tile([s0, 0.0, 0.0], (nP, 1))
        vpar = np.full(nP, 0.5 * v0)

        res_tys, _ = trace_particles_boozer_with_collisions(
            _field(),
            stz,
            vpar,
            backgrounds=electron_bg,
            tmax=tmax,
            mass=m,
            charge=q,
            Ekin=Ekin,
            tol=1e-9,
            dt_save=tmax,
            forget_exact_path=True,
            rng_seed=0,
            comm=_COMM,
            DP_hmin=1e-10,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )

        v_final = np.array([t[-1, 5] for t in res_tys])
        mean_dv = np.mean(v_final) - v0

        # Normalise by the time actually integrated, not by tmax: the domain
        # guard stops these particles at s = 1 well before tmax, and dividing
        # by tmax rescales the result by ~10x, which happens to land noise
        # inside a [0.3, 3.0] band and made this assertion meaningless.
        t_elapsed = np.mean([t[-1, 0] for t in res_tys])
        self.assertGreater(t_elapsed, 0.0)

        self.assertTrue(np.all(np.isfinite(v_final)), "non-finite final speed")
        self.assertLess(mean_dv, 0, "Mean speed must decrease under electron drag")

        # Wide band by construction: per the module docstring the SNR here is
        # below 1, so this checks sign and order of magnitude only.  The
        # coefficients themselves are pinned by TestCollisionCoefficients.
        ratio = mean_dv / (K_theory * t_elapsed)
        self.assertGreater(
            ratio, 0.2, f"Drag too weak: mean_dv/(K_theory*t) = {ratio:.2f} < 0.2"
        )
        self.assertLess(
            ratio, 30.0, f"Drag too strong: mean_dv/(K_theory*t) = {ratio:.2f} > 30"
        )


# ---------------------------------------------------------------------------
# Pitch isotropization (ASCOT5 thermal-equilibration analog)
# ---------------------------------------------------------------------------


class TestPitchIsotropization(unittest.TestCase):
    """
    Pitch-angle scattering tests, from qualitative to quantitative.

    The exact Ito result for the pitch SDE
      dξ = −ν_D ξ dt + √(ν_D(1−ξ²)) dW
    is d⟨ξ⟩/dt = −ν_D ⟨ξ⟩  (E[f(ξ(t)) dW] = 0 in Ito calculus),
    giving ⟨ξ(t)⟩ = ξ₀ exp(−ν_D t) exactly when ν_D is constant.
    This is the ASCOT5 pitch-angle benchmark: start a beam at ξ₀ ≠ 0
    and verify the exponential decay rate against the analytical ν_D.

    Two backgrounds are used across these tests:

      n = 10²³ m⁻³, T = 1 keV D  (ν_D ≈ 47 s⁻¹):
        Used for the qualitative test (ξ₀ = 0, tmax = 1e-4 s → σ ≈ 0.07).

      n = 10²⁵ m⁻³, T = 1 keV D  (ν_D ≈ 4700 s⁻¹):
        Used for the quantitative decay test (tmax = 0.5/ν_D ≈ 107 μs,
        ν_D × tmax = 0.5, N = 50 → SNR ≈ 5 on ⟨ξ⟩ change).
    """

    @staticmethod
    def _uniform_field():
        """Near-uniform |B| (etabar ~ 0): no mirror force and no trapped
        cone, so the uniform-plasma pitch-decay law applies exactly.
        In a toroidal field the trapped-passing boundary at s = 0.3 sits
        at xi_t ~ 0.65, just below the xi0 = 0.8 beam: scattered
        particles fall into the trapped cone where their orbit-sampled
        xi averages to zero, and <xi> decays ~2x faster than the
        uniform-plasma law (measured exponent 1.05 vs m_D/m_alpha =
        0.50) -- real physics, but not what this test verifies.
        """
        return BoozerAnalytic(1e-8, 5.3, 0, 32.86, 10.6, 1.0, Bbar=5.3)

    @staticmethod
    def _confining_field():
        """ITER-scale near-axis field (R0 = 6.2 m, B0 = 5.3 T, psi0 = 10.6).

        3.5 MeV alphas have radial orbit widths delta-s ~ 0.15 here, so
        markers started at s = 0.3 stay inside the plasma.  In the
        small-psi0 _field() used elsewhere, alpha orbit widths are order
        unity: markers escape through s = 1 within a few transits and
        their states corrupt in the analytic continuation of the field.
        """
        return BoozerAnalytic(0.25, 5.3, 0, 32.86, 10.6, 1.0, Bbar=5.3)

    @staticmethod
    def _assert_confined(res_tys, tmax):
        t_end = np.array([t[-1, 0] for t in res_tys])
        v_end = np.array([t[-1, 5] for t in res_tys])
        assert np.all(t_end >= 0.999 * tmax), "particles lost before tmax"
        assert np.all(np.isfinite(v_end)), "non-finite final state"

    def _background_low_n(self):
        return ThermalBackground(
            n_profile=lambda s: 1e23,
            T_profile=lambda s: 1e3 * ONE_EV,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

    def _background_high_n(self):
        return ThermalBackground(
            n_profile=lambda s: 1e25,
            T_profile=lambda s: 1e3 * ONE_EV,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

    def test_pitch_scattering_broadens_distribution(self):
        """
        Starting at ξ₀ = 0 with 30 particles:
          * std(ξ_final) > 0.05 (distribution has broadened)
          * Both positive and negative ξ values appear (symmetric scatter)
        """
        bg = self._background_low_n()
        tmax = 1e-4

        nP = 30
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        stz = np.tile([0.3, 0.0, 0.0], (nP, 1))
        vpar = np.zeros(nP)  # ξ₀ = 0: perpendicular particles

        res_tys, _ = trace_particles_boozer_with_collisions(
            self._confining_field(),
            stz,
            vpar,
            backgrounds=bg,
            tmax=tmax,
            mass=m,
            charge=q,
            Ekin=Ekin,
            tol=1e-8,
            dt_save=tmax,
            forget_exact_path=True,
            rng_seed=0,
            comm=_COMM,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        self._assert_confined(res_tys, tmax)

        xi_f = np.array([t[-1, 4] / t[-1, 5] for t in res_tys])
        std_xi = np.std(xi_f)

        self.assertGreater(
            std_xi,
            0.05,
            f"ξ did not broaden from ξ₀=0 (std={std_xi:.4f}); "
            "pitch-angle scattering may be broken",
        )
        self.assertTrue(
            np.any(xi_f > 0.01) and np.any(xi_f < -0.01),
            "All ξ_final have the same sign — scattering is not symmetric",
        )

    def test_pitch_angle_exponential_decay(self):
        """
        ⟨ξ⟩ ∝ v^{m_b/m_a}: pitch decay of fast alphas on deuterium.

        For a fast test particle (x = v/v_th >> 1) on a single ion
        background, pitch scattering CANNOT be isolated from drag: both
        rates scale as 1/v^3 with the fixed ratio
        nu_drag/nu_D = m_a/m_b (= 2 for alphas on D).  Over nu_D t = 0.5
        the alphas lose most of their speed, nu_D(v) explodes, and <xi>
        isotropizes completely -- the original form of this test, which
        assumed constant v, failed with <xi> = 0.

        Eliminating time between d<xi>/dt = -nu_D <xi> and
        dv/dt = -(m_a/m_b) Gamma / v^2 gives the classical relation
            <xi(t)> / <xi(t1)> = ( v(t) / v(t1) )^{m_b/m_a},
        which is checked here using the simulation's own mean speed.
        Referencing the first SAVED snapshot (t1 = tmax/8) rather than
        the initial condition cancels the geometric offset from sampling
        xi along orbits (xi varies with B along a field line).

        tmax = 0.15/nu_D(v0) keeps x >> 1 throughout (v_end ~ 0.6 v0).
        With N = 128 the accumulated pitch scatter gives a ~2-3 % standard
        error on the ratio; the +/-15 % bracket is a >5 sigma window that
        still catches an O(1) error in nu_D or the drag.
        """
        bg = self._background_high_n()
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        xi0 = 0.8
        s0 = 0.3

        nu_D_theory = _analytical_nu_D(v0, s0, m, q, [bg])
        tmax = 0.15 / nu_D_theory

        nP = 128
        stz = np.tile([s0, 0.0, 0.0], (nP, 1))
        vpar = np.full(nP, xi0 * v0)

        res_tys, _ = trace_particles_boozer_with_collisions(
            self._uniform_field(),
            stz,
            vpar,
            backgrounds=bg,
            tmax=tmax,
            mass=m,
            charge=q,
            Ekin=Ekin,
            tol=1e-8,
            dt_save=tmax / 8,
            forget_exact_path=False,
            rng_seed=0,
            comm=_COMM,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        self._assert_confined(res_tys, tmax)

        # First saved snapshot (t = tmax/8) as reference, last as end
        xi_1 = np.array([t[1, 4] / t[1, 5] for t in res_tys])
        v_1 = np.array([t[1, 5] for t in res_tys])
        xi_2 = np.array([t[-1, 4] / t[-1, 5] for t in res_tys])
        v_2 = np.array([t[-1, 5] for t in res_tys])

        measured = np.mean(xi_2) / np.mean(xi_1)
        predicted = (np.mean(v_2) / np.mean(v_1)) ** (
            (2 * PROTON_MASS) / ALPHA_PARTICLE_MASS
        )

        # The speed must have decayed appreciably (drag active) while
        # the alphas remain fast (asymptotic relation applicable)
        self.assertLess(np.mean(v_2) / np.mean(v_1), 0.95, "no drag signal")
        self.assertGreater(np.mean(v_2) / np.mean(v_1), 0.4, "over-slowed")

        self.assertGreater(
            measured,
            0.85 * predicted,
            f"pitch decayed too fast: <xi_2>/<xi_1> = {measured:.3f}, "
            f"(v_2/v_1)^(m_D/m_alpha) = {predicted:.3f}",
        )
        self.assertLess(
            measured,
            1.15 * predicted,
            f"pitch decayed too slow: <xi_2>/<xi_1> = {measured:.3f}, "
            f"(v_2/v_1)^(m_D/m_alpha) = {predicted:.3f}",
        )

    def test_nu_D_positive(self):
        nu_D = _analytical_nu_D(
            np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS),
            0.3,
            ALPHA_PARTICLE_MASS,
            ALPHA_PARTICLE_CHARGE,
            [self._background_low_n()],
        )
        self.assertGreater(nu_D, 0.0)

    def test_nu_D_zero_for_n_zero(self):
        nu_D = _analytical_nu_D(
            np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS),
            0.3,
            ALPHA_PARTICLE_MASS,
            ALPHA_PARTICLE_CHARGE,
            [_zero_background()],
        )
        self.assertEqual(nu_D, 0.0)


if __name__ == "__main__":
    unittest.main()
