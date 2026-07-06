"""
Tests for Monte Carlo Coulomb collision tracing.

All tests use BoozerAnalytic (no file I/O) and short integration times.

Quantitative drag-rate test — why electrons, not ions
------------------------------------------------------
For 3.52 MeV fusion alphas (v₀ ≈ 1.3×10⁷ m/s) the signal-to-noise ratio for
measuring the mean drag in a finite-particle simulation depends on the background
species through the Chandrasekhar G function:

  D-ion background  (10 keV D): x_D = v₀/v_th,D ≈ 13  →  G ≈ 1/(2x²) ≈ 0.003
    K_D ≈ −5×10³ m/s², noise dominates, SNR ≪ 1 even with N = 1000 particles.

  Electron background (10 keV e): x_e = v₀/v_th,e ≈ 0.22  →  G ≈ 0.046 (near max)
    K_e ≈ −3.5×10⁸ m/s² (enhanced by m_α/m_e ≈ 7344 in the friction term),
    SNR ≈ 5 with N = 50 particles × tmax = 5 μs.

TestCollisionPhysics.test_electron_drag_mean_velocity runs this quantitative test.
Ion-background drag is verified analytically in TestCollisionCoefficients instead.

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

import numpy as np
import pytest
import scipy.special
import scipy.stats

from firm3d.field.boozermagneticfield import BoozerAnalytic
from firm3d.field.collisions import (
    ThermalBackground,
    trace_particles_boozer_with_collisions,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELEMENTARY_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
    ONE_EV,
    PROTON_MASS,
    VACUUM_PERMITTIVITY,
)

ELECTRON_MASS = 9.1093837015e-31  # kg

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
    def test_speed_exactly_constant(self):
        """
        v(t) must equal v(0) to machine precision when n = 0.

        Physical basis: K = 0 and D_par = 0 exactly, so the v-component of
        the ODE RHS is identically zero.  Dormand-Prince cannot modify a state
        component whose derivative is always zero.
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
        np.testing.assert_allclose(
            traj[:, 5],
            v0,
            rtol=0,
            atol=1e-12 * v0,
            err_msg="Speed must be bitwise constant when n = 0 (K = 0 everywhere)",
        )

    # ------------------------------------------------------------------
    # 2. Magnetic moment conserved
    # ------------------------------------------------------------------
    def test_magnetic_moment_conserved(self):
        """
        μ = (v² − v∥²) / (2B) is the adiabatic invariant in vacuum GC theory.
        Verify it drifts less than 1e-5 relative over 5×10⁻⁷ s.
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
        pts = np.column_stack([traj[:, 1], traj[:, 2], traj[:, 3]])
        field.set_points(pts)
        B = np.array(field.modB()).flatten()

        v, vpar = traj[:, 5], traj[:, 4]
        mu = (v**2 - vpar**2) / (2.0 * B)
        rel_drift = np.abs(mu - mu[0]) / mu[0]

        self.assertLess(
            np.max(rel_drift),
            1e-5,
            f"μ drifted by {np.max(rel_drift):.2e}; expected < 1e-5",
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
# Collision coefficients (analytical)
# ---------------------------------------------------------------------------


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
    @pytest.mark.slow
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
            DP_hmin=1e-10,
        )
        v_final = np.array([t[-1, 5] for t in res_tys])
        self.assertLess(
            np.mean(v_final), v0, "Mean speed must decrease under electron drag"
        )

    def test_speed_non_negative(self):
        """v ≥ 0 must hold under extreme drag."""
        traj = _coll_trace(_field(), _cold_background(n=1e22, T_keV=0.1), tmax=2e-6)
        self.assertTrue(np.all(traj[:, 5] >= 0))

    @pytest.mark.slow
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
            DP_hmin=1e-10,
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
            DP_hmin=1e-10,
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

    @pytest.mark.slow
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
            DP_hmin=1e-10,
        )

        v_final = np.array([t[-1, 5] for t in res_tys])
        mean_dv = np.mean(v_final) - v0

        self.assertLess(mean_dv, 0, "Mean speed must decrease under electron drag")
        ratio = mean_dv / (K_theory * tmax)
        self.assertGreater(
            ratio, 0.3, f"Drag too weak: mean_dv/K_theory/tmax = {ratio:.2f} < 0.3"
        )
        self.assertLess(
            ratio, 3.0, f"Drag too strong: mean_dv/K_theory/tmax = {ratio:.2f} > 3.0"
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

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_pitch_angle_exponential_decay(self):
        """
        ⟨ξ(t)⟩ = ξ₀ exp(−ν_D t): the ASCOT5 pitch-angle benchmark.

        Starting from ξ₀ = 0.8 with N = 50 particles, the ensemble mean
        pitch must decay to ξ₀ exp(−0.5) ≈ 0.485 after tmax = 0.5/ν_D.

        Statistical tolerance [0.5, 1.5] × ξ_expected gives a 4.5σ window
        (false-failure probability < 10⁻⁵ for a correct implementation).

        Note: ν_D × tmax = 0.5 is chosen so (a) the signal is large enough
        to measure (⟨ξ⟩ drops by ~37%), (b) v barely changes (Δv/v₀ < 0.1%),
        keeping ν_D effectively constant throughout the integration.
        """
        bg = self._background_high_n()
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        m, q = ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE
        v0 = np.sqrt(2 * Ekin / m)
        xi0 = 0.8
        s0 = 0.3

        nu_D_theory = _analytical_nu_D(v0, s0, m, q, [bg])
        tmax = 0.5 / nu_D_theory  # ν_D × tmax = 0.5 exactly

        nP = 50
        stz = np.tile([s0, 0.0, 0.0], (nP, 1))
        vpar = np.full(nP, xi0 * v0)

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
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        )
        self._assert_confined(res_tys, tmax)

        xi_final = np.array([t[-1, 4] / t[-1, 5] for t in res_tys])
        mean_xi = np.mean(xi_final)
        xi_expected = xi0 * np.exp(-nu_D_theory * tmax)  # = xi0 * exp(-0.5)

        ratio = mean_xi / xi_expected
        self.assertGreater(
            ratio,
            0.5,
            f"Mean pitch decayed too fast: ⟨ξ⟩/ξ_theory = {ratio:.2f} "
            f"(expected ≈ {xi_expected:.3f}, got {mean_xi:.3f})",
        )
        self.assertLess(
            ratio,
            1.5,
            f"Mean pitch decayed too slow: ⟨ξ⟩/ξ_theory = {ratio:.2f} "
            f"(expected ≈ {xi_expected:.3f}, got {mean_xi:.3f})",
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
