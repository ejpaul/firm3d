"""
Fast, local velocity-space collision tests (no compiled firm3d module, no
magnetic field, seconds of wall time).

Strategy: all collision frequencies scale linearly with background density
n_b (through Gamma), while the dimensionless physics in x = v/v_th is
unchanged.  Boosting n_b to 1e23 m^-3 shrinks the collisional relaxation
time to ~0.4 us, so an ensemble equilibrates within a few thousand fixed
Milstein steps.

The key physics checks:

* An ensemble started far from equilibrium (too cold / too hot) must relax
  to the Maxwellian of the background temperature: <E> -> (3/2) T_b and the
  speed distribution must match the Maxwell speed law (KS test).  This is a
  direct test of the Einstein-relation drag Q = -(m_a v/T_b) D_par: with
  a wrong drag the stationary temperature is wrong (the pre-fix drag
  equilibrated a proton ensemble to 3.7 T_b instead of 1.5 T_b, KS p = 0).

* An ensemble started AT the Maxwellian must stay there (detailed balance).

* Pitch angles must isotropize: <xi> -> 0, <xi^2> -> 1/3, xi ~ U(-1, 1).

* The short-time mean drift of an alpha ensemble must match K(v0).

All RNG seeds are fixed, so the assertions are deterministic.
"""

import unittest

import numpy as np
from scipy.stats import kstest, maxwell

from firm3d.util.constants import (  # noqa: E402
    ELECTRON_MASS,
    ONE_EV,
    PROTON_MASS,
)
from firm3d.util.constants import (
    ELEMENTARY_CHARGE as E_CHARGE,
)
from tests.field.collision_helpers import (
    collision_coefficients,
    evolve_velocity_ensemble,
)

# The measured alpha mass, which is 0.76% below the library's
# ALPHA_PARTICLE_MASS (2*m_p + 2*m_n, i.e. the free constituents with no
# binding-energy defect).  Kept local deliberately: these helpers check
# against analytic velocity-space solutions where the physical value is the
# meaningful one, and m_a enters Gamma quadratically.
ALPHA_MASS = 6.644657e-27

# Proton test particles on a proton background, T = 1 keV, boosted density.
T_B = 1e3 * ONE_EV
N_B = 1e23  # m^-3: the "skew" -- rates ~us^-1 instead of s^-1
SPECIES = [(PROTON_MASS, E_CHARGE, N_B, T_B)]
M_A, Q_A = PROTON_MASS, E_CHARGE
V_TH = np.sqrt(2.0 * T_B / M_A)

N_PART = 4000


def _characteristic_rate():
    """max(nu_D, 2 D_par/v^2) at v = v_th: sets dt and run length."""
    _, D_par, _, nu_D = collision_coefficients(np.array([V_TH]), M_A, Q_A, SPECIES)
    return max(nu_D[0], 2.0 * D_par[0] / V_TH**2)


NU_CHAR = _characteristic_rate()
DT = 1e-3 / NU_CHAR
N_RELAX = 8000  # 8 relaxation times at DT


def _maxwell_dist():
    """Maxwell speed distribution for M_A at T_B (scipy convention)."""
    return maxwell(scale=np.sqrt(T_B / M_A))


class TestMaxwellianRelaxation(unittest.TestCase):
    """Ensemble relaxation of (v, xi) under the collisional SDE alone."""

    def _relax_and_check(self, v0, seed):
        rng = np.random.default_rng(seed)
        v, xi = evolve_velocity_ensemble(
            np.full(N_PART, v0),
            np.ones(N_PART),
            M_A,
            Q_A,
            SPECIES,
            DT,
            N_RELAX,
            rng,
        )
        E_mean = 0.5 * M_A * np.mean(v**2) / T_B
        self.assertAlmostEqual(
            E_mean,
            1.5,
            delta=0.15,
            msg=f"<E>/T_b = {E_mean:.3f}, expected 3/2 at equilibrium",
        )
        p = kstest(v, _maxwell_dist().cdf).pvalue
        self.assertGreater(
            p, 0.01, f"KS test against Maxwell speed law failed (p = {p:.2e})"
        )
        return v, xi

    def test_equilibration_from_cold_start(self):
        """Monoenergetic ensemble at E0 = 0.45 T_b heats up to <E> = 1.5 T_b."""
        v0 = np.sqrt(2.0 * 0.45 * T_B / M_A)
        self._relax_and_check(v0, seed=1)

    def test_equilibration_from_hot_start(self):
        """Monoenergetic ensemble at E0 = 4.5 T_b cools down to <E> = 1.5 T_b."""
        v0 = np.sqrt(2.0 * 4.5 * T_B / M_A)
        self._relax_and_check(v0, seed=2)

    def test_stationarity_of_maxwellian(self):
        """
        An ensemble sampled from the Maxwellian stays Maxwellian (detailed
        balance).  This is the most direct regression test for the
        Einstein-relation drag.
        """
        rng = np.random.default_rng(3)
        v0 = _maxwell_dist().rvs(size=N_PART, random_state=42)
        xi0 = rng.uniform(-1.0, 1.0, N_PART)
        v, _ = evolve_velocity_ensemble(
            v0, xi0, M_A, Q_A, SPECIES, DT, N_RELAX // 2, rng
        )
        E_mean = 0.5 * M_A * np.mean(v**2) / T_B
        self.assertAlmostEqual(
            E_mean,
            1.5,
            delta=0.1,
            msg=f"Maxwellian drifted to <E>/T_b = {E_mean:.3f}",
        )
        p = kstest(v, _maxwell_dist().cdf).pvalue
        self.assertGreater(p, 0.01, f"stationary KS failed (p = {p:.2e})")

    def test_pitch_isotropization(self):
        """xi = 1 initially; after t >> 1/nu_D, xi ~ U(-1, 1)."""
        _, xi = self._relax_and_check(np.sqrt(2.0 * 1.5 * T_B / M_A), seed=4)
        self.assertLess(abs(np.mean(xi)), 0.05, f"<xi> = {np.mean(xi):.3f}")
        self.assertAlmostEqual(
            np.mean(xi**2),
            1.0 / 3.0,
            delta=0.03,
            msg=f"<xi^2> = {np.mean(xi**2):.4f}, expected 1/3",
        )
        p = kstest(xi, lambda t: np.clip((t + 1.0) / 2.0, 0.0, 1.0)).pvalue
        self.assertGreater(
            p, 0.01, f"KS test of xi against U(-1,1) failed (p = {p:.2e})"
        )


class TestUnphysicalCoulombLog(unittest.TestCase):
    """ln Lambda <= 0 must raise (mirrors the collisions.h exception)."""

    def test_coulomb_log_raises(self):
        from tests.field.collision_helpers import collision_coefficients

        cold_dense = [(ELECTRON_MASS, -E_CHARGE, 1e30, 1e-3 * ONE_EV)]
        v = np.array([np.sqrt(2.0 * 3.52e6 * ONE_EV / ALPHA_MASS)])
        with self.assertRaises(ValueError):
            collision_coefficients(v, ALPHA_MASS, 2 * E_CHARGE, cold_dense)


class TestDeterministicDrift(unittest.TestCase):
    """Short-time mean drift of the ensemble must match K(v0)."""

    def test_alpha_on_electrons_drift(self):
        """
        3.52 MeV alphas on 10 keV electrons: over a window short enough
        that K(v) is nearly constant, d<v>/dt = K(v0) (noise averages out).
        """
        species = [(ELECTRON_MASS, -E_CHARGE, 1e23, 10e3 * ONE_EV)]
        v0 = np.sqrt(2.0 * 3.52e6 * ONE_EV / ALPHA_MASS)
        (K0,), _, _, _ = collision_coefficients(
            np.array([v0]), ALPHA_MASS, 2 * E_CHARGE, species
        )
        nsteps = 100
        dt = 0.02 * v0 / abs(K0) / nsteps  # total window: 2% speed change
        rng = np.random.default_rng(5)
        v, _ = evolve_velocity_ensemble(
            np.full(N_PART, v0),
            np.zeros(N_PART),
            ALPHA_MASS,
            2 * E_CHARGE,
            species,
            dt,
            nsteps,
            rng,
        )
        drift = (np.mean(v) - v0) / (nsteps * dt)
        self.assertAlmostEqual(
            drift / K0,
            1.0,
            delta=0.05,
            msg=f"d<v>/dt / K(v0) = {drift / K0:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
