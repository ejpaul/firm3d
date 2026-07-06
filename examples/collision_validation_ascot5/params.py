"""
Shared parameters for the firm3d <-> ASCOT5 collision validation.

The case mirrors the alpha slowing-down branch of ASCOT5's built-in
Coulomb-collision physics test (a5py/testascot/physicstests.py,
init_ccoll): fusion alphas slowing down in a uniform hydrogen plasma
inside an ITER-like circular tokamak.  The two codes use different
magnetic geometries (ASCOT5: analytical Grad-Shafranov; firm3d:
BoozerAnalytic near-axis), but the plasma is uniform so the
velocity-space moments <E(t)> and <xi(t)> and the slowing-down time are
geometry-insensitive.

Density is a free knob: all collision rates scale linearly with n_b, so
n = 1e21 m^-3 gives a ~3 ms slowing-down window (fast local runs) with
physics identical to the published ASCOT5 test point (1e20) up to the
~10 % change in the Coulomb logarithm, which both codes compute
consistently.
"""

import numpy as np

# Plasma (uniform, single hydrogen ion species, T_e = T_i)
DENSITY = 1e21  # m^-3
TEMPERATURE_EV = 1e3  # eV

# Alpha markers
N_MARKERS = 100
E0_EV = 3.5e6
EMIN_EV = 50e3  # slowing-down end condition
SEED = 20260705

# Physical constants (SI)
EV = 1.602176634e-19
M_ALPHA = 6.6446573357e-27
M_ELECTRON = 9.1093837015e-31
M_PROTON = 1.67262192369e-27
EPS0 = 8.8541878188e-12
HBAR = 1.054571817e-34

# Tokamak-ish scale shared by both geometries
R0 = 6.2  # m
B0 = 5.3  # T


def coulomb_log_alpha_electron():
    """ln Lambda for alpha-electron collisions at the birth speed
    (ASCOT5 convention, same as firm3d collisions.h)."""
    v0 = np.sqrt(2 * E0_EV * EV / M_ALPHA)
    T = TEMPERATURE_EV * EV
    lam_d = np.sqrt(EPS0 * T / (2 * DENSITY * EV**2))
    vth = np.sqrt(2 * T / M_ELECTRON)
    mr = M_ALPHA * M_ELECTRON / (M_ALPHA + M_ELECTRON)
    ve2 = v0**2 + vth**2
    bcl = 2 * EV**2 / (4 * np.pi * EPS0 * mr * ve2)
    bqm = HBAR / (2 * mr * np.sqrt(ve2))
    return np.log(lam_d / max(bcl, bqm))


def spitzer_ts():
    """Spitzer slowing-down rate ts [s] (as in ASCOT5's check_ccoll)."""
    T = TEMPERATURE_EV * EV
    clog = coulomb_log_alpha_electron()
    return (
        3
        * np.sqrt((2 * np.pi * T) ** 3 / M_ELECTRON)
        * EPS0**2
        * M_ALPHA
        / (4 * EV**4 * DENSITY * clog)
    )


def analytic_slowing_time():
    """Mean time to slow from E0 to EMIN: 0.5 ts ln(E0/EMIN)."""
    return 0.5 * spitzer_ts() * np.log(E0_EV / EMIN_EV)


# Simulation window: generous margin over the analytic slowing time
TMAX = 1.8 * analytic_slowing_time()
N_SAVE = 100
DT_SAVE = TMAX / N_SAVE


def set_case(density=None, n_markers=None):
    """Override the default case (e.g. the published ASCOT5 test point
    n = 1e20 with more markers on an HPC system) and recompute the
    derived quantities."""
    global DENSITY, N_MARKERS, TMAX, DT_SAVE
    if density is not None:
        DENSITY = density
    if n_markers is not None:
        N_MARKERS = n_markers
    TMAX = 1.8 * analytic_slowing_time()
    DT_SAVE = TMAX / N_SAVE


def add_case_arguments(parser):
    """Common CLI arguments for the run scripts."""
    parser.add_argument("--density", type=float, default=None)
    parser.add_argument("--nmarkers", type=int, default=None)


def initial_pitches():
    """Shared initial pitch samples (uniform in [-1, 1], fixed seed)."""
    rng = np.random.default_rng(SEED)
    return 1.0 - 2.0 * rng.random(N_MARKERS)
