#pragma once

#include <vector>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <functional>

using std::vector;

// Physical constants (SI)
static constexpr double COLL_PI          = 3.14159265358979323846;
static constexpr double COLL_EPSILON0    = 8.8541878188e-12;   // F/m
static constexpr double COLL_SQRT_PI     = 1.7724538509055159; // sqrt(pi)
static constexpr double COLL_HBAR        = 1.054571817e-34;    // J·s (reduced Planck)

// Reflecting speed boundary as a fraction of sqrt(T_b / m_a), following
// ASCOT5 (MCCC_CUTOFF in mccc.h): "if the guiding center energy goes below
// this, it is mirrored to prevent collision coefficients from diverging."
static constexpr double COLL_CUTOFF      = 0.1;

// --------------------------------------------------------------------------
// Chandrasekhar G function: G(x) = [erf(x) - (2x/sqrt(pi)) exp(-x^2)] / (2x^2)
// Satisfies G'(x) = (2/sqrt(pi)) exp(-x^2) - 2 G(x)/x
// --------------------------------------------------------------------------
inline double chandrasekhar_G(double x) {
    if (x == 0.0) return 0.0;
    return (std::erf(x) - (2.0 * x / COLL_SQRT_PI) * std::exp(-x * x)) / (2.0 * x * x);
}

// dG/dx = (2/sqrt(pi)) exp(-x^2) - 2 G(x)/x
inline double chandrasekhar_G_deriv(double x) {
    if (x == 0.0) return 2.0 / (3.0 * COLL_SQRT_PI);
    return (2.0 / COLL_SQRT_PI) * std::exp(-x * x) - 2.0 * chandrasekhar_G(x) / x;
}

// --------------------------------------------------------------------------
// Background species specification.
//
// Density n_grid (m^-3) and temperature T_grid (J) are stored on a uniform
// grid in s in [0, 1]. Simple linear interpolation is used.
// --------------------------------------------------------------------------
struct ThermalBackground {
    vector<double> s_grid;   // uniform s in [0,1], length >= 2
    vector<double> n_grid;   // number density m^-3 at each s
    vector<double> T_grid;   // temperature J at each s
    double mass;             // kg
    double charge;           // C (signed)

    // Linear interpolation helper
    double interp(const vector<double>& vals, double s) const {
        s = std::max(s_grid.front(), std::min(s_grid.back(), s));
        int n = (int)s_grid.size();
        // Uniform grid assumed
        double ds = (s_grid.back() - s_grid.front()) / (n - 1);
        int i = (int)((s - s_grid.front()) / ds);
        i = std::max(0, std::min(n - 2, i));
        double t = (s - s_grid[i]) / ds;
        return vals[i] * (1.0 - t) + vals[i + 1] * t;
    }

    double n(double s) const { return interp(n_grid, s); }
    double T(double s) const { return interp(T_grid, s); }
};

// --------------------------------------------------------------------------
// Collision coefficients for EP (species a) against one Maxwellian background.
//
// Returns the summed coefficients for the collision kick in (v, xi):
//   D_par      : parallel velocity diffusion  [m^2/s^3]
//   dD_par_dv  : d(D_par)/dv               [m/s^2]  (for Milstein)
//   nu_D       : pitch-angle scattering freq [s^-1]; also the deterministic
//                pitch drift, dxi/dt|_coll = -xi * nu_D
//   K          : deterministic drift in v   [m/s^2]
//                K = Q + d(D_par)/dv + 2*D_par/v
//   v_cutoff   : reflecting speed boundary  [m/s]
// --------------------------------------------------------------------------
struct CollisionCoefficients {
    double D_par;       // m^2/s^3
    double dD_par_dv;   // m/s^2
    double nu_D;        // s^-1
    double K;           // m/s^2  (total deterministic drift in v)
    double v_cutoff;    // m/s   (reflecting speed boundary, ASCOT5 style)
};

inline CollisionCoefficients compute_collision_coefficients(
    double v,                     // EP speed [m/s]
    double s,                     // flux surface label
    double m_a,                   // EP mass [kg]
    double q_a,                   // EP charge [C]
    const vector<ThermalBackground>& backgrounds)
{
    CollisionCoefficients c = {0.0, 0.0, 0.0, 0.0, 0.0};

    // Pre-pass: total Debye length from all species (used when coulomb_log <= 0).
    // 1/lambda_D^2 = sum_b n_b q_b^2 / (eps0 T_b)
    // Also track the coldest active species for the reflecting speed
    // boundary v_cutoff = COLL_CUTOFF * sqrt(T_min / m_a).  ASCOT5 uses the
    // first background species (Tb[0]); the coldest one is the conservative
    // generalization when species temperatures differ.
    double inv_lD_sq = 0.0;
    double T_min = std::numeric_limits<double>::infinity();
    for (const auto& bg : backgrounds) {
        double n_b = bg.n(s), T_b = bg.T(s);
        if (n_b > 0.0 && T_b > 0.0) {
            inv_lD_sq += n_b * bg.charge * bg.charge / (COLL_EPSILON0 * T_b);
            T_min = std::min(T_min, T_b);
        }
    }
    double lambda_D = (inv_lD_sq > 0.0) ? 1.0 / std::sqrt(inv_lD_sq) : 0.0;
    if (T_min < std::numeric_limits<double>::infinity())
        c.v_cutoff = COLL_CUTOFF * std::sqrt(T_min / m_a);

    for (const auto& bg : backgrounds) {
        double n_b = bg.n(s);
        double T_b = bg.T(s);
        if (n_b <= 0.0 || T_b <= 0.0) continue;

        double m_b    = bg.mass;
        double q_b    = bg.charge;

        // Background thermal speed and normalised EP speed
        double v_th   = std::sqrt(2.0 * T_b / m_b);
        double x      = v / v_th;

        // ln(4 pi eps0 lambda_D m_r v_eff^2 / |q_a q_b|)
        // v_eff^2 = v^2 + v_th^2 handles slow EP (v << v_th,e) and fast EP (v >> v_th,b)
        // Coulomb logarithm: ln(lambda_D / b_min) where b_min = max(b_cl, b_qm).
        //
        // b_cl  = |qa qb| / (4 pi eps0 mr vbar)   classical 90-deg deflection
        // b_qm  = hbar / (2 mr sqrt(vbar))         de Broglie wavelength
        // vbar  = v^2 + v_th_b^2   (v_th_b = sqrt(2 T_b / m_b))
        //
        // Taking max(b_cl, b_qm) handles the quantum regime (fast EP against
        // electrons) where the de Broglie wavelength exceeds the classical
        // distance of closest approach.  This matches the ASCOT5 convention
        // (Hirvijoki et al., Comput. Phys. Commun. 185, 1310, 2014).
        //
        // See also: NRL Plasma Formulary (Huba), "Collision Parameters" section;
        // Spitzer, Physics of Fully Ionized Gases (1962), Ch. 5.
        double m_r      = m_a * m_b / (m_a + m_b);
        double v_eff_sq = v * v + v_th * v_th;
        double bcl      = std::abs(q_a * q_b)
                          / (4.0 * COLL_PI * COLL_EPSILON0 * m_r * v_eff_sq);
        double bqm      = COLL_HBAR / (2.0 * m_r * std::sqrt(v_eff_sq));
        double b_min    = std::max(bcl, bqm);
        double lnL      = std::log(lambda_D / b_min);
        if (lnL <= 0.0) {
            // The Debye length is smaller than the minimum impact
            // parameter: the binary-collision model is undefined and the
            // coefficients would change sign.  This happens when the
            // background temperature goes to zero at finite density --
            // fail loudly (ASCOT5 aborts such markers via its input
            // evaluation errors) rather than integrate garbage.
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                "collisions: ln_Lambda = %.3f <= 0 (v=%.3e m/s, s=%.3f, "
                "n_b=%.3e m^-3, T_b=%.3e J): background profiles give an "
                "unphysical Coulomb logarithm; keep T finite where n > 0",
                lnL, v, s, n_b, T_b);
            throw std::runtime_error(msg);
        }
        // A marginal-but-positive ln_Lambda (< 2) is reported once, up front,
        // by _validate_coulomb_log() in collisions.py.  Warning from here is
        // not viable: this runs once per species per collision sub-step, so a
        // single cold region would emit millions of duplicate lines and
        // dominate the run.

        double G      = chandrasekhar_G(x);
        double Gp     = chandrasekhar_G_deriv(x);   // dG/dx
        double erf_x  = std::erf(x);

        // Prefactor: Gamma = n_b q_a^2 q_b^2 lnL / (4 pi eps0^2 m_a^2)  [m^3/s^4]
        double Gamma  = n_b * q_a * q_a * q_b * q_b * lnL
                        / (4.0 * COLL_PI * COLL_EPSILON0 * COLL_EPSILON0 * m_a * m_a);

        // Parallel diffusion and its v-derivative
        double D_par_b   = Gamma * G / v;
        // d(G/v)/dv = G'/(v_th * v) - G/v^2  =>  d(D_par)/dv = Gamma*(G'/v_th - G/v)/v
        double dD_par_dv_b = Gamma * (Gp / v_th - G / v) / v;

        // Pitch-angle scattering frequency: nu_D = Gamma*(erf(x) - G(x)) / v^3
        double nu_D_b = Gamma * (erf_x - G) / (v * v * v);

        // Drag (Einstein relation): Q = -(m_a v / T_b) * D_par
        //                             = -Gamma * G * m_a / T_b.
        // Matches ASCOT5 mccc_coefs_Q and guarantees relaxation to the
        // background Maxwellian.  This is not the full dynamical friction
        // F = -(1 + m_a/m_b) * 2 x^2 G(x) * Gamma / v^2; the two are related
        // by F = Q + dD_par/dv + 2*(D_par - D_perp)/v.
        double Q_b = -Gamma * G * m_a / T_b;

        // Total deterministic drift in v: K = Q + d(D_par)/dv + 2*D_par/v
        double K_b = Q_b + dD_par_dv_b + 2.0 * D_par_b / v;

        c.D_par     += D_par_b;
        c.dD_par_dv += dD_par_dv_b;
        c.nu_D      += nu_D_b;
        c.K         += K_b;
    }
    return c;
}

// --------------------------------------------------------------------------
// Apply one collision step to (v, xi) after an accepted orbit step of size h:
// deterministic drift by explicit Euler, plus the Milstein noise term.
// Modifies v and xi in-place.
//
// This matches the (v, xi) half of ASCOT5's mccc_gc_milstein.c.  ASCOT5
// additionally displaces the guiding centre by the spatial diffusion DX and
// works in the plasma flow frame; both are omitted here (see the notes'
// Validity and Limitations section):
//
//   vout  = vin  + K*h        + sqrt(2 D_par) dW_v  + (1/2) dD_par/dv (dW_v^2 - h)
//   xiout = xiin - xi*nu_D*h  + sqrt(nu_D(1-xi^2)) dW_xi - (1/2) xi nu_D (dW_xi^2 - h)
//
// Keeping the drift here rather than inside the adaptive orbit right-hand
// side is what lets the orbit equations stay in (s, theta, zeta, v_par) with
// mu fixed across a step, so any static-field guiding-centre right-hand side
// can be driven by this operator without being rewritten in (v, xi).
// --------------------------------------------------------------------------
inline void milstein_collision_step(
    double& v,
    double& xi,
    const CollisionCoefficients& coef,
    double h,                    // accepted orbit step size [s] (physical time)
    double dW_v,                 // N(0,h) increment for v
    double dW_xi)                // N(0,h) increment for xi
{
    // Noise amplitudes
    double g_v  = std::sqrt(std::max(0.0, 2.0 * coef.D_par));
    double g_xi = std::sqrt(std::max(0.0, coef.nu_D * (1.0 - xi * xi)));

    // Deterministic drift, evaluated at the pre-step (v, xi) as in ASCOT5
    double drift_v  = coef.K * h;
    double drift_xi = -xi * coef.nu_D * h;

    // Milstein corrections
    double milstein_v  = 0.5 * coef.dD_par_dv * (dW_v * dW_v - h);
    double milstein_xi = -0.5 * xi * coef.nu_D  * (dW_xi * dW_xi - h);

    v  += drift_v  + g_v  * dW_v  + milstein_v;
    xi += drift_xi + g_xi * dW_xi + milstein_xi;

    // Boundary conditions, following ASCOT5 (mccc_gc_milstein.c):
    //  - the speed reflects off the thermal cutoff v_cutoff, so the
    //    collision coefficients, which diverge as v -> 0, are never
    //    evaluated deep below the background thermal speed;
    //  - the pitch mirrors at |xi| = 1 (a hard clamp would pile up
    //    probability at exactly +-1).
    // The final clamp on xi only acts on pathologically large steps.
    if (v < coef.v_cutoff)
        v = 2.0 * coef.v_cutoff - v;
    if (std::fabs(xi) > 1.0)
        xi = ((xi > 0.0) - (xi < 0.0)) * (2.0 - std::fabs(xi));
    xi = std::max(-1.0, std::min(1.0, xi));
}

// --------------------------------------------------------------------------
// Sub-stepping control for the collision kick.
//
// The orbit integrator chooses its step from orbit dynamics alone; the
// collision terms get no vote.  Applying them as one explicit-Euler kick over
// that whole step is badly inaccurate whenever the collision rates are fast
// compared with it -- which is exactly the thermal regime, where nu_D and K
// diverge as v -> 0.  Measured on the Maxwellian-equilibration test, a single
// kick per orbit step gave <E>/T_b = 7.8 instead of 1.5 at tol = 1e-6, the
// error shrinking only as the orbit tolerance (and hence h) was tightened.
//
// Sub-cycling fixes this at almost no cost: the particle position is frozen
// during a kick, so a sub-step needs no field evaluation, only a coefficient
// re-evaluation at the updated speed.  ASCOT5 reaches the same place from the
// other direction, with a dedicated error estimate that shortens its
// collision step (mccc_gc_milstein.c, the kappa_k drift/diffusion limits).
//
// Splitting is exact for the noise: a Wiener increment over h is the sum of n
// independent increments over h/n.
// --------------------------------------------------------------------------

// Fraction of a collision timescale a single sub-step may cover.
static constexpr double COLL_SUBSTEP_SAFETY = 0.05;
// Runaway guard.  Hitting this means the orbit step is enormously longer than
// the collision timescale; accuracy then degrades rather than the run hanging.
static constexpr int COLL_MAX_SUBSTEPS = 10000;

inline int collision_substeps(
    double v,
    const CollisionCoefficients& coef,
    double h)
{
    if (!(h > 0.0) || !(v > 0.0)) return 1;   // also rejects NaN

    // Drift terms scale as h, so dividing the step by n divides them by n.
    double n_needed = std::max(coef.nu_D * h,                    // pitch drift
                               std::abs(coef.K) * h / v)         // speed drift
                      / COLL_SUBSTEP_SAFETY;

    // The diffusive excursion scales as sqrt(h), so it falls only as
    // sqrt(n): bounding it to the same fraction needs n = (rate/safety)^2,
    // not rate/safety.  Getting this wrong misses the stated bound by the
    // ratio itself -- a factor of 100 at rate = 5.
    double diff = std::sqrt(std::max(0.0, 2.0 * coef.D_par * h)) / v;
    double n_diff = diff / COLL_SUBSTEP_SAFETY;
    n_needed = std::max(n_needed, n_diff * n_diff);

    // Compare in double before converting: casting a non-finite or
    // out-of-range double to int is undefined behaviour, and on x86-64 it
    // yields INT_MIN, which the n < 1 clamp below would then turn into a
    // single sub-step -- the runaway guard failing open in exactly the
    // regime it exists for.
    if (!(n_needed > 1.0)) return 1;                      // NaN lands here
    if (n_needed >= (double)COLL_MAX_SUBSTEPS) return COLL_MAX_SUBSTEPS;
    return (int)std::ceil(n_needed);
}
