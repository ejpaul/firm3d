#pragma once

#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <functional>

using std::vector;

// Physical constants (SI)
static constexpr double COLL_PI          = 3.14159265358979323846;
static constexpr double COLL_EPSILON0    = 8.8541878188e-12;   // F/m
static constexpr double COLL_SQRT_PI     = 1.7724538509055159; // sqrt(pi)
static constexpr double COLL_HBAR        = 1.054571817e-34;    // J·s (reduced Planck)

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
// Returns the summed coefficients needed for the GC SDE in (v, xi) space:
//   D_par      : parallel velocity diffusion  [m^2/s^3]
//   dD_par_dv  : d(D_par)/dv               [m/s^3]  (for Milstein)
//   nu_D       : pitch-angle scattering freq [s^-1]
//   K          : deterministic drift in v   [m/s^2]
//                K = Q + d(D_par)/dv + 2*D_par/v
//   nu_D_det   : deterministic drift coeff for xi: dxi/dt|_coll = -xi * nu_D
// --------------------------------------------------------------------------
struct CollisionCoefficients {
    double D_par;       // m^2/s^3
    double dD_par_dv;   // m/s^3
    double nu_D;        // s^-1
    double K;           // m/s^2  (total deterministic drift in v)
};

inline CollisionCoefficients compute_collision_coefficients(
    double v,                     // EP speed [m/s]
    double s,                     // flux surface label
    double m_a,                   // EP mass [kg]
    double q_a,                   // EP charge [C]
    const vector<ThermalBackground>& backgrounds)
{
    CollisionCoefficients c = {0.0, 0.0, 0.0, 0.0};

    // Pre-pass: total Debye length from all species (used when coulomb_log <= 0).
    // 1/lambda_D^2 = sum_b n_b q_b^2 / (eps0 T_b)
    double inv_lD_sq = 0.0;
    for (const auto& bg : backgrounds) {
        double n_b = bg.n(s), T_b = bg.T(s);
        if (n_b > 0.0 && T_b > 0.0)
            inv_lD_sq += n_b * bg.charge * bg.charge / (COLL_EPSILON0 * T_b);
    }
    double lambda_D = (inv_lD_sq > 0.0) ? 1.0 / std::sqrt(inv_lD_sq) : 0.0;

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
        if (lnL < 2.0) {
            std::fprintf(stderr,
                "collisions: warning: ln_Lambda = %.3f < 2 "
                "(v=%.3e m/s, s=%.3f, species m=%.3e kg, q=%.3e C)\n",
                lnL, v, s, m_b, q_b);
        }

        double G      = chandrasekhar_G(x);
        double Gp     = chandrasekhar_G_deriv(x);   // dG/dx
        double erf_x  = std::erf(x);

        // Prefactor: Gamma = n_b q_a^2 q_b^2 lnL / (4 pi eps0^2 m_a^2)  [m^2/s^3 * m]
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
// Apply one Milstein noise step to (v, xi) after an accepted DP step of size h.
// Modifies v and xi in-place.
// --------------------------------------------------------------------------
inline void milstein_collision_step(
    double& v,
    double& xi,
    const CollisionCoefficients& coef,
    double h,                    // accepted DP step size [s] (physical time)
    double dW_v,                 // N(0,h) increment for v
    double dW_xi)                // N(0,h) increment for xi
{
    // Noise amplitudes
    double g_v  = std::sqrt(std::max(0.0, 2.0 * coef.D_par));
    double g_xi = std::sqrt(std::max(0.0, coef.nu_D * (1.0 - xi * xi)));

    // Milstein corrections
    double milstein_v  = 0.5 * coef.dD_par_dv * (dW_v * dW_v - h);
    double milstein_xi = -0.5 * xi * coef.nu_D  * (dW_xi * dW_xi - h);

    v  += g_v  * dW_v  + milstein_v;
    xi += g_xi * dW_xi + milstein_xi;

    // Enforce physical bounds.  The speed reflects at v = 0: the exact
    // process never reaches the origin (the 2 D_par / v drift repels it),
    // but a discrete step can overshoot.  Clamping to exactly zero would
    // kill the particle -- the collision coefficients are skipped at v = 0,
    // every orbit term vanishes, and xi = v_par / v is undefined.
    v  = std::fabs(v);
    xi = std::max(-1.0, std::min(1.0, xi));
}
