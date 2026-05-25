#include "tracing_helpers.h"
#include "boozermagneticfield.h"
#include "shearalfvenwave.h"
#include "tracing.h"
#include "ode_solvers.h"
#include "collisions.h"
#ifdef USE_GSL
    #include "symplectic.h"
#endif

#include <memory>
#include <vector>
#include <functional>
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>

using std::shared_ptr;
using std::tuple;
using std::function;
using std::array;

using boost::math::tools::toms748_solve;
using namespace boost::numeric::odeint;
using Array2 = BoozerMagneticField::Array2;

class GuidingCenterVacuumBoozerRHS : public BaseRHS {
    /*
     * The state consists of :math:`[s, theta, zeta, v_par]` with
     *
     *    \dot s = -|B|_{,\theta} m(v_{||}^2/|B| + \mu)/(q \psi_0)
     *    \dot \theta = |B|_{,s} m(v_{||}^2/|B| + \mu)/(q \psi_0) + \iota v_{||} |B|/G
     *    \dot \zeta = v_{||}|B|/G
     *    \dot v_{||} = -(\iota |B|_{,\theta} + |B|_{,\zeta})\mu |B|/G,
     *
     *  where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     *
     */
    private:
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q, mu;
    public:
        int axis;
        double vnorm, tnorm;
        static constexpr int Size = 4;

        GuidingCenterVacuumBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis, double vnorm=1, double tnorm=1)
            : field(field), m(m), q(q), mu(mu), axis(axis), vnorm(vnorm), tnorm(tnorm) {
            }

        int get_state_size() const override {
            return Size;
        }

        void operator()(const vector<double> &ys, vector<double> &dydt, const double t) override {
            vector<double> stzv(Size), stzvdot(Size);
            y_to_stzvt(ys, stzv, axis, vnorm, tnorm);

            stz(0, 0) = stzv[0];
            stz(0, 1) = stzv[1];
            stz(0, 2) = stzv[2];
            double v_par = stzv[3];

            field->set_points(stz);
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double iota = field->iota_ref()(0);
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBds = modB_derivs(0);
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;

            stzvdot[0] = -dmodBdtheta*fak1/(q*psi0);
            stzvdot[1] =  dmodBds*fak1/(q*psi0) + iota*v_par*modB/G;
            stzvdot[2] = v_par*modB/G;
            stzvdot[3] = -(iota*dmodBdtheta + dmodBdzeta)*mu*modB/G;

            stzvtdot_to_ydot(stzvdot, stzv, dydt, axis, vnorm, tnorm);
        }
};

class GuidingCenterVacuumBoozerPerturbedRHS : public BaseRHS {
    /*
     * The state consists of :math:`[s, theta, zeta, v_par, t]` with
     *
     *    \dot s      = (-|B|_{,\theta} m(v_{||}^2/|B| + \mu)/q
     *                  + \alpha_{,\theta}|B|v_{||} - \Phi_{\theta})/psi0;
     *    \dot \theta = |B|_{,\psi} m (v_{||}^2/|B| + \mu)/q
     *                  + (\iota - \alpha_{,psi} G) v_{||}|B|/G + \Phi_{,\psi};
     *    \dot \zeta  = v_{||}|B|/G
     *    \dot v_{||} = -|B|/(Gm) (m\mu(|B|_{,\zeta}
     *                          + \alpha_{,\theta}|B|_{,\psi}G
     *                          + |B|_{,\theta}(\iota - \alpha_{,\psi}G))
     *                  + q(\dot\alpha G + \alpha_{,\theta}G\Phi_{,\psi}
     *                  + (\iota - \alpha_{\psi}*G)*\Phi_{\theta}
     *                  + \Phi_{,\zeta}))
     *                  + v_{||}/|B|(|B|_{,\theta}\Phi_{,\psi}
     *                             - |B|_{,\psi} \Phi_{,\theta})
     *
     *  where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     *
     */
    private:
        Array2 stzt = xt::zeros<double>({1, 4});
        shared_ptr<ShearAlfvenWave> perturbed_field;
        double m, q, mu;
    public:
        int axis;
        double vnorm, tnorm;
        static constexpr int Size = 5;

        GuidingCenterVacuumBoozerPerturbedRHS(
            shared_ptr<ShearAlfvenWave> perturbed_field,
            double m,
            double q,
            double mu,
            int axis,
            double vnorm=1,
            double tnorm=1
        ):
            perturbed_field(perturbed_field),
            m(m),
            q(q),
            mu(mu),
            axis(axis),
            vnorm(vnorm),
            tnorm(tnorm) {}

        int get_state_size() const override {
            return Size;
        }

        void operator()(const vector<double> &ys, vector<double> &dydt, const double t) override {
            vector<double> stzvt(Size), stzvtdot(Size);
            y_to_stzvt(ys, stzvt, axis, vnorm, tnorm);

            stzt(0, 0) = stzvt[0];
            stzt(0, 1) = stzvt[1];
            stzt(0, 2) = stzvt[2];
            stzt(0, 3) = stzvt[4];
            double v_par = stzvt[3];
            double time = stzvt[4];

            perturbed_field->set_points(stzt);
            auto field = perturbed_field->get_B0();
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double iota = field->iota_ref()(0);
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double dPhidpsi = perturbed_field->dPhidpsi_ref()(0);
            double dPhidtheta = perturbed_field->dPhidtheta_ref()(0);
            double dPhidzeta = perturbed_field->dPhidzeta_ref()(0);
            double alphadot = perturbed_field->alphadot_ref()(0);
            double dalphadpsi = perturbed_field->dalphadpsi_ref()(0);
            double dalphadtheta = perturbed_field->dalphadtheta_ref()(0);

            stzvtdot[0] = (-dmodBdtheta*fak1/q + dalphadtheta*modB*v_par - dPhidtheta)/psi0;
            stzvtdot[1] = dmodBdpsi*fak1/q + (iota - dalphadpsi*G)*v_par*modB/G + dPhidpsi;
            stzvtdot[2] = v_par*modB/G;
            stzvtdot[3] = -modB/(G*m) * (m*mu*(dmodBdzeta + dalphadtheta*dmodBdpsi*G \
                    + dmodBdtheta*(iota - dalphadpsi*G)) + q*(alphadot*G \
                    + dalphadtheta*G*dPhidpsi + (iota - dalphadpsi*G)*dPhidtheta + dPhidzeta)) \
                    + v_par/modB * (dmodBdtheta*dPhidpsi - dmodBdpsi*dPhidtheta);
            stzvtdot[4] = 1;

            stzvtdot_to_ydot(stzvtdot, stzvt, dydt, axis, vnorm, tnorm);
        }
};

class GuidingCenterNoKBoozerPerturbedRHS : public BaseRHS {
    /*
     * The state consists of :math:`[s, theta, zeta, v_par, t]` with
     *
     *    \dot s = (-G \Phi_{,\theta}q + I\Phi_{,\zeta}q
     *               + |B|qv_{||}(\alpha_{\theta}G-\alpha_{,\zeta}I)
     *               + (-|B|_{,\theta}G + |B|_{,\zeta}I)
     *               * (mv_{||}}^2/|B| + m\mu))/(D psi0)
     *    \dot theta = (G q \Phi_{,\psi}
     *               + |B| q v_{||} (-\alpha_{,\psi} G - \alpha G_{,\psi} + \iota)
     *               - G_{,\psi} m v_{||}^2 + |B|_{,\psi} G (mv_{||}}^2/|B| + m\mu))/D
     *    \dot \zeta = (-I (|B|_{,\psi} m \mu + \Phi_{,\psi} q)
     *               + |B| q v_{||} (1 + \alpha_{,\psi}) I + \alpha I'(\psi))
     *               + m v_{||}^2/|B| (|B| I'(\psi) - |B|_{,\psi} I))/D
     *    \dot v_{||} = (|B|q/m ( -m mu (|B|_{,\zeta}(1 + \alpha_{,\psi} I + \alpha I'(\psi))
     *                + |B|_{,\psi} (\alpha_{,\theta} G - \alpha_{,\zeta} I)
     *                + |B|_{,\theta} (\iota - \alpha G'(\psi) - \alpha_{,\psi} G))
     *                - q (\dot \alpha (G + I (\iota - \alpha G'(\psi)) + \alpha G I'(\psi))
     *                + (\alpha_{,\theta} G - \alpha_{,\zeta} I) \Phi_{,\psi}
     *                + (\iota - \alpha G_{,\psi} - \alpha_{,\psi} G) \Phi_{,\theta}
     *                + (1 + \alpha I'(\psi) + \alpha_{,\psi} I) Phi_{,\zeta}))
     *                + q v_{||}/|B| ((|B|_{,\theta} G - |B|_{,\zeta} I) \Phi_{,\psi}
     *                + |B|_{,\psi} (I \Phi_{,\zeta} - G \Phi_{,\theta}))
     *                + v_{||} (m \mu (|B|_{,\theta} G'(\psi) - |B|_{,\zeta} I'(\psi))
     *                + q (\dot \alpha (G'(\psi) I - G I'(\psi))
     *                + G'(\psi) \Phi_{,\theta} - I'(\psi)\Phi_{,\zeta})))/D
     *    D = (q(G + I(-\alpha G_{,\psi} + \iota) + \alpha G I'(\psi)
     *          + mv_{||}/|B| (-G'(\psi) I + G I'(\psi)))
     *  where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     *
     */
    private:
        Array2 stzt = xt::zeros<double>({1, 4});
        shared_ptr<ShearAlfvenWave> perturbed_field;
        double m, q, mu;
    public:
        int axis;
        double vnorm, tnorm;
        static constexpr int Size = 5;

        GuidingCenterNoKBoozerPerturbedRHS(
            shared_ptr<ShearAlfvenWave> perturbed_field,
            double m,
            double q,
            double mu,
            int axis,
            double vnorm=1,
            double tnorm=1
        ):
        perturbed_field(perturbed_field),
        m(m),
        q(q),
        mu(mu),
        axis(axis),
        vnorm(vnorm),
        tnorm(tnorm) {}

        int get_state_size() const override {
            return Size;
        }

        void operator()(const vector<double> &ys, vector<double> &dydt, const double t) override {
            vector<double> stzvt(Size), stzvtdot(Size);
            y_to_stzvt(ys, stzvt, axis, vnorm, tnorm);

            stzt(0, 0) = stzvt[0];
            stzt(0, 1) = stzvt[1];
            stzt(0, 2) = stzvt[2];
            stzt(0, 3) = stzvt[4];
            double v_par = stzvt[3];

            perturbed_field->set_points(stzt);
            auto field = perturbed_field->get_B0();
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double I = field->I_ref()(0);
            double dGdpsi = field->dGds_ref()(0)/psi0;
            double dIdpsi = field->dIds_ref()(0)/psi0;
            double iota = field->iota_ref()(0);
            double diotadpsi = field->diotads_ref()(0)/psi0;
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);


            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double dPhidpsi = perturbed_field->dPhidpsi_ref()(0);
            double dPhidtheta = perturbed_field->dPhidtheta_ref()(0);
            double dPhidzeta = perturbed_field->dPhidzeta_ref()(0);
            double alpha = perturbed_field->alpha_ref()(0);
            double alphadot = perturbed_field->alphadot_ref()(0);
            double dalphadpsi = perturbed_field->dalphadpsi_ref()(0);
            double dalphadtheta = perturbed_field->dalphadtheta_ref()(0);
            double dalphadzeta = perturbed_field->dalphadzeta_ref()(0);
            double denom = (q*(G + I*(-alpha*dGdpsi + iota) + alpha*G*dIdpsi)
                + m*v_par/modB * (-dGdpsi*I + G*dIdpsi)); // q*G in vacuum


            stzvtdot[0] = (-G*dPhidtheta*q + I*dPhidzeta*q + modB*q*v_par*(dalphadtheta*G-dalphadzeta*I) + (-dmodBdtheta*G + dmodBdzeta*I)*fak1)/(denom*psi0);
            stzvtdot[1] = (G*q*dPhidpsi + modB*q*v_par*(-dalphadpsi*G - alpha*dGdpsi + iota) - dGdpsi*m*v_par*v_par \
                      + dmodBdpsi*G*fak1)/denom;
            stzvtdot[2] = (-I*(dmodBdpsi*m*mu + dPhidpsi*q) + modB*q*v_par*(1 + dalphadpsi*I + alpha*dIdpsi) \
                      + m*v_par*v_par/modB * (modB*dIdpsi - dmodBdpsi*I))/denom;
            stzvtdot[3] = (modB*q/m * ( -m*mu * (dmodBdzeta*(1 + dalphadpsi*I + alpha*dIdpsi) \
                      + dmodBdpsi*(dalphadtheta*G - dalphadzeta*I) + dmodBdtheta*(iota - alpha*dGdpsi - dalphadpsi*G)) \
                      - q*(alphadot*(G + I*(iota - alpha*dGdpsi) + alpha*G*dIdpsi) \
                      + (dalphadtheta*G - dalphadzeta*I)*dPhidpsi \
                      + (iota - alpha*dGdpsi - dalphadpsi*G)*dPhidtheta \
                      + (1 + alpha*dIdpsi + dalphadpsi*I)*dPhidzeta)) \
                      + q*v_par/modB * ((dmodBdtheta*G - dmodBdzeta*I)*dPhidpsi \
                      + dmodBdpsi*(I*dPhidzeta - G*dPhidtheta)) \
                      + v_par*(m*mu*(dmodBdtheta*dGdpsi - dmodBdzeta*dIdpsi) \
                      + q*(alphadot*(dGdpsi*I-G*dIdpsi) + dGdpsi*dPhidtheta - dIdpsi*dPhidzeta)))/denom;
            stzvtdot[4] = 1;

            stzvtdot_to_ydot(stzvtdot, stzvt, dydt, axis, vnorm, tnorm);
        }
};

class GuidingCenterNoKBoozerRHS : public BaseRHS {
    /*
     * The state consists of :math:`[s, t, z, v_par]` with
     *
     *  \dot s = (I |B|_{,\zeta} - G |B|_{,\theta})m(v_{||}^2/|B| + \mu)/(\iota D \psi_0)
     *  \dot \theta = (G |B|_{,\psi} m(v_{||}^2/|B| + \mu) - (-q \iota + m v_{||} G' / |B|) v_{||} |B|)/(\iota D)
     *  \dot \zeta = \left((q + m v_{||} I'/|B|) v_{||} |B| - |B|_{,\psi} m(\rho_{||}^2 |B| + \mu) I\right)/(\iota D)
     *  \dot v_{||} = ((-q\iota + m v_{||} G'/|B|)|B|_{,\theta} - (q + m v_{||}I'/|B|)|B|_{,\zeta})\mu |B|/(\iota D)
     *  D = ((q + m v_{||} I'/|B|)*G - (-q \iota + m v_{||} G'/|B|) I)/\iota
     *
     *  where primes indicate differentiation wrt :math:`\psi`, :math:`q` is the charge,
     *  :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`. This corresponds
     *  with the limit K = 0.
     */
    private:
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q, mu;
    public:
        int axis;
        double vnorm, tnorm;
        static constexpr int Size = 4;

        GuidingCenterNoKBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis, double vnorm=1, double tnorm=1)
            : field(field), m(m), q(q), mu(mu), axis(axis), vnorm(vnorm), tnorm(tnorm) {
            }

        int get_state_size() const override {
            return Size;
        }

        void operator()(const vector<double> &ys, vector<double> &dydt, const double t) override {
            vector<double> stzv(Size), stzvdot(Size);
            y_to_stzvt(ys, stzv, axis, vnorm, tnorm);

            stz(0, 0) = stzv[0];
            stz(0, 1) = stzv[1];
            stz(0, 2) = stzv[2];
            double v_par = stzv[3];

            field->set_points(stz);
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double I = field->I_ref()(0);
            double dGdpsi = field->dGds_ref()(0)/psi0;
            double dIdpsi = field->dIds_ref()(0)/psi0;
            double iota = field->iota_ref()(0);
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double D = ((q + m*v_par*dIdpsi/modB)*G - (-q*iota + m*v_par*dGdpsi/modB)*I)/iota;
            double F = (q + m*v_par*dIdpsi/modB);
            double C = (-q*iota + m*v_par*dGdpsi/modB);

            stzvdot[0] = (I*dmodBdzeta - G*dmodBdtheta)*fak1/(D*iota*psi0);
            stzvdot[1] = (G*dmodBdpsi*fak1 - (-q*iota + m*v_par*dGdpsi/modB)*v_par*modB)/(D*iota);
            stzvdot[2] = ((q + m*v_par*dIdpsi/modB)*v_par*modB - dmodBdpsi*fak1*I)/(D*iota);
            stzvdot[3] = modB*mu*(dmodBdtheta*C - dmodBdzeta*F)/(F*G-C*I);

            stzvtdot_to_ydot(stzvdot, stzv, dydt, axis, vnorm, tnorm);
        }
};

class GuidingCenterBoozerRHS : public BaseRHS {
    /*
     * The state consists of :math:`[s, t, z, v_par]` with
     *
     *  \dot s = (I |B|_{,\zeta} - G |B|_{,\theta})m(v_{||}^2/|B| + \mu)/(\iota D \psi_0)
     *  \dot \theta = ((G |B|_{,\psi} - K |B|_{,\zeta}) m(v_{||}^2/|B| + \mu) - C v_{||} |B|)/(\iota D)
     *  \dot \zeta = (F v_{||} |B| - (|B|_{,\psi} I - |B|_{,\theta} K) m(\rho_{||}^2 |B| + \mu) )/(\iota D)
     *  \dot v_{||} = (C|B|_{,\theta} - F|B|_{,\zeta})\mu |B|/(\iota D)
     *  C = - m v_{||} K_{,\zeta}/|B|  - q \iota + m v_{||}G'/|B|
     *  F = - m v_{||} K_{,\theta}/|B| + q + m v_{||}I'/|B|
     *  D = (F G - C I))/\iota
     *
     *  where primes indicate differentiation wrt :math:`\psi`, :math:`q` is the charge,
     *  :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     */
    private:
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q, mu;
    public:
        static constexpr int Size = 4;
        int axis;
        double vnorm, tnorm;

        GuidingCenterBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis, double vnorm=1, double tnorm=1)
            : field(field), m(m), q(q), mu(mu), axis(axis), vnorm(vnorm), tnorm(tnorm) {
            }

        int get_state_size() const override {
            return Size;
        }

        void operator()(const vector<double> &ys, vector<double> &dydt, const double t) override {
            vector<double> stzv(Size), stzvdot(Size);
            y_to_stzvt(ys, stzv, axis, vnorm, tnorm);

            stz(0, 0) = stzv[0];
            stz(0, 1) = stzv[1];
            stz(0, 2) = stzv[2];
            double v_par = stzv[3];

            field->set_points(stz);
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double K = field->K_ref()(0);
            auto K_derivs = field->K_derivs_ref();
            double dKdtheta = K_derivs(0);
            double dKdzeta = K_derivs(1);

            double G = field->G_ref()(0);
            double I = field->I_ref()(0);
            double dGdpsi = field->dGds_ref()(0)/psi0;
            double dIdpsi = field->dIds_ref()(0)/psi0;
            double iota = field->iota_ref()(0);
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu; // dHdB
            double C = -m*v_par*(dKdzeta-dGdpsi)/modB - q*iota;
            double F = -m*v_par*(dKdtheta-dIdpsi)/modB + q;
            double D = (F*G-C*I)/iota;

            stzvdot[0] = (I*dmodBdzeta - G*dmodBdtheta)*fak1/(D*iota*psi0);
            stzvdot[1] = (G*dmodBdpsi*fak1 - C*v_par*modB - K*fak1*dmodBdzeta)/(D*iota);
            stzvdot[2] = (F*v_par*modB - dmodBdpsi*fak1*I + K*fak1*dmodBdtheta)/(D*iota);
            // dydt[3] = - (mu / v_par) * (dmodBdpsi * sdot * psi0 + dmodBdtheta * tdot + dmodBdzeta * dydt[2]);
            stzvdot[3] = modB*mu*(dmodBdtheta*C - dmodBdzeta*F)/(F*G-C*I);

            stzvtdot_to_ydot(stzvdot, stzv, dydt, axis, vnorm, tnorm);
        }
};

tuple<vector<vector<double>>, vector<vector<double>>>
solve(
    BaseRHS& rhs,
    vector<double> stzvt,
    double tau_max,
    double dtau,
    double dtau_max,
    double abstol,
    double reltol,
    vector<double> phases,
    vector<double> n_zetas,
    vector<double> m_thetas,
    vector<double> omegas,
    vector<shared_ptr<StoppingCriterion>> stopping_criteria,
    double dtau_save,
    vector<double> vpars,
    bool phases_stop,
    bool vpars_stop,
    bool forget_exact_path,
    int axis,
    double vnorm,
    double tnorm,
    string ode_solver,
    double DP_hmin) {
    if (phases.size() != n_zetas.size() || phases.size() != m_thetas.size() || phases.size() != omegas.size()) {
        throw std::invalid_argument("phases, n_zetas, m_thetas, and omegas need to have matching length.");
    }

    int state_size = rhs.get_state_size();

    vector<vector<double>> res = {};
    vector<vector<double>> res_hits = {};
    vector<double> y(state_size), temp(state_size);

    std::unique_ptr<ODESolver> solver;
    if (ode_solver == "dormand_prince") {
        solver = create_dormand_prince_solver(
            abstol,
            reltol,
            dtau_max,
            DP_hmin
        );
    } else if (ode_solver == "boost") {
        solver = create_dopri_boost_solver(
            abstol,
            reltol,
            dtau_max
        );
    } else {
        throw std::invalid_argument(
            "ode_solver is \"boost\", \"dormand_prince\" of \"symplectic\""
        );
    }

    double tau = 0;
    int iter = 0;
    bool stop = false;
    double tau_last = 0;
    double tau_current;
    tau_last = tau;

    // Save initial state
    vector<double> initial_state = {0};
    initial_state.insert(initial_state.end(), stzvt.begin(), stzvt.end());
    res.push_back(initial_state);

    stzvt_to_y(stzvt, y, axis, vnorm, tnorm);
    solver->initialize(y, tau, dtau, rhs);

    do {
        tuple<double, double> step = solver->do_step(rhs);
        iter++;
        tau = solver->current_time();
        y = solver->current_state();
        tau_last = std::get<0>(step);
        tau_current = std::get<1>(step);
        dtau = tau_current - tau_last;

        // Check if we have hit a stopping criterion between tau_last and tau_current
        stop = check_stopping_criteria(
            state_size,
            iter,
            res_hits,
            *solver,
            tau_last,
            tau_current,
            dtau,
            abstol,
            phases,
            n_zetas,
            m_thetas,
            omegas,
            stopping_criteria,
            vpars,
            phases_stop,
            vpars_stop,
            axis,
            vnorm,
            tnorm
        );

        // Save path if forget_exact_path = False
        if (forget_exact_path == 0 && !stop) {
            // This will give the first save point after tau_last
            double tau_save_last = (std::floor(tau_last/dtau_save) + 1) * dtau_save;
            for (double tau_save = tau_save_last; tau_save <= std::min(tau_current, tau_max); tau_save += dtau_save) {
                if (tau_save != 0) {  // tau = 0 is already saved.
                    solver->calc_state(tau_save, temp);
                    double t_save = tau_save * tnorm;
                    y_to_stzvt(temp, stzvt, axis, vnorm, tnorm);
                    vector<double> save_state = {t_save};
                    save_state.insert(save_state.end(), stzvt.begin(), stzvt.end());
                    res.push_back(save_state);
                }
            }
        }
    } while(tau < tau_max && !stop);

    // Save t = tmax or time when StoppingCriterion is hit if we not already saved it
    if (stop) {
        tau_max = tau_last;
    }
    double t_max = tau_max * tnorm;
    if (t_max - res.back()[0] > 1e-15) {
        solver->calc_state(tau_max, y);
        y_to_stzvt(y, stzvt, axis, vnorm, tnorm);
        vector<double> final_state = {t_max};
        final_state.insert(final_state.end(), stzvt.begin(), stzvt.end());
        res.push_back(final_state);
    }

    return std::make_tuple(res, res_hits);
}



/**
See trace_particles_boozer() defined in tracing.py for details on the parameters.
**/
tuple<vector<vector<double>>, vector<vector<double>>>
particle_guiding_center_boozer_perturbed_tracing(
        shared_ptr<ShearAlfvenWave> perturbed_field,
        vector<double> stz_init,
        double m,
        double q,
        double vtotal,
        double vtang,
        double mu,
        double tmax,
        double abstol,
        double reltol,
        bool vacuum,
        bool noK,
        vector<double> phases,
        vector<double> n_zetas,
        vector<double> m_thetas,
        vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        double dt_save,
        bool phases_stop,
        bool vpars_stop,
        bool forget_exact_path,
        int axis,
        vector<double> vpars,
        string ode_solver,
        double DP_hmin
        )
{
    if (ode_solver != "boost" && ode_solver != "dormand_prince") {
        throw std::invalid_argument("ode_solver must be either \"boost\" or \"dormand_prince\" for perturbed tracing");
    }
    Array2 stzt({{stz_init[0], stz_init[1], stz_init[2], 0.0}});
    perturbed_field->set_points(stzt);
    auto field = perturbed_field->get_B0();
    double modB = field->modB()(0);
    vector<double> stzvt(5);
    double G0 = std::abs(field->G()(0));
    double r0 = G0/modB;
    double vnorm = vtotal; // Normalizing velocity = vtotal
    double tnorm = r0*2*M_PI/vtotal; // Normalizing time = time for one toroidal revolution
    double dtau, dtau_max;
    dtau_max = 0.25; // can at most do quarter of a revolution per step
    dtau = 1e-3 * dtau_max; // initial guess for first timestep, will be adjusted by adaptive timestepper

    if (dtau<0) {
        throw std::invalid_argument("dtau needs to be positive.");
    }

    // Normalize tmax and dt_save
    double tau_max = tmax / tnorm;
    double dtau_save = dt_save / tnorm;

    // Initial conditions are passed as (s, theta, zeta, v_par, t)
    // While, tracing is done in mapped coordinates y
    stzvt[0] = stz_init[0];
    stzvt[1] = stz_init[1];
    stzvt[2] = stz_init[2];
    stzvt[3] = vtang;
    stzvt[4] = 0;

    if (vacuum) {
      auto rhs_class = GuidingCenterVacuumBoozerPerturbedRHS(
          perturbed_field, m, q, mu, axis, vnorm, tnorm
      );
      return solve(
        rhs_class,
        stzvt,
        tau_max,
        dtau,
        dtau_max,
        abstol,
        reltol,
        phases,
        n_zetas,
        m_thetas,
        omegas,
        stopping_criteria,
        dtau_save,
        vpars,
        phases_stop,
        vpars_stop,
        forget_exact_path,
        axis,
        vnorm,
        tnorm,
        ode_solver,
        DP_hmin
      );
  } else {
      auto rhs_class = GuidingCenterNoKBoozerPerturbedRHS(
          perturbed_field, m, q, mu, axis, vnorm, tnorm
      );
      return solve(
        rhs_class,
        stzvt,
        tau_max,
        dtau,
        dtau_max,
        abstol,
        reltol,
        phases,
        n_zetas,
        m_thetas,
        omegas,
        stopping_criteria,
        dtau_save,
        vpars,
        phases_stop,
        vpars_stop,
        forget_exact_path,
        axis,
        vnorm,
        tnorm,
        ode_solver,
        DP_hmin
      );
  }
}

/**
See trace_particles_boozer() defined in tracing.py for details on the parameters.
**/
tuple<vector<vector<double>>, vector<vector<double>>>
particle_guiding_center_boozer_tracing(
        shared_ptr<BoozerMagneticField> field,
        vector<double> stz_init,
        double m,
        double q,
        double vtotal,
        double vtang,
        double tmax,
        bool vacuum,
        bool noK,
        vector<double> phases,
        vector<double> n_zetas,
        vector<double> m_thetas,
        vector<double> omegas,
        vector<double> vpars,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        double dt_save,
        bool forget_exact_path,
        bool phases_stop,
        bool vpars_stop,
        int axis,
        double abstol,
        double reltol,
        string ode_solver,
        bool predictor_step,
        double roottol,
        double dt,
        double DP_hmin
        )
{
    if (ode_solver != "boost" && ode_solver != "dormand_prince" && ode_solver != "symplectic") {
        throw std::invalid_argument("ode_solver must be either \"boost\", \"dormand_prince\", or \"symplectic\" for unperturbed tracing");
    }
    Array2 stz({{stz_init[0], stz_init[1], stz_init[2]}});
    field->set_points(stz);
    double modB = field->modB()(0);
    double vperp2 = vtotal*vtotal - vtang*vtang;
    double mu = vperp2/(2*modB);
    vector<double> stzv(4);
    double vnorm, tnorm, dtau_max, dtau;

    double G0 = std::abs(field->G()(0));
    double r0 = G0/modB;
    vnorm = vtotal; // Normalizing velocity = vtotal
    tnorm = r0*2*M_PI/vtotal; // Normalizing time = time for one toroidal revolution
    if (ode_solver == "symplectic") {
        dtau = dt / tnorm;
    } else {
        dtau_max = 0.25; // can at most do quarter of a revolution per step
        dtau = 1e-3 * dtau_max; // initial guess for first timestep, will be adjusted by adaptive timestepper
    }
    if (dtau<0) {
        throw std::invalid_argument("dtau needs to be positive.");
    }
    // Normalize tmax and dt_save
    double tau_max = tmax / tnorm;
    double dtau_save = dt_save / tnorm;

    stzv[0] = stz_init[0];
    stzv[1] = stz_init[1];
    stzv[2] = stz_init[2];
    stzv[3] = vtang;

    if (ode_solver == "symplectic") {
#ifdef USE_GSL
        auto f = SymplField(field, m, q, mu, vnorm, tnorm);
        return solve_sympl_vector(
            f,
            stzv,
            tau_max,
            dtau,
            roottol,
            phases,
            n_zetas,
            m_thetas,
            omegas,
            stopping_criteria,
            vpars,
            phases_stop,
            vpars_stop,
            forget_exact_path,
            predictor_step,
            dtau_save
        );
#else
        throw std::invalid_argument("Symplectic solver not available. Please recompile with GSL support.");
#endif
    } else {
        if (vacuum) {
          auto rhs_class = GuidingCenterVacuumBoozerRHS(field, m, q, mu, axis, vnorm, tnorm);
          return solve(
              rhs_class,
              stzv,
              tau_max,
              dtau,
              dtau_max,
              abstol,
              reltol,
              phases,
              n_zetas,
              m_thetas,
              omegas,
              stopping_criteria,
              dtau_save,
              vpars,
              phases_stop,
              vpars_stop,
              forget_exact_path,
              axis,
              vnorm,
              tnorm,
              ode_solver,
              DP_hmin
          );
        } else if (noK) {
          auto rhs_class = GuidingCenterNoKBoozerRHS(field, m, q, mu, axis, vnorm, tnorm);
          return solve(
              rhs_class,
              stzv,
              tau_max,
              dtau,
              dtau_max,
              abstol,
              reltol,
              phases,
              n_zetas,
              m_thetas,
              omegas,
              stopping_criteria,
              dtau_save,
              vpars,
              phases_stop,
              vpars_stop,
              forget_exact_path,
              axis,
              vnorm,
              tnorm,
              ode_solver,
              DP_hmin
          );
        } else {
          auto rhs_class = GuidingCenterBoozerRHS(field, m, q, mu, axis, vnorm, tnorm);
          return solve(
              rhs_class,
              stzv,
              tau_max,
              dtau,
              dtau_max,
              abstol,
              reltol,
              phases,
              n_zetas,
              m_thetas,
              omegas,
              stopping_criteria,
              dtau_save,
              vpars,
              phases_stop,
              vpars_stop,
              forget_exact_path,
              axis,
              vnorm,
              tnorm,
              ode_solver,
              DP_hmin
          );
        }
    }
}
#ifdef USE_GSL
// Wrapper function to convert vector to array for symplectic solver
tuple<vector<vector<double>>, vector<vector<double>>>
solve_sympl_wrapper(
    SymplField f,
    vector<double> y,
    double tau_max,
    double dtau,
    double roottol,
    vector<double> phases,
    vector<double> n_zetas,
    vector<double> m_thetas,
    vector<double> omegas,
    vector<shared_ptr<StoppingCriterion>> stopping_criteria,
    vector<double> vpars,
    bool phases_stop=false,
    bool vpars_stop=false,
    bool forget_exact_path=false,
    bool predictor_step=true,
    double dtau_save=1e-6
) {
    // Call the vector-based symplectic solver directly
    return solve_sympl_vector(
        f,
        y,
        tau_max,
        dtau,
        roottol,
        phases,
        n_zetas,
        m_thetas,
        omegas,
        stopping_criteria,
        vpars,
        phases_stop,
        vpars_stop,
        forget_exact_path,
        predictor_step,
        dtau_save
    );
}
#endif


// ==========================================================================
// Collision integrator: GC equations in (v, xi) space
// ==========================================================================

// --------------------------------------------------------------------------
// RHS for vacuum GC equations with collision drift, state = [s,theta,zeta,v,xi]
// --------------------------------------------------------------------------
class GuidingCenterVacuumCollisionBoozerRHS : public BaseRHS {
    /*
     * State: [s, theta, zeta, v, xi]  where v = |v_total|, xi = v_par / v.
     *
     * Deterministic drift (orbit + collision drag):
     *   sdot   = -dBdtheta * m v^2(1+xi^2)/(2 B q psi0)
     *   thdot  =  dBds    * m v^2(1+xi^2)/(2 B q psi0) + iota*xi*v*B/G
     *   zetdot =  xi*v*B/G
     *   vdot   =  K(v,s)   [collision drag only; zero collisionlessly]
     *   xidot  = -(iota*dBdtheta + dBdzeta)*v*(1-xi^2)/(2G)  - xi*nu_D
     *
     * mu = v^2*(1-xi^2)/(2B) is recomputed at each evaluation.
     */
    private:
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q;
        vector<ThermalBackground> backgrounds;
        double m_a, q_a;
    public:
        int axis;
        double vnorm, tnorm;
        static constexpr int Size = 5;

        GuidingCenterVacuumCollisionBoozerRHS(
            shared_ptr<BoozerMagneticField> field,
            double m, double q,
            const vector<ThermalBackground>& backgrounds,
            int axis, double vnorm, double tnorm)
            : field(field), m(m), q(q), backgrounds(backgrounds),
              m_a(m), q_a(q), axis(axis), vnorm(vnorm), tnorm(tnorm) {}

        int get_state_size() const override { return Size; }

        void operator()(const vector<double>& ys, vector<double>& dydt, const double t) override {
            vector<double> stzv_xi(Size), dot(Size);
            y_to_stzv_xi(ys, stzv_xi, axis, vnorm);

            stz(0, 0) = stzv_xi[0];
            stz(0, 1) = stzv_xi[1];
            stz(0, 2) = stzv_xi[2];
            double v   = stzv_xi[3];
            double xi  = stzv_xi[4];

            field->set_points(stz);
            double psi0       = field->psi0;
            double modB       = field->modB_ref()(0);
            double G          = field->G_ref()(0);
            double iota       = field->iota_ref()(0);
            auto modB_derivs  = field->modB_derivs_ref();
            double dmodBds    = modB_derivs(0);
            double dmodBdtheta= modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);

            double s = stzv_xi[0];
            double fak1 = m * v * v * (1.0 + xi * xi) / (2.0 * modB);  // m*(vpar^2/B + mu)

            // Spatial orbit equations (vacuum)
            dot[0] = -dmodBdtheta * fak1 / (q * psi0);
            dot[1] =  dmodBds    * fak1 / (q * psi0) + iota * xi * v * modB / G;
            dot[2] =  xi * v * modB / G;

            // Mirror force on xi (orbit)
            dot[4] = -(iota * dmodBdtheta + dmodBdzeta) * v * (1.0 - xi * xi) / (2.0 * G);

            // Collision drift terms (deterministic)
            if (!backgrounds.empty() && v > 0.0) {
                auto coef = compute_collision_coefficients(v, s, m_a, q_a, backgrounds);
                dot[3]  = coef.K;           // vdot from drag
                dot[4] += -xi * coef.nu_D;  // xidot from deterministic pitch drift
            } else {
                dot[3] = 0.0;
            }

            stzv_xi_dot_to_ydot(dot, stzv_xi, dydt, axis, vnorm, tnorm);
        }
};

// --------------------------------------------------------------------------
// solve_sde: like solve() but applies Milstein noise after each accepted step.
// Output rows are [t, s, theta, zeta, v_par, v] (6 elements).
// --------------------------------------------------------------------------
tuple<vector<vector<double>>, vector<vector<double>>>
solve_sde(
    BaseRHS& rhs,
    const vector<ThermalBackground>& backgrounds,
    double m_a, double q_a,
    vector<double> stzv_xi_init,   // [s, theta, zeta, v, xi]
    double tau_max,
    double dtau,
    double dtau_max,
    double abstol,
    double reltol,
    vector<double> phases,
    vector<double> n_zetas,
    vector<double> m_thetas,
    vector<double> omegas,
    vector<shared_ptr<StoppingCriterion>> stopping_criteria,
    double dtau_save,
    vector<double> vpars,
    bool phases_stop,
    bool vpars_stop,
    bool forget_exact_path,
    int axis,
    double vnorm,
    double tnorm,
    string ode_solver,
    double DP_hmin,
    uint64_t rng_seed)
{
    const int state_size = 5;

    // Set up RNG
    std::mt19937_64 rng(rng_seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    vector<vector<double>> res;
    vector<vector<double>> res_hits;
    vector<double> y(state_size), stzv_xi(state_size);

    // Factory: only dormand_prince or boost for SDE
    std::unique_ptr<ODESolver> solver;
    if (ode_solver == "boost") {
        solver = create_dopri_boost_solver(abstol, reltol, dtau_max);
    } else {
        solver = create_dormand_prince_solver(abstol, reltol, dtau_max, DP_hmin);
    }

    // Helper: build a 6-element save record [t, s, theta, zeta, v_par, v]
    auto make_record = [&](double t_phys, const vector<double>& sv) -> vector<double> {
        // sv = [s, theta, zeta, v, xi]
        double v_par = sv[3] * sv[4];
        return {t_phys, sv[0], sv[1], sv[2], v_par, sv[3]};
    };

    // Save initial state
    stzv_xi_to_y(stzv_xi_init, y, axis, vnorm);
    res.push_back(make_record(0.0, stzv_xi_init));

    solver->initialize(y, 0.0, dtau, rhs);

    double tau = 0.0;
    bool stop = false;

    do {
        auto step = solver->do_step(rhs);
        double tau_last    = std::get<0>(step);
        double tau_current = std::get<1>(step);
        double h_taken     = tau_current - tau_last;  // normalised step
        tau = tau_current;

        // ---- Milstein noise applied to (v, xi) ----
        // Evaluate collision coefficients at post-step state
        {
            vector<double> y_now = solver->current_state();
            y_to_stzv_xi(y_now, stzv_xi, axis, vnorm);
            double v_now  = stzv_xi[3];
            double xi_now = stzv_xi[4];
            double s_now  = stzv_xi[0];

            if (!backgrounds.empty() && v_now > 0.0) {
                auto coef = compute_collision_coefficients(v_now, s_now, m_a, q_a, backgrounds);
                double h_phys = h_taken * tnorm;
                double sqrt_h = std::sqrt(h_phys);
                double dW_v  = normal_dist(rng) * sqrt_h;
                double dW_xi = normal_dist(rng) * sqrt_h;
                milstein_collision_step(v_now, xi_now, coef, h_phys, dW_v, dW_xi);
                // Write updated v and xi back into the solver state
                stzv_xi[3] = v_now;
                stzv_xi[4] = xi_now;
                stzv_xi_to_y(stzv_xi, y_now, axis, vnorm);
                // Force solver's internal state to the post-noise state.
                // DormandPrinceSolver and DopriBoostSolver both expose current_state_
                // through current_state(); we update via re-initializing for next step.
                // Since we only need the updated state for the *next* step's k1 (FSAL),
                // we re-initialize the solver at the current time with the noisy state.
                solver->initialize(y_now, tau_current, solver->get_hnext(), rhs);
            }
        }

        // ---- Check stopping criteria ----
        // Re-read state (may have been updated by noise step)
        {
            vector<double> y_check(state_size);
            solver->calc_state(tau_current, y_check);
            y_to_stzv_xi(y_check, stzv_xi, axis, vnorm);
            double t_current  = tau_current * tnorm;
            double s_current  = stzv_xi[0];
            double th_current = stzv_xi[1];
            double z_current  = stzv_xi[2];
            double vpar_current = stzv_xi[3] * stzv_xi[4];  // v * xi

            for (int i = 0; i < (int)stopping_criteria.size(); ++i) {
                if (stopping_criteria[i] && (*stopping_criteria[i])(
                        (int)(tau / (dtau_max * 1e-3)), h_taken * tnorm,
                        t_current, s_current, th_current, z_current, vpar_current)) {
                    stop = true;
                    vector<double> hit = {t_current, -1.0 - double(i)};
                    hit.push_back(s_current); hit.push_back(th_current);
                    hit.push_back(z_current); hit.push_back(vpar_current);
                    hit.push_back(stzv_xi[3]);  // v
                    res_hits.push_back(hit);
                    break;
                }
            }
        }

        // ---- Save path ----
        if (!forget_exact_path && !stop) {
            double tau_save_last = (std::floor(tau_last / dtau_save) + 1) * dtau_save;
            vector<double> y_save(state_size), sv_save(state_size);
            for (double tau_save = tau_save_last;
                 tau_save <= std::min(tau_current, tau_max);
                 tau_save += dtau_save) {
                if (tau_save != 0.0) {
                    solver->calc_state(tau_save, y_save);
                    y_to_stzv_xi(y_save, sv_save, axis, vnorm);
                    res.push_back(make_record(tau_save * tnorm, sv_save));
                }
            }
        }

    } while (tau < tau_max && !stop);

    // Save final state
    if (stop) tau_max = tau;
    double t_end = tau_max * tnorm;
    if (t_end - res.back()[0] > 1e-15) {
        vector<double> y_fin(state_size), sv_fin(state_size);
        solver->calc_state(tau_max, y_fin);
        y_to_stzv_xi(y_fin, sv_fin, axis, vnorm);
        res.push_back(make_record(t_end, sv_fin));
    }

    return std::make_tuple(res, res_hits);
}

// --------------------------------------------------------------------------
// Public entry point: collision tracing (vacuum GC only for now)
// --------------------------------------------------------------------------
tuple<vector<vector<double>>, vector<vector<double>>>
particle_guiding_center_boozer_collision_tracing(
    shared_ptr<BoozerMagneticField> field,
    vector<double> stz_init,      // [s, theta, zeta]
    double m,
    double q,
    double vtotal,                // initial total speed [m/s]
    double vtang,                 // initial v_par [m/s]
    double tmax,
    const vector<ThermalBackground>& backgrounds,
    vector<shared_ptr<StoppingCriterion>> stopping_criteria,
    double dt_save,
    bool forget_exact_path,
    int axis,
    double abstol,
    double reltol,
    string ode_solver,
    double DP_hmin,
    uint64_t rng_seed)
{
    if (ode_solver != "boost" && ode_solver != "dormand_prince")
        throw std::invalid_argument("collision tracing requires ode_solver \"boost\" or \"dormand_prince\"");

    Array2 stz({{stz_init[0], stz_init[1], stz_init[2]}});
    field->set_points(stz);
    double modB = field->modB()(0);

    double G0    = std::abs(field->G()(0));
    double r0    = G0 / modB;
    double vnorm = vtotal;
    double tnorm = r0 * 2.0 * M_PI / vtotal;

    double dtau_max = 0.25;
    double dtau     = 1e-3 * dtau_max;

    double tau_max  = tmax / tnorm;
    double dtau_save = dt_save / tnorm;

    // Initial xi = v_par / v_total; clamp to [-1, 1]
    double xi_init = vtang / vtotal;
    xi_init = std::max(-1.0, std::min(1.0, xi_init));

    vector<double> stzv_xi_init = {
        stz_init[0], stz_init[1], stz_init[2], vtotal, xi_init
    };

    auto rhs = GuidingCenterVacuumCollisionBoozerRHS(
        field, m, q, backgrounds, axis, vnorm, tnorm);

    return solve_sde(
        rhs,
        backgrounds,
        m, q,
        stzv_xi_init,
        tau_max,
        dtau,
        dtau_max,
        abstol,
        reltol,
        {}, {}, {}, {},  // phases/n_zetas/m_thetas/omegas (not used yet)
        stopping_criteria,
        dtau_save,
        {},              // vpars
        false, false,    // phases_stop, vpars_stop
        forget_exact_path,
        axis,
        vnorm,
        tnorm,
        ode_solver,
        DP_hmin,
        rng_seed
    );
}

// compute derivative for a single point, with vacuum switch
void particle_guiding_center_boozer_derivs(
        shared_ptr<BoozerMagneticField> field, array<double, 3> stz_init, vector<double>&  out,
        double m, double q, double vtotal, double vtang, bool vacuum)
{
    Array2 stz({{stz_init[0], stz_init[1], stz_init[2]}});
    field->set_points(stz);
    double modB = field->modB()(0);
    double vperp2 = vtotal*vtotal - vtang*vtang;
    double mu = vperp2/(2*modB);

    double s = stz_init[0];
    double t = stz_init[1];

    vector<double> y = {s*cos(t), s*sin(t), stz_init[2], vtang};
    if (vacuum) {
        auto rhs_class = GuidingCenterVacuumBoozerRHS(field, m, q, mu, 2);
        rhs_class(y, out, 0.0);
    } else {
        auto rhs_class = GuidingCenterBoozerRHS(field, m, q, mu, 2);
        rhs_class(y, out, 0.0);
    }
}

// Add vacuum argument to select correct RHS
vector<double> simsopt_derivs_boozer(shared_ptr<BoozerMagneticField> field, vector<double> loc, double m, double q, double vtotal, double vtang, bool vacuum){
    vector<double> out(4);
    array<double, 3> stz = {loc[0], loc[1], loc[2]};
    vector<double> derivs(4);
    particle_guiding_center_boozer_derivs(field, stz, derivs, m, q, vtotal, vtang, vacuum);
    for(int i=0; i<4; ++i){
        out[i] = derivs[i];
    }
    return out;
}

// compute derivative for a single point
void particle_guiding_center_saw_derivs(
        shared_ptr<ShearAlfvenWave> perturbed_field, array<double, 5> stz_init, vector<double>&  out,
        double m, double q, double vtotal, double vtang, double time, std::string rhs)
{
    Array2 stz({{stz_init[0], stz_init[1], stz_init[2]}});
    auto field = perturbed_field->get_B0();
    field->set_points(stz);
    double modB = field->modB()(0);
    double vperp2 = vtotal*vtotal - vtang*vtang;
    double mu = vperp2/(2*modB);

    double s = stz_init[0];
    double t = stz_init[1];

    vector<double> y = {s*cos(t), s*sin(t), stz_init[2], vtang, time};

    if(rhs == "vacuum_saw"){
        auto rhs_class = GuidingCenterVacuumBoozerPerturbedRHS(perturbed_field, m, q, mu, 2);
        rhs_class(y, out, time);
    } else if (rhs == "nok_saw"){
        auto rhs_class = GuidingCenterNoKBoozerPerturbedRHS(perturbed_field, m, q, mu, 2);
        rhs_class(y, out, time);
    }
}

vector<double> simsopt_derivs_saw(shared_ptr<ShearAlfvenWave> perturbed_field, vector<double> loc, double m, double q, double vtotal, double vtang, double time, std::string rhs){

    // py::buffer_info loc_buf = loc.request();
    // double* loc_arr = static_cast<double*>(loc_buf.ptr);

    vector<double> out(4);
    array<double, 5> stzvt = {loc[0], loc[1], loc[2], vtang, time};

    vector<double> derivs(5);

    particle_guiding_center_saw_derivs(perturbed_field, stzvt, derivs, m, q, vtotal, vtang, time, rhs);

    for(int i=0; i<4; ++i){
        out[i] = derivs[i];
    }

    // auto result = py::array_t<double>(4, out);
    return out;
}
