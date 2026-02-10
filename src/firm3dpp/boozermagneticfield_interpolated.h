#pragma once

#include "boozermagneticfield.h"
#include "xtensor/xlayout.hpp"
#include "regular_grid_interpolant_3d.h"
#include <string>
#include <nlohmann/json.hpp>  // JSON serialization for save/load
#include <fstream>
using std::string;

class InterpolatedBoozerField : public BoozerMagneticField {
    public:
        using typename BoozerMagneticField::Array2;
        bool status_modB = false, status_dmodBdtheta = false, status_dmodBdzeta = false, \
          status_dmodBds = false, status_G = false, status_I = false, status_iota = false,
          status_dGds = false, status_dIds = false, status_diotads = false, status_psip = false,
          status_R = false, status_Z = false, status_nu = false, status_K = false, \
          status_dRdtheta = false, status_dRdzeta = false, status_dRds = false, \
          status_dZdtheta = false, status_dZdzeta = false, status_dZds = false, \
          status_dnudtheta = false, status_dnudzeta = false, status_dnuds = false, \
          status_dKdtheta = false, status_dKdzeta = false, status_K_derivs = false, \
          status_R_derivs = false, status_Z_derivs = false, status_nu_derivs = false, \
          status_modB_derivs = false;
    private:
        shared_ptr<RegularGridInterpolant3D<Array2>> interp_modB, interp_dmodBdtheta, \
          interp_dmodBdzeta, interp_dmodBds, interp_G, interp_iota, interp_dGds, \
          interp_I, interp_dIds, interp_diotads, interp_psip, interp_R, interp_Z, \
          interp_nu, interp_K, interp_dRdtheta, interp_dRdzeta, interp_dRds, \
          interp_dZdtheta, interp_dZdzeta, interp_dZds, interp_dnudtheta, \
          interp_dnudzeta, interp_dnuds, interp_dKdtheta, interp_dKdzeta, interp_K_derivs, \
          interp_nu_derivs, interp_R_derivs, interp_Z_derivs, interp_modB_derivs;
        const bool extrapolate;
        const bool stellsym = false;
        const int nfp = 1;
        vector<bool> symmetries = vector<bool>(1, false);

    protected:
      void _psip_impl(Array2& psip) override {
          if(!interp_psip)
              interp_psip = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
          if(!status_psip) {
              Array2 old_points = this->field->get_points();
              string which_scalar = "psip";
              std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                return fbatch_scalar(s,theta,zeta,which_scalar);
              };
              interp_psip->interpolate_batch(fbatch);
              Array2 old_points_py(old_points);
              this->field->set_points(old_points_py);
              status_psip = true;
          }
          Array2 stz = this->get_points_ref();
          points_sym.resize({npoints, 3});
          Array2 stz0 = this->get_sym_points_ref();
          exploit_fluxfunction_points(stz, stz0);
          interp_psip->evaluate_batch(stz0, psip);
      }

        void _G_impl(Array2& G) override {
            if(!interp_G)
                interp_G = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_G) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "G";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_G->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_G = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz0 = this->get_sym_points_ref();
            exploit_fluxfunction_points(stz, stz0);
            interp_G->evaluate_batch(stz0, G);
        }

        void _I_impl(Array2& I) override {
            if(!interp_I)
                interp_I = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_I) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "I";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_I->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_I = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz0 = this->get_sym_points_ref();
            exploit_fluxfunction_points(stz, stz0);
            interp_I->evaluate_batch(stz0, I);
        }

        void _iota_impl(Array2& iota) override {
            if(!interp_iota)
                interp_iota = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_iota) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "iota";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_iota->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_iota = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz0 = this->get_sym_points_ref();
            exploit_fluxfunction_points(stz, stz0);
            interp_iota->evaluate_batch(stz0, iota);
        }

        void _dGds_impl(Array2& dGds) override {
            if(!interp_dGds)
                interp_dGds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_dGds) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dGds";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dGds->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dGds = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz0 = this->get_sym_points_ref();
            exploit_fluxfunction_points(stz, stz0);
            interp_dGds->evaluate_batch(stz0, dGds);
        }

        void _dIds_impl(Array2& dIds) override {
            if(!interp_dIds)
                interp_dIds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_dIds) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dIds";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dIds->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dIds = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz0 = this->get_sym_points_ref();
            exploit_fluxfunction_points(stz, stz0);
            interp_dIds->evaluate_batch(stz0, dIds);
        }

        void _diotads_impl(Array2& diotads) override {
            if(!interp_diotads)
                interp_diotads = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_diotads) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "diotads";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_diotads->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_diotads = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz0 = this->get_sym_points_ref();
            exploit_fluxfunction_points(stz, stz0);
            interp_diotads->evaluate_batch(stz0, diotads);
        }

        void _K_impl(Array2& K) override {
            if(!interp_K)
                interp_K = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_K) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "K";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_K->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_K = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz_sym = this->get_sym_points_ref();
            exploit_symmetries_points(stz, stz_sym);
            interp_K->evaluate_batch(stz_sym, K);
            if(stellsym){
              apply_odd_symmetry(K);
            }
        }

        void _dKdtheta_impl(Array2& dKdtheta) override {
            if(!interp_dKdtheta)
                interp_dKdtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dKdtheta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dKdtheta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dKdtheta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dKdtheta = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz_sym = this->get_sym_points_ref();
            exploit_symmetries_points(stz, stz_sym);
            interp_dKdtheta->evaluate_batch(stz_sym, dKdtheta);
        }

        void _dKdzeta_impl(Array2& dKdzeta) override {
            if(!interp_dKdzeta)
                interp_dKdzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dKdzeta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dKdzeta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dKdzeta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dKdzeta = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz_sym = this->get_sym_points_ref();
            exploit_symmetries_points(stz, stz_sym);
            interp_dKdzeta->evaluate_batch(stz_sym, dKdzeta);
        }

        void _K_derivs_impl(Array2& K_derivs) override {
            if(!interp_K_derivs)
                interp_K_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 2, extrapolate);
            if(!status_K_derivs) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "K_derivs";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_K_derivs->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_K_derivs = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz_sym = this->get_sym_points_ref();
            exploit_symmetries_points(stz, stz_sym);
            interp_K_derivs->evaluate_batch(stz_sym, K_derivs);
        }

        void _nu_impl(Array2& nu) override {
            if(!interp_nu)
                interp_nu = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_nu) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "nu";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_nu->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_nu = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz_sym = this->get_sym_points_ref();
            exploit_symmetries_points(stz, stz_sym);
            interp_nu->evaluate_batch(stz_sym, nu);
            if (stellsym) {
              apply_odd_symmetry(nu);
            }
        }

        void _dnudtheta_impl(Array2& dnudtheta) override {
            if(!interp_dnudtheta)
                interp_dnudtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dnudtheta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dnudtheta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dnudtheta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dnudtheta = true;
            }
            Array2 stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2 stz_sym = this->get_sym_points_ref();
            exploit_symmetries_points(stz, stz_sym);
            interp_dnudtheta->evaluate_batch(stz_sym, dnudtheta);
        }

        void _dnudzeta_impl(Array2& dnudzeta) override {
            if(!interp_dnudzeta)
                interp_dnudzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dnudzeta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dnudzeta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dnudzeta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dnudzeta = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dnudzeta->evaluate_batch(stz_sym, dnudzeta);
        }

        void _dnuds_impl(Array2& dnuds) override {
            if(!interp_dnuds)
                interp_dnuds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dnuds) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dnuds";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dnuds->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dnuds = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dnuds->evaluate_batch(stz_sym, dnuds);
            if (stellsym) {
              apply_odd_symmetry(dnuds);
            }
        }

        void _nu_derivs_impl(Array2& nu_derivs) override {
            if(!interp_nu_derivs)
                interp_nu_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            if(!status_nu_derivs) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "nu_derivs";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_nu_derivs->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_nu_derivs = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_nu_derivs->evaluate_batch(stz_sym, nu_derivs);
            if (stellsym) {
              apply_odd_symmetry(nu_derivs);
            }
        }

        void _R_impl(Array2& R) override {
            if(!interp_R)
                interp_R = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_R) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "R";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_R->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_R = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_R->evaluate_batch(stz_sym, R);
        }

        void _dRdtheta_impl(Array2& dRdtheta) override {
            if(!interp_dRdtheta)
                interp_dRdtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dRdtheta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dRdtheta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dRdtheta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dRdtheta = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dRdtheta->evaluate_batch(stz_sym, dRdtheta);
            if (stellsym) {
              apply_odd_symmetry(dRdtheta);
            }
        }

        void _dRdzeta_impl(Array2& dRdzeta) override {
            if(!interp_dRdzeta)
                interp_dRdzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dRdzeta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dRdzeta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dRdzeta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dRdzeta = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dRdzeta->evaluate_batch(stz_sym, dRdzeta);
            if (stellsym) {
              apply_odd_symmetry(dRdzeta);
            }
        }

        void _dRds_impl(Array2& dRds) override {
            if(!interp_dRds)
                interp_dRds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dRds) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dRds";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dRds->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dRds = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dRds->evaluate_batch(stz_sym, dRds);
        }

        void _R_derivs_impl(Array2& R_derivs) override {
            if(!interp_R_derivs)
                interp_R_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            if(!status_R_derivs) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "R_derivs";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_R_derivs->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_R_derivs = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_R_derivs->evaluate_batch(stz_sym, R_derivs);
            if (stellsym) {
              apply_even_symmetry(R_derivs);
            }
        }

        void _Z_impl(Array2& Z) override {
            if(!interp_Z)
                interp_Z = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_Z) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "Z";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_Z->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_Z = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_Z->evaluate_batch(stz_sym, Z);
            if (stellsym) {
              apply_odd_symmetry(Z);
            }
        }

        void _dZdtheta_impl(Array2& dZdtheta) override {
            if(!interp_dZdtheta)
                interp_dZdtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dZdtheta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dZdtheta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dZdtheta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dZdtheta = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dZdtheta->evaluate_batch(stz_sym, dZdtheta);
        }

        void _dZdzeta_impl(Array2& dZdzeta) override {
            if(!interp_dZdzeta)
                interp_dZdzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dZdzeta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dZdzeta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dZdzeta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dZdzeta = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dZdzeta->evaluate_batch(stz_sym, dZdzeta);
        }

        void _dZds_impl(Array2& dZds) override {
            if(!interp_dZds)
                interp_dZds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dZds) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dZds";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dZds->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dZds = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dZds->evaluate_batch(stz_sym, dZds);
            if (stellsym) {
              apply_odd_symmetry(dZds);
            }
        }

        void _Z_derivs_impl(Array2& Z_derivs) override {
            if(!interp_Z_derivs)
                interp_Z_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            if(!status_Z_derivs) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "Z_derivs";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_Z_derivs->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_Z_derivs = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_Z_derivs->evaluate_batch(stz_sym, Z_derivs);
            if (stellsym) {
              apply_odd_symmetry(Z_derivs);
            }
        }

        void _modB_impl(Array2& modB) override {
            if(!interp_modB)
                interp_modB = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_modB) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "modB";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_modB->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_modB = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_modB->evaluate_batch(stz_sym, modB);
        }

        void _dmodBdtheta_impl(Array2& dmodBdtheta) override {
            if(!interp_dmodBdtheta)
                interp_dmodBdtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dmodBdtheta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dmodBdtheta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dmodBdtheta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dmodBdtheta = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dmodBdtheta->evaluate_batch(stz_sym, dmodBdtheta);
            if (stellsym) {
              apply_odd_symmetry(dmodBdtheta);
            }
        }

        void _dmodBdzeta_impl(Array2& dmodBdzeta) override {
            if(!interp_dmodBdzeta)
                interp_dmodBdzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dmodBdzeta) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dmodBdzeta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dmodBdzeta->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dmodBdzeta = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dmodBdzeta->evaluate_batch(stz_sym, dmodBdzeta);
            if (stellsym) {
              apply_odd_symmetry(dmodBdzeta);
            }
        }

        void _dmodBds_impl(Array2& dmodBds) override {
            if(!interp_dmodBds)
                interp_dmodBds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dmodBds) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "dmodBds";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dmodBds->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_dmodBds = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_dmodBds->evaluate_batch(stz_sym, dmodBds);
        }

        void _modB_derivs_impl(Array2& modB_derivs) override {
            if(!interp_modB_derivs)
                interp_modB_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            if(!status_modB_derivs) {
                Array2 old_points = this->field->get_points();
                string which_scalar = "modB_derivs";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_modB_derivs->interpolate_batch(fbatch);
                Array2 old_points_py(old_points);
                this->field->set_points(old_points_py);
                status_modB_derivs = true;
            }
            Array2& stz = this->get_points_ref();
            points_sym.resize({npoints, 3});
            Array2& stz_sym = this->get_sym_points_ref();               exploit_symmetries_points(stz, stz_sym);
            interp_modB_derivs->evaluate_batch(stz_sym, modB_derivs);
            if (stellsym) {
              apply_even_symmetry(modB_derivs);
            }
        }

        void exploit_fluxfunction_points(Array2& stz, Array2& stz0){
            int npoints = stz.shape(0);
            double* dataptr = &(stz(0, 0));
            double* datasymptr = &(stz0(0, 0));
            for (int i = 0; i < npoints; ++i) {
                double s = dataptr[3*i+0];
                datasymptr[3*i+0] = s;
                datasymptr[3*i+1] = 0.;
                datasymptr[3*i+2] = 0.;
            }
        }

        void exploit_symmetries_points(Array2& stz, Array2& stz_sym){
            int npoints = stz.shape(0);
            if(symmetries.size() != npoints)
                symmetries = vector<bool>(npoints, false);
            double period = (2*M_PI)/nfp;
            double* dataptr = &(stz(0, 0));
            double* datasymptr = &(stz_sym(0, 0));
            for (int i = 0; i < npoints; ++i) {
                double s = dataptr[3*i+0];
                double theta = dataptr[3*i+1];
                double zeta = dataptr[3*i+2];
                // Restrict theta to [0,2 pi]
                int theta_mult = int(theta/(2*M_PI));
                theta = theta - theta_mult * 2*M_PI;
                if (theta < 0) {
                  theta = theta + 2*M_PI;
                }
                if (theta > 2*M_PI) {
                  theta = theta - 2*M_PI;
                }
                // Restrict zeta to [0,2 pi/nfp]
                int zeta_mult = int(zeta/period);
                zeta = zeta - zeta_mult * period;
                if (zeta < 0) {
                  zeta = zeta + period;
                }
                if (zeta > period) {
                  zeta = zeta - period;
                }
                assert(theta >= 0);
                assert(theta <= 2*M_PI);
                assert(zeta >= 0);
                assert(zeta <= period);
                if(theta > M_PI && stellsym) {
                    zeta = period-zeta;
                    theta = 2*M_PI-theta;
                    symmetries[i] = true;
                    assert(theta >= 0);
                    assert(theta <= M_PI);
                    assert(zeta >= 0);
                    assert(zeta <= period);
                } else{
                    symmetries[i] = false;
                }
                datasymptr[3*i+0] = s;
                datasymptr[3*i+1] = theta;
                datasymptr[3*i+2] = zeta;
            }
        }

        void apply_odd_symmetry(Array2& field){
            int npoints = field.shape(0);
            for (int i = 0; i < npoints; ++i) {
                if(symmetries[i]) {
                  if (field.shape(1)==1) {
                    field(i, 0) = -field(i, 0);
                  } else if (field.shape(1)==3) {
                    field(i, 0) = -field(i, 0);
                  }
                }
            }
        }

        void apply_even_symmetry(Array2& field){
            int npoints = field.shape(0);
            for (int i = 0; i < npoints; ++i) {
                if(symmetries[i] && field.shape(1)==3) {
                    field(i, 1) = -field(i, 1);
                    field(i, 2) = -field(i, 2);
                }
            }
        }

        Vec fbatch_scalar(Vec s, Vec theta, Vec zeta, string which_scalar) {
            int npoints = s.size();
            Array2 points = xt::zeros<double>({npoints, 3});
            for(int i=0; i<npoints; i++) {
                points(i, 0) = s[i];
                if ((which_scalar != "G") && (which_scalar != "I") && (which_scalar != "iota") && (which_scalar != "dGds") && (which_scalar != "dIds") && (which_scalar != "diotads")) {
                  points(i, 1) = theta[i];
                  points(i, 2) = zeta[i];
                }
            }
            Array2 points_py(points);
            this->field->set_points(points_py);
            Array2 scalar;
            if (which_scalar == "modB") {
              scalar = this->field->modB();
            } else if (which_scalar == "K") {
              scalar = this->field->K();
            } else if (which_scalar == "dKdtheta") {
              scalar = this->field->dKdtheta();
            } else if (which_scalar == "dKdzeta") {
              scalar = this->field->dKdzeta();
            } else if (which_scalar == "K_derivs") {
              scalar = this->field->K_derivs();
              npoints = 2*npoints;
            } else if (which_scalar == "nu") {
              scalar = this->field->nu();
            } else if (which_scalar == "dnudtheta") {
              scalar = this->field->dnudtheta();
            } else if (which_scalar == "dnudzeta") {
              scalar = this->field->dnudzeta();
            } else if (which_scalar == "dnuds") {
              scalar = this->field->dnuds();
            } else if (which_scalar == "nu_derivs") {
              scalar = this->field->nu_derivs();
              npoints = 3*npoints;
            } else if (which_scalar == "R") {
              scalar = this->field->R();
            } else if (which_scalar == "dRdtheta") {
              scalar = this->field->dRdtheta();
            } else if (which_scalar == "dRdzeta") {
              scalar = this->field->dRdzeta();
            } else if (which_scalar == "dRds") {
              scalar = this->field->dRds();
            } else if (which_scalar == "R_derivs") {
              scalar = this->field->R_derivs();
              npoints = 3*npoints;
            } else if (which_scalar == "Z") {
              scalar = this->field->Z();
            } else if (which_scalar == "dZdtheta") {
              scalar = this->field->dZdtheta();
            } else if (which_scalar == "dZdzeta") {
              scalar = this->field->dZdzeta();
            } else if (which_scalar == "dZds") {
              scalar = this->field->dZds();
            } else if (which_scalar == "Z_derivs") {
              scalar = this->field->Z_derivs();
              npoints = 3*npoints;
            } else if (which_scalar == "dmodBdtheta") {
              scalar = this->field->dmodBdtheta();
            } else if (which_scalar == "dmodBdzeta") {
              scalar = this->field->dmodBdzeta();
            } else if (which_scalar == "dmodBds") {
              scalar = this->field->dmodBds();
            } else if (which_scalar == "modB_derivs") {
              scalar = this->field->modB_derivs();
              npoints = 3*npoints;
            } else if (which_scalar == "G") {
              scalar = this->field->G();
            } else if (which_scalar == "I") {
              scalar = this->field->I();
            } else if (which_scalar == "psip") {
              scalar = this->field->psip();
            } else if (which_scalar == "iota") {
              scalar = this->field->iota();
            } else if (which_scalar == "dGds") {
              scalar = this->field->dGds();
            } else if (which_scalar == "dIds") {
              scalar = this->field->dIds();
            } else if (which_scalar == "diotads") {
              scalar = this->field->diotads();
            } else {
              throw std::runtime_error("Incorrect value for which_scalar.");
            }
            return Vec(scalar.data(), scalar.data()+npoints);
        }

    public:
        const shared_ptr<BoozerMagneticField> field;
        const RangeTriplet s_range, theta_range, zeta_range, angle0_range = {0., M_PI, 1};
        using BoozerMagneticField::npoints;
        const InterpolationRule rule;

        InterpolatedBoozerField(
                shared_ptr<BoozerMagneticField> field, InterpolationRule rule,
                RangeTriplet s_range, RangeTriplet theta_range, RangeTriplet zeta_range,
                bool extrapolate, int nfp, bool stellsym, string field_type) :
            BoozerMagneticField(field->psi0, field->field_type), field(field), rule(rule), s_range(s_range), theta_range(theta_range), zeta_range(zeta_range), extrapolate(extrapolate), nfp(nfp), stellsym(stellsym)
        {}

        InterpolatedBoozerField(
                shared_ptr<BoozerMagneticField> field, int degree,
                RangeTriplet s_range, RangeTriplet theta_range, RangeTriplet zeta_range,
                bool extrapolate, int nfp, bool stellsym, string field_type) : InterpolatedBoozerField(field, UniformInterpolationRule(degree), s_range, theta_range, zeta_range, extrapolate, nfp, stellsym, field_type) {}

                std::pair<double, double> estimate_error_modB(int samples) {
                    if(!interp_modB) {
                      interp_modB = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"modB");
                    };
                    if(!status_modB) {
                        Array2 old_points = this->field->get_points();
                        interp_modB->interpolate_batch(fbatch);
                        Array2 old_points_py(old_points);
                        this->field->set_points(old_points_py);
                        status_modB = true;
                    }
                    return interp_modB->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_K(int samples) {
                    if(!interp_K) {
                      interp_K = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"K");
                    };
                    if(!status_K) {
                        Array2 old_points = this->field->get_points();
                        interp_K->interpolate_batch(fbatch);
                        Array2 old_points_py(old_points);
                        this->field->set_points(old_points_py);
                        status_K = true;
                    }
                    return interp_K->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_R(int samples) {
                    if(!interp_R) {
                      interp_R = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"R");
                    };
                    if(!status_R) {
                        Array2 old_points = this->field->get_points();
                        interp_R->interpolate_batch(fbatch);
                        Array2 old_points_py(old_points);
                        this->field->set_points(old_points_py);
                        status_R = true;
                    }
                    return interp_R->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_Z(int samples) {
                    if(!interp_Z) {
                      interp_Z = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"Z");
                    };
                    if(!status_Z) {
                        Array2 old_points = this->field->get_points();
                        interp_Z->interpolate_batch(fbatch);
                        Array2 old_points_py(old_points);
                        this->field->set_points(old_points_py);
                        status_Z = true;
                    }
                    return interp_Z->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_nu(int samples) {
                    if(!interp_nu) {
                      interp_nu = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"nu");
                    };
                    if(!status_nu) {
                        Array2 old_points = this->field->get_points();
                        interp_nu->interpolate_batch(fbatch);
                        Array2 old_points_py(old_points);
                        this->field->set_points(old_points_py);
                        status_nu = true;
                    }
                    return interp_nu->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_G(int samples) {
                    if(!interp_G) {
                      interp_G = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"G");
                    };
                    if(!status_G) {
                        Array2 old_points = this->field->get_points();
                        interp_G->interpolate_batch(fbatch);
                        Array2 old_points_py(old_points);
                        this->field->set_points(old_points_py);
                        status_G = true;
                    }
                    return interp_G->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_I(int samples) {
                    if(!interp_I) {
                      interp_I = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"I");
                    };
                    if(!status_I) {
                        Array2 old_points = this->field->get_points();
                        interp_I->interpolate_batch(fbatch);
                        Array2 old_points_py(old_points);
                        this->field->set_points(old_points_py);
                        status_I = true;
                    }
                    return interp_I->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_iota(int samples) {
                    if(!interp_iota) {
                      interp_iota = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"iota");
                    };
                    if(!status_iota) {
                        Array2 old_points = this->field->get_points();
                        interp_iota->interpolate_batch(fbatch);
                        Array2 old_points_py(old_points);
                        this->field->set_points(old_points_py);
                        status_iota = true;
                    }
                    return interp_iota->estimate_error(fbatch, samples);
                }

        /**
         * JSON Save/Load for InterpolatedBoozerField
         * 
         * Computing interpolants can take 10+ minutes for large grids. This save/load
         * mechanism stores pre-computed data to JSON for instant reload:
         * 
         *   field.to_json("field.json");                          // save
         *   auto loaded = InterpolatedBoozerField("field.json");  // load
         * 
         * The JSON file contains field parameters (psi0, nfp, grid ranges, etc.) and
         * the vals array from each computed interpolant. When loading, status flags
         * are set to true for each quantity, which prevents recomputation: the
         * _*_impl() methods check if(!status_X) before calling interpolate_batch().
         */
        InterpolatedBoozerField(const std::string& json_path) :
            BoozerMagneticField(0.0, ""),
            field(nullptr),
            rule(UniformInterpolationRule(1)),
            s_range({0,1,1}), theta_range({0,1,1}), zeta_range({0,1,1}),
            extrapolate(true)
        {
            std::ifstream file(json_path);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open JSON file: " + json_path);
            }
            nlohmann::json j;
            file >> j;
            file.close();

            // Require keys written by to_json(); throw clear error if any are missing
            const std::vector<std::string> required_keys = {
                "psi0", "field_type", "nfp", "stellsym", "extrapolate",
                "s_range", "theta_range", "zeta_range", "degree"
            };
            for (const auto& key : required_keys) {
                if (!j.contains(key)) {
                    throw std::runtime_error("JSON missing required key: " + key);
                }
            }

            const_cast<double&>(psi0) = j["psi0"].get<double>();
            const_cast<string&>(field_type) = j["field_type"].get<string>();
            const_cast<int&>(nfp) = j["nfp"].get<int>();
            const_cast<bool&>(stellsym) = j["stellsym"].get<bool>();
            const_cast<bool&>(extrapolate) = j["extrapolate"].get<bool>();

            auto s_vec = j["s_range"].get<std::vector<double>>();
            auto theta_vec = j["theta_range"].get<std::vector<double>>();
            auto zeta_vec = j["zeta_range"].get<std::vector<double>>();
            const_cast<RangeTriplet&>(s_range) = {s_vec[0], s_vec[1], static_cast<int>(s_vec[2])};
            const_cast<RangeTriplet&>(theta_range) = {theta_vec[0], theta_vec[1], static_cast<int>(theta_vec[2])};
            const_cast<RangeTriplet&>(zeta_range) = {zeta_vec[0], zeta_vec[1], static_cast<int>(zeta_vec[2])};

            int degree = j["degree"].get<int>();
            new (const_cast<InterpolationRule*>(&rule)) UniformInterpolationRule(degree);

            if (j.contains("interpolants")) {
                auto& interps = j["interpolants"];
                for (auto& [name, data] : interps.items()) {
                    load_single_interpolant(name, data);
                }
            }
        }
        
        /// Save all computed interpolants to a JSON file for later loading.
        void to_json(const std::string& json_path) const {
            nlohmann::json j;
            
            j["psi0"] = psi0;
            j["field_type"] = field_type;
            j["nfp"] = nfp;
            j["stellsym"] = stellsym;
            j["extrapolate"] = extrapolate;
            j["degree"] = rule.degree;
            j["s_range"] = {std::get<0>(s_range), std::get<1>(s_range), std::get<2>(s_range)};
            j["theta_range"] = {std::get<0>(theta_range), std::get<1>(theta_range), std::get<2>(theta_range)};
            j["zeta_range"] = {std::get<0>(zeta_range), std::get<1>(zeta_range), std::get<2>(zeta_range)};
            
            j["interpolants"] = nlohmann::json::object();
            save_interpolant_if_computed(j, "modB", interp_modB, status_modB);
            save_interpolant_if_computed(j, "dmodBdtheta", interp_dmodBdtheta, status_dmodBdtheta);
            save_interpolant_if_computed(j, "dmodBdzeta", interp_dmodBdzeta, status_dmodBdzeta);
            save_interpolant_if_computed(j, "dmodBds", interp_dmodBds, status_dmodBds);
            save_interpolant_if_computed(j, "modB_derivs", interp_modB_derivs, status_modB_derivs);
            save_interpolant_if_computed(j, "G", interp_G, status_G);
            save_interpolant_if_computed(j, "I", interp_I, status_I);
            save_interpolant_if_computed(j, "iota", interp_iota, status_iota);
            save_interpolant_if_computed(j, "dGds", interp_dGds, status_dGds);
            save_interpolant_if_computed(j, "dIds", interp_dIds, status_dIds);
            save_interpolant_if_computed(j, "diotads", interp_diotads, status_diotads);
            save_interpolant_if_computed(j, "psip", interp_psip, status_psip);
            save_interpolant_if_computed(j, "K", interp_K, status_K);
            save_interpolant_if_computed(j, "dKdtheta", interp_dKdtheta, status_dKdtheta);
            save_interpolant_if_computed(j, "dKdzeta", interp_dKdzeta, status_dKdzeta);
            save_interpolant_if_computed(j, "K_derivs", interp_K_derivs, status_K_derivs);
            save_interpolant_if_computed(j, "nu", interp_nu, status_nu);
            save_interpolant_if_computed(j, "dnudtheta", interp_dnudtheta, status_dnudtheta);
            save_interpolant_if_computed(j, "dnudzeta", interp_dnudzeta, status_dnudzeta);
            save_interpolant_if_computed(j, "dnuds", interp_dnuds, status_dnuds);
            save_interpolant_if_computed(j, "nu_derivs", interp_nu_derivs, status_nu_derivs);
            save_interpolant_if_computed(j, "R", interp_R, status_R);
            save_interpolant_if_computed(j, "dRdtheta", interp_dRdtheta, status_dRdtheta);
            save_interpolant_if_computed(j, "dRdzeta", interp_dRdzeta, status_dRdzeta);
            save_interpolant_if_computed(j, "dRds", interp_dRds, status_dRds);
            save_interpolant_if_computed(j, "R_derivs", interp_R_derivs, status_R_derivs);
            save_interpolant_if_computed(j, "Z", interp_Z, status_Z);
            save_interpolant_if_computed(j, "dZdtheta", interp_dZdtheta, status_dZdtheta);
            save_interpolant_if_computed(j, "dZdzeta", interp_dZdzeta, status_dZdzeta);
            save_interpolant_if_computed(j, "dZds", interp_dZds, status_dZds);
            save_interpolant_if_computed(j, "Z_derivs", interp_Z_derivs, status_Z_derivs);
            
            std::ofstream file(json_path);
            const int json_indent = 2;  // spaces per indent level so that JSON is readable
            file << j.dump(json_indent);
            file.close();
        }

        // Getters for Python from_json(): nfp, stellsym, extrapolate are private
        // so pybind11 cannot expose them with def_readonly.
        int get_nfp() const { return nfp; }
        bool get_stellsym() const { return stellsym; }
        bool get_extrapolate() const { return extrapolate; }
        double get_psi0() const { return psi0; }
        string get_field_type() const { return field_type; }

    private:
        // Serialization helpers for to_json() and the JSON constructor.
        // save_interpolant_if_computed: exports one interpolant's data if status flag is true.
        // load_single_interpolant: creates interpolant from JSON, sets status flag to prevent recomputation.
        
        void save_interpolant_if_computed(nlohmann::json& j, const std::string& name,
                const shared_ptr<RegularGridInterpolant3D<Array2>>& interp, bool is_computed) const {
            // In this class is_computed and interp are always set together, so is_computed true
            // implies interp non-null. We check interp anyway to avoid null dereference
            // if that invariant is ever broken. interp is a std::shared_ptr; the standard
            // defines explicit operator bool() for it (true when non-null), so in
            // "if (interp)" the compiler converts the pointer to bool, not a function call.
            if (is_computed && interp) {
                auto data = interp->get_interpolant_data();
                for (const auto& [key, vec] : data) {
                    j["interpolants"][name][key] = vec;
                }
            }
        }
        
        void load_single_interpolant(const std::string& name, const nlohmann::json& data) {
            std::map<std::string, std::vector<double>> data_map;
            for (auto& [key, value] : data.items()) {
                if (value.is_array()) {
                    data_map[key] = value.get<std::vector<double>>();
                } else if (value.is_number()) {
                    data_map[key] = {value.get<double>()}; 
                    //.get<double> from nlohmann library safely converts
                    // acceptable types to double. But just in case,
                    // we provide explicity numeric check
                } else {
                    throw std::runtime_error("Interpolant data key '" + key + "': expected array or number, got non-numeric JSON type");
                }
            }
            
            // Flux functions (G, I, iota, psip, etc.) depend only on s, so they use
            // a degenerate grid (angle0_range). Other quantities use the full 3D grid.
            bool is_flux_function = (name == "G" || name == "I" || name == "iota" ||
                                     name == "dGds" || name == "dIds" || name == "diotads" ||
                                     name == "psip");
            RangeTriplet y_range = is_flux_function ? angle0_range : theta_range;
            RangeTriplet z_range = is_flux_function ? angle0_range : zeta_range;
            
            int value_size = 1;
            if (name == "K_derivs") value_size = 2;
            else if (name.find("_derivs") != std::string::npos) value_size = 3;
            
            auto interp = std::make_shared<RegularGridInterpolant3D<Array2>>(
                rule, s_range, y_range, z_range, value_size, extrapolate);
            interp->set_interpolant_data(data_map);
            
            // Set status_X = true (same meaning as is_computed) so _*_impl() skips interpolate_batch().
            // Member names stay status_* to match the rest of the class (e.g. if(!status_modB)).
            if (name == "modB") { interp_modB = interp; status_modB = true; }
            else if (name == "dmodBdtheta") { interp_dmodBdtheta = interp; status_dmodBdtheta = true; }
            else if (name == "dmodBdzeta") { interp_dmodBdzeta = interp; status_dmodBdzeta = true; }
            else if (name == "dmodBds") { interp_dmodBds = interp; status_dmodBds = true; }
            else if (name == "modB_derivs") { interp_modB_derivs = interp; status_modB_derivs = true; }
            else if (name == "G") { interp_G = interp; status_G = true; }
            else if (name == "I") { interp_I = interp; status_I = true; }
            else if (name == "iota") { interp_iota = interp; status_iota = true; }
            else if (name == "dGds") { interp_dGds = interp; status_dGds = true; }
            else if (name == "dIds") { interp_dIds = interp; status_dIds = true; }
            else if (name == "diotads") { interp_diotads = interp; status_diotads = true; }
            else if (name == "psip") { interp_psip = interp; status_psip = true; }
            else if (name == "K") { interp_K = interp; status_K = true; }
            else if (name == "dKdtheta") { interp_dKdtheta = interp; status_dKdtheta = true; }
            else if (name == "dKdzeta") { interp_dKdzeta = interp; status_dKdzeta = true; }
            else if (name == "K_derivs") { interp_K_derivs = interp; status_K_derivs = true; }
            else if (name == "nu") { interp_nu = interp; status_nu = true; }
            else if (name == "dnudtheta") { interp_dnudtheta = interp; status_dnudtheta = true; }
            else if (name == "dnudzeta") { interp_dnudzeta = interp; status_dnudzeta = true; }
            else if (name == "dnuds") { interp_dnuds = interp; status_dnuds = true; }
            else if (name == "nu_derivs") { interp_nu_derivs = interp; status_nu_derivs = true; }
            else if (name == "R") { interp_R = interp; status_R = true; }
            else if (name == "dRdtheta") { interp_dRdtheta = interp; status_dRdtheta = true; }
            else if (name == "dRdzeta") { interp_dRdzeta = interp; status_dRdzeta = true; }
            else if (name == "dRds") { interp_dRds = interp; status_dRds = true; }
            else if (name == "R_derivs") { interp_R_derivs = interp; status_R_derivs = true; }
            else if (name == "Z") { interp_Z = interp; status_Z = true; }
            else if (name == "dZdtheta") { interp_dZdtheta = interp; status_dZdtheta = true; }
            else if (name == "dZdzeta") { interp_dZdzeta = interp; status_dZdzeta = true; }
            else if (name == "dZds") { interp_dZds = interp; status_dZds = true; }
            else if (name == "Z_derivs") { interp_Z_derivs = interp; status_Z_derivs = true; }
        }
};
