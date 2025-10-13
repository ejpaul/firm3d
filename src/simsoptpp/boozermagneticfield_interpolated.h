#pragma once

#include "boozermagneticfield.h"
#include "xtensor/xlayout.hpp"
#include "regular_grid_interpolant_3d.h"
#include <string>
#include <nlohmann/json.hpp>
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
          
          // In load mode constructor, skip expensive computation and return zeros
          // This prevents recomputation during field loading from saved data
          if (is_load_mode_constructor) {
              // Create new array and fill with zeros
              psip = Array2({static_cast<long>(npoints), 1});
              psip.fill(0.0);
              return;
          }
          
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                G = Array2({static_cast<long>(npoints), 1});
                G.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                I = Array2({static_cast<long>(npoints), 1});
                I.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                iota = Array2({static_cast<long>(npoints), 1});
                iota.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dGds = Array2({static_cast<long>(npoints), 1});
                dGds.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dIds = Array2({static_cast<long>(npoints), 1});
                dIds.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                diotads = Array2({static_cast<long>(npoints), 1});
                diotads.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                K = Array2({static_cast<long>(npoints), 1});
                K.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dKdtheta = Array2({static_cast<long>(npoints), 1});
                dKdtheta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dKdzeta = Array2({static_cast<long>(npoints), 1});
                dKdzeta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                K_derivs = Array2({static_cast<long>(npoints), 2});
                K_derivs.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                nu = Array2({static_cast<long>(npoints), 1});
                nu.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dnudtheta = Array2({static_cast<long>(npoints), 1});
                dnudtheta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dnudzeta = Array2({static_cast<long>(npoints), 1});
                dnudzeta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dnuds = Array2({static_cast<long>(npoints), 1});
                dnuds.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                nu_derivs = Array2({static_cast<long>(npoints), 3});
                nu_derivs.fill(0.0);
                return;
            }
            
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
            // Always create the interpolant object if it doesn't exist
            // This ensures the object exists for both normal operation and load mode
            if(!interp_R)
                interp_R = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                R = Array2({static_cast<long>(npoints), 1});
                R.fill(0.0);
                return;
            }
            
            // Normal operation: compute R interpolant if not already computed
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dRdtheta = Array2({static_cast<long>(npoints), 1});
                dRdtheta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dRdzeta = Array2({static_cast<long>(npoints), 1});
                dRdzeta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dRds = Array2({static_cast<long>(npoints), 1});
                dRds.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                R_derivs = Array2({static_cast<long>(npoints), 3});
                R_derivs.fill(0.0);
                return;
            }
            
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
            // Always create the interpolant object if it doesn't exist
            // This ensures the object exists for both normal operation and load mode
            if(!interp_Z)
                interp_Z = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                Z = Array2({static_cast<long>(npoints), 1});
                Z.fill(0.0);
                return;
            }
            
            // Normal operation: compute Z interpolant if not already computed
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dZdtheta = Array2({static_cast<long>(npoints), 1});
                dZdtheta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dZdzeta = Array2({static_cast<long>(npoints), 1});
                dZdzeta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dZds = Array2({static_cast<long>(npoints), 1});
                dZds.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                Z_derivs = Array2({static_cast<long>(npoints), 3});
                Z_derivs.fill(0.0);
                return;
            }
            
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
            // Always create the interpolant object if it doesn't exist
            // This ensures the object exists for both normal operation and load mode
            if(!interp_modB)
                interp_modB = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                modB = Array2({static_cast<long>(npoints), 1});
                modB.fill(0.0);
                return;
            }
            
            // Normal operation: compute modB interpolant if not already computed
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dmodBdtheta = Array2({static_cast<long>(npoints), 1});
                dmodBdtheta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dmodBdzeta = Array2({static_cast<long>(npoints), 1});
                dmodBdzeta.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                dmodBds = Array2({static_cast<long>(npoints), 1});
                dmodBds.fill(0.0);
                return;
            }
            
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
            
            // In load mode constructor, skip expensive computation and return zeros
            // This prevents recomputation during field loading from saved data
            if (is_load_mode_constructor) {
                // Create new array and fill with zeros
                modB_derivs = Array2({static_cast<long>(npoints), 3});
                modB_derivs.fill(0.0);
                return;
            }
            
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
            // Create new array and fill with zeros
            Array2 points = Array2({static_cast<long>(npoints), 3});
            // Copy your data manually if needed
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

    private:
        bool is_load_mode_constructor = false;  // Flag to track if this is a load mode constructor

    protected:
        // Override _set_points_cb to prevent computation during load mode construction
        // This prevents the base class from triggering field evaluation during loading
        void _set_points_cb() override {
            if (is_load_mode_constructor) {
                // Do nothing during load mode construction - prevents computation
                return;
            }
            // Normal behavior (base class is empty anyway)
            return;
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
        {
            // CRITICAL: Ensure normal operation mode for regular field construction
            // This prevents the static load mode flag from affecting normal field construction
            // The static flag might be set to true from previous save/load operations
            RegularGridInterpolant3D<Array2>::set_load_mode(false);
            is_load_mode_constructor = false;
        }

        InterpolatedBoozerField(
                shared_ptr<BoozerMagneticField> field, int degree,
                RangeTriplet s_range, RangeTriplet theta_range, RangeTriplet zeta_range,
                bool extrapolate, int nfp, bool stellsym, string field_type) : InterpolatedBoozerField(field, UniformInterpolationRule(degree), s_range, theta_range, zeta_range, extrapolate, nfp, stellsym, field_type)         {
        }


        // Constructor for loading from saved data (no computation)
        // This constructor takes only the JSON file path and handles all loading internally
        InterpolatedBoozerField(const std::string& json_file_path) : 
            BoozerMagneticField(0.0, ""),  // Will be set from JSON data
            rule(UniformInterpolationRule(1)),  // Temporary, will be set from JSON
            s_range(0.0, 1.0, 10),  // Temporary, will be set from JSON
            theta_range(0.0, 2*M_PI, 10),  // Temporary, will be set from JSON
            zeta_range(0.0, 2*M_PI, 10),  // Temporary, will be set from JSON
            extrapolate(false),  // Temporary, will be set from JSON
            nfp(1),  // Temporary, will be set from JSON
            stellsym(false)  // Temporary, will be set from JSON
        {
            // Read and parse JSON file
            std::ifstream file(json_file_path);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open JSON file: " + json_file_path);
            }
            
            nlohmann::json data;
            file >> data;
            file.close();
            
            // Extract configuration data
            auto config = data["config"];
            auto grid_info = data["grid_info"];
            auto json_interpolant_data = data["interpolant_data"];
            auto status_flags = data["status_flags"];
            
            // Extract grid parameters
            std::vector<double> s_range_vec = grid_info["s_range"];
            std::vector<double> theta_range_vec = grid_info["theta_range"];
            std::vector<double> zeta_range_vec = grid_info["zeta_range"];
            
            RangeTriplet s_range_new(s_range_vec[0], s_range_vec[1], s_range_vec[2]);
            RangeTriplet theta_range_new(theta_range_vec[0], theta_range_vec[1], theta_range_vec[2]);
            RangeTriplet zeta_range_new(zeta_range_vec[0], zeta_range_vec[1], zeta_range_vec[2]);
            
            // Extract other parameters
            int degree = config["degree"];
            bool extrapolate_val = config["extrapolate"];
            int nfp_val = config["nfp"];
            bool stellsym_val = config["stellsym"];
            std::string field_type = config["field_type"];
            double psi0 = config["psi0"];
            
            // Extract rule data
            std::vector<double> rule_nodes = grid_info["rule_nodes"];
            std::vector<double> rule_scalings = grid_info["rule_scalings"];
            
            // Note: We cannot reassign const members, so we need to use placement new
            // This is safe because we're in the constructor and know the values
            // Create a custom InterpolationRule with the exact same nodes and scalings
            new (const_cast<InterpolationRule*>(&this->rule)) InterpolationRule(degree);
            // Now copy the saved nodes and scalings
            const_cast<InterpolationRule*>(&this->rule)->nodes = rule_nodes;
            const_cast<InterpolationRule*>(&this->rule)->scalings = rule_scalings;
            const_cast<RangeTriplet&>(this->s_range) = s_range_new;
            const_cast<RangeTriplet&>(this->theta_range) = theta_range_new;
            const_cast<RangeTriplet&>(this->zeta_range) = zeta_range_new;
            const_cast<bool&>(this->extrapolate) = extrapolate_val;
            const_cast<int&>(this->nfp) = nfp_val;
            const_cast<bool&>(this->stellsym) = stellsym_val;
            
            // Set base class values
            BoozerMagneticField::psi0 = psi0;  // Load psi0 from JSON
            BoozerMagneticField::field_type = field_type;
            
            // ACTIVATE LOAD MODE - prevent computation during construction
            RegularGridInterpolant3D<Array2>::set_load_mode(true);
            is_load_mode_constructor = true;
            
            // Set all status flags to FALSE to prevent computation
            status_modB = false;
            status_dmodBdtheta = false;
            status_dmodBdzeta = false;
            status_dmodBds = false;
            status_G = false;
            status_I = false;
            status_iota = false;
            status_dGds = false;
            status_dIds = false;
            status_diotads = false;
            status_psip = false;
            status_R = false;
            status_Z = false;
            status_nu = false;
            status_K = false;
            status_dRdtheta = false;
            status_dRdzeta = false;
            status_dRds = false;
            status_dZdtheta = false;
            status_dZdzeta = false;
            status_dZds = false;
            status_dnudtheta = false;
            status_dnudzeta = false;
            status_dnuds = false;
            status_dKdtheta = false;
            status_dKdzeta = false;
            status_K_derivs = false;
            status_R_derivs = false;
            status_Z_derivs = false;
            status_nu_derivs = false;
            status_modB_derivs = false;
            
            // Convert JSON interpolant data to C++ format
            std::map<std::string, std::map<std::string, std::vector<double>>> interpolant_data;
            for (auto& [quantity, json_data] : json_interpolant_data.items()) {
                std::map<std::string, std::vector<double>> data_map;
                for (auto& [key, value] : json_data.items()) {
                    if (value.is_array()) {
                        std::vector<double> vec;
                        for (auto& item : value) {
                            vec.push_back(item.get<double>());
                        }
                        data_map[key] = vec;
                    } else {
                        // Handle scalar values
                        std::vector<double> vec = {value.get<double>()};
                        data_map[key] = vec;
                    }
                }
                interpolant_data[quantity] = data_map;
            }
            
            // Load interpolant data (this will create interpolant objects and load data)
            set_all_interpolant_data(interpolant_data);
            
            // Load status flags
            std::map<std::string, bool> flags_map;
            for (auto& [key, value] : status_flags.items()) {
                flags_map[key] = value.get<bool>();
            }
            set_status_flags(flags_map);
            
            // Load mode is automatically cleared by set_all_interpolant_data()
            // Field is now ready for normal operation
        }

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
                
                // Save/load methods for all interpolant data
                std::map<std::string, std::map<std::string, std::vector<double>>> get_all_interpolant_data() const;
                void set_all_interpolant_data(const std::map<std::string, std::map<std::string, std::vector<double>>>& data);
                std::map<std::string, bool> get_status_flags() const;
                void set_status_flags(const std::map<std::string, bool>& flags);
                
                // Save field data to JSON file
                void to_json(const std::string& json_file_path) const;
};

// Include the implementation
#include "boozermagneticfield_interpolated_impl.h"
