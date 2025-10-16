#pragma once

#include "boozermagneticfield_interpolated.h"
#include <nlohmann/json.hpp>
#include <fstream>

// Implementation of save/load methods for InterpolatedBoozerField
// These methods enable efficient serialization of interpolated field data to avoid recomputation

std::map<std::string, std::map<std::string, std::vector<double>>> InterpolatedBoozerField::get_all_interpolant_data() const {
    std::map<std::string, std::map<std::string, std::vector<double>>> all_data;
    
    // Save data for each interpolant that has been computed
    // Use status flags for efficient checking - no need for is_computed() calls
    if (status_modB) {
        all_data["modB"] = interp_modB->get_interpolant_data();
    }
    if (status_dmodBdtheta) {
        all_data["dmodBdtheta"] = interp_dmodBdtheta->get_interpolant_data();
    }
    if (status_dmodBdzeta) {
        all_data["dmodBdzeta"] = interp_dmodBdzeta->get_interpolant_data();
    }
    if (status_dmodBds) {
        all_data["dmodBds"] = interp_dmodBds->get_interpolant_data();
    }
    if (status_G) {
        all_data["G"] = interp_G->get_interpolant_data();
    }
    if (status_I) {
        all_data["I"] = interp_I->get_interpolant_data();
    }
    if (status_iota) {
        all_data["iota"] = interp_iota->get_interpolant_data();
    }
    if (status_dGds) {
        all_data["dGds"] = interp_dGds->get_interpolant_data();
    }
    if (status_dIds) {
        all_data["dIds"] = interp_dIds->get_interpolant_data();
    }
    if (status_diotads) {
        all_data["diotads"] = interp_diotads->get_interpolant_data();
    }
    if (status_psip) {
        all_data["psip"] = interp_psip->get_interpolant_data();
    }
    if (status_R) {
        all_data["R"] = interp_R->get_interpolant_data();
    }
    if (status_Z) {
        all_data["Z"] = interp_Z->get_interpolant_data();
    }
    if (status_nu) {
        all_data["nu"] = interp_nu->get_interpolant_data();
    }
    if (status_K) {
        all_data["K"] = interp_K->get_interpolant_data();
    }
    if (status_dRdtheta) {
        all_data["dRdtheta"] = interp_dRdtheta->get_interpolant_data();
    }
    if (status_dRdzeta) {
        all_data["dRdzeta"] = interp_dRdzeta->get_interpolant_data();
    }
    if (status_dRds) {
        all_data["dRds"] = interp_dRds->get_interpolant_data();
    }
    if (status_dZdtheta) {
        all_data["dZdtheta"] = interp_dZdtheta->get_interpolant_data();
    }
    if (status_dZdzeta) {
        all_data["dZdzeta"] = interp_dZdzeta->get_interpolant_data();
    }
    if (status_dZds) {
        all_data["dZds"] = interp_dZds->get_interpolant_data();
    }
    if (status_dnudtheta) {
        all_data["dnudtheta"] = interp_dnudtheta->get_interpolant_data();
    }
    if (status_dnudzeta) {
        all_data["dnudzeta"] = interp_dnudzeta->get_interpolant_data();
    }
    if (status_dnuds) {
        all_data["dnuds"] = interp_dnuds->get_interpolant_data();
    }
    if (status_dKdtheta) {
        all_data["dKdtheta"] = interp_dKdtheta->get_interpolant_data();
    }
    if (status_dKdzeta) {
        all_data["dKdzeta"] = interp_dKdzeta->get_interpolant_data();
    }
    if (status_K_derivs) {
        all_data["K_derivs"] = interp_K_derivs->get_interpolant_data();
    }
    if (status_nu_derivs) {
        all_data["nu_derivs"] = interp_nu_derivs->get_interpolant_data();
    }
    if (status_R_derivs) {
        all_data["R_derivs"] = interp_R_derivs->get_interpolant_data();
    }
    if (status_Z_derivs) {
        all_data["Z_derivs"] = interp_Z_derivs->get_interpolant_data();
    }
    if (status_modB_derivs) {
        all_data["modB_derivs"] = interp_modB_derivs->get_interpolant_data();
    }
    
    return all_data;
}

void InterpolatedBoozerField::set_all_interpolant_data(const std::map<std::string, std::map<std::string, std::vector<double>>>& data) {
    // Load data for each interpolant, creating interpolant objects as needed
    // This method is called during field loading to restore saved interpolant data
    for (const auto& pair : data) {
        const std::string& quantity = pair.first;
        const std::map<std::string, std::vector<double>>& interpolant_data = pair.second;
        
        // Create interpolant object if it doesn't exist, then load data
        // This lazy creation matches the original behavior where interpolants are created on demand
        if (quantity == "modB") {
            if (!interp_modB) {
                interp_modB = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_modB->set_interpolant_data(interpolant_data);
        } else if (quantity == "dmodBdtheta") {
            if (!interp_dmodBdtheta) {
                interp_dmodBdtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dmodBdtheta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dmodBdzeta") {
            if (!interp_dmodBdzeta) {
                interp_dmodBdzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dmodBdzeta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dmodBds") {
            if (!interp_dmodBds) {
                interp_dmodBds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dmodBds->set_interpolant_data(interpolant_data);
        } else if (quantity == "G") {
            if (!interp_G) {
                interp_G = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_G->set_interpolant_data(interpolant_data);
        } else if (quantity == "I") {
            if (!interp_I) {
                interp_I = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_I->set_interpolant_data(interpolant_data);
        } else if (quantity == "iota") {
            if (!interp_iota) {
                interp_iota = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_iota->set_interpolant_data(interpolant_data);
        } else if (quantity == "dGds") {
            if (!interp_dGds) {
                interp_dGds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dGds->set_interpolant_data(interpolant_data);
        } else if (quantity == "dIds") {
            if (!interp_dIds) {
                interp_dIds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dIds->set_interpolant_data(interpolant_data);
        } else if (quantity == "diotads") {
            if (!interp_diotads) {
                interp_diotads = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_diotads->set_interpolant_data(interpolant_data);
        } else if (quantity == "psip") {
            if (!interp_psip) {
                interp_psip = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_psip->set_interpolant_data(interpolant_data);
        } else if (quantity == "R") {
            if (!interp_R) {
                interp_R = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_R->set_interpolant_data(interpolant_data);
        } else if (quantity == "Z") {
            if (!interp_Z) {
                interp_Z = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_Z->set_interpolant_data(interpolant_data);
        } else if (quantity == "nu") {
            if (!interp_nu) {
                interp_nu = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_nu->set_interpolant_data(interpolant_data);
        } else if (quantity == "K") {
            if (!interp_K) {
                interp_K = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_K->set_interpolant_data(interpolant_data);
        } else if (quantity == "dRdtheta") {
            if (!interp_dRdtheta) {
                interp_dRdtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dRdtheta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dRdzeta") {
            if (!interp_dRdzeta) {
                interp_dRdzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dRdzeta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dRds") {
            if (!interp_dRds) {
                interp_dRds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dRds->set_interpolant_data(interpolant_data);
        } else if (quantity == "dZdtheta") {
            if (!interp_dZdtheta) {
                interp_dZdtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dZdtheta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dZdzeta") {
            if (!interp_dZdzeta) {
                interp_dZdzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dZdzeta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dZds") {
            if (!interp_dZds) {
                interp_dZds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dZds->set_interpolant_data(interpolant_data);
        } else if (quantity == "dnudtheta") {
            if (!interp_dnudtheta) {
                interp_dnudtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dnudtheta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dnudzeta") {
            if (!interp_dnudzeta) {
                interp_dnudzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dnudzeta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dnuds") {
            if (!interp_dnuds) {
                interp_dnuds = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dnuds->set_interpolant_data(interpolant_data);
        } else if (quantity == "dKdtheta") {
            if (!interp_dKdtheta) {
                interp_dKdtheta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dKdtheta->set_interpolant_data(interpolant_data);
        } else if (quantity == "dKdzeta") {
            if (!interp_dKdzeta) {
                interp_dKdzeta = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            }
            interp_dKdzeta->set_interpolant_data(interpolant_data);
        } else if (quantity == "K_derivs") {
            if (!interp_K_derivs) {
                interp_K_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            }
            interp_K_derivs->set_interpolant_data(interpolant_data);
        } else if (quantity == "nu_derivs") {
            if (!interp_nu_derivs) {
                interp_nu_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            }
            interp_nu_derivs->set_interpolant_data(interpolant_data);
        } else if (quantity == "R_derivs") {
            if (!interp_R_derivs) {
                interp_R_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            }
            interp_R_derivs->set_interpolant_data(interpolant_data);
        } else if (quantity == "Z_derivs") {
            if (!interp_Z_derivs) {
                interp_Z_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            }
            interp_Z_derivs->set_interpolant_data(interpolant_data);
        } else if (quantity == "modB_derivs") {
            if (!interp_modB_derivs) {
                interp_modB_derivs = std::make_shared<RegularGridInterpolant3D<Array2>>(rule, s_range, theta_range, zeta_range, 3, extrapolate);
            }
            interp_modB_derivs->set_interpolant_data(interpolant_data);
        }
    }
    
    // CRITICAL: Reset load mode flag and clear static load mode after data is loaded
    // This allows the field to function normally for calculations
    // Without this, the field would always return zeros instead of actual values
    is_load_mode_constructor = false;
    
    // Automatically clear the static load mode to restore normal operation
    RegularGridInterpolant3D<Array2>::set_load_mode(false);
}

std::map<std::string, bool> InterpolatedBoozerField::get_status_flags() const {
    // Return all status flags indicating which interpolants have been computed
    // These flags are used to restore the field state after loading
    std::map<std::string, bool> flags;
    flags["status_modB"] = status_modB;
    flags["status_dmodBdtheta"] = status_dmodBdtheta;
    flags["status_dmodBdzeta"] = status_dmodBdzeta;
    flags["status_dmodBds"] = status_dmodBds;
    flags["status_G"] = status_G;
    flags["status_I"] = status_I;
    flags["status_iota"] = status_iota;
    flags["status_dGds"] = status_dGds;
    flags["status_dIds"] = status_dIds;
    flags["status_diotads"] = status_diotads;
    flags["status_psip"] = status_psip;
    flags["status_R"] = status_R;
    flags["status_Z"] = status_Z;
    flags["status_nu"] = status_nu;
    flags["status_K"] = status_K;
    flags["status_dRdtheta"] = status_dRdtheta;
    flags["status_dRdzeta"] = status_dRdzeta;
    flags["status_dRds"] = status_dRds;
    flags["status_dZdtheta"] = status_dZdtheta;
    flags["status_dZdzeta"] = status_dZdzeta;
    flags["status_dZds"] = status_dZds;
    flags["status_dnudtheta"] = status_dnudtheta;
    flags["status_dnudzeta"] = status_dnudzeta;
    flags["status_dnuds"] = status_dnuds;
    flags["status_dKdtheta"] = status_dKdtheta;
    flags["status_dKdzeta"] = status_dKdzeta;
    flags["status_K_derivs"] = status_K_derivs;
    flags["status_R_derivs"] = status_R_derivs;
    flags["status_Z_derivs"] = status_Z_derivs;
    flags["status_nu_derivs"] = status_nu_derivs;
    flags["status_modB_derivs"] = status_modB_derivs;
    return flags;
}

void InterpolatedBoozerField::set_status_flags(const std::map<std::string, bool>& flags) {
    // Restore status flags after loading interpolant data
    // This ensures the field knows which quantities are available for evaluation
    if (flags.find("status_modB") != flags.end()) status_modB = flags.at("status_modB");
    if (flags.find("status_dmodBdtheta") != flags.end()) status_dmodBdtheta = flags.at("status_dmodBdtheta");
    if (flags.find("status_dmodBdzeta") != flags.end()) status_dmodBdzeta = flags.at("status_dmodBdzeta");
    if (flags.find("status_dmodBds") != flags.end()) status_dmodBds = flags.at("status_dmodBds");
    if (flags.find("status_G") != flags.end()) status_G = flags.at("status_G");
    if (flags.find("status_I") != flags.end()) status_I = flags.at("status_I");
    if (flags.find("status_iota") != flags.end()) status_iota = flags.at("status_iota");
    if (flags.find("status_dGds") != flags.end()) status_dGds = flags.at("status_dGds");
    if (flags.find("status_dIds") != flags.end()) status_dIds = flags.at("status_dIds");
    if (flags.find("status_diotads") != flags.end()) status_diotads = flags.at("status_diotads");
    if (flags.find("status_psip") != flags.end()) status_psip = flags.at("status_psip");
    if (flags.find("status_R") != flags.end()) status_R = flags.at("status_R");
    if (flags.find("status_Z") != flags.end()) status_Z = flags.at("status_Z");
    if (flags.find("status_nu") != flags.end()) status_nu = flags.at("status_nu");
    if (flags.find("status_K") != flags.end()) status_K = flags.at("status_K");
    if (flags.find("status_dRdtheta") != flags.end()) status_dRdtheta = flags.at("status_dRdtheta");
    if (flags.find("status_dRdzeta") != flags.end()) status_dRdzeta = flags.at("status_dRdzeta");
    if (flags.find("status_dRds") != flags.end()) status_dRds = flags.at("status_dRds");
    if (flags.find("status_dZdtheta") != flags.end()) status_dZdtheta = flags.at("status_dZdtheta");
    if (flags.find("status_dZdzeta") != flags.end()) status_dZdzeta = flags.at("status_dZdzeta");
    if (flags.find("status_dZds") != flags.end()) status_dZds = flags.at("status_dZds");
    if (flags.find("status_dnudtheta") != flags.end()) status_dnudtheta = flags.at("status_dnudtheta");
    if (flags.find("status_dnudzeta") != flags.end()) status_dnudzeta = flags.at("status_dnudzeta");
    if (flags.find("status_dnuds") != flags.end()) status_dnuds = flags.at("status_dnuds");
    if (flags.find("status_dKdtheta") != flags.end()) status_dKdtheta = flags.at("status_dKdtheta");
    if (flags.find("status_dKdzeta") != flags.end()) status_dKdzeta = flags.at("status_dKdzeta");
    if (flags.find("status_K_derivs") != flags.end()) status_K_derivs = flags.at("status_K_derivs");
    if (flags.find("status_R_derivs") != flags.end()) status_R_derivs = flags.at("status_R_derivs");
    if (flags.find("status_Z_derivs") != flags.end()) status_Z_derivs = flags.at("status_Z_derivs");
    if (flags.find("status_nu_derivs") != flags.end()) status_nu_derivs = flags.at("status_nu_derivs");
    if (flags.find("status_modB_derivs") != flags.end()) status_modB_derivs = flags.at("status_modB_derivs");
}

// Implementation of to_json method
void InterpolatedBoozerField::to_json(const std::string& json_file_path) const {
    // Get the actual interpolated data from C++ objects (only already computed ones)
    auto interpolant_data = get_all_interpolant_data();
    auto status_flags = get_status_flags();
    
    // Find which quantities are actually computed
    std::vector<std::string> computed_quantities;
    for (const auto& [quantity, data] : interpolant_data) {
        if (!data.empty()) {
            computed_quantities.push_back(quantity);
        }
    }
    
    // Get the interpolation grid information
    auto s_range = this->s_range;
    auto theta_range = this->theta_range;  
    auto zeta_range = this->zeta_range;
    auto rule = this->rule;
    
    // Save grid and rule information
    nlohmann::json grid_info = {
        {"s_range", {std::get<0>(s_range), std::get<1>(s_range), std::get<2>(s_range)}},
        {"theta_range", {std::get<0>(theta_range), std::get<1>(theta_range), std::get<2>(theta_range)}}, 
        {"zeta_range", {std::get<0>(zeta_range), std::get<1>(zeta_range), std::get<2>(zeta_range)}},
        {"rule_degree", rule.degree},
        {"rule_nodes", rule.nodes},
        {"rule_scalings", rule.scalings}
    };
    
    // Convert interpolant data to JSON-serializable format
    nlohmann::json json_interpolant_data;
    for (const auto& [quantity, data] : interpolant_data) {
        nlohmann::json json_data;
        for (const auto& [key, value] : data) {
            json_data[key] = value;
        }
        json_interpolant_data[quantity] = json_data;
    }
    
    // Save configuration, interpolant data, and status
    nlohmann::json save_dict = {
        {"config", {
            {"boozmn_filename", "saved_field"},  // Placeholder since we don't have access to original filename
            {"order", 3},  // Placeholder
            {"no_K", true},  // Placeholder
            {"degree", rule.degree},
            {"ns_interp", std::get<2>(s_range)},    
            {"ntheta_interp", std::get<2>(theta_range)}, 
            {"nzeta_interp", std::get<2>(zeta_range)},
            {"extrapolate", extrapolate},
            {"nfp", nfp},
            {"stellsym", stellsym},
            {"field_type", field_type},
            {"psi0", psi0}  // Save the psi0 value from the original field
        }},
        {"grid_info", grid_info},
        {"interpolant_data", json_interpolant_data},
        {"status_flags", status_flags},
        {"computed_quantities", computed_quantities}
    };
    
    // Write to file
    std::ofstream file(json_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON file for writing: " + json_file_path);
    }
    file << save_dict.dump(2);
    file.close();
}
