#pragma once

#include "boozermagneticfield_interpolated.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <cstdint>

// ============================================================================
// SAVE/LOAD IMPLEMENTATION FOR InterpolatedBoozerField
// 
// PURPOSE: Enable serialization of interpolated magnetic field data to JSON,
//          avoiding the 10+ minute computation time for large grids.
// 
// KEY METHODS:
// - get_all_interpolant_data(): Extracts all 31 interpolant arrays for saving
// - set_all_interpolant_data(): Reconstructs interpolants from saved data
// - get/set_status_flags(): Track which quantities are computed
// - to_json(): Main entry point for saving field to JSON file
// 
// OPTIMIZATION: Mapping arrays (reduced_to_full_map, etc.) are saved only once
//               under "shared_maps" key, reducing JSON size by ~95%.
// ============================================================================

std::map<std::string, std::map<std::string, std::vector<double>>> InterpolatedBoozerField::get_all_interpolant_data() const {
    std::map<std::string, std::map<std::string, std::vector<double>>> all_data;
    
    // Save mapping arrays only once (they're identical for all quantities)
    bool saved_shared_maps = false;
    
    auto save_quantity = [&](bool status, auto& interp, const std::string& name) {
        if (status) {
            auto data = interp->get_interpolant_data();
            if (!saved_shared_maps) {
                // First computed quantity: extract and save shared maps once
                all_data["shared_maps"]["reduced_to_full_map"] = data["reduced_to_full_map"];
                all_data["shared_maps"]["full_to_reduced_map"] = data["full_to_reduced_map"];
                all_data["shared_maps"]["skip_cell"] = data["skip_cell"];
                saved_shared_maps = true;
            }
            data.erase("reduced_to_full_map");
            data.erase("full_to_reduced_map");
            data.erase("skip_cell");
            all_data[name] = data;
        }
    };
    
    // Save all quantities
    save_quantity(status_modB, interp_modB, "modB");
    save_quantity(status_dmodBdtheta, interp_dmodBdtheta, "dmodBdtheta");
    save_quantity(status_dmodBdzeta, interp_dmodBdzeta, "dmodBdzeta");
    save_quantity(status_dmodBds, interp_dmodBds, "dmodBds");
    save_quantity(status_modB_derivs, interp_modB_derivs, "modB_derivs");
    save_quantity(status_G, interp_G, "G");
    save_quantity(status_I, interp_I, "I");
    save_quantity(status_iota, interp_iota, "iota");
    save_quantity(status_dGds, interp_dGds, "dGds");
    save_quantity(status_dIds, interp_dIds, "dIds");
    save_quantity(status_diotads, interp_diotads, "diotads");
    save_quantity(status_psip, interp_psip, "psip");
    save_quantity(status_R, interp_R, "R");
    save_quantity(status_Z, interp_Z, "Z");
    save_quantity(status_nu, interp_nu, "nu");
    save_quantity(status_K, interp_K, "K");
    save_quantity(status_dRdtheta, interp_dRdtheta, "dRdtheta");
    save_quantity(status_dRdzeta, interp_dRdzeta, "dRdzeta");
    save_quantity(status_dRds, interp_dRds, "dRds");
    save_quantity(status_dZdtheta, interp_dZdtheta, "dZdtheta");
    save_quantity(status_dZdzeta, interp_dZdzeta, "dZdzeta");
    save_quantity(status_dZds, interp_dZds, "dZds");
    save_quantity(status_dnudtheta, interp_dnudtheta, "dnudtheta");
    save_quantity(status_dnudzeta, interp_dnudzeta, "dnudzeta");
    save_quantity(status_dnuds, interp_dnuds, "dnuds");
    save_quantity(status_dKdtheta, interp_dKdtheta, "dKdtheta");
    save_quantity(status_dKdzeta, interp_dKdzeta, "dKdzeta");
    save_quantity(status_K_derivs, interp_K_derivs, "K_derivs");
    save_quantity(status_nu_derivs, interp_nu_derivs, "nu_derivs");
    save_quantity(status_R_derivs, interp_R_derivs, "R_derivs");
    save_quantity(status_Z_derivs, interp_Z_derivs, "Z_derivs");
    
    return all_data;
}

void InterpolatedBoozerField::set_all_interpolant_data(const std::map<std::string, std::map<std::string, std::vector<double>>>& data) {
    // Load shared mapping arrays once and inject into all quantities
    std::map<std::string, std::vector<double>> shared_maps;
    auto shared_it = data.find("shared_maps");
    if (shared_it != data.end()) {
        shared_maps = shared_it->second;
    }
    
    // Value sizes: most quantities are 1, exceptions listed here
    std::map<std::string, int> value_size_map = {
        {"K_derivs", 2},
        {"R_derivs", 3},
        {"Z_derivs", 3},
        {"nu_derivs", 3},
        {"modB_derivs", 3}
    };
    
    // Map quantity names to interpolant pointers
    std::map<std::string, std::shared_ptr<RegularGridInterpolant3D<Array2>>*> interp_map = {
        {"modB", &interp_modB}, {"dmodBdtheta", &interp_dmodBdtheta}, {"dmodBdzeta", &interp_dmodBdzeta},
        {"dmodBds", &interp_dmodBds}, {"modB_derivs", &interp_modB_derivs}, {"G", &interp_G}, {"I", &interp_I},
        {"iota", &interp_iota}, {"dGds", &interp_dGds}, {"dIds", &interp_dIds}, {"diotads", &interp_diotads},
        {"psip", &interp_psip}, {"R", &interp_R}, {"Z", &interp_Z}, {"nu", &interp_nu}, {"K", &interp_K},
        {"dRdtheta", &interp_dRdtheta}, {"dRdzeta", &interp_dRdzeta}, {"dRds", &interp_dRds},
        {"dZdtheta", &interp_dZdtheta}, {"dZdzeta", &interp_dZdzeta}, {"dZds", &interp_dZds},
        {"dnudtheta", &interp_dnudtheta}, {"dnudzeta", &interp_dnudzeta}, {"dnuds", &interp_dnuds},
        {"dKdtheta", &interp_dKdtheta}, {"dKdzeta", &interp_dKdzeta}, {"K_derivs", &interp_K_derivs},
        {"nu_derivs", &interp_nu_derivs}, {"R_derivs", &interp_R_derivs}, {"Z_derivs", &interp_Z_derivs}
    };
    
    for (const auto& pair : data) {
        const std::string& quantity = pair.first;
        
        // Skip the shared_maps entry itself
        if (quantity == "shared_maps") continue;
        
        auto interp_it = interp_map.find(quantity);
        if (interp_it == interp_map.end()) continue;
        
        // Inject shared maps into this quantity's data
        std::map<std::string, std::vector<double>> interpolant_data = pair.second;
        if (!shared_maps.empty()) {
            if (interpolant_data.find("reduced_to_full_map") == interpolant_data.end()) {
                interpolant_data["reduced_to_full_map"] = shared_maps["reduced_to_full_map"];
            }
            if (interpolant_data.find("full_to_reduced_map") == interpolant_data.end()) {
                interpolant_data["full_to_reduced_map"] = shared_maps["full_to_reduced_map"];
            }
            if (interpolant_data.find("skip_cell") == interpolant_data.end()) {
                interpolant_data["skip_cell"] = shared_maps["skip_cell"];
            }
        }
        
        std::shared_ptr<RegularGridInterpolant3D<Array2>>* interp_ptr = interp_it->second;
        
        if (!(*interp_ptr)) {
            int value_size = 1;
            auto vs_it = value_size_map.find(quantity);
            if (vs_it != value_size_map.end()) {
                value_size = vs_it->second;
            }
            if (quantity == "modB" && interpolant_data.find("value_size") != interpolant_data.end()) {
                value_size = static_cast<int>(interpolant_data.at("value_size")[0]);
            }
            *interp_ptr = std::make_shared<RegularGridInterpolant3D<Array2>>(
                rule, s_range, theta_range, zeta_range, value_size, extrapolate
            );
        }
        (*interp_ptr)->set_interpolant_data(interpolant_data);
    }
    
    // Reset load mode to allow normal field evaluation
    is_load_mode_constructor = false;
    RegularGridInterpolant3D<Array2>::set_load_mode(false);
}

std::map<std::string, bool> InterpolatedBoozerField::get_status_flags() const {
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

void InterpolatedBoozerField::to_json(const std::string& json_file_path) const {
    auto interpolant_data = get_all_interpolant_data();
    auto status_flags = get_status_flags();
    
    // Extract tuple elements once to avoid repeated std::get calls
    int ns_interp = std::get<2>(this->s_range);
    int ntheta_interp = std::get<2>(this->theta_range);
    int nzeta_interp = std::get<2>(this->zeta_range);
    
    nlohmann::json grid_info = {
        {"s_range", {std::get<0>(this->s_range), std::get<1>(this->s_range), ns_interp}},
        {"theta_range", {std::get<0>(this->theta_range), std::get<1>(this->theta_range), ntheta_interp}}, 
        {"zeta_range", {std::get<0>(this->zeta_range), std::get<1>(this->zeta_range), nzeta_interp}},
        {"rule_degree", this->rule.degree},
        {"rule_nodes", this->rule.nodes},
        {"rule_scalings", this->rule.scalings}
    };
    
    // nlohmann::json can directly convert nested maps
    nlohmann::json json_interpolant_data = interpolant_data;
    
    nlohmann::json save_dict = {
        {"config", {
            {"degree", this->rule.degree},
            {"ns_interp", ns_interp},
            {"ntheta_interp", ntheta_interp},
            {"nzeta_interp", nzeta_interp},
            {"extrapolate", extrapolate},
            {"nfp", nfp},
            {"stellsym", stellsym},
            {"field_type", field_type},
            {"psi0", psi0}
        }},
        {"grid_info", grid_info},
        {"interpolant_data", json_interpolant_data},
        {"status_flags", status_flags}
    };
    
    std::ofstream file(json_file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + json_file_path);
    }
    
    // Use MessagePack binary format for faster save/load vs text JSON
    std::vector<std::uint8_t> msgpack = nlohmann::json::to_msgpack(save_dict);
    file.write(reinterpret_cast<const char*>(msgpack.data()), msgpack.size());
    file.close();
}
