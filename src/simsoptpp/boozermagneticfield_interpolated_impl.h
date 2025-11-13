#pragma once

#include "boozermagneticfield_interpolated.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

// Implementation of save/load methods for InterpolatedBoozerField
// These methods enable efficient serialization of interpolated field data to avoid recomputation

std::map<std::string, std::map<std::string, std::vector<double>>> InterpolatedBoozerField::get_all_interpolant_data() const {
    std::map<std::string, std::map<std::string, std::vector<double>>> all_data;
    
    // OPTIMIZATION: Save mapping arrays only ONCE (they're identical for all quantities)
    // The mapping arrays (reduced_to_full_map, full_to_reduced_map, skip_cell) are ~50MB each
    // Saving them 30+ times would waste ~1.5GB! Instead, save once as "shared_maps"
    bool saved_shared_maps = false;
    
    // Helper lambda to save quantity data without redundant mapping arrays
    // This extracts shared maps from the first quantity, then strips them from all quantities
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
            // Remove redundant mapping arrays from this quantity (already saved in shared_maps)
            data.erase("reduced_to_full_map");
            data.erase("full_to_reduced_map");
            data.erase("skip_cell");
            all_data[name] = data;
        }
    };
    
    // Save all 31 quantities using the helper (order matches header declaration)
    // This removes ~1.5GB of redundant mapping array data from the JSON!
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
    // STRATEGY: We create interpolant objects using field's RangeTriplets (not saved values)
    // This ensures consistency with normal operation and avoids floating-point rounding errors
    // All interpolants share the same grid structure defined by s_range, theta_range, zeta_range
    
    // OPTIMIZATION: Load shared mapping arrays once and inject into all quantities
    // These mapping arrays are ~50MB each and are identical for all quantities
    std::map<std::string, std::vector<double>> shared_maps;
    auto shared_it = data.find("shared_maps");
    if (shared_it != data.end()) {
        shared_maps = shared_it->second;
    }
    
    // VALUE_SIZE LOOKUP TABLE (only 5 quantities differ from the default of 1)
    // Most quantities are scalars (value_size=1), so we use a map to store only exceptions
    // modB: dynamically determined from saved data (usually 1)
    // K_derivs: 2 (dK/dθ, dK/dζ)
    // R_derivs, Z_derivs, nu_derivs, modB_derivs: 3 (ds, dθ, dζ)
    std::map<std::string, int> value_size_map = {
        {"K_derivs", 2},
        {"R_derivs", 3},
        {"Z_derivs", 3},
        {"nu_derivs", 3},
        {"modB_derivs", 3}
    };
    
    // MAPPING TABLE: Quantity name → interpolant pointer reference
    // This avoids the massive if-else chain and makes the code maintainable
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
    
    // LOAD EACH QUANTITY
    for (const auto& pair : data) {
        const std::string& quantity = pair.first;
        
        // Skip the shared_maps entry itself
        if (quantity == "shared_maps") continue;
        
        // Check if this is a known quantity
        auto interp_it = interp_map.find(quantity);
        if (interp_it == interp_map.end()) {
            // Unknown quantity - skip it (could be future extension or typo in JSON)
            continue;
        }
        
        // Make a copy and inject shared maps into this quantity's data
        std::map<std::string, std::vector<double>> interpolant_data = pair.second;
        if (!shared_maps.empty()) {
            // Inject shared mapping arrays if they're not already in this quantity's data
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
        
        // Get the interpolant pointer reference
        std::shared_ptr<RegularGridInterpolant3D<Array2>>* interp_ptr = interp_it->second;
        
        // Create interpolant object if it doesn't exist
        if (!(*interp_ptr)) {
            // Determine value_size: check lookup table first, then default to 1
            int value_size = 1; // Default for most quantities
            auto vs_it = value_size_map.find(quantity);
            if (vs_it != value_size_map.end()) {
                value_size = vs_it->second;
            }
            
            // Special case: modB can have dynamic value_size (though usually 1)
            if (quantity == "modB" && interpolant_data.find("value_size") != interpolant_data.end()) {
                value_size = static_cast<int>(interpolant_data.at("value_size")[0]);
            }
            
            // Create the interpolant using field's grid parameters for consistency
            *interp_ptr = std::make_shared<RegularGridInterpolant3D<Array2>>(
                rule, s_range, theta_range, zeta_range, value_size, extrapolate
            );
        }
        
        // Load the data into the interpolant
        (*interp_ptr)->set_interpolant_data(interpolant_data);
    }
    
    // CRITICAL: Reset load mode to allow normal field evaluation
    // During loading, load_mode=true prevents expensive computation
    // Now that data is loaded, set load_mode=false to enable normal operation
    is_load_mode_constructor = false;
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
      // PERFORMANCE: Use compact format (no indentation) for ~2x faster save/load
      // Keep full precision (default) to maintain numerical accuracy
      file << save_dict.dump();
      file.close();
}
