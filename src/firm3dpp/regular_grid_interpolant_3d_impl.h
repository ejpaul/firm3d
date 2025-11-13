#include "regular_grid_interpolant_3d.h"
#include <xtensor/xarray.hpp>
#include "xtensor/xlayout.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/format.hpp>
#include <stdexcept>
#include <iostream>

#define _EPS_ 1e-13

template<class Array>
const int RegularGridInterpolant3D<Array>::simdcount;

template<class Array>
void RegularGridInterpolant3D<Array>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f) {
    // Check if we're in load mode - if so, skip expensive computation
    // This prevents recomputation during field loading from saved data
    if (get_load_mode()) {
        return;
    }
    int BATCH_SIZE = 16384;
    int NUM_BATCHES = dofs_to_keep/BATCH_SIZE + (dofs_to_keep % BATCH_SIZE != 0);
    for (int i = 0; i < NUM_BATCHES; ++i) {
        uint32_t first = i * BATCH_SIZE;
        uint32_t last = std::min((uint32_t)((i+1) * BATCH_SIZE), dofs_to_keep);
        Vec xsub(xdoftensor_reduced.begin() + first, xdoftensor_reduced.begin() + last);
        Vec ysub(ydoftensor_reduced.begin() + first, ydoftensor_reduced.begin() + last);
        Vec zsub(zdoftensor_reduced.begin() + first, zdoftensor_reduced.begin() + last);
        Vec fxyzsub  = f(xsub, ysub, zsub);
        for (int j = 0; j < last-first; ++j) {
            for (int l = 0; l < value_size; ++l) {
                vals[first * value_size + j * value_size + l] = fxyzsub[j * value_size + l];
            }
        }
    }
    int degree = rule.degree;
    all_local_vals_map = std::unordered_map<int, AlignedPaddedVec>();
    all_local_vals_map.reserve(cells_to_keep);

    for (int xidx = 0; xidx < nx; ++xidx) {
        for (int yidx = 0; yidx < ny; ++yidx) {
            for (int zidx = 0; zidx < nz; ++zidx) {
                int meshidx = idx_cell(xidx, yidx, zidx);
                if(skip_cell[meshidx])
                    continue;
                // Zero-initialize ALL memory including SIMD padding
                AlignedPaddedVec local_vals(local_vals_size, 0.);
                // Explicitly zero any SIMD padding beyond local_vals_size
                size_t simdcount = XSIMD_DEFAULT_ALIGNMENT / sizeof(double);
                size_t allocated_size = (local_vals_size + simdcount) - (local_vals_size % simdcount);
                for (size_t idx = local_vals_size; idx < allocated_size && idx < local_vals.size(); ++idx) {
                    local_vals[idx] = 0.0;
                }
                for (int i = 0; i < degree+1; ++i) {
                    for (int j = 0; j < degree+1; ++j) {
                        for (int k = 0; k < degree+1; ++k) {
                            int dof_global_idx = idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+k);
                            if(dof_global_idx < 0 || dof_global_idx >= full_to_reduced_map.size()) {
                                throw std::runtime_error("ERROR: Invalid dof_global_idx in interpolate_batch: " + std::to_string(dof_global_idx) + 
                                                       " (full_to_reduced_map.size()=" + std::to_string(full_to_reduced_map.size()) + ")");
                            }
                            int dof_idx = full_to_reduced_map[dof_global_idx];
                            int offset_local = padded_value_size * idx_dof_local(i, j, k);
                            for (int l = 0; l < value_size; ++l) {
                                local_vals[offset_local + l] = vals[dof_idx * value_size + l];
                            }
                        }
                    }
                }
                all_local_vals_map.insert({meshidx, local_vals});
            }
        }
    }
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_batch(Array& xyz, Array& fxyz){
    if(fxyz.layout() != xt::layout_type::row_major)
          throw std::runtime_error("fxyz needs to be in row-major storage order");
    int npoints = xyz.shape(0);
    for (int i = 0; i < npoints; ++i) {
        evaluate_inplace(xyz(i, 0), xyz(i, 1), xyz(i, 2), fxyz.data() + value_size*i);
    }
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_batch_1D(Array& xyz, Array& fxyz){
    if(fxyz.layout() != xt::layout_type::row_major)
          throw std::runtime_error("fxyz needs to be in row-major storage order");
    int npoints = xyz.shape(0);
    for (int i = 0; i < npoints; ++i) {
        evaluate_inplace(xyz(i, 0), fxyz.data() + value_size*i);
    }
}

template<class Array>
Vec RegularGridInterpolant3D<Array>::evaluate(double x, double y, double z){
    Vec fxyz(value_size, 0.);
    evaluate_inplace(x, y, z, fxyz.data());
    return fxyz;

}

template<class Array>
int RegularGridInterpolant3D<Array>::locate_unsafe(double x, double y, double z){
    int xidx = int(nx*(x-xmin)/(xmax-xmin)); // find idx so that xmesh[xidx] <= x <= xs[xidx+1]
    int yidx = int(ny*(y-ymin)/(ymax-ymin));
    int zidx = int(nz*(z-zmin)/(zmax-zmin));
    return idx_cell(xidx, yidx, zidx);
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_inplace(double x, double y, double z, double* res){

    // to avoid funny business when the data is just a tiny bit out of bounds
    // due to machine precision, we perform this check and shift
    if(x >= xmax) x -= _EPS_;
    else if (x <= xmin) x += _EPS_;
    if(y >= ymax) y -= _EPS_;
    else if (y <= ymin) y += _EPS_;
    if(z >= zmax) z -= _EPS_;
    else if (z <= zmin) z += _EPS_;

    int xidx = int(nx*(x-xmin)/(xmax-xmin)); // find idx so that xmesh[xidx] <= x <= xs[xidx+1]
    int yidx = int(ny*(y-ymin)/(ymax-ymin));
    int zidx = int(nz*(z-zmin)/(zmax-zmin));
    if(!out_of_bounds_ok){
        if(xidx < 0 || xidx >= nx)
            throw std::runtime_error((boost::format("xidxs={} not within [0, {}]") % xidx % (nx-1)).str());
        if(yidx < 0 || yidx >= ny)
            throw std::runtime_error((boost::format("yidxs={} not within [0, {}]") % yidx % (ny-1)).str());
        if(zidx < 0 || zidx >= nz)
            throw std::runtime_error((boost::format("zidxs={} not within [0, {}]") % zidx % (nz-1)).str());
    } else {
    }
    int cell_idx = idx_cell(xidx, yidx, zidx);
    
    
    double xlocal = (x-xmesh[xidx])/hx;
    double ylocal = (y-ymesh[yidx])/hy;
    double zlocal = (z-zmesh[zidx])/hz;
    return evaluate_local(xlocal, ylocal, zlocal, cell_idx, res);
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_inplace(double x, double* res){

    // to avoid funny business when the data is just a tiny bit out of bounds
    // due to machine precision, we perform this check and shift
    if(x >= xmax) x -= _EPS_;
    else if (x <= xmin) x += _EPS_;

    int xidx = int(nx*(x-xmin)/(xmax-xmin)); // find idx so that xmesh[xidx] <= x <= xs[xidx+1]

    if(!out_of_bounds_ok){
        if(xidx < 0 || xidx >= nx)
            throw std::runtime_error((boost::format("xidxs={} not within [0, {}]") % xidx % (nx-1)).str());
    }
    double xlocal = (x-xmesh[xidx])/hx;
    return evaluate_local(xlocal, idx_cell(xidx, 0, 0), res);
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_local(double x, double y, double z, int cell_idx, double* res)
{
    int degree = rule.degree;
    auto got = all_local_vals_map.find(cell_idx);
    if (got == all_local_vals_map.end()) {
        if(out_of_bounds_ok)
            return;
        else
            throw std::runtime_error((boost::format("cell_idx={} not in all_local_vals_map") % cell_idx).str());
    }
    
    
    

    double* vals_local = got->second.data();
    
    if(xsimd::simd_type<double>::size >= 3){
        simd_t xyz;
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
        for (int k = 0; k < degree+1; ++k) {
            simd_t temp = this->rule.basis_fun(k, xyz);
            pkxs[k] = temp[0];
            pkys[k] = temp[1];
            pkzs[k] = temp[2];
        }
    } else {
        for (int k = 0; k < degree+1; ++k) {
            pkxs[k] = this->rule.basis_fun(k, x);
            pkys[k] = this->rule.basis_fun(k, y);
            pkzs[k] = this->rule.basis_fun(k, z);
        }
    }

    // Potential optimization: use barycentric interpolation here right now the
    // implementation in O(degree^3) in memory and O(degree^4) in computation,
    // using Barycentric interpolation this could be reduced to O(degree^3) in
    // memory and O(degree^3) in computation.
    for(int l=0; l<padded_value_size; l += simdcount) {
        simd_t sumi(0.);
        for (int i = 0; i < degree+1; ++i) {
            simd_t sumj(0.);
            for (int j = 0; j < degree+1; ++j) {
                simd_t sumk(0.);
                for (int k = 0; k < degree+1; ++k) {
                    int offset_local = padded_value_size * idx_dof_local(i, j, k) + l;
                    if (offset_local < 0 || offset_local + simdcount > got->second.size()) {
                        throw std::runtime_error("ERROR: Invalid offset_local: " + std::to_string(offset_local) + 
                                               " (size=" + std::to_string(got->second.size()) + ")");
                    }
                    double* val_ptr = &(vals_local[offset_local]);
                    double pkz = pkzs[k];
                    
                    
                    sumk = xsimd::fma(xsimd::load_aligned(val_ptr), simd_t(pkz), sumk);
                }
                double pjy = pkys[j];
                sumj = xsimd::fma(sumk, simd_t(pjy), sumj);
            }
            double pix = pkxs[i];
            sumi = xsimd::fma(sumj, simd_t(pix), sumi);
        }
        for (int ll = 0; ll < std::min(simdcount, value_size-l); ++ll) {
            res[l+ll] = sumi[ll];
        }
    }
    
}
//TODO memory usage not fixed
template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_local(double x, int cell_idx, double* res)
{
    int degree = rule.degree;
    auto got = all_local_vals_map.find(cell_idx);
    if (got == all_local_vals_map.end()) {
        if(out_of_bounds_ok)
            return;
        else
            throw std::runtime_error((boost::format("cell_idx={} not in all_local_vals_map") % cell_idx).str());
    }

    double* vals_local = got->second.data();

    for (int k = 0; k < degree+1; ++k) {
        pkxs[k] = this->rule.basis_fun(k, x);
    }

    for(int l=0; l<padded_value_size; l += simdcount) {
        simd_t sumi(0.);
        int offset_local = l;
        double* val_ptr = &(vals_local[offset_local]);
        for (int i = 0; i < degree+1; ++i) {
            double pkx = pkxs[i];
            sumi = xsimd::fma(xsimd::load_aligned(val_ptr), simd_t(pkx), sumi);
            val_ptr += padded_value_size * (degree+1) * (degree+1);
        }
        for (int ll = 0; ll < std::min(simdcount, value_size-l); ++ll) {
            res[l+ll] = sumi[ll];
        }
    }
}

template<class Array>
std::pair<double, double> RegularGridInterpolant3D<Array>::estimate_error(std::function<Vec(Vec, Vec, Vec)> &f, int samples) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, +1.0);
    double err = 0;
    double errsq = 0;
    Vec xs(samples, 0.);
    Vec ys(samples, 0.);
    Vec zs(samples, 0.);
    Array xyz = xt::zeros<double>({samples, 3});
    Array fhxyz = xt::zeros<double>({samples, value_size});
    for (int i = 0; i < samples; ++i) {
        xs[i] = xmin + distribution(generator)*(xmax-xmin);
        ys[i] = ymin + distribution(generator)*(ymax-ymin);
        zs[i] = zmin + distribution(generator)*(zmax-zmin);
        xyz(i, 0) = xs[i];
        xyz(i, 1) = ys[i];
        xyz(i, 2) = zs[i];
    }
    Vec fx = f(xs, ys, zs);
    this->evaluate_batch(xyz, fhxyz);
    for (int i = 0; i < samples; ++i) {
        double diff = 0.;
        for (int l = 0; l < value_size; ++l) {
            diff += std::pow(fx[value_size*i+l]-fhxyz(i, l), 2);
        }
        diff = std::sqrt(diff);
        err += diff;
        errsq += diff*diff;
    }
    double mean = err/samples;
    double std = std::sqrt((errsq - err*err/samples)/(samples-1)/samples);
    return std::make_pair(mean-std, mean+std);
}



Vec linspace(double min, double max, int n, bool endpoint) {
    Vec res(n, 0.);
    if(endpoint) {
        double h = (max-min)/(n-1);
        for (int i = 0; i < n; ++i)
            res[i] = min + i*h;
    } else {
        double h = (max-min)/n;
        for (int i = 0; i < n; ++i)
            res[i] = min + i*h;
    }
    return res;
}

template<class Array>
std::map<std::string, std::vector<double>> RegularGridInterpolant3D<Array>::get_interpolant_data() const {
    std::map<std::string, std::vector<double>> data;
    
    // Save ONLY the essential interpolated values - this is what we need to restore the field
    // The vals array contains the actual interpolated function values on the grid
    data["vals"] = vals;
    
    // Save grid parameters (for verification, but these are const and can't be modified during load)
    // These parameters define the grid structure and are needed to reconstruct the interpolant
    data["nx"] = {static_cast<double>(nx)};
    data["ny"] = {static_cast<double>(ny)};
    data["nz"] = {static_cast<double>(nz)};
    data["hx"] = {hx};
    data["hy"] = {hy};
    data["hz"] = {hz};
    data["xmin"] = {xmin};
    data["ymin"] = {ymin};
    data["zmin"] = {zmin};
    data["xmax"] = {xmax};
    data["ymax"] = {ymax};
    data["zmax"] = {zmax};
    data["value_size"] = {static_cast<double>(value_size)};
    data["padded_value_size"] = {static_cast<double>(padded_value_size)};
    data["dofs_to_keep"] = {static_cast<double>(dofs_to_keep)};
    data["cells_to_keep"] = {static_cast<double>(cells_to_keep)};
    data["local_vals_size"] = {static_cast<double>(local_vals_size)};
    
    // Save interpolation rule (needed for verification, but rule is const and can't be modified)
    data["rule_degree"] = {static_cast<double>(rule.degree)};
    data["rule_nodes"] = rule.nodes;
    data["rule_scalings"] = rule.scalings;
    
    // Save the mapping arrays for complete reconstruction
    std::vector<double> reduced_to_full_double(reduced_to_full_map.begin(), reduced_to_full_map.end());
    std::vector<double> full_to_reduced_double(full_to_reduced_map.begin(), full_to_reduced_map.end());
    std::vector<double> skip_cell_double(skip_cell.begin(), skip_cell.end());
    
    data["reduced_to_full_map"] = reduced_to_full_double;
    data["full_to_reduced_map"] = full_to_reduced_double;
    data["skip_cell"] = skip_cell_double;
    
    // NOTE: We do NOT save all_local_vals_map to keep JSON small and fast
    // It will be reconstructed from vals during loading using the fallback path
    // This reconstruction is fast (just memory operations) compared to saving/loading 100x more data
    
    return data;
}

template<class Array>
void RegularGridInterpolant3D<Array>::set_interpolant_data(const std::map<std::string, std::vector<double>>& data) {
    // CRITICAL: Load vals array first - this contains the actual interpolated function values
    auto vals_it = data.find("vals");
    if (vals_it != data.end()) {
        vals = vals_it->second;
    } else {
        throw std::runtime_error("ERROR: vals array not found in saved data!");
    }
    
    // Verify that vals was loaded correctly
    if (vals.size() == 0) {
        throw std::runtime_error("ERROR: vals array is empty after loading from JSON!");
    }
    
    // Verify that vals array size is consistent with expected size
    // The vals array should have size dofs_to_keep * value_size
    if (data.find("dofs_to_keep") != data.end()) {
        uint32_t expected_dofs = static_cast<uint32_t>(data.at("dofs_to_keep")[0]);
        size_t expected_size = expected_dofs * value_size;
        if (vals.size() != expected_size) {
            throw std::runtime_error("ERROR: vals array size mismatch! Expected: " + std::to_string(expected_size) + 
                                   ", Got: " + std::to_string(vals.size()) + 
                                   " (dofs_to_keep=" + std::to_string(expected_dofs) + 
                                   ", value_size=" + std::to_string(value_size) + ")");
        }
    }
    
    // Load only the non-const grid parameters that can be modified
    // Const parameters (nx, ny, nz, xmin, etc.) are already set during construction
    if (data.find("hx") != data.end()) hx = data.at("hx")[0];
    if (data.find("hy") != data.end()) hy = data.at("hy")[0];
    if (data.find("hz") != data.end()) hz = data.at("hz")[0];
    // Note: value_size is const and set during construction, cannot be modified here
    // These values must match exactly what was used during saving to ensure compatibility
    // The constructor's calculated values might be different due to different simdcount or other factors
    if (data.find("padded_value_size") != data.end()) padded_value_size = static_cast<int>(data.at("padded_value_size")[0]);
    if (data.find("local_vals_size") != data.end()) local_vals_size = static_cast<int>(data.at("local_vals_size")[0]);
    
    // Validate that the saved value_size matches the constructor parameter
    if (data.find("value_size") != data.end()) {
        int saved_value_size = static_cast<int>(data.at("value_size")[0]);
        if (saved_value_size != value_size) {
            throw std::runtime_error("ERROR: Saved value_size (" + std::to_string(saved_value_size) + 
                                   ") does not match constructor value_size (" + std::to_string(value_size) + ")");
        }
    }
    
    if (data.find("dofs_to_keep") != data.end()) dofs_to_keep = static_cast<uint32_t>(data.at("dofs_to_keep")[0]);
    if (data.find("cells_to_keep") != data.end()) cells_to_keep = static_cast<uint32_t>(data.at("cells_to_keep")[0]);
    
    // Safety checks to prevent segfault
    // Note: value_size is const and set during construction, so we can't check it here
    if (padded_value_size <= 0 || padded_value_size > 1000) {
        throw std::runtime_error("ERROR: Invalid padded_value_size value: " + std::to_string(padded_value_size));
    }
    if (dofs_to_keep == 0 || dofs_to_keep > 10000000) {
        throw std::runtime_error("ERROR: Invalid dofs_to_keep value: " + std::to_string(dofs_to_keep));
    }
    if (cells_to_keep == 0 || cells_to_keep > 10000000) {
        throw std::runtime_error("ERROR: Invalid cells_to_keep value: " + std::to_string(cells_to_keep));
    }
    if (local_vals_size <= 0 || local_vals_size > 10000) {
        throw std::runtime_error("ERROR: Invalid local_vals_size value: " + std::to_string(local_vals_size));
    }
    
    // CRITICAL: Validate that local_vals_size is consistent with padded_value_size and degree
    // This validation is now done using the loaded values from saved data
    int expected_local_vals_size = padded_value_size * (rule.degree + 1) * (rule.degree + 1) * (rule.degree + 1);
    if (local_vals_size != expected_local_vals_size) {
        throw std::runtime_error("ERROR: local_vals_size mismatch! Loaded local_vals_size=" + std::to_string(local_vals_size) + 
                               " but expected " + std::to_string(expected_local_vals_size) + 
                               " based on padded_value_size=" + std::to_string(padded_value_size) + 
                               " and degree=" + std::to_string(rule.degree) + 
                               ". This indicates incompatible saved data.");
    }
    
    // CRITICAL: Do NOT reconstruct xmesh, ymesh, zmesh, xdof, ydof, zdof!
    // The RegularGridInterpolant3D constructor already created these correctly
    // when the object was created with the loaded RangeTriplet values.
    // Reconstructing them here would introduce floating-point errors from saved hx, hy, hz values.
    // The constructor logic (regular_grid_interpolant_3d.h lines 161-224) already handles everything.
    
    // Only verify that hx, hy, hz match what the constructor calculated
    // This is just for sanity checking - we don't actually need to load them
    double expected_hx = (xmax - xmin) / nx;
    double expected_hy = (ymax - ymin) / ny;
    double expected_hz = (zmax - zmin) / nz;
    
    // Small tolerance for floating point comparison
    if (std::abs(hx - expected_hx) > 1e-10 || 
        std::abs(hy - expected_hy) > 1e-10 || 
        std::abs(hz - expected_hz) > 1e-10) {
        // This is OK - just use what the constructor calculated
        // The constructor's values are more accurate
        hx = expected_hx;
        hy = expected_hy;
        hz = expected_hz;
    }
    
    // Load the mapping arrays from saved data
    auto reduced_it = data.find("reduced_to_full_map");
    if (reduced_it != data.end()) {
        const std::vector<double>& reduced_double = reduced_it->second;
        reduced_to_full_map.assign(reduced_double.begin(), reduced_double.end());
    } else {
        reduced_to_full_map.resize(dofs_to_keep);
        for (int i = 0; i < dofs_to_keep; ++i) {
            reduced_to_full_map[i] = i;
        }
    }
    
    auto full_it = data.find("full_to_reduced_map");
    if (full_it != data.end()) {
        const std::vector<double>& full_double = full_it->second;
        full_to_reduced_map.assign(full_double.begin(), full_double.end());
    } else {
       
        // The total number of degrees of freedom is (nx*degree+1) * (ny*degree+1) * (nz*degree+1)
        int total_dofs = (nx * rule.degree + 1) * (ny * rule.degree + 1) * (nz * rule.degree + 1);
        
        // Safety check to prevent segfault
        if (total_dofs <= 0 || total_dofs > 10000000) {  // Reasonable upper bound
            throw std::runtime_error("ERROR: Invalid total_dofs calculation: " + std::to_string(total_dofs) + 
                                   " (nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny) + 
                                   ", nz=" + std::to_string(nz) + ", degree=" + std::to_string(rule.degree) + ")");
        }
        
        full_to_reduced_map.resize(total_dofs);
        for (int i = 0; i < total_dofs; ++i) {
            if (i < dofs_to_keep) {
                full_to_reduced_map[i] = i;
            } else {
                full_to_reduced_map[i] = -1; // Not kept
            }
        }
    }
    
    auto skip_it = data.find("skip_cell");
    if (skip_it != data.end()) {
        const std::vector<double>& skip_double = skip_it->second;
        skip_cell.assign(skip_double.begin(), skip_double.end());
    } else {
        // Fallback: assume no cells are skipped
        skip_cell.resize(nx * ny * nz, false);
    }
    
    // CRITICAL: Do NOT reconstruct xdoftensor_reduced, ydoftensor_reduced, zdoftensor_reduced!
    // These arrays are ONLY used during interpolate_batch() to evaluate the function at DOF points.
    // After interpolate_batch() completes, they are NEVER used again!
    // The actual interpolation uses all_local_vals_map, which we reconstruct below.
    // 
    // When loading from JSON:
    // 1. interpolate_batch() is skipped (load_mode prevents it)
    // 2. We load vals directly
    // 3. We reconstruct all_local_vals_map from vals
    // 
    // So xdoftensor_reduced is irrelevant for loaded fields!
    
    
    
    // FAST RECONSTRUCTION: Build all_local_vals_map from vals array
    // This is much faster than saving/loading 100x more data to/from JSON
    // The reconstruction is just memory operations - no expensive physics calculations
    all_local_vals_map.clear();
    int degree = rule.degree;
    
    for (int xidx = 0; xidx < nx; ++xidx) {
        for (int yidx = 0; yidx < ny; ++yidx) {
            for (int zidx = 0; zidx < nz; ++zidx) {
                int meshidx = idx_cell(xidx, yidx, zidx);
                if(meshidx < 0 || meshidx >= skip_cell.size()) {
                    throw std::runtime_error("ERROR: Invalid meshidx: " + std::to_string(meshidx) + 
                                           " (skip_cell.size()=" + std::to_string(skip_cell.size()) + ")");
                }
                if(skip_cell[meshidx])
                    continue;
                // Zero-initialize including SIMD padding
                AlignedPaddedVec local_vals(local_vals_size, 0.);
                // Explicitly zero any SIMD padding beyond local_vals_size
                size_t simdcount = XSIMD_DEFAULT_ALIGNMENT / sizeof(double);
                size_t allocated_size = (local_vals_size + simdcount) - (local_vals_size % simdcount);
                for (size_t idx = local_vals_size; idx < allocated_size && idx < local_vals.size(); ++idx) {
                    local_vals[idx] = 0.0;
                }
                for (int i = 0; i < degree+1; ++i) {
                    for (int j = 0; j < degree+1; ++j) {
                        for (int k = 0; k < degree+1; ++k) {
                            int dof_global_idx = idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+k);
                            if(dof_global_idx < 0 || dof_global_idx >= full_to_reduced_map.size()) {
                                throw std::runtime_error("ERROR: Invalid dof_global_idx: " + std::to_string(dof_global_idx) + 
                                                       " (full_to_reduced_map.size()=" + std::to_string(full_to_reduced_map.size()) + ")");
                            }
                            int dof_idx = full_to_reduced_map[dof_global_idx];
                            int offset_local = padded_value_size * idx_dof_local(i, j, k);
                            
                            // CRITICAL: If dof_idx < 0, this DOF was skipped (likely in a neighboring skipped cell)
                            // This should NEVER happen for DOFs in a non-skipped cell!
                            if (dof_idx < 0) {
                                throw std::runtime_error("ERROR: Cell " + std::to_string(meshidx) + " is not skipped, but DOF " + 
                                                       std::to_string(dof_global_idx) + " (i=" + std::to_string(i) + 
                                                       ", j=" + std::to_string(j) + ", k=" + std::to_string(k) + 
                                                       ") at position (" + std::to_string(xidx*degree+i) + ", " + 
                                                       std::to_string(yidx*degree+j) + ", " + std::to_string(zidx*degree+k) + 
                                                       ") was skipped! This indicates inconsistent skip_cell and full_to_reduced_map.");
                            }
                            
                            for (int l = 0; l < value_size; ++l) {
                                if (dof_idx * value_size + l >= vals.size()) {
                                    throw std::runtime_error("ERROR: Trying to access vals[" + std::to_string(dof_idx * value_size + l) + 
                                                           "] but vals.size()=" + std::to_string(vals.size()));
                                }
                                local_vals[offset_local + l] = vals[dof_idx * value_size + l];
                            }
                        }
                    }
                }
                all_local_vals_map.insert({meshidx, local_vals});
            }
        }
    }
    
    // Note: The const parameters (nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax, value_size, rule)
    // cannot be modified after construction. They are saved for verification purposes only.
    // The InterpolationRule is const and cannot be modified after construction.
}

// Static method implementations for load mode control
// Use a single static variable that both methods can access
template<class Array>
bool& RegularGridInterpolant3D<Array>::get_load_mode_flag() {
    // Static variable shared across all instances of RegularGridInterpolant3D
    // This ensures consistent load mode state across all interpolants
    static bool in_load_mode = false;
    return in_load_mode;
}

template<class Array>
void RegularGridInterpolant3D<Array>::set_load_mode(bool load_mode) {
    // Set the global load mode flag to prevent expensive computation during data loading
    get_load_mode_flag() = load_mode;
}

template<class Array>
bool RegularGridInterpolant3D<Array>::get_load_mode() {
    // Get the current load mode state
    return get_load_mode_flag();
}
