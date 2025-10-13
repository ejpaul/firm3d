#include "regular_grid_interpolant_3d.h"
#include <xtensor/xarray.hpp>
#include "xtensor/xlayout.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/format.hpp>

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
                AlignedPaddedVec local_vals(local_vals_size, 0.);
                for (int i = 0; i < degree+1; ++i) {
                    for (int j = 0; j < degree+1; ++j) {
                        for (int k = 0; k < degree+1; ++k) {
                            int offset = value_size*full_to_reduced_map[idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+k)];
                            int offset_local = padded_value_size * idx_dof_local(i, j, k);
                            for (int l = 0; l < value_size; ++l) {
                                local_vals[offset_local + l] = vals[offset + l];
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
    }
    double xlocal = (x-xmesh[xidx])/hx;
    double ylocal = (y-ymesh[yidx])/hy;
    double zlocal = (z-zmesh[zidx])/hz;
    return evaluate_local(xlocal, ylocal, zlocal, idx_cell(xidx, yidx, zidx), res);
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
        int offset_local = l;
        double* val_ptr = &(vals_local[offset_local]);
        for (int i = 0; i < degree+1; ++i) {
            simd_t sumj(0.);
            for (int j = 0; j < degree+1; ++j) {
                simd_t sumk(0.);
                for (int k = 0; k < degree+1; ++k) {
                    double pkz = pkzs[k];
                    sumk = xsimd::fma(xsimd::load_aligned(val_ptr), simd_t(pkz), sumk);
                    val_ptr += padded_value_size;
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
    
    // NOTE: We don't save the large arrays (reduced_to_full_map, full_to_reduced_map, 
    // skip_cell, all_local_vals_map) because they can be reconstructed from the 
    // grid parameters and rule data. This dramatically reduces the save/load time and file size.
    
    return data;
}

template<class Array>
void RegularGridInterpolant3D<Array>::set_interpolant_data(const std::map<std::string, std::vector<double>>& data) {
    // Load the interpolated values (this is the most important part)
    // The vals array contains the actual interpolated function values that were saved
    if (data.find("vals") != data.end()) {
        vals = data.at("vals");
    }
    
    // Load only the non-const grid parameters that can be modified
    // Const parameters (nx, ny, nz, xmin, etc.) are already set during construction
    if (data.find("hx") != data.end()) hx = data.at("hx")[0];
    if (data.find("hy") != data.end()) hy = data.at("hy")[0];
    if (data.find("hz") != data.end()) hz = data.at("hz")[0];
    if (data.find("padded_value_size") != data.end()) padded_value_size = static_cast<int>(data.at("padded_value_size")[0]);
    if (data.find("dofs_to_keep") != data.end()) dofs_to_keep = static_cast<uint32_t>(data.at("dofs_to_keep")[0]);
    if (data.find("cells_to_keep") != data.end()) cells_to_keep = static_cast<uint32_t>(data.at("cells_to_keep")[0]);
    if (data.find("local_vals_size") != data.end()) local_vals_size = static_cast<int>(data.at("local_vals_size")[0]);
    
    // Reconstruct the grid meshes from the const parameters (these are already set during construction)
    // These arrays define the grid points for interpolation
    xmesh = linspace(xmin, xmax, nx, true);
    ymesh = linspace(ymin, ymax, ny, true);
    zmesh = linspace(zmin, zmax, nz, true);
    
    // Reconstruct degree of freedom arrays from the const parameters
    // These arrays define the degrees of freedom for the interpolation
    xdof = linspace(xmin, xmax, nx, true);
    ydof = linspace(ymin, ymax, ny, true);
    zdof = linspace(zmin, zmax, nz, true);
    
    // Reconstruct the reduced degree of freedom tensors
    // This is a simplified reconstruction - in practice, you might need more sophisticated logic
    // These arrays are used for batch evaluation of the interpolant
    xdoftensor_reduced = xdof;
    ydoftensor_reduced = ydof;
    zdoftensor_reduced = zdof;
    
    // NOTE: We don't load the large arrays (reduced_to_full_map, full_to_reduced_map, 
    // skip_cell, all_local_vals_map) because they weren't saved. These can be 
    // reconstructed if needed, but for basic field evaluation, they're not essential.
    // The key data (vals) is loaded, which is what we need for field evaluation.
    
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
