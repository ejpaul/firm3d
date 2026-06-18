// #include "simdhelpers.h" // import above cuda_runtime to prevent collision for rsqrt
#include <cuda_runtime.h>
#include <iostream>
#include "tracing.h"
#include <math.h>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
using std::shared_ptr;
using std::vector;
namespace py = pybind11;

#define THREADS_PER_BLOCK 64
#define PARTICLES_PER_BLOCK 8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Improvement notes:
// - Rewrite build state to be a loop i=0,1,2 with _constant__ xrange as one array

// enum used for templating
// https://stackoverflow.com/questions/9116267/how-can-i-use-an-enumeration-as-a-template-parameter
enum class RHS {GC_CartesianVacuum, GC_BoozerVacuum, GC_Boozer, GC_BoozerVacuumSAW, GC_BoozerNoKSAW};

enum class CoordSys {Cartesian, Boozer};

template<RHS id>
__host__ __device__ constexpr CoordSys map_rhs_to_coord(){
    if constexpr(id == RHS::GC_BoozerVacuum || id == RHS::GC_Boozer || id == RHS::GC_BoozerVacuumSAW || id == RHS::GC_BoozerNoKSAW){
        return CoordSys::Boozer;
    } else if constexpr (id == RHS::GC_CartesianVacuum) {
        return CoordSys::Cartesian;
    }
}

template<RHS id>
__host__ __device__ constexpr int map_rhs_to_n_interpolants(){
    if constexpr(id == RHS::GC_CartesianVacuum){
        return 7;
    } else if constexpr(id == RHS::GC_BoozerVacuum){
        return 6;
    } else if constexpr(id == RHS::GC_Boozer){
        return 12;
    } else if constexpr(id == RHS::GC_BoozerVacuumSAW || id == RHS::GC_BoozerNoKSAW){
        return 10; 
    }
}

// this is a helper function to convert python arrays to C++ arrays
template <typename T>
__host__ T* create_array(py::array_t<T> x){
    py::buffer_info buf = x.request();
    T* arr = static_cast<T*>(buf.ptr);
    return arr;
} 


/* below are declarations of data that are stored in the constant memory cache on the GPU
 * they are referenced like global variables and need to be set here, or copied to before the kernel is launched
 * accesses to constant memory are serialized and broadcasted across a warp
 * used when data doesn't change for the lifetime of the program and all threads use the same value
 */

// position weights for Dormand-Prince 5 timesteps
__constant__ double dp5_wgts[7][7] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
    {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0},
    {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0},
    {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,-5103.0 / 18656.0, 0.0, 0.0},
    {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0}
};
// position weights for Dormand-Prince 5 timesteps
__constant__ double dp5_t_wgts[7] = {
    0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0
};

// each RHS has an associated 3 dimensional coordinates system (x1, x2, x3)
// xi_range_d containts the beginning, end, number of grid points, and grid step size for an interpolant
// __constant__ double x1_range_d[4], x2_range_d[4], x3_range_d[4]; // contains start, end, number of points, grid size
__constant__ double grid_ranges_d[12];

// store the particle's mass, charge, max tracing time, absolute and relative tolerances for timestepping
__constant__ double mass_d, charge_d, tmax_d, atol_d, rtol_d; 
__constant__ int n_x2_d, n_x3_d, n_x23_d; // stores the number of interpolant cells in x2 and x3 direction, along with their product
__constant__ int nparticles_d; // number of particles being traced
__constant__ double v_total_d; // initial velocity

__constant__ double psi0_d; // used for Boozer RHS only
__constant__ double inv_psi0_charge_d; // used for Boozer RHS only, precompute 1 / charge_d * psi0_d
__constant__ double saw_srange_d[4]; // used for SAW RHS only

__constant__ bool rescale_abstol_var_d = true;
__constant__ bool is_test_d = false;


// interpolate performs tricubic interpolation in the r, phi, z coordinates
// which we assume is on a regular grid
// the name of these coordinates only reflects the original cylindircal coordinates
// this function works in general
// 
// the n interpolant elements are written to out in order
// interpolant data is stored in data in 64 interpolation pt windows 
// with n contiguous entries at each point
// 
// shape values are precomputed in build_state and here we are computing the needed inner product
// cell_index_start stores the grid index for interpolation in the r, phi, z coordinates
// r_shape, phi_shape, z_shape store shape function elements
// nphi and nz indicate how many grid pts there are in phi and z directions
// nparticles_blk store the number of *actual* particles in the current block
//
// note that nparticles_blk isn't always equal to PARTICLES_PER_BLOCK
template <typename T, int n> __device__ void interpolate(T*  out, const T* __restrict__ data, const int* __restrict__ cell_index_start, 
    const T* __restrict__ shape_fun_vals, int nparticles_blk){
    for(int idx=threadIdx.x; idx<nparticles_blk*n; idx+= THREADS_PER_BLOCK){
        int zz = idx % n;
        int particle_id = idx / n;

        int i = cell_index_start[3*particle_id];
        int j = cell_index_start[3*particle_id + 1];
        int k = cell_index_start[3*particle_id + 2];

        int row_start = 64*(i*n_x23_d + j*n_x3_d + k);

        T local_val = 0.0;
        for(int ii=0; ii<4; ++ii){
            for(int jj=0; jj<4; ++jj){
                for(int kk=0; kk<4; ++kk){
                    int row_idx = row_start + 16*ii + 4*jj + kk;
                    T shape_val = shape_fun_vals[ii*PARTICLES_PER_BLOCK + particle_id] * shape_fun_vals[(4 + jj)*PARTICLES_PER_BLOCK + particle_id] * shape_fun_vals[(8 + kk)*PARTICLES_PER_BLOCK + particle_id];
                    local_val += data[n*row_idx + zz] * shape_val;

                }
            }
        }
        out[PARTICLES_PER_BLOCK*zz + particle_id] = local_val;

    }
}


// calc_derivs implementation for guiding center cartesian vacuum tracing
template <typename T, int deriv_id>
__device__ void rhs_GC_CartesianVacuum(T* derivs, T* x_temp, T* block_interpolants, bool* symmetry_exploited, T* mu){

    T x = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T y = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T z = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK + threadIdx.x];

    T B_r = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
    T B_phi = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T B_z = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T GradAbsB_r = block_interpolants[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T GradAbsB_phi = block_interpolants[4*PARTICLES_PER_BLOCK + threadIdx.x];
    T GradAbsB_z = block_interpolants[5*PARTICLES_PER_BLOCK + threadIdx.x];

    if(symmetry_exploited[threadIdx.x]){
        B_r *= -1.0;
        GradAbsB_phi *= -1.0;
        GradAbsB_z *= -1.0;
    }

    T phi = atan2(y, x);
    T B_x = cos(phi) * B_r - sin(phi) * B_phi;
    T B_y = sin(phi) * B_r + cos(phi) * B_phi;
    T GradAbsB_x = cos(phi) * GradAbsB_r - sin(phi) * GradAbsB_phi;
    T GradAbsB_y = sin(phi) * GradAbsB_r + cos(phi) * GradAbsB_phi;

    T AbsB = sqrt(B_x*B_x + B_y*B_y + B_z*B_z);
    T v_perp2 = 2*mu[threadIdx.x]*AbsB;
    T fak1 = (v_par/AbsB);
    T fak2 = (mass_d/(charge_d*pow(AbsB, 3)))*(0.5*v_perp2 + v_par*v_par);

    T BcrossGradAbsB_elt = B_y*GradAbsB_z - B_z*GradAbsB_y;
    derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_x + fak2*BcrossGradAbsB_elt;
    BcrossGradAbsB_elt = B_z*GradAbsB_x - B_x*GradAbsB_z;
    derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_y + fak2*BcrossGradAbsB_elt;
    BcrossGradAbsB_elt = B_x*GradAbsB_y - B_y*GradAbsB_x;
    derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_z + fak2*BcrossGradAbsB_elt;
    derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x] = -mu[threadIdx.x]*(B_x*GradAbsB_x + B_y*GradAbsB_y + B_z*GradAbsB_z)/AbsB;
    derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK + threadIdx.x] = AbsB; // AbsB
    derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = block_interpolants[6*PARTICLES_PER_BLOCK + threadIdx.x]; // boundary dist fn

}


// calc_derivs implementation for guiding center boozer vacuum tracing
template <typename T, int deriv_id> 
__device__ void rhs_GC_BoozerVacuum(T* derivs, T* x_temp, T* block_interpolants, bool* symmetry_exploited, T* mu){

    T x1 = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];

    T inv_s = rsqrt(x1*x1 + x2*x2);
    T zeta = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK + threadIdx.x];


    T modB = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBds = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T G = block_interpolants[4*PARTICLES_PER_BLOCK + threadIdx.x];
    T iota = block_interpolants[5*PARTICLES_PER_BLOCK + threadIdx.x];
    T mu_val = mu[threadIdx.x];

    T modB_inv_G = modB / G;

    T sign = symmetry_exploited[threadIdx.x] ? (T)-1.0 : (T)1.0;
    dmodBdtheta *= sign;
    dmodBdzeta *= sign;


    T fak1 = mass_d*v_par*v_par/modB + mass_d*mu_val;
    T sdot = -dmodBdtheta*fak1 * inv_psi0_charge_d;
    T tdot = dmodBds*fak1 * inv_psi0_charge_d + iota*v_par*modB_inv_G;

    derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*x1*inv_s - x2*tdot;
    derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*x2*inv_s + x1*tdot;
    derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x] = v_par*modB_inv_G;
    derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x] = -(iota*dmodBdtheta + dmodBdzeta)*mu_val*modB_inv_G;
    derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK + threadIdx.x] = modB; // modB for setting mu
    derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = G;
    // derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = // no boundary dist fn
    
}


// calc_derivs implementation for general guiding center Boozer tracing (with K != 0)
// The equations in this function match those for the CPU tracing at
// tracing.cpp::GuidingCenterBoozerRHS
template<typename T, int deriv_id>
__device__ void rhs_GC_Boozer(T* derivs, T* x_temp, T* block_interpolants, bool* symmetry_exploited, T* mu){

    T x1 = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];

    T s = sqrt(x1*x1 + x2*x2);
    T theta = atan2(x2, x1);
    T zeta = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK + threadIdx.x];

    T modB = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBdpsi = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T G = block_interpolants[4*PARTICLES_PER_BLOCK + threadIdx.x];
    T dGdpsi = block_interpolants[5*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T I = block_interpolants[6*PARTICLES_PER_BLOCK + threadIdx.x];
    T dIdpsi = block_interpolants[7*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T iota = block_interpolants[8*PARTICLES_PER_BLOCK + threadIdx.x];
    T K = block_interpolants[9*PARTICLES_PER_BLOCK + threadIdx.x];
    T dKdtheta = block_interpolants[10*PARTICLES_PER_BLOCK + threadIdx.x];
    T dKdzeta = block_interpolants[11*PARTICLES_PER_BLOCK + threadIdx.x];

    T mu_val = mu[threadIdx.x];

    if(symmetry_exploited[threadIdx.x]){
        dmodBdtheta *= -1.0;
        dmodBdzeta *= -1.0;
        K *= -1.0;
    }

    // General guiding center equations (mode='gc')
    // C = - m v|| K,zeta /|B| - q iota + m v|| G' / |B|
    // F = - m v|| K,theta /|B| + q + m v|| I' / |B|
    // D = (F G - C I) / iota
    
    T C = -mass_d * v_par * dKdzeta / modB - charge_d * iota + mass_d * v_par * dGdpsi / modB;
    T F = -mass_d * v_par * dKdtheta / modB + charge_d + mass_d * v_par * dIdpsi / modB;
    T D = (F * G - C * I) / iota;

    T fak1 = mass_d * v_par * v_par / modB + mass_d * mu_val;
    
    // sdot = (I |B|,zeta - G |B|,theta) m (v||^2/|B| + mu) / (iota D psi0)
    T sdot = (I * dmodBdzeta - G * dmodBdtheta) * fak1 / (iota * D * psi0_d);
    
    // tdot = ((G |B|,psi - K |B|,zeta) m (v||^2/|B| + mu) - C v|| |B|) / (iota D)
    T tdot = ((G * dmodBdpsi - K * dmodBdzeta) * fak1 - C * v_par * modB) / (iota * D);
    
    // zetadot = (F v|| |B| - (|B|,psi I - |B|,theta K) m (v||^2/|B| + mu)) / (iota D)
    T zetadot = (F * v_par * modB - (dmodBdpsi * I - dmodBdtheta * K) * fak1) / (iota * D);
    
    // v||dot = (C |B|,theta - F |B|,zeta) mu |B| / (iota D)
    T vpardot = (C * dmodBdtheta - F * dmodBdzeta) * mu_val * modB / (iota * D);

    derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*cos(theta) - s*sin(theta)*tdot;
    derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*sin(theta) + s*cos(theta)*tdot;
    derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x] = zetadot;
    derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x] = vpardot;
    derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK + threadIdx.x] = modB; // modB for setting mu
    derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = G;
};

// calc_derivs implementation for guiding center boozer vacuum tracing with Shear Alfven Waves
template <typename T, int deriv_id>
__device__ void rhs_GC_BoozerVacuumSAW(T* derivs, T* x_temp, T* block_interpolants, bool* symmetry_exploited, T* mu,
                                         T saw_omega, int* saw_m, int* saw_n, T* saw_phihats, int saw_nharmonics){

    T time = x_temp[threadIdx.x];
    T x1 = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];

    T s = sqrt(x1*x1 + x2*x2);
    T theta = atan2(x2, x1);
    T zeta = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK + threadIdx.x];

    T modB = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBdpsi = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T G = block_interpolants[4*PARTICLES_PER_BLOCK + threadIdx.x];
    T dGdpsi = block_interpolants[5*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T I = block_interpolants[6*PARTICLES_PER_BLOCK + threadIdx.x];
    T dIdpsi = block_interpolants[7*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T iota = block_interpolants[8*PARTICLES_PER_BLOCK + threadIdx.x];
    T diotadpsi = block_interpolants[9*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;

    T mu_val = mu[threadIdx.x];

    if(symmetry_exploited[threadIdx.x]){
        dmodBdtheta *= -1.0;
        dmodBdzeta *= -1.0;
    }

    // accumulate over harmonics
    int s_index = (s - saw_srange_d[0]) / (saw_srange_d[3]);
    s_index = min(s_index, (int)saw_srange_d[2]-1);
    T s_diff = s - s_index*saw_srange_d[3];

    // rhs values from SAW 
    T dphidpsi = 0.0;
    T dphidtheta = 0.0;
    T dphidzeta = 0.0;

    T dalphadpsi = 0.0;
    T dalphadtheta = 0.0;
    T alphadot = 0.0;

    for(int i=0; i<saw_nharmonics; ++i){
        T left_phihat = saw_phihats[s_index*saw_nharmonics + i];
        T right_phihat = saw_phihats[min(s_index+1, (int)saw_srange_d[2]-1)*saw_nharmonics + i];
        T s_slope = (right_phihat - left_phihat) / saw_srange_d[3];

        int m = saw_m[i];
        int n = saw_n[i];
        T alpha_fac = (iota *m - n) / (saw_omega * G);
        T dalpha_fac_dpsi = diotadpsi * m / (saw_omega * G);

        T pt_cos = cos(m*theta - n*zeta + saw_omega*time);
        T pt_sin = sin(m*theta - n*zeta + saw_omega*time);

        T phihat_i = left_phihat + s_slope*(s_diff);
        T dphihatdpsi = s_slope / psi0_d;

        T phi_i = phihat_i * pt_sin;
        T dphidpsi_i = dphihatdpsi * pt_sin;
        T phidot_i = phihat_i * pt_cos * saw_omega;
        T dphidtheta_i = phidot_i * (m / saw_omega);
        T dphidzeta_i = -phidot_i * (n / saw_omega);

        T alphadot_i = -phidot_i * alpha_fac;
        T dalphadpsi_i = -dphidpsi_i * alpha_fac - phi_i*dalpha_fac_dpsi;
        T dalphadtheta_i = -dphidtheta_i * alpha_fac;

        dphidpsi += dphidpsi_i;
        dphidtheta += dphidtheta_i;
        dphidzeta += dphidzeta_i;

        alphadot += alphadot_i;
        dalphadpsi += dalphadpsi_i;
        dalphadtheta += dalphadtheta_i;
        
    }

    T fak1 = mass_d*v_par*v_par/modB + mass_d*mu_val;

    T sdot = (-dmodBdtheta*fak1/charge_d + dalphadtheta*modB*v_par - dphidtheta) / psi0_d;
    T tdot = (dmodBdpsi*fak1 / charge_d) + (iota - dalphadpsi*G)*v_par*modB / G + dphidpsi;

    derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*cos(theta) - s * sin(theta) * tdot;
    derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*sin(theta) + s*cos(theta)*tdot;
    derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x] = v_par*modB/G;
    derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x] = -modB/(G*mass_d) * (mass_d*mu_val*(dmodBdzeta + dalphadtheta*dmodBdpsi*G \
                + dmodBdtheta*(iota - dalphadpsi*G)) + charge_d*(alphadot*G \
                + dalphadtheta*G*dphidpsi + (iota - dalphadpsi*G)*dphidtheta + dphidzeta)) \
                + v_par/modB * (dmodBdtheta*dphidpsi - dmodBdpsi*dphidtheta);
    derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK + threadIdx.x] = modB; // modB for setting mu
    derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = G;

};


// calc_derivs implementation for guiding center boozer NoK tracing with Shear Alfven Waves
template <typename T, int deriv_id> 
__device__ void rhs_GC_BoozerNoKSAW(T* derivs, T* x_temp, T* block_interpolants, bool* symmetry_exploited, T* mu,
                                     T saw_omega, int* saw_m, int* saw_n, T* saw_phihats, int saw_nharmonics){

    T time = x_temp[threadIdx.x];
    T x1 = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];

    T s = sqrt(x1*x1 + x2*x2);
    T theta = atan2(x2, x1);
    T zeta = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T v_par = x_temp[4*PARTICLES_PER_BLOCK + threadIdx.x];

    T modB = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBdpsi = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T G = block_interpolants[4*PARTICLES_PER_BLOCK + threadIdx.x];
    T dGdpsi = block_interpolants[5*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T I = block_interpolants[6*PARTICLES_PER_BLOCK + threadIdx.x];
    T dIdpsi = block_interpolants[7*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;
    T iota = block_interpolants[8*PARTICLES_PER_BLOCK + threadIdx.x];
    T diotadpsi = block_interpolants[9*PARTICLES_PER_BLOCK + threadIdx.x] / psi0_d;

    T mu_val = mu[threadIdx.x];

    if(symmetry_exploited[threadIdx.x]){
        dmodBdtheta *= -1.0;
        dmodBdzeta *= -1.0;
    }

    // accumulate over harmonics
    int s_index = (s - saw_srange_d[0]) / (saw_srange_d[3]);
    s_index = min(s_index, (int)saw_srange_d[2]-1);
    T s_diff = s - s_index*saw_srange_d[3];

    // rhs values from SAW 
    T dphidpsi = 0.0;
    T dphidtheta = 0.0;
    T dphidzeta = 0.0;
    T dalphadzeta = 0.0;

    T alpha = 0.0;
    T dalphadpsi = 0.0;
    T dalphadtheta = 0.0;
    T alphadot = 0.0;

    for(int i=0; i<saw_nharmonics; ++i){
        T left_phihat = saw_phihats[s_index*saw_nharmonics + i];
        T right_phihat = saw_phihats[min(s_index+1, (int)saw_srange_d[2]-1)*saw_nharmonics + i];
        T s_slope = (right_phihat - left_phihat) / saw_srange_d[3];

        int m = saw_m[i];
        int n = saw_n[i];
        T alpha_fac = (iota *m - n) / (saw_omega * (G + iota*I));
        T dalpha_fac_dpsi = diotadpsi * m / (saw_omega * (G + iota*I)) - alpha_fac / (G+iota*I) * (dGdpsi + diotadpsi*I + iota*dIdpsi);

        T pt_cos = cos(m*theta - n*zeta + saw_omega*time);
        T pt_sin = sin(m*theta - n*zeta + saw_omega*time);

        T phihat_i = left_phihat + s_slope*(s_diff);
        T dphihatdpsi = s_slope / psi0_d;

        T phi_i = phihat_i * pt_sin;
        T dphidpsi_i = dphihatdpsi * pt_sin;
        T phidot_i = phihat_i * pt_cos * saw_omega;
        T dphidtheta_i = phidot_i * (m / saw_omega);
        T dphidzeta_i = -phidot_i * (n / saw_omega);

        T alpha_i = -phi_i*alpha_fac;
        T alphadot_i = -phidot_i * alpha_fac;
        T dalphadpsi_i = -dphidpsi_i * alpha_fac - phi_i*dalpha_fac_dpsi;
        T dalphadtheta_i = -dphidtheta_i * alpha_fac;
        T dalphadzeta_i = -dphidzeta_i*alpha_fac;

        dphidpsi += dphidpsi_i;
        dphidtheta += dphidtheta_i;
        dphidzeta += dphidzeta_i;

        alpha += alpha_i;
        alphadot += alphadot_i;
        dalphadpsi += dalphadpsi_i;
        dalphadtheta += dalphadtheta_i;
        dalphadzeta += dalphadzeta_i;
    }
    T fak1 = mass_d*v_par*v_par/modB + mass_d*mu_val;
    T denom = (charge_d*(G + I*(-alpha*dGdpsi + iota) + alpha*G*dIdpsi)
            + mass_d*v_par/modB * (-dGdpsi*I + G*dIdpsi)); 
    T sdot = (-G*dphidtheta*charge_d + I*dphidzeta*charge_d + modB*charge_d*v_par*(dalphadtheta*G-dalphadzeta*I) + (-dmodBdtheta*G + dmodBdzeta*I)*fak1)/(denom*psi0_d);
    T tdot = (G*charge_d*dphidpsi + modB*charge_d*v_par*(-dalphadpsi*G - alpha*dGdpsi + iota) - dGdpsi*mass_d*v_par*v_par \
                    + dmodBdpsi*G*fak1)/denom;
    derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*cos(theta) - s * sin(theta) * tdot;
    derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*sin(theta) + s*cos(theta)*tdot;
    derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x] = v_par*modB/G;
    derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x] = (modB*charge_d/mass_d * ( -mass_d*mu_val * (dmodBdzeta*(1 + dalphadpsi*I + alpha*dIdpsi) \
                    + dmodBdpsi*(dalphadtheta*G - dalphadzeta*I) + dmodBdtheta*(iota - alpha*dGdpsi - dalphadpsi*G)) \
                    - charge_d*(alphadot*(G + I*(iota - alpha*dGdpsi) + alpha*G*dIdpsi) \
                    + (dalphadtheta*G - dalphadzeta*I)*dphidpsi \
                    + (iota - alpha*dGdpsi - dalphadpsi*G)*dphidtheta \
                    + (1 + alpha*dIdpsi + dalphadpsi*I)*dphidzeta)) \
                    + charge_d*v_par/modB * ((dmodBdtheta*G - dmodBdzeta*I)*dphidpsi \
                    + dmodBdpsi*(I*dphidzeta - G*dphidtheta)) \
                    + v_par*(mass_d*mu_val*(dmodBdtheta*dGdpsi - dmodBdzeta*dIdpsi) \
                    + charge_d*(alphadot*(dGdpsi*I-G*dIdpsi) + dGdpsi*dphidtheta - dIdpsi*dphidzeta)))/denom;
    derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK + threadIdx.x] = modB; // modB for setting mu
    derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = G;
    // derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = // no boundary dist fn
    

};

// calc_derivs computes the derivatives at points for which the corresponding
// i,j,k indices and shape functions have been precomputed
// the results are stored in the appropriate region of derivs
// nparticles_blk stores the number of actual particles in the block
//
// this function is templated across rhs options
template<typename T, RHS id, int deriv_id, typename... Args>  
__device__ void calc_derivs(T* derivs, const T* __restrict__ quadpts_arr, T* x_temp, bool* symmetry_exploited, 
                                    int* cell_index_start, T* shape_fun_vals, T* mu, int nparticles_blk, 
                                // optional parameters for SAW cases
                                T saw_omega = 0, int* saw_m = nullptr, int* saw_n = nullptr, 
                                T* saw_phihats = nullptr, int saw_nharmonics = 0){
    
    
    constexpr int n = map_rhs_to_n_interpolants<id>();
    __shared__ T block_interpolants[n*PARTICLES_PER_BLOCK];
    interpolate<T, n>(block_interpolants, quadpts_arr, cell_index_start, shape_fun_vals, nparticles_blk);
    __syncthreads();

    if(threadIdx.x < nparticles_blk){
        if constexpr (id == RHS::GC_CartesianVacuum){
            rhs_GC_CartesianVacuum<T, deriv_id>(derivs, x_temp, block_interpolants, symmetry_exploited, mu);
        } else if constexpr(id == RHS::GC_BoozerVacuum){
            rhs_GC_BoozerVacuum<T, deriv_id>(derivs, x_temp, block_interpolants, symmetry_exploited, mu);
        } else if constexpr(id == RHS::GC_Boozer){
            rhs_GC_Boozer<T, deriv_id>(derivs, x_temp, block_interpolants, symmetry_exploited, mu);
        } else if constexpr(id == RHS::GC_BoozerVacuumSAW){
            rhs_GC_BoozerVacuumSAW<T, deriv_id>(derivs, x_temp, block_interpolants, symmetry_exploited, mu,
                saw_omega, saw_m, saw_n, saw_phihats, saw_nharmonics);
        } else if constexpr(id == RHS::GC_BoozerNoKSAW){
            rhs_GC_BoozerNoKSAW<T, deriv_id>(derivs, x_temp, block_interpolants, symmetry_exploited, mu,
                saw_omega, saw_m, saw_n, saw_phihats,saw_nharmonics);
        }
    }
};




// map to grid takes a particle location and maps it to the interpolation grid
// this is where stellarator symmetry is exploited
// and points outside of the magnetic field are mapped to the nearest cell
//
// this function is templated across rhs implementations but it's possible
// that this should be considered to be the implementation for a coordinate
// system independent of rhs
//
// interp_pt stores where the interpolant should be evaluated (in the grid)
// xyz stores the particle's current location via the x_temp array in shared
//  memory. 
// symmetry_exploited is a bool indicating whether stellarator symmetry was exploited
// there's an option for optional parameters

// map_to_grid implementation for Cartesian tracing
template <typename T>
__device__ void map_to_grid_cartesian(T* interp_pt, T* x_temp, bool* symmetry_exploited){
    T x = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T y = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T z = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];

    // convert to cylindrical coordinates for interpolation
    T r = sqrt(x*x + y*y);
    T phi = atan2(y, x);

    // restrict phi to [0, 2pi / nfp]
    T period = grid_ranges_d[5];
    phi = fmod(phi, period);
    phi += period*(phi < 0);

    // exploit stellarator symmetry
    symmetry_exploited[threadIdx.x] = z < 0;
    if(symmetry_exploited[threadIdx.x]){
        z = -z;
        phi = 2*M_PI - phi;
        phi = fmod(phi, period);
        phi += period*(phi < 0);
    }

    interp_pt[threadIdx.x] = r;
    interp_pt[PARTICLES_PER_BLOCK + threadIdx.x] = phi;
    interp_pt[2*PARTICLES_PER_BLOCK + threadIdx.x] = z;
} 

// map_to_grid implementation for Boozer tracing
template <typename T>
__device__ void map_to_grid_boozer(T* interp_pt, T* x_temp, bool* symmetry_exploited){

    T x1 = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T x2 = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T s = sqrt(x1*x1 + x2*x2);
    T theta = atan2(x2, x1);
    T z = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x]; // zeta

    // we want to exploit periodicity in the B-field, but leave sine(theta) unchanged
    T t = fmod(theta, 2*M_PI);
    t += 2*M_PI*(t < 0);

    // we can modify z because it's only used to access the B-field location
    T period = grid_ranges_d[9];
    z = fmod(z, period);
    z += period*(z < 0);

    // exploit stellarator symmetry
    symmetry_exploited[threadIdx.x] = t > M_PI;
    if(symmetry_exploited[threadIdx.x]){
        z = period - z;
        t = 2*M_PI - t;

    }
    interp_pt[threadIdx.x] = s;
    interp_pt[PARTICLES_PER_BLOCK + threadIdx.x] = t;
    interp_pt[2*PARTICLES_PER_BLOCK + threadIdx.x] = z;
}


template<typename T, CoordSys coord>
__device__ void map_to_grid(T* interp_pt, T* xyz, bool* symmetry_exploited){
    if constexpr (coord == CoordSys::Cartesian){
        map_to_grid_cartesian(interp_pt, xyz, symmetry_exploited);
    } else if constexpr (coord == CoordSys::Boozer){
        map_to_grid_boozer(interp_pt, xyz, symmetry_exploited);
    }
};                                    



// build_state is part of the DP5 implementation
template <typename T, RHS id, int deriv_id>
__device__ void build_state(T* x_temp, bool* symmetry_exploited, int* cell_index_start,
                            T* shape_fun_vals, T* state, T* derivs, T* t, T* dt, int nparticles_blk){

    
    // store time
    if(threadIdx.x < nparticles_blk){
        x_temp[threadIdx.x] = t[threadIdx.x] + dp5_t_wgts[deriv_id]*dt[threadIdx.x];
    }

    // store current location in x_temp
    for (int idx=threadIdx.x; idx<4*PARTICLES_PER_BLOCK; idx+=PARTICLES_PER_BLOCK) {
        int particle_id = idx % PARTICLES_PER_BLOCK;
        int state_var = idx / PARTICLES_PER_BLOCK;

        x_temp[(state_var+1)*PARTICLES_PER_BLOCK + particle_id] = state[state_var*PARTICLES_PER_BLOCK + particle_id];
        T dt_particle = dt[particle_id];
        for(int j=0; j<deriv_id; ++j){
            x_temp[(state_var+1)*PARTICLES_PER_BLOCK + particle_id] += dt_particle * dp5_wgts[deriv_id][j] * derivs[(6*j+state_var)*PARTICLES_PER_BLOCK + particle_id];
        }
    }
    __syncthreads();
    
    // one thread per particle maps the position to the interpolant grid.

    __shared__ T interp_pt[3*PARTICLES_PER_BLOCK];

    if(threadIdx.x < nparticles_blk){
        constexpr CoordSys coord = map_rhs_to_coord<id>();
        map_to_grid<T, coord>(interp_pt, x_temp, symmetry_exploited);
    }
    __syncthreads();

    // threads map each coordinate for each particle and compute its shape functions.
    if(threadIdx.x < 3*PARTICLES_PER_BLOCK){ // assumes 64 threads per block, 8 particles per block (3*8 = 24 < 64)
        int particle_id = threadIdx.x % PARTICLES_PER_BLOCK;
        int coord_id = threadIdx.x / PARTICLES_PER_BLOCK;

        T value = interp_pt[threadIdx.x];
        T grid_size = grid_ranges_d[coord_id*4 + 3];
        T min_bound = grid_ranges_d[coord_id*4 + 0];

        T inv_grid_size = (T)1.0 / grid_size;

        int index = 3*((int) ((value - min_bound) * inv_grid_size) /3);
        index = min(index, (int)grid_ranges_d[coord_id*4 + 2]-4);
        index = max(index, 0);

        T value_rel = (value - index*grid_size - min_bound) * inv_grid_size;

         // compute 4 shape values for this coordinate and particle
        T x_minus_1 = value_rel - (T)1.0;
        T x_minus_2 = value_rel - (T)2.0;
        T x_minus_3 = value_rel - (T)3.0;

        constexpr T one_sixth = (T) (1.0/6.0);
        constexpr T one_half = (T) 0.5;

        // compute shared terms for shape functions
        T prod23 = x_minus_2 * x_minus_3;
        T prodx1 = value_rel * x_minus_1;
        int base_idx = (coord_id*4)*PARTICLES_PER_BLOCK + particle_id;

        shape_fun_vals[base_idx + 0] = - one_sixth * x_minus_1 * prod23;
        shape_fun_vals[base_idx + 1*PARTICLES_PER_BLOCK] = one_half * value_rel * prod23;
        shape_fun_vals[base_idx + 2*PARTICLES_PER_BLOCK] = -one_half * prodx1 * x_minus_3;
        shape_fun_vals[base_idx + 3*PARTICLES_PER_BLOCK] = one_sixth * prodx1 * x_minus_2;

        cell_index_start[3*particle_id + coord_id] = index/3;
    }
    __syncthreads();

};


template<typename T>
__device__ void max_stepsize_cartesian(T* dtmax, T* loc, T* derivs){
    T x = loc[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T y = loc[2*PARTICLES_PER_BLOCK + threadIdx.x];
    T z = loc[3*PARTICLES_PER_BLOCK + threadIdx.x];
    T v_par = loc[4*PARTICLES_PER_BLOCK + threadIdx.x];

    T r = sqrt(x*x + y*y);
    dtmax[threadIdx.x] = r*0.5*M_PI / v_total_d;
}


template<typename T>
__device__ void max_stepsize_boozer(T* dtmax, T* loc, T* derivs){
    T modB = derivs[(6*0 + 4)*PARTICLES_PER_BLOCK + threadIdx.x];
    T G = derivs[(6*0 + 5)*PARTICLES_PER_BLOCK + threadIdx.x];
    dtmax[threadIdx.x] = (G / modB)*0.5*M_PI / v_total_d;
}

// calculate maximum allowable timestep to allow at most a quarter of a revolution per step
template<typename T, CoordSys coord>
__device__ void calc_max_timestep_size(T* dtmax, T* loc, T* derivs){
    if constexpr (coord == CoordSys::Cartesian){
        max_stepsize_cartesian(dtmax, loc, derivs);
    } else if constexpr (coord == CoordSys::Boozer){
        max_stepsize_boozer(dtmax, loc, derivs);
    } else{
        printf("max timestep size calculation not implemented for this coordinate system\n");
    }
};



// set up particles for tracing
// use the derivatives function to calculate mu, max step size
// store these values for the remainder of tracing

template<typename T, RHS id, typename... Args>
__device__ void setup_particle(T* mu, T* t, T* dt, T* dtmax, T* x_temp, bool* symmetry_exploited, int* cell_index_start,
                            const T* __restrict__ quad_pts, T* shape_fun_vals, T* state, T* derivs,
                            int nparticles_blk, Args... args){

    
    if(threadIdx.x < nparticles_blk){
        t[threadIdx.x] = 0.0;
        symmetry_exploited[threadIdx.x] = false;
        // dummy call to get norm B
        mu[threadIdx.x] = -1.0; // initialize mu
    }
    build_state<T, id, 0>(x_temp, symmetry_exploited, cell_index_start,
                        shape_fun_vals, state, derivs, t, dt, nparticles_blk);
    __syncthreads();
    calc_derivs<T, id, 0>(derivs, quad_pts, x_temp, symmetry_exploited, cell_index_start,
                     shape_fun_vals, mu, nparticles_blk, args...);
    __syncthreads();

    if(threadIdx.x < nparticles_blk){
        T v_par = state[3*PARTICLES_PER_BLOCK + threadIdx.x];
        T v_perp2 = v_total_d*v_total_d - v_par*v_par;
        
        T modB = derivs[4*PARTICLES_PER_BLOCK + threadIdx.x];
        mu[threadIdx.x] = v_perp2 / (2*modB);

        constexpr CoordSys coord = map_rhs_to_coord<id>();
        calc_max_timestep_size<T, coord>(dtmax, x_temp, derivs);
        dtmax[threadIdx.x] = fmin(dtmax[threadIdx.x], tmax_d);

        if(dt[threadIdx.x] == -1.0){ // dummy value from python when dt needs to be computed
            dt[threadIdx.x] = 1e-3*dtmax[threadIdx.x];
        }
    }
}

template<typename T>
__device__ void check_has_left_cartesian(bool* has_left, T* state, T* derivs){
    has_left[threadIdx.x] = derivs[(6*6 + 5)*PARTICLES_PER_BLOCK + threadIdx.x] < 0; // boundary dist fn at new location
}

template<typename T>
__device__ void check_has_left_boozer(bool* has_left, T* state, T* derivs){
    T x1 = state[0*PARTICLES_PER_BLOCK + threadIdx.x];
    T x2 = state[1*PARTICLES_PER_BLOCK + threadIdx.x];
    T s = sqrt(x1*x1 + x2*x2);

    has_left[threadIdx.x] = s >= 1; 
}


// determine whether a particle has been lost or not
// in cartesian coordinates, we check the signed distance function
// in boozer coordinates we check for s >= 1
template<typename T, CoordSys coord>
__device__ void check_has_left(bool* has_left, T* state, T* derivs){
    if constexpr (coord == CoordSys::Cartesian){
        check_has_left_cartesian(has_left, state, derivs);
    } else if constexpr (coord == CoordSys::Boozer){
        check_has_left_boozer(has_left, state, derivs);
    } else{
        printf("default check_has_left not implemented\n");
    }
};


// this function estimates error, accepts/rejects the proposed step
// and adjust the step size
template<typename T, RHS id>
__device__ void adjust_time(T* t, T* dt, T* state, T* derivs, T* x_temp, bool* has_left, T* dtmax){
    if(has_left[threadIdx.x]){
        return;
    }
    const T bhat1 = 71.0 / 57600.0, bhat3 = -71.0 / 16695.0, bhat4 = 71.0 / 1920.0, bhat5 = -17253.0 / 339200.0, bhat6 = 22.0 / 525.0, bhat7 = -1.0 / 40.0;
    // Compute  error
    // https://live.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
    // resolve typo in boost docs: https://numerical.recipes/book.html
    T max_err = 0.0;
    T err_elt;
    for(int i = 0; i < 4; i++) {
        T state_i = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        T deriv_i = derivs[(6*0 + i)*PARTICLES_PER_BLOCK + threadIdx.x];
        err_elt = dt[threadIdx.x]*(bhat1 * deriv_i
                                 + bhat3 * derivs[(6*2 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat4 * derivs[(6*3 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat5 * derivs[(6*4 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat6 * derivs[(6*5 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat7 * derivs[(6*6 + i)*PARTICLES_PER_BLOCK + threadIdx.x]);
        T atol_i = (rescale_abstol_var_d) && (i == 3) ?  atol_d * v_total_d : atol_d;
        err_elt = fabs(err_elt) / (atol_i + rtol_d*(fabs(state_i) + dt[threadIdx.x]*fabs(deriv_i)));
        max_err = fmax(max_err, err_elt);
    }

    // Compute new step size
    T dt_new = dt[threadIdx.x]*0.9;
    T exponent = 0.0;
    if(max_err > 1.0){
        exponent = -1.0/3.0;
    } 
    if(max_err < 0.5) {
        exponent = -1.0/5.0;
    }
    dt_new *= pow(max_err, exponent);
    dt_new = fmax(dt_new, 0.2 * dt[threadIdx.x]);
    dt_new = fmin(dt_new, 5.0 * dt[threadIdx.x]);   

    if(max_err <= 1.0) {
        // if the error is moderate, don't use a new step size
        if (0.5 < max_err){
            dt_new = dt[threadIdx.x];
        }
        // Accept the step
        t[threadIdx.x] += dt[threadIdx.x];

        if(t[threadIdx.x] < tmax_d){
            dt[threadIdx.x] = fmin(dt_new, tmax_d - t[threadIdx.x]);
        }

        for(int i = 0; i < 4; i++) {
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = x_temp[(i+1)*PARTICLES_PER_BLOCK + threadIdx.x];
        }
        // check if particle has left the device
        constexpr CoordSys coord = map_rhs_to_coord<id>();
        check_has_left<T, coord>(has_left, state, derivs);
    } else {
        // Reject the step and try again with smaller dt
        dt[threadIdx.x] = dt_new;
    }
}

// helper function for a single DP5 evaluation
template<typename T, RHS id, int deriv_id, typename... Args>
__device__ void dp5_one_step(T* x_temp, T* derivs, const T* __restrict__ quadpts_arr, int* cell_index_start,
                            T* shape_fun_vals, T* t, T* dt,
                            bool* symmetry_exploited, T* state, T* mu, int nparticles_blk, Args... args){
    // if the thread is responsible for a particle, compute the point at which the derivative will be computed
    build_state<T, id, deriv_id>(x_temp, symmetry_exploited, cell_index_start, shape_fun_vals, state, derivs, t, dt, nparticles_blk);
    // ensure that all threads have updated x_temp before calculating derivatives, where a data race would occur
    __syncthreads();
    calc_derivs<T, id, deriv_id>(derivs, quadpts_arr, x_temp, symmetry_exploited, cell_index_start, shape_fun_vals, mu, nparticles_blk, args...);

    // ensure all particles have derivative calculations before accepting/rejecting timestep
    __syncthreads();
}


/*
 * This function puts it all together. The while loop keeps track of the work the block has remaining
 * The inner loop computes the 7 Dormand Prince derivative estimates.
 * Everything lives in shared memory except the data for the interpolant
 */
template<typename T, RHS id, typename... Args>
__global__ void particle_trace_kernel(T* out, T* init_pos, const T* __restrict__ quadpts_arr, T* dt_in, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;

    __shared__ T x_temp[5 * PARTICLES_PER_BLOCK];
    __shared__ T derivs[42 * PARTICLES_PER_BLOCK];
    __shared__ T dt[PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int cell_index_start[3*PARTICLES_PER_BLOCK];
    __shared__ T shape_fun_vals[12*PARTICLES_PER_BLOCK]; // 4 shape function values for each of the 3 coordinates
    __shared__ T mu[PARTICLES_PER_BLOCK];
    __shared__ T t[PARTICLES_PER_BLOCK];
    __shared__ T dtmax[PARTICLES_PER_BLOCK];
    __shared__ T state[4 * PARTICLES_PER_BLOCK];
    __shared__ bool has_left[PARTICLES_PER_BLOCK];


    bool is_valid = idx < nparticles_d && threadIdx.x < PARTICLES_PER_BLOCK;
    int nparticles_blk = __syncthreads_count(is_valid);

    // if thread is responsible for a valid particle id, load that particle's data
    if(is_valid){
        has_left[threadIdx.x] = true;
        t[threadIdx.x] = 0.0;
        has_left[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = init_pos[4*idx + i];
        }
        dt[threadIdx.x] = dt_in[threadIdx.x]; // copy input dt
    }
    __syncthreads();

    // calculate the particle's magnetic moment mu, dt, dtmax
    setup_particle<T, id>(mu, t, dt, dtmax, x_temp, symmetry_exploited, cell_index_start,
                        quadpts_arr, shape_fun_vals, state, derivs, nparticles_blk, args...);
    __syncthreads();

    // if there exists a particle which is real and hasn't not reached tmax or left, keep tracing
    while(__syncthreads_count(is_valid && !(t[threadIdx.x] >= tmax_d || has_left[threadIdx.x])) > 0){
        dp5_one_step<T, id, 0>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<T, id, 1>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<T, id, 2>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<T, id, 3>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<T, id, 4>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<T, id,  5>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<T, id, 6>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        if(is_valid){
            adjust_time<T, id>(t, dt, state, derivs, x_temp, has_left, dtmax);
        }
        __syncthreads();
    }
    __syncthreads();
    if(is_valid){
        out[6*idx] = t[threadIdx.x];
        for(int i=0; i<4; ++i){
            out[6*idx + i + 1] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }
        out[6*idx + 5] = dt[threadIdx.x];
    }
    return;
}


template<typename T, RHS id, typename... Args>
vector<T> gpu_tracing(py::array_t<T> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
    py::array_t<T> loc_init, double m, double q, double vtotal, py::array_t<T> vtang, double tmax, double tol, py::array_t<T> dt_in, int nparticles, Args... args){

    //  read data in from python
    T* loc_init_arr = create_array(loc_init);
    T* vtang_arr = create_array(vtang);
    T* quadpts_arr = create_array(quad_pts);
    double* x1_range_arr = create_array(x1_range);
    double* x2_range_arr = create_array(x2_range);
    double* x3_range_arr = create_array(x3_range);
    T* dt_in_arr = create_array(dt_in);

    // allocate and copy to device memory
    double x1_range_ext[4];
    double x2_range_ext[4];
    double x3_range_ext[4];

    for(int i=0; i<3; ++i){
        x1_range_ext[i] = x1_range_arr[i];
        x2_range_ext[i] = x2_range_arr[i];
        x3_range_ext[i] = x3_range_arr[i];
    }
    x1_range_ext[3] = (x1_range_ext[1] - x1_range_ext[0]) / (x1_range_ext[2] - 1);
    x2_range_ext[3] = (x2_range_ext[1] - x2_range_ext[0]) / (x2_range_ext[2] - 1);
    x3_range_ext[3] = (x3_range_ext[1] - x3_range_ext[0]) / (x3_range_ext[2] - 1);

    int n_x2 = (x2_range_ext[2]-1)/3;
    int n_x3 = (x3_range_ext[2]-1)/3;
    int n_x23 = n_x2*n_x3;

    // combine ranges into one array
    double grid_ranges[12];
    for(int i=0; i<4; ++i){
        grid_ranges[i] = x1_range_ext[i];
        grid_ranges[4 + i] = x2_range_ext[i];
        grid_ranges[8 + i] = x3_range_ext[i];
    }

    gpuErrchk(cudaMemcpyToSymbol(grid_ranges_d, grid_ranges, 12*sizeof(double)) );
    // gpuErrchk(cudaMemcpyToSymbol(x2_range_d, x2_range_ext, 4*sizeof(double)) );
    // gpuErrchk(cudaMemcpyToSymbol(x3_range_d, x3_range_ext, 4*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(tmax_d, &tmax, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(mass_d, &m, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(charge_d, &q, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(atol_d, &tol, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(rtol_d, &tol, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(v_total_d, &vtotal, sizeof(double)));


    gpuErrchk(cudaMemcpyToSymbol(n_x2_d, &n_x2, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x3_d, &n_x3, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x23_d, &n_x23, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(nparticles_d, &nparticles, sizeof(int)));

    T init_pos[4*nparticles];
    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        T s = loc_init_arr[start];
        T theta = loc_init_arr[start+1];
        
        for(int j=0; j<3; j++){
            init_pos[4*i + j] = loc_init_arr[start + j];
        }
        init_pos[4*i + 3] = vtang_arr[i];
    }
   
    T* init_pos_d;
    gpuErrchk(cudaMalloc((void**)&init_pos_d, 4 * nparticles * sizeof(T)) );
    gpuErrchk(cudaMemcpy(init_pos_d, init_pos, 4 * nparticles * sizeof(T), cudaMemcpyHostToDevice) );

    T* quadpts_d;
    gpuErrchk(cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(T)) ); 
    gpuErrchk(cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(T), cudaMemcpyHostToDevice) );

    T* dt_in_d;
    gpuErrchk(cudaMalloc((void**)&dt_in_d, dt_in.size() * sizeof(T)) ); 
    gpuErrchk(cudaMemcpy(dt_in_d, dt_in_arr, dt_in.size() * sizeof(T), cudaMemcpyHostToDevice) );
    T* out_d;
    gpuErrchk(cudaMalloc((void**)&out_d, 6 * nparticles * sizeof(T)) ); 


    int nthreads = THREADS_PER_BLOCK;

    int nblks = nparticles  / PARTICLES_PER_BLOCK + 1;
    // std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    particle_trace_kernel<T, id><<<nblks, nthreads>>>(out_d, init_pos_d, quadpts_d, dt_in_d, args...);

    T out[6*nparticles];
    gpuErrchk(cudaMemcpy(out, out_d, 6 * nparticles * sizeof(T), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(quadpts_d) );
    gpuErrchk( cudaFree(init_pos_d) );
    gpuErrchk( cudaFree(out_d) );
    vector<T> particle_output(6*nparticles);
    for(int i=0; i<6*nparticles; ++i){
        particle_output[i] = out[i];
    }

    return particle_output;
}

vector<double> cartesian_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<double> xyz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, py::array_t<double> dt_in, int nparticles){
            return gpu_tracing<double, RHS::GC_CartesianVacuum>(quad_pts, rrange, phirange, zrange, xyz_init, m, q, vtotal, vtang, tmax, tol, dt_in, nparticles);
        }


vector<double> boozer_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, py::array_t<double> dt_in, double psi0, int nparticles, bool vacuum){

    //  read data in from python
    double* stz_init_arr = create_array(stz_init);
    
    for(int i=0; i<nparticles; ++i){
        double s = stz_init_arr[3*i];
        double theta = stz_init_arr[3*i+1];

        stz_init_arr[3*i] = s*cos(theta);
        stz_init_arr[3*i+1] = s*sin(theta);
    }
    double inv_psi0_charge = 1.0 / (psi0*q);
    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(inv_psi0_charge_d, &inv_psi0_charge, sizeof(double)));

    std::vector<double> results;
    if (vacuum) {
        results = gpu_tracing<double, RHS::GC_BoozerVacuum>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, dt_in, nparticles);
    } else {
        results = gpu_tracing<double, RHS::GC_Boozer>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, dt_in, nparticles);
    }

    for(int i=0; i<nparticles; ++i){
        double x1 = results[6*i+1];
        double x2 = results[6*i+2];

        results[6*i+1] = sqrt(x1*x1 + x2*x2);
        results[6*i+2] = atan2(x2, x1);
    }

    return results;
}


vector<double> boozer_saw_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, double tmax, double tol, py::array_t<double> dt_in, double psi0, int nparticles){

    //  read data in from python
    double* stz_init_arr = create_array(stz_init);
    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    gpuErrchk( cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int)) );
    gpuErrchk( cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice) );

    int* saw_n_d;
    gpuErrchk( cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int)) );
    gpuErrchk( cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice) );

    double* saw_phihats_d;
    gpuErrchk( cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice) );
    
    for(int i=0; i<nparticles; ++i){
        double s = stz_init_arr[3*i];
        double theta = stz_init_arr[3*i+1];

        stz_init_arr[3*i] = s*cos(theta);
        stz_init_arr[3*i+1] = s*sin(theta);
    }
    // copy saw s_range to constant memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];   
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));

    std::vector<double> results =  gpu_tracing<double, RHS::GC_BoozerVacuumSAW>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, dt_in, nparticles,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);

    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );

    for(int i=0; i<nparticles; ++i){
        double x1 = results[6*i+1];
        double x2 = results[6*i+2];

        results[6*i+1] = sqrt(x1*x1 + x2*x2);
        results[6*i+2] = atan2(x2, x1);
    }

    return results;
}

vector<double> boozer_saw_nok_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, double tmax, double tol, py::array_t<double> dt_in, double psi0, int nparticles){

    //  read data in from python
    double* stz_init_arr = create_array(stz_init);
    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);
    
    for(int i=0; i<nparticles; ++i){
        double s = stz_init_arr[3*i];
        double theta = stz_init_arr[3*i+1];

        stz_init_arr[3*i] = s*cos(theta);
        stz_init_arr[3*i+1] = s*sin(theta);
    }
    // copy saw s_range to constant memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];   
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));

    std::vector<double> results =  gpu_tracing<double, RHS::GC_BoozerNoKSAW>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, dt_in, nparticles,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);

    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );

    for(int i=0; i<nparticles; ++i){
        double x1 = results[6*i+1];
        double x2 = results[6*i+2];

        results[6*i+1] = sqrt(x1*x1 + x2*x2);
        results[6*i+2] = atan2(x2, x1);
    }

    return results;
}


/*
 * This function accounts for exploiting stellarator symmetry
 * It is only used in the interpolant test.
 */ 

template<CoordSys coord>
__device__ void account_for_symmetry(double* interpolants, bool* symmetry_exploited){
    printf("default account_for_symmetry not implemented\n");
};

template<>
__device__ void account_for_symmetry<CoordSys::Cartesian>(double* interpolants, bool* symmetry_exploited){
    if(symmetry_exploited[threadIdx.x]){
        interpolants[0] *= -1.0;
        interpolants[4] *= -1.0;
        interpolants[5] *= -1.0;
    }
}

template<>
__device__ void account_for_symmetry<CoordSys::Boozer>(double* interpolants, bool* symmetry_exploited){
    // modB, dmodBds, dmodBdtheta, dmodBdzeta, G, iota
    if(symmetry_exploited[threadIdx.x]){
        interpolants[2] *= -1.0;
        interpolants[3] *= -1.0;
    }
}

// RHS-aware symmetry correction used by the interpolation test helper
template<RHS id, int n>
__device__ void account_for_symmetry_rhs(double* interpolants, bool* symmetry_exploited){
    if(!symmetry_exploited[threadIdx.x]) return;
    if constexpr (id == RHS::GC_CartesianVacuum){
        interpolants[0] *= -1.0;
        interpolants[4] *= -1.0;
        interpolants[5] *= -1.0;
    } else if constexpr (id == RHS::GC_BoozerVacuum || id == RHS::GC_BoozerVacuumSAW){
        // Only theta/zeta derivatives flip sign
        interpolants[2] *= -1.0;
        interpolants[3] *= -1.0;
    } else if constexpr (id == RHS::GC_Boozer){
        // 12-field ordering: flip dB/dtheta, dB/dzeta, and K
        interpolants[2] *= -1.0;  // d|B|/dtheta
        interpolants[3] *= -1.0;  // d|B|/dzeta
        if constexpr (n >= 12) {
            interpolants[9] *= -1.0; // K
        }
    }
}


template <RHS id, int n>
__global__ void test_gpu_interpolation_kernel(double* quad_pts, double* loc, double* out, int n_points){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;

    __shared__ double x_temp[5 * PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int cell_index_start[3*PARTICLES_PER_BLOCK];
    __shared__ double shape_fun_vals[12*PARTICLES_PER_BLOCK];
    __shared__ double state[4 * PARTICLES_PER_BLOCK];
    __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
    __shared__ double dt[PARTICLES_PER_BLOCK];
    __shared__ double t[PARTICLES_PER_BLOCK];

    __shared__ double block_interpolants[n*PARTICLES_PER_BLOCK];


    double* loc_arr = loc + 3*idx;
    double* out_arr  =  out + idx*n;

    bool is_valid = idx < n_points && threadIdx.x < PARTICLES_PER_BLOCK;
    int nparticles_blk = __syncthreads_count(is_valid);
    if(is_valid){
        dt[threadIdx.x] = 1e-3; // needed for build_state
        symmetry_exploited[threadIdx.x] = false;
        for(int i=0; i<3; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = loc_arr[i];
        }
        state[3*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0; // dummy vpar value
        t[threadIdx.x] = 0.0; // dummy time value


        for(int i=0; i<n; ++i){
            block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0;
        }
    } 

    build_state<double, id, 0>(x_temp, symmetry_exploited, cell_index_start, shape_fun_vals, state, derivs, t, dt, nparticles_blk);

    __syncthreads();
    interpolate<double, n>(block_interpolants, quad_pts, cell_index_start, shape_fun_vals, nparticles_blk);
    __syncthreads();
    
    if(is_valid){
        for(int i=0; i<n; ++i){
            out_arr[i] = block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x];

        }
        // Apply symmetry fixes with RHS/layout awareness
        account_for_symmetry_rhs<id, n>(out_arr, symmetry_exploited);
    }
}



py::array_t<double> test_gpu_interpolation(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc, std::string rhs, int n_points){
    // read data in from python
    double* quadpts_arr = create_array(quad_pts);
    double* x1_range_arr = create_array(x1_range);
    double* x2_range_arr = create_array(x2_range);
    double* x3_range_arr = create_array(x3_range);
    double* loc_arr = create_array(loc);
    
    // map input data
    // Cartesian Coordinates
    if(rhs == "cartesian_vacuum"){
        for(int i=0; i<n_points; ++i){
            double x = loc_arr[3*i] * cos(loc_arr[3*i + 1]);
            double y = loc_arr[3*i] * sin(loc_arr[3*i + 1]);
            
            loc_arr[3*i] = x;
            loc_arr[3*i+1] = y;
        }
    }

    // Boozer Coordinates
    if((rhs == "boozer_vacuum") || (rhs == "boozer_saw_vacuum") || (rhs == "boozer")) {
        for(int i=0; i<n_points; ++i){
            double x1 = loc_arr[3*i] * cos(loc_arr[3*i + 1]);
            double x2 = loc_arr[3*i] * sin(loc_arr[3*i + 1]);
            
            loc_arr[3*i] = x1;
            loc_arr[3*i+1] = x2;
        }
    }

    int n;
    if(rhs == "cartesian_vacuum"){
        n = 7;
    } else if(rhs == "boozer_vacuum"){
        n = 6;
    } else if(rhs == "boozer_saw_vacuum"){
        n = 10;
    } else if(rhs == "boozer"){
        n = 12;
    }

    // allocate and copy to device memory
    double x1_range_ext[4];
    double x2_range_ext[4];
    double x3_range_ext[4];

    for(int i=0; i<3; ++i){
        x1_range_ext[i] = x1_range_arr[i];
        x2_range_ext[i] = x2_range_arr[i];
        x3_range_ext[i] = x3_range_arr[i];
    }
    x1_range_ext[3] = (x1_range_ext[1] - x1_range_ext[0]) / (x1_range_ext[2] - 1);
    x2_range_ext[3] = (x2_range_ext[1] - x2_range_ext[0]) / (x2_range_ext[2] - 1);
    x3_range_ext[3] = (x3_range_ext[1] - x3_range_ext[0]) / (x3_range_ext[2] - 1);

    int n_x2 = (x2_range_ext[2]-1)/3;
    int n_x3 = (x3_range_ext[2]-1)/3;
    int n_x23 = n_x2*n_x3;

    double grid_ranges[12];
    for(int i=0; i<4; ++i){
        grid_ranges[i] = x1_range_ext[i];
        grid_ranges[4 + i] = x2_range_ext[i];
        grid_ranges[8 + i] = x3_range_ext[i];
    }

    gpuErrchk(cudaMemcpyToSymbol(grid_ranges_d, grid_ranges, 12*sizeof(double)) );

    gpuErrchk(cudaMemcpyToSymbol(n_x2_d, &n_x2, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x3_d, &n_x3, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x23_d, &n_x23, sizeof(int)) );

    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(double));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    cudaMalloc((void**)&out_d, n*n_points * sizeof(double));

    int nthreads = THREADS_PER_BLOCK;
    int nblks = n_points / PARTICLES_PER_BLOCK + 1;

    if(rhs == "cartesian_vacuum"){
        test_gpu_interpolation_kernel<RHS::GC_CartesianVacuum, 7><<<nblks, nthreads>>>(quadpts_d, loc_d, out_d, n_points);
    } else if(rhs == "boozer_vacuum") {
        test_gpu_interpolation_kernel<RHS::GC_BoozerVacuum, 6><<<nblks, nthreads>>>(quadpts_d, loc_d, out_d, n_points);
    } else if(rhs == "boozer_saw_vacuum") {
        test_gpu_interpolation_kernel<RHS::GC_BoozerVacuumSAW, 10><<<nblks, nthreads>>>(quadpts_d, loc_d, out_d, n_points);
    } else if(rhs == "boozer") {
        test_gpu_interpolation_kernel<RHS::GC_Boozer, 12><<<nblks, nthreads>>>(quadpts_d, loc_d, out_d, n_points);
    }
    double out[n*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, n*n_points * sizeof(double), cudaMemcpyDeviceToHost) );

    auto result = py::array_t<double>(n*n_points, out);

    gpuErrchk( cudaFree(quadpts_d) );
    gpuErrchk( cudaFree(loc_d) );
    gpuErrchk( cudaFree(out_d) );

    return result;

}


template<RHS id, typename... Args>
__global__ void test_gpu_derivs_kernel(double* quad_pts, double* loc, double* vpar, double* time, double* out, int n_points, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;    
    double* loc_arr = loc + 3*idx;
    double* out_arr  =  out + 4*idx;

    __shared__ double x_temp[5 * PARTICLES_PER_BLOCK];
    __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
    __shared__ double dt[PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int cell_index_start[3*PARTICLES_PER_BLOCK];
    __shared__ double shape_fun_vals[12*PARTICLES_PER_BLOCK];
    __shared__ double mu[PARTICLES_PER_BLOCK];
    __shared__ double t[PARTICLES_PER_BLOCK];
    __shared__ double dtmax[PARTICLES_PER_BLOCK];
    __shared__ double state[4 * PARTICLES_PER_BLOCK];

    bool is_valid = idx < n_points && threadIdx.x < PARTICLES_PER_BLOCK;
    int nparticles_blk = __syncthreads_count(is_valid);

    if(is_valid){
        double vpar_val = vpar[idx];
        double r = loc_arr[0];
        double phi = loc_arr[1];
        double z = loc_arr[2];

        state[threadIdx.x] = r*cos(phi);
        state[PARTICLES_PER_BLOCK + threadIdx.x] = r*sin(phi);
        state[2*PARTICLES_PER_BLOCK + threadIdx.x] = z;
        state[3*PARTICLES_PER_BLOCK + threadIdx.x] = vpar_val;

        t[threadIdx.x] = time[idx];
    }
    __syncthreads();

    setup_particle<double, id>(mu, t, dt, dtmax, x_temp, symmetry_exploited, cell_index_start,
                        quad_pts, shape_fun_vals, state, derivs, nparticles_blk, args...);

    __syncthreads();

    // set non-zero time
    if(is_valid){
        t[threadIdx.x] = time[idx];
    }
    build_state<double, id, 0>(x_temp, symmetry_exploited, cell_index_start,
            shape_fun_vals, state, derivs, t, dt, nparticles_blk);
    __syncthreads();
    calc_derivs<double, id, 0>(derivs, quad_pts, x_temp, symmetry_exploited, cell_index_start, shape_fun_vals, mu, nparticles_blk, args...);
    __syncthreads();

    if(is_valid){
        // copy back
        for(int i=0; i<4; ++i){
            out_arr[i] = derivs[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }

    }
}

template<RHS id, typename... Args>
py::array_t<double> test_gpu_derivatives(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range,
                                 py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q, int n_points, Args... args){

    double* quadpts_arr = create_array(quad_pts);
    double* x1_range_arr = create_array(x1_range);
    double* x2_range_arr = create_array(x2_range);
    double* x3_range_arr = create_array(x3_range);
    double* loc_arr = create_array(loc);
    double* vpar_arr = create_array(vpar);
    double* time_arr = create_array(time);
    
    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(double));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* vpar_d;
    cudaMalloc((void**)&vpar_d, vpar.size() * sizeof(double));
    cudaMemcpy(vpar_d, vpar_arr, vpar.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* time_d;
    cudaMalloc((void**)&time_d, n_points*sizeof(double));
    cudaMemcpy(time_d, time_arr, n_points * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    cudaMalloc((void**)&out_d, 4*n_points * sizeof(double));

    // allocate and copy to device memory
    double x1_range_ext[4];
    double x2_range_ext[4];
    double x3_range_ext[4];

    for(int i=0; i<3; ++i){
        x1_range_ext[i] = x1_range_arr[i];
        x2_range_ext[i] = x2_range_arr[i];
        x3_range_ext[i] = x3_range_arr[i];
    }
    x1_range_ext[3] = (x1_range_ext[1] - x1_range_ext[0]) / (x1_range_ext[2] - 1);
    x2_range_ext[3] = (x2_range_ext[1] - x2_range_ext[0]) / (x2_range_ext[2] - 1);
    x3_range_ext[3] = (x3_range_ext[1] - x3_range_ext[0]) / (x3_range_ext[2] - 1);

    int n_x2 = (x2_range_ext[2]-1)/3;
    int n_x3 = (x3_range_ext[2]-1)/3;
    int n_x23 = n_x2*n_x3;
    double tmax = 1e-2; // needed for setup_particle

    double grid_ranges[12];
    for(int i=0; i<4; ++i){
        grid_ranges[i] = x1_range_ext[i];
        grid_ranges[4 + i] = x2_range_ext[i];
        grid_ranges[8 + i] = x3_range_ext[i];
    }

    gpuErrchk(cudaMemcpyToSymbol(grid_ranges_d, grid_ranges, 12*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(tmax_d, &tmax, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(mass_d, &m, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(charge_d, &q, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(v_total_d, &v_total, sizeof(double)));

    gpuErrchk(cudaMemcpyToSymbol(n_x2_d, &n_x2, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x3_d, &n_x3, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x23_d, &n_x23, sizeof(int)) );

    bool is_test = true;
    gpuErrchk(cudaMemcpyToSymbol(is_test_d, &is_test, sizeof(bool)) ); 

    int nthreads = THREADS_PER_BLOCK;
    int nblks = n_points / PARTICLES_PER_BLOCK + 1;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
        
    test_gpu_derivs_kernel<id><<<nblks, nthreads>>>(quadpts_d, loc_d, vpar_d, time_d, out_d, n_points, args...);
    
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "derivatives kernel time (ms): " << milliseconds<< "\n";
    
    double out[4*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, 4*n_points * sizeof(double), cudaMemcpyDeviceToHost) );
    auto result = py::array_t<double>(4*n_points, out);
    
    gpuErrchk( cudaFree(quadpts_d) );
    gpuErrchk( cudaFree(loc_d) );
    gpuErrchk( cudaFree(vpar_d) );
    gpuErrchk( cudaFree(time_d) );
    gpuErrchk( cudaFree(out_d) );

    return result;
}


py::array_t<double> test_derivatives_cartesian(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, int n_points){        
    py::array_t<double> time = py::array_t<double>(n_points); // dummy time
    return test_gpu_derivatives<RHS::GC_CartesianVacuum>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points);
}



py::array_t<double> test_derivatives_boozer(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, double psi0, int n_points, bool vacuum){
    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    double inv_psi0_charge = 1.0 / (psi0*q);
    gpuErrchk(cudaMemcpyToSymbol(inv_psi0_charge_d, &inv_psi0_charge, sizeof(double)));
    py::array_t<double> time = py::array_t<double>(n_points); // dummy time
    if (vacuum) {
        return test_gpu_derivatives<RHS::GC_BoozerVacuum>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points);
    } else {
        return test_gpu_derivatives<RHS::GC_Boozer>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points);
    }
}

py::array_t<double> test_derivatives_saw(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q,  double psi0, int n_points){

    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);
    
    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);

    // allocate and copy to device memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];
        
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    
    py::array_t<double> out = test_gpu_derivatives<RHS::GC_BoozerVacuumSAW>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);

    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );
    return out;
}

py::array_t<double> test_derivatives_saw_nok(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time, double v_total, double m, double q,  double psi0, int n_points){

    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);

    // allocate and copy to device memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
        
    py::array_t<double> out = test_gpu_derivatives<RHS::GC_BoozerNoKSAW>(quad_pts, x1_range, x2_range, x3_range, loc, vpar, time, v_total, m, q, n_points,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);
    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );

    return out;
}

template<RHS id, typename... Args>
__global__ void test_gpu_timestep_kernel(double* out, double* init_pos, double* quadpts_arr, int nparticles, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;

    __shared__ double x_temp[5 * PARTICLES_PER_BLOCK];
    __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
    __shared__ double dt[PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int cell_index_start[3 * PARTICLES_PER_BLOCK];
    __shared__ double shape_fun_vals[12 * PARTICLES_PER_BLOCK];
    __shared__ double mu[PARTICLES_PER_BLOCK];
    __shared__ double t[PARTICLES_PER_BLOCK];
    __shared__ double dtmax[PARTICLES_PER_BLOCK];
    __shared__ double state[4 * PARTICLES_PER_BLOCK];
    __shared__ bool has_left[PARTICLES_PER_BLOCK];

    bool is_valid = idx < nparticles && threadIdx.x < PARTICLES_PER_BLOCK;
    int nparticles_blk = __syncthreads_count(is_valid);

    // if thread is responsible for a valid particle id, load that particle's data
    if(is_valid){
        has_left[threadIdx.x] = true;
        t[threadIdx.x] = 0.0;
        dt[threadIdx.x] = -1.0; //there is no dt input for time step test
        has_left[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = init_pos[4*idx+i];
        }
    }
    __syncthreads();

    // calculate the particle's magnetic moment mu, dt, dtmax
    setup_particle<double, id>(mu, t, dt, dtmax, x_temp, symmetry_exploited, cell_index_start,
                        quadpts_arr, shape_fun_vals, state, derivs, nparticles_blk, args...);
    __syncthreads();

    // if there exists a particle at t=0, which is a real particle, then keep tracing
    while(__syncthreads_count(t[threadIdx.x] == 0.0  && is_valid) > 0){
        // calculate the 7 Dormand-Prince 5 derivatives
        dp5_one_step<double, id, 0>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<double, id, 1>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<double, id, 2>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<double, id, 3>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<double, id, 4>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<double, id, 5>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        dp5_one_step<double, id, 6>(x_temp, derivs, quadpts_arr, cell_index_start, shape_fun_vals, t, dt,
                            symmetry_exploited, state, mu, nparticles_blk, args...);
        if(is_valid && t[threadIdx.x] == 0.0){
            adjust_time<double, id>(t, dt, state, derivs, x_temp, has_left, dtmax);
        }
        __syncthreads();
    }
    __syncthreads();
    if(is_valid){
        out[5*idx] = t[threadIdx.x];
        for(int i=0; i<4; ++i){
            out[5*idx + i + 1] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }
    }
    return;
}

template<RHS id, typename... Args>
vector<double> test_gpu_timestep(py::array_t<double> quad_pts, py::array_t<double> x1_range,
        py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, int nparticles, Args... args){

    //  read data in from python
    double* loc_init_arr = create_array(loc_init);
    double* vtang_arr = create_array(vtang);

    double* quadpts_arr = create_array(quad_pts);
    double* x1_range_arr = create_array(x1_range);
    double* x2_range_arr = create_array(x2_range);
    double* x3_range_arr = create_array(x3_range);

    // allocate and copy to device memory
    double x1_range_ext[4];
    double x2_range_ext[4];
    double x3_range_ext[4];

    for(int i=0; i<3; ++i){
        x1_range_ext[i] = x1_range_arr[i];
        x2_range_ext[i] = x2_range_arr[i];
        x3_range_ext[i] = x3_range_arr[i];
    }
    x1_range_ext[3] = (x1_range_ext[1] - x1_range_ext[0]) / (x1_range_ext[2] - 1);
    x2_range_ext[3] = (x2_range_ext[1] - x2_range_ext[0]) / (x2_range_ext[2] - 1);
    x3_range_ext[3] = (x3_range_ext[1] - x3_range_ext[0]) / (x3_range_ext[2] - 1);

    int n_x2 = (x2_range_ext[2]-1)/3;
    int n_x3 = (x3_range_ext[2]-1)/3;
    int n_x23 = n_x2*n_x3;
    double tmax = 1e-2; // needed for setup_particle

    double grid_ranges[12];
    for(int i=0; i<4; ++i){
        grid_ranges[i] = x1_range_ext[i];
        grid_ranges[4 + i] = x2_range_ext[i];
        grid_ranges[8 + i] = x3_range_ext[i];
    }

    gpuErrchk(cudaMemcpyToSymbol(grid_ranges_d, grid_ranges, 12*sizeof(double)) );
    gpuErrchk(cudaMemcpyToSymbol(tmax_d, &tmax, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(mass_d, &m, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(charge_d, &q, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(atol_d, &tol, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(rtol_d, &tol, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(v_total_d, &vtotal, sizeof(double)));

    gpuErrchk(cudaMemcpyToSymbol(n_x2_d, &n_x2, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x3_d, &n_x3, sizeof(int)) );
    gpuErrchk(cudaMemcpyToSymbol(n_x23_d, &n_x23, sizeof(int)) );

    double init_pos[4*nparticles];
    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;
        double r = loc_init_arr[start];
        double phi =  loc_init_arr[start + 1];
        
        init_pos[4*i] = r*cos(phi);
        init_pos[4*i+1] = r*sin(phi);
        init_pos[4*i+2] = loc_init_arr[start+2];
        init_pos[4*i + 3] = vtang_arr[i];
    }

   
    double* init_pos_d;
    gpuErrchk(cudaMalloc((void**)&init_pos_d, 4 * nparticles * sizeof(double)) );
    gpuErrchk(cudaMemcpy(init_pos_d, init_pos, 4 * nparticles * sizeof(double), cudaMemcpyHostToDevice) );

    double* quadpts_d;
    gpuErrchk( cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice) );

    double* out_d;
    gpuErrchk( cudaMalloc((void**)&out_d, 5 * nparticles * sizeof(double)) );

    int nthreads = THREADS_PER_BLOCK;
    int nblks = nparticles / PARTICLES_PER_BLOCK + 1;

    // std::cout << "starting particle tracing kernel\n";
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    test_gpu_timestep_kernel<id><<<nblks, nthreads>>>(out_d, init_pos_d, quadpts_d, nparticles, args...);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    double out[5*nparticles];
    gpuErrchk( cudaMemcpy(out, out_d, 5 * nparticles * sizeof(double), cudaMemcpyDeviceToHost) );

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";
    
    vector<double> particle_output(5*nparticles);
    for(int i=0; i<5*nparticles; ++i){
        particle_output[i] = out[i];
    }

    gpuErrchk( cudaFree(init_pos_d) );
    gpuErrchk( cudaFree(quadpts_d) );
    gpuErrchk( cudaFree(out_d) );

    return particle_output;
}

vector<double> test_timestep_cartesian(py::array_t<double> quad_pts, py::array_t<double> x1_range,
        py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, int nparticles){
    bool rescale_abstol_var = false;
    gpuErrchk(cudaMemcpyToSymbol(rescale_abstol_var_d, &rescale_abstol_var, sizeof(bool)) );
    return test_gpu_timestep<RHS::GC_CartesianVacuum>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, vtotal, vtang, tol, nparticles);
}

vector<double> test_timestep_boozer(py::array_t<double> quad_pts, py::array_t<double> x1_range,
        py::array_t<double> x2_range, py::array_t<double> x3_range, py::array_t<double> loc_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, double psi0, int nparticles, bool vacuum){

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    double inv_psi0_charge = 1.0 / (psi0*q);
    gpuErrchk(cudaMemcpyToSymbol(inv_psi0_charge_d, &inv_psi0_charge, sizeof(double)));
    vector<double> particle_output;
    if (vacuum) {
        particle_output = test_gpu_timestep<RHS::GC_BoozerVacuum>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, vtotal, vtang, tol, nparticles);
    } else {
        particle_output = test_gpu_timestep<RHS::GC_Boozer>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, vtotal, vtang, tol, nparticles);
    }

    for(int i=0; i<nparticles; ++i){
        double x1 = particle_output[5*i + 1];
        double x2 = particle_output[5*i + 2];
        double s = sqrt(x1*x1 + x2*x2);
        double theta = atan2(x2, x1);

        particle_output[5*i] = particle_output[5*i];
        particle_output[5*i+1] = s;
        particle_output[5*i+2] = theta;
        particle_output[5*i+3] = particle_output[5*i+3];
        particle_output[5*i+4] = particle_output[5*i+4];
    }

    return particle_output;
}

vector<double> test_timestep_saw(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc_init, double m, double q, double v_total, py::array_t<double> vtang, py::array_t<double> time,
        double tol, double psi0, int nparticles){
 
    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    gpuErrchk( cudaMalloc((void**)&out_d, 5 * nparticles * sizeof(double)) );

    // allocate and copy to device memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    vector<double> particle_output = test_gpu_timestep<RHS::GC_BoozerVacuumSAW>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, v_total, vtang, tol, nparticles,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);
    for(int i=0; i<nparticles; ++i){
        double x1 = particle_output[5*i + 1];
        double x2 = particle_output[5*i + 2];
        double s = sqrt(x1*x1 + x2*x2);
        double theta = atan2(x2, x1);

        particle_output[5*i] = particle_output[5*i];
        particle_output[5*i+1] = s;
        particle_output[5*i+2] = theta;
        particle_output[5*i+3] = particle_output[5*i+3];
        particle_output[5*i+4] = particle_output[5*i+4];
    }
    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );
    gpuErrchk( cudaFree(out_d) );
 
    return particle_output;

};

vector<double> test_timestep_saw_nok(py::array_t<double> quad_pts, py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range, 
        double saw_omega, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, int saw_nharmonics,
        py::array_t<double> loc_init, double m, double q, double v_total, py::array_t<double> vtang, py::array_t<double> time,
        double tol, double psi0, int nparticles){
 
    double* saw_srange_arr = create_array(saw_srange);
    int* saw_m_arr = create_array(saw_m);
    int* saw_n_arr = create_array(saw_n);
    double* saw_phihats_arr = create_array(saw_phihats);

    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size() * sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size() * sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size() * sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size() * sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    gpuErrchk( cudaMalloc((void**)&out_d, 5 * nparticles * sizeof(double)) );

    // allocate and copy to device memory
    double saw_srange_ext[4];
    for(int i=0; i<3; ++i){
        saw_srange_ext[i] = saw_srange_arr[i];
    }
    saw_srange_ext[3] = (saw_srange_ext[1] - saw_srange_ext[0]) / (saw_srange_ext[2] - 1);

    gpuErrchk(cudaMemcpyToSymbol(psi0_d, &psi0, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(saw_srange_d, saw_srange_ext, 4*sizeof(double)) );
    vector<double> particle_output = test_gpu_timestep<RHS::GC_BoozerNoKSAW>(quad_pts, x1_range, x2_range, x3_range, loc_init, m, q, v_total, vtang, tol, nparticles,
                                                                        saw_omega, saw_m_d, saw_n_d, saw_phihats_d, saw_nharmonics);
    for(int i=0; i<nparticles; ++i){
        double x1 = particle_output[5*i + 1];
        double x2 = particle_output[5*i + 2];
        double s = sqrt(x1*x1 + x2*x2);
        double theta = atan2(x2, x1);

        particle_output[5*i] = particle_output[5*i];
        particle_output[5*i+1] = s;
        particle_output[5*i+2] = theta;
        particle_output[5*i+3] = particle_output[5*i+3];
        particle_output[5*i+4] = particle_output[5*i+4];
    }
    gpuErrchk( cudaFree(saw_m_d) );
    gpuErrchk( cudaFree(saw_n_d) );
    gpuErrchk( cudaFree(saw_phihats_d) );
    gpuErrchk( cudaFree(out_d) );

    return particle_output;

};