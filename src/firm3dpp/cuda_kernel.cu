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

// enum used for templating
// https://stackoverflow.com/questions/9116267/how-can-i-use-an-enumeration-as-a-template-parameter
enum class RHS {GC_CartesianVacuum, GC_BoozerVacuum};


// Particle Data Structure
// This should eventually be removed
typedef struct particle_t {
    double state[4];
    double v_perp; // Velocity perpendicular
    double v_total;
    bool has_left;
    double dt;
    double dtmax;
    double t;
    double mu;
    double derivs[42] = {0.0};
    double x_temp[4], x_err[4];
    double r_shape[4], phi_shape[4], z_shape[4];
    int i, j, k;
    double interpolation_loc[3];
    bool symmetry_exploited;
    int id;
    int step_attempt, step_accept;
    double surf_dist;
} particle_t;

/* shape computes shape functions for cubic interpolation on a a regular grid
 * we assume the point x has been rescaled to be on the grid 0, 1, 2, 3
 * i indicates which shape function we are computing
 *
 * This could potentially be optimized. It is called millions of times.
 */
__host__ __device__ void shape(double& x, double& output, int i) {
    switch (i) {
        case 0:
            output = (1.0 - x) * (2.0 - x) * (3.0 - x) / 6.0;
            break;
        case 1:
            output = x * (2.0 - x) * (3.0 - x) / 2.0;
            break;
        case 2:
            output = x * (x - 1.0) * (3.0 - x) / 2.0;
            break;
        case 3:
            output = x * (x - 1.0) * (x - 2.0) / 6.0;
            break;
        default:
            output = 0.0;
            break;
    }
}

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
// index_i, index_j, index_k store the grid index for interpolation in the r, phi, z coordinates
// r_shape, phi_shape, z_shape store shape function elements
// nphi and nz indicate how many grid pts there are in phi and z directions
// nparticles_blk store the number of *actual* particles in the current block
//
// note that nparticles_blk isn't always equal to PARTICLES_PER_BLOCK
template <int n> __device__ void interpolate(double*  out, const double* __restrict__ data, const int* __restrict__ index_i, const int* __restrict__ index_j, const int* __restrict__ index_k, 
    const double* __restrict__ r_shape, const double* __restrict__ phi_shape, const double* __restrict__ z_shape, int nphi, int nz, int nparticles_blk){

    for(int idx=threadIdx.x; idx<nparticles_blk*n; idx+= THREADS_PER_BLOCK){
        int zz = idx % n;
        int particle_id = idx / n;
        int i = index_i[particle_id];
        int j = index_j[particle_id];
        int k = index_k[particle_id];
        double local_val = 0.0;
        for(int ii=0; ii<4; ++ii){
            for(int jj=0; jj<4; ++jj){
                for(int kk=0; kk<4; ++kk){
                    int row_idx = 64*(i*nphi*nz + j*nz + k) + 16*ii + 4*jj + kk;
                    double shape_val = r_shape[ii*PARTICLES_PER_BLOCK + particle_id] * phi_shape[jj*PARTICLES_PER_BLOCK + particle_id] * z_shape[kk*PARTICLES_PER_BLOCK + particle_id];
                    local_val += data[n*row_idx + zz] * shape_val;

                }
            }
        }
        out[PARTICLES_PER_BLOCK*zz + particle_id] = local_val;

    }
}

// calc_derivs computes the derivatives at points for which the corresponding
// i,j,k indices and shape functions have been precomputed
// the results are stored in the appropriate region of derivs
// nparticles_blk stores the number of actual particles in the block
//
// this function is templated across rhs options
template<RHS id, typename... Args>  
__device__ void calc_derivs(double* derivs, int deriv_id, double* quadpts_arr, double* x_temp, bool* symmetry_exploited, 
                                    int* index_i, int* index_j, int* index_k, double* r_shape, double* phi_shape, double* z_shape,
                                    double* mu, double m, double q, int nphi, int nz, int nparticles_blk, Args... args){};


// calc_derivs implementation for guiding center cartesian vacuum tracing
template <> 
__device__ void calc_derivs<RHS::GC_CartesianVacuum>(double* derivs, int deriv_id, double* quadpts_arr, double* x_temp, bool* symmetry_exploited, 
                                    int* index_i, int* index_j, int* index_k, double* r_shape, double* phi_shape, double* z_shape,
                                    double* mu, double m, double q, int nphi, int nz, int nparticles_blk){
    __shared__ double block_interpolants[7*PARTICLES_PER_BLOCK];

    if(threadIdx.x < nparticles_blk){
        for(int i=0; i<7; ++i){
            block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0;
        }
    }
    __syncthreads();
    interpolate<7>(block_interpolants, quadpts_arr, index_i, index_j, index_k, r_shape, phi_shape, z_shape, nphi, nz, nparticles_blk);
    __syncthreads();

    if(threadIdx.x < nparticles_blk){
        double x = x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x];
        double y = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
        double z = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
        double v_par = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];

        double B_r = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
        double B_phi = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x];
        double B_z = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];
        double GradAbsB_r = block_interpolants[3*PARTICLES_PER_BLOCK + threadIdx.x];
        double GradAbsB_phi = block_interpolants[4*PARTICLES_PER_BLOCK + threadIdx.x];
        double GradAbsB_z = block_interpolants[5*PARTICLES_PER_BLOCK + threadIdx.x];

        if(symmetry_exploited[threadIdx.x]){
            B_r *= -1.0;
            GradAbsB_phi *= -1.0;
            GradAbsB_z *= -1.0;
        }

        double phi = atan2(y, x);
        double B_x = cos(phi) * B_r - sin(phi) * B_phi;
        double B_y = sin(phi) * B_r + cos(phi) * B_phi;
        double GradAbsB_x = cos(phi) * GradAbsB_r - sin(phi) * GradAbsB_phi;
        double GradAbsB_y = sin(phi) * GradAbsB_r + cos(phi) * GradAbsB_phi;

        double AbsB = sqrt(B_x*B_x + B_y*B_y + B_z*B_z);
        double v_perp2 = 2*mu[threadIdx.x]*AbsB;
        double fak1 = (v_par/AbsB);
        double fak2 = (m/(q*pow(AbsB, 3)))*(0.5*v_perp2 + v_par*v_par);

        double BcrossGradAbsB_elt = B_y*GradAbsB_z - B_z*GradAbsB_y;
        derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_x + fak2*BcrossGradAbsB_elt;
        BcrossGradAbsB_elt = B_z*GradAbsB_x - B_x*GradAbsB_z;
        derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_y + fak2*BcrossGradAbsB_elt;
        BcrossGradAbsB_elt = B_x*GradAbsB_y - B_y*GradAbsB_x;
        derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_z + fak2*BcrossGradAbsB_elt;
        derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x] = -mu[threadIdx.x]*(B_x*GradAbsB_x + B_y*GradAbsB_y + B_z*GradAbsB_z)/AbsB;
        derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK + threadIdx.x] = AbsB; // AbsB
        derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = block_interpolants[6*PARTICLES_PER_BLOCK + threadIdx.x]; // boundary dist fn
    }
}


// calc_derivs implementation for guiding center boozer vacuum tracing
template <> 
__device__ void calc_derivs<RHS::GC_BoozerVacuum>(double* derivs, int deriv_id, double* quadpts_arr, double* x_temp, bool* symmetry_exploited, 
                                    int* index_i, int* index_j, int* index_k, double* s_shape, double* t_shape, double* z_shape,
                                    double* mu, double m, double q, int nt, int nz, int nparticles_blk, double psi0){

   __shared__ double block_interpolants[6*PARTICLES_PER_BLOCK];

    if(threadIdx.x < nparticles_blk){
        for(int i=0; i<6; ++i){
            block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0;
        }
    }
    __syncthreads();
    interpolate<6>(block_interpolants, quadpts_arr, index_i, index_j, index_k, s_shape, t_shape, z_shape, nt, nz, nparticles_blk);
    __syncthreads();

    if(threadIdx.x < nparticles_blk){
        double x1 = x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x];
        double x2 = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];

        double s = sqrt(x1*x1 + x2*x2);
        double theta = atan2(x2, x1);
        double zeta = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
        double v_par = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];

        double modB = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
        double dmodBds = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x];
        double dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];
        double dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK + threadIdx.x];
        double G = block_interpolants[4*PARTICLES_PER_BLOCK + threadIdx.x];
        double iota = block_interpolants[5*PARTICLES_PER_BLOCK + threadIdx.x];

        double mu_val = mu[threadIdx.x];

        if(symmetry_exploited[threadIdx.x]){
            dmodBdtheta *= -1.0;
            dmodBdzeta *= -1.0;
        }

        double fak1 = m*v_par*v_par/modB + m*mu_val;
        double sdot = -dmodBdtheta*fak1 / (q*psi0);
        double tdot = dmodBds*fak1 / (q*psi0) + iota*v_par*modB / G;

        derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*cos(theta) - s*sin(theta)*tdot;
        derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x] = sdot*sin(theta) + s*cos(theta)*tdot;
        derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x] = v_par*modB/G;
        derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x] = -(iota*dmodBdtheta + dmodBdzeta)*mu_val*modB / G;
        derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK + threadIdx.x] = modB; // modB for setting mu
        derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = G;
        // derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = // no boundary dist fn
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

template<RHS id, typename... Args>
__device__ void map_to_grid(double* interp_pt, double * xyz, bool* symmetry_exploited, Args... args);                                    


// map_to_grid implementation for Cartesian tracing
template <>
__device__ void map_to_grid<RHS::GC_CartesianVacuum>(double* interp_pt, double* x_temp, bool* symmetry_exploited, double* rrange_arr, double* phirange_arr, double* zrange_arr){
    double x = x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x];
    double y = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    double z = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];


    // convert to cylindrical coordinates for interpolation
    double r = sqrt(x*x + y*y);
    double phi = atan2(y, x);

    // restrict phi to [0, 2pi / nfp]
    double period = phirange_arr[1];
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

    interp_pt[0] = r;
    interp_pt[1] = phi;
    interp_pt[2] = z;
} 

// map_to_grid implementation for Boozer tracing
template <>
__device__ void map_to_grid<RHS::GC_BoozerVacuum>(double* interp_pt, double* x_temp, bool* symmetry_exploited, double* srange_arr, double* trange_arr, double* zrange_arr){

    double x1 = x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x];
    double x2 = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    double s = sqrt(x1*x1 + x2*x2);
    double theta = atan2(x2, x1);
    double z = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x]; // zeta

    // we want to exploit periodicity in the B-field, but leave sine(theta) unchanged
    double t = fmod(theta, 2*M_PI);
    t += 2*M_PI*(t < 0);

    // we can modify z because it's only used to access the B-field location
    double period = zrange_arr[1];
    z = fmod(z, period);
    z += period*(z < 0);

    // exploit stellarator symmetry
    symmetry_exploited[threadIdx.x] = t > M_PI;
    if(symmetry_exploited[threadIdx.x]){
        z = period - z;
        t = 2*M_PI - t;

    }
    interp_pt[0] = s;
    interp_pt[1] = t;
    interp_pt[2] = z;
}


// build_state is part of the DP5 implementation
template <RHS id>
__device__ void build_state(double* x_temp, int deriv_id, bool* symmetry_exploited, int* index_i, int* index_j, int* index_k,
                            double* r_shape, double* phi_shape, double* z_shape, double* state, double* derivs, double* dt,
                            double* rrange_arr, double* phirange_arr, double* zrange_arr){
    // const double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    double wgts[6] = {0.0}; 
    for (int i = 0; i < 4; i++) {
        x_temp[i*PARTICLES_PER_BLOCK + threadIdx.x] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
    }

    switch(deriv_id){
        case 0:
            // wgts = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            break;
        case 1:
            // wgts = {1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            wgts[0] = 1.0/5.0;
            break;
        case 2:
            // wgts = {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0};
            wgts[0] = 3.0 / 40.0;
            wgts[1] = 9.0 / 40.0;
            break;
        case 3:
            // wgts = {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0};
            wgts[0] = 44.0 / 45.0;
            wgts[1] = -56.0 / 15.0;
            wgts[2] = 32.0 / 9.0;
            break;
        case 4:
            // wgts = {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0};
            wgts[0] = 19372.0 / 6561.0;
            wgts[1] = -25360.0 / 2187.0;
            wgts[2] = 64448.0 / 6561.0;
            wgts[3] = -212.0 / 729.0;
            break;
        case 5:
            // wgts = {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,-5103.0 / 18656.0, 0.0, 0.0};
            wgts[0] = 9017.0 / 3168.0;
            wgts[1] = -355.0 / 33.0;
            wgts[2] = 46732.0 / 5247.0;
            wgts[3] = 49.0 / 176.0;
            wgts[4] = -5103.0 / 18656.0;
            break;
        case 6:
            // wgts = {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0};
            wgts[0] = 35.0 / 384.0;
            wgts[2] = 500.0 / 1113.0;
            wgts[3] = 125.0 / 192.0; 
            wgts[4] = -2187.0 / 6784.0;
            wgts[5] = 11.0 / 84.0;
            break;
        default:
            break;
    }

    for (int j=0; j<6; ++j){
        for(int i=0; i<4; ++i){
            x_temp[i*PARTICLES_PER_BLOCK + threadIdx.x] += dt[threadIdx.x] * wgts[j] * derivs[(6*j+i)*PARTICLES_PER_BLOCK + threadIdx.x];
        }
    } 

    // double x = x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x];
    // double y = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    // double z = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
    // double v_par = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];


    // // convert to cylindrical coordinates for interpolation
    // double r = sqrt(x*x + y*y);
    // double phi = atan2(y, x);

    // // restrict phi to [0, 2pi / nfp]
    // double period = phirange_arr[1];
    // phi = fmod(phi, period);
    // phi += period*(phi < 0);

    // // exploit stellarator symmetry
    // symmetry_exploited[threadIdx.x] = z < 0;
    // if(symmetry_exploited[threadIdx.x]){
    //     z = -z;
    //     phi = 2*M_PI - phi;
    //     phi = fmod(phi, period);
    //     phi += period*(phi < 0);
    // }

    // printf("location = %.15e, %.15e, %.15e, %.15e \n", x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x],x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x],x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x],x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x]);

    double interp_pt[3];
    map_to_grid<id>(interp_pt, x_temp, symmetry_exploited, rrange_arr, phirange_arr, zrange_arr);
    // printf("interp pt = %.15e, %.15e, %.15e, %.15e \n", interp_pt[0], interp_pt[1], interp_pt[2]);
    // printf("location = %.15e, %.15e, %.15e, %.15e \n", x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x],x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x],x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x],x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x]);

    double r = interp_pt[0];
    double phi = interp_pt[1];
    double z = interp_pt[2];

    // printf("s, theta, zeta in grid= %.15e, %.15e, %.15e\n", r, phi, z);
    /*
    * index into the grid and calculate weights
    */ 
    double r_grid_size = (rrange_arr[1]-rrange_arr[0]) / (rrange_arr[2]-1);
    double phi_grid_size = (phirange_arr[1]-phirange_arr[0]) / (phirange_arr[2]-1);
    double z_grid_size = (zrange_arr[1]-zrange_arr[0]) / (zrange_arr[2]-1);

    // printf("grid sizes = %.15e, %.15e, %.15e\n", r_grid_size, phi_grid_size, z_grid_size);

    int i = 3*((int) ((r - rrange_arr[0]) / r_grid_size) / 3);
    int j = 3*((int) ((phi - phirange_arr[0]) / phi_grid_size) / 3);
    int k = 3*((int) ((z - zrange_arr[0]) / z_grid_size) / 3);

    i = min(i, (int)rrange_arr[2]-4);
    j = min(j, (int)phirange_arr[2]-4);
    k = min(k, (int)zrange_arr[2]-4);

    i = max(i, 0); // if r too small to be in the device, extrapolate

    // normalized positions in local grid wrt e.g. r at index i
    // maps the position to [0,3] in the "meta grid"
    double r_rel = (r -  i*r_grid_size - rrange_arr[0]) / r_grid_size;
    double phi_rel = (phi -  j*phi_grid_size - phirange_arr[0]) / phi_grid_size;
    double z_rel = (z - k*z_grid_size - zrange_arr[0]) / z_grid_size;
   
    for(int i=0; i<4; ++i){
        shape(r_rel, r_shape[i*PARTICLES_PER_BLOCK + threadIdx.x], i);
        shape(phi_rel, phi_shape[i*PARTICLES_PER_BLOCK + threadIdx.x], i);
        shape(z_rel, z_shape[i*PARTICLES_PER_BLOCK + threadIdx.x], i);
    }

    // convert to cell id
    index_i[threadIdx.x] = i/3;
    index_j[threadIdx.x] = j/3;
    index_k[threadIdx.x] = k/3;
}


// calculate maximum allowable timestep to allow at most a quarter of a revolution per stel
template<RHS id>
__device__ void calc_max_timestep_size(double* dtmax, double* loc, double* derivs, double vtotal);

template<>
__device__ void calc_max_timestep_size<RHS::GC_CartesianVacuum>(double* dtmax, double* loc, double* derivs, double vtotal){
    double x = loc[0*PARTICLES_PER_BLOCK + threadIdx.x];
    double y = loc[1*PARTICLES_PER_BLOCK + threadIdx.x];
    double z = loc[2*PARTICLES_PER_BLOCK + threadIdx.x];
    double v_par = loc[3*PARTICLES_PER_BLOCK + threadIdx.x];

    double r = sqrt(x*x + y*y);
    dtmax[threadIdx.x] = r*0.5*M_PI / vtotal;
}


template<>
__device__ void calc_max_timestep_size<RHS::GC_BoozerVacuum>(double* dtmax, double* loc, double* derivs, double vtotal){
    double modB = derivs[(6*0 + 4)*PARTICLES_PER_BLOCK + threadIdx.x];
    double G = derivs[(6*0 + 5)*PARTICLES_PER_BLOCK + threadIdx.x];
    dtmax[threadIdx.x] = (G / modB)*0.5*M_PI / vtotal;
}

template<RHS id, typename... Args>
__device__ void setup_particle(double* mu, double* t, double* dt, double* dtmax, double* x_temp, bool* symmetry_exploited, int* index_i, int* index_j, int* index_k,
                            double* quad_pts, double* r_shape, double* phi_shape, double* z_shape, double* state, double* derivs,
                            double* rrange_arr, double* phirange_arr, double* zrange_arr, double vtotal, double tmax, double m, double q, int nparticles_blk, Args... args){

    if(threadIdx.x < nparticles_blk){
        t[threadIdx.x] = 0.0;
        dt[threadIdx.x] = 0.0;
        symmetry_exploited[threadIdx.x] = false;
        build_state<id>(x_temp, 0, symmetry_exploited, index_i, index_j, index_k,
                                r_shape, phi_shape, z_shape, state, derivs, dt,
                                rrange_arr, phirange_arr, zrange_arr);
        // dummy call to get norm B
        mu[threadIdx.x] = -1.0; // initialize mu
    }


    int nphi = (phirange_arr[2]-1)/3;
    int nz = (zrange_arr[2]-1)/3;

    __syncthreads();
    calc_derivs<id>(derivs, 0, quad_pts, x_temp, symmetry_exploited, index_i, index_j, index_k,
                     r_shape, phi_shape, z_shape, mu, m, q, nphi, nz, nparticles_blk, args...);
    __syncthreads();

    if(threadIdx.x < nparticles_blk){
        double v_par = state[3*PARTICLES_PER_BLOCK + threadIdx.x];
        double v_perp2 = vtotal*vtotal - v_par*v_par;
        
        double modB = derivs[4*PARTICLES_PER_BLOCK + threadIdx.x];
        double denom = 1 / (2*modB);
        mu[threadIdx.x] = v_perp2 * denom;

        // double x = state[0*PARTICLES_PER_BLOCK + threadIdx.x];
        // double y = state[1*PARTICLES_PER_BLOCK + threadIdx.x];
        // // can at most do quarter of a revolution per step
        // double r = sqrt(x*x + y*y);
        // dtmax[threadIdx.x] = r*0.5*M_PI/vtotal;

        calc_max_timestep_size<id>(dtmax, x_temp, derivs, vtotal);
        dtmax[threadIdx.x] = fmin(dtmax[threadIdx.x], tmax);

        dt[threadIdx.x] = 1e-3*dtmax[threadIdx.x];
    }
}


// determine whether a particle has been lost or not
// in cartesian coordinates, we check the signed distance function
// in boozer coordinates we check for s >= 1
template<RHS id>
__device__ void check_has_left(bool* has_left, double* state, double* derivs);

template<>
__device__ void check_has_left<RHS::GC_CartesianVacuum>(bool* has_left, double* state, double* derivs){
    has_left[threadIdx.x] = derivs[(6*6 + 5)*PARTICLES_PER_BLOCK + threadIdx.x] < 0; // boundary dist fn at new location
}

template<>
__device__ void check_has_left<RHS::GC_BoozerVacuum>(bool* has_left, double* state, double* derivs){
    double x1 = state[0*PARTICLES_PER_BLOCK + threadIdx.x];
    double x2 = state[1*PARTICLES_PER_BLOCK + threadIdx.x];
    double s = sqrt(x1*x1 + x2*x2);


    has_left[threadIdx.x] = s >= 1; 
}

template<RHS id>
__device__ void adjust_time(double* t, double* dt, double* state, double* derivs, double* x_temp, bool* has_left, double atol, double rtol, double tmax, double* dtmax){
    if(has_left[threadIdx.x]){
        return;
    }
    const double bhat1 = 71.0 / 57600.0, bhat3 = -71.0 / 16695.0, bhat4 = 71.0 / 1920.0, bhat5 = -17253.0 / 339200.0, bhat6 = 22.0 / 525.0, bhat7 = -1.0 / 40.0;
    // Compute  error
    // https://live.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
    // resolve typo in boost docs: https://numerical.recipes/book.html
    double max_err = 0.0;
    double err_elt;
    for(int i = 0; i < 4; i++) {
        double state_i = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        double deriv_i = derivs[(6*0 + i)*PARTICLES_PER_BLOCK + threadIdx.x];
        err_elt = dt[threadIdx.x]*(bhat1 * deriv_i
                                 + bhat3 * derivs[(6*2 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat4 * derivs[(6*3 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat5 * derivs[(6*4 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat6 * derivs[(6*5 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat7 * derivs[(6*6 + i)*PARTICLES_PER_BLOCK + threadIdx.x]);
        err_elt = fabs(err_elt) / (atol + rtol*(fabs(state_i) + dt[threadIdx.x]*fabs(deriv_i)));
        max_err = fmax(max_err, err_elt);
    }

    // Compute new step size
    double dt_new = dt[threadIdx.x]*0.9*pow(max_err, -1.0/3.0);
    dt_new = fmax(dt_new, 0.2 * dt[threadIdx.x]);  // Limit step size reduction
    dt_new = fmin(dt_new, 5.0 * dt[threadIdx.x]);  // Limit step size increase
    dt_new = fmin(dtmax[threadIdx.x], dt_new);
    if ((0.5 < max_err) & (max_err < 1.0)){
        dt_new = dt[threadIdx.x];
    }

    if(max_err <= 1.0) {
        // Accept the step
        t[threadIdx.x] += dt[threadIdx.x];
        dt[threadIdx.x] = fmin(dt_new, tmax - t[threadIdx.x]);

        for(int i = 0; i < 4; i++) {
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = x_temp[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }
        // check if particle has left the device
        check_has_left<id>(has_left, state, derivs);

        // double x1 = state[0*PARTICLES_PER_BLOCK + threadIdx.x];
        // double x2 = state[1*PARTICLES_PER_BLOCK + threadIdx.x];
        // double s = sqrt(x1*x1 + x2*x2);
        // printf("particle %d, time = %.15e, s=%.15e\n", threadIdx.x, t[threadIdx.x], s);

    } else {
        // Reject the step and try again with smaller dt
        dt[threadIdx.x] = dt_new;
    }
}


template<RHS id, typename... Args>
__global__ void particle_trace_kernel(particle_t* particles, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double tmax, double m, double q, double atol, double rtol, int nparticles, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;
    particle_t p;

    __shared__ double x_temp[4 * PARTICLES_PER_BLOCK];
    __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
    __shared__ double dt[PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int index_i[PARTICLES_PER_BLOCK];
    __shared__ int index_j[PARTICLES_PER_BLOCK];
    __shared__ int index_k[PARTICLES_PER_BLOCK];
    __shared__ double r_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double phi_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double z_shape[4 * PARTICLES_PER_BLOCK];
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
        p = particles[idx];
        has_left[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.state[i];
        }
    }
    __syncthreads();

    // calculate the particle's magnetic moment mu, dt, dtmax
    setup_particle<id>(mu, t, dt, dtmax, x_temp, symmetry_exploited, index_i, index_j, index_k,
                        quadpts_arr, r_shape, phi_shape, z_shape, state, derivs,
                        srange_arr, trange_arr, zrange_arr, p.v_total, tmax, m, q, nparticles_blk, args...);
    int nphi = (trange_arr[2]-1)/3;
    int nz = (zrange_arr[2]-1)/3;
    __syncthreads();

    // if there exists a particle which is real and hasn't not reached tmax or left, keep tracing
    while(__syncthreads_count(is_valid && !(t[threadIdx.x] >= tmax || has_left[threadIdx.x])) > 0){

        // calculate the 7 Dormand-Prince 5 derivatives
        for(int k=0; k<7; ++k){
            // if the thread is responsible for a particle, compute the point at which the derivative will be computed
            if(is_valid){
                build_state<id>(x_temp, k, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, state, derivs, dt,
                            srange_arr, trange_arr, zrange_arr);
            }

            // ensure that all threads have updated x_temp before calculating derivatives, where a data race would occur
            __syncthreads();
            calc_derivs<id>(derivs, k, quadpts_arr, x_temp, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, mu, m, q,
                        nphi, nz, nparticles_blk, args...);

            // ensure all particles have derivative calculations before accepting/rejecting timestep
            __syncthreads();

        }

        __syncthreads();
        if(is_valid){
            adjust_time<id>(t, dt, state, derivs, x_temp, has_left, atol, rtol, tmax, dtmax);
        }
        __syncthreads();
    }
    __syncthreads();
    if(is_valid){
        particles[idx].dt = dt[threadIdx.x];
        particles[idx].t = t[threadIdx.x];
        particles[idx].has_left = has_left[threadIdx.x];
        for(int i=0; i<4; ++i){
            particles[idx].state[i] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }
    }
    return;
}


template<RHS id, typename... Args>
vector<double> gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, int nparticles, Args... args){

    //  read data in from python
    py::buffer_info stz_init_buf = stz_init.request();
    double* stz_init_arr = static_cast<double*>(stz_init_buf.ptr);
    
    py::buffer_info vtang_buf = vtang.request();
    double* vtang_arr = static_cast<double*>(vtang_buf.ptr);

    // contains b field
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);


    particle_t* particles =  new particle_t[nparticles];

    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        double s = stz_init_arr[start];
        double theta = stz_init_arr[start+1];
        
        // convert to alternative coordinates
        particles[i].state[0] = stz_init_arr[start]; // x
        particles[i].state[1] = stz_init_arr[start+1];// y
        
        particles[i].state[2] = stz_init_arr[start+2]; // z
        particles[i].state[3] = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  vtang_arr[i]*vtang_arr[i]);
        particles[i].v_total = vtotal;
        particles[i].has_left = false;
        particles[i].t = 0;
        
        particles[i].step_accept = 0;
        particles[i].step_attempt = 0;
        particles[i].id = i;

    }

   
    particle_t* particles_d;
    gpuErrchk(cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t)) );
    gpuErrchk(cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice) );

    double* srange_d;
    gpuErrchk(cudaMalloc((void**)&srange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* zrange_d;
    gpuErrchk(cudaMalloc((void**)&zrange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);


    double* quadpts_d;
    gpuErrchk(cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double)) ); 
    gpuErrchk(cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice) );

    int nthreads = THREADS_PER_BLOCK;

    int nblks = nparticles  / PARTICLES_PER_BLOCK + 1;
    std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    particle_trace_kernel<id><<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, tmax, m, q, tol, tol, nparticles, args...);

    gpuErrchk(cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost) );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";


    gpuErrchk( cudaFree(particles_d) );

    gpuErrchk( cudaFree(srange_d) );

    gpuErrchk( cudaFree(trange_d) );

    gpuErrchk( cudaFree(zrange_d) );

    gpuErrchk( cudaFree(quadpts_d) );

    vector<double> particle_output(7*nparticles);
    for(int i=0; i<nparticles; ++i){
        double y1 = particles[i].state[0];
        double y2 = particles[i].state[1];
        double z = particles[i].state[2];
        double v_par = particles[i].state[3];

        // last location in Boozer coordinates
        particle_output[7*i] = y1;
        particle_output[7*i + 1] = y2;
        particle_output[7*i + 2] = z;
        particle_output[7*i + 3] = v_par;
        particle_output[7*i + 4] = particles[i].t;
        particle_output[7*i + 5] = particles[i].step_accept;
        particle_output[7*i + 6] = particles[i].step_attempt;
    }


    delete[] particles;

    return particle_output;
}

extern "C" vector<double> cartesian_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, int nparticles){
            return gpu_tracing<RHS::GC_CartesianVacuum>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, nparticles);
        }

extern "C" vector<double> boozer_gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, double psi0, int nparticles){

            //  read data in from python
            py::buffer_info stz_init_buf = stz_init.request();
            double* stz_init_arr = static_cast<double*>(stz_init_buf.ptr);
            
            for(int i=0; i<nparticles; ++i){
                double s = stz_init_arr[3*i];
                double theta = stz_init_arr[3*i+1];

                stz_init_arr[3*i] = s*cos(theta);
                stz_init_arr[3*i+1] = s*sin(theta);
            }

            std::vector<double> results =  gpu_tracing<RHS::GC_BoozerVacuum>(quad_pts, srange, trange, zrange, stz_init, m, q, vtotal, vtang, tmax, tol, nparticles, psi0);

            for(int i=0; i<nparticles; ++i){
                double x1 = results[7*i];
                double x2 = results[7*i+1];

                results[7*i] = sqrt(x1*x1 + x2*x2);
                results[7*i+1] = atan2(x2, x1);
            }

            return results;
        }


template<RHS id>
__device__ void account_for_symmetry(double* interpolants, bool* symmetry_exploited);

template<>
__device__ void account_for_symmetry<RHS::GC_CartesianVacuum>(double* interpolants, bool* symmetry_exploited){
    if(symmetry_exploited[threadIdx.x]){
        interpolants[0] *= -1.0;
        interpolants[4] *= -1.0;
        interpolants[5] *= -1.0;
    }
}

template<>
__device__ void account_for_symmetry<RHS::GC_BoozerVacuum>(double* interpolants, bool* symmetry_exploited){
    if(symmetry_exploited[threadIdx.x]){
        interpolants[2] *= -1.0;
        interpolants[3] *= -1.0;
    }
}

template <RHS id, int n>
__global__ void test_gpu_interpolation_kernel(double* quad_pts, double* srange, double* trange, double* zrange, double* loc, double* out, int n_points){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;
    particle_t p;
    __shared__ double x_temp[4 * PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int index_i[PARTICLES_PER_BLOCK];
    __shared__ int index_j[PARTICLES_PER_BLOCK];
    __shared__ int index_k[PARTICLES_PER_BLOCK];
    __shared__ double r_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double phi_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double z_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double state[4 * PARTICLES_PER_BLOCK];
    __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
    __shared__ double dt[PARTICLES_PER_BLOCK];

    __shared__ double block_interpolants[n*PARTICLES_PER_BLOCK];


    double* loc_arr = loc + 3*idx;
    double* out_arr  =  out + idx*n;

    bool is_valid = idx < n_points && threadIdx.x < PARTICLES_PER_BLOCK;
    int nparticles_blk = __syncthreads_count(is_valid);
    // printf("test_gpu_interpolation_kernel called with idx=%d, n_points=%d, nparticles_blk=%d\n", idx, n_points, nparticles_blk);
    if(is_valid){
        // double x = loc_arr[0]*cos(loc_arr[1]);
        // double y = loc_arr[0]*sin(loc_arr[1]);
        // double z = loc_arr[2];

        p.state[0] = loc_arr[0];
        p.state[1] = loc_arr[1];
        p.state[2] = loc_arr[2];
        p.state[3] = 0.0; // v_par

        p.dt = 1e-3; //needed for build_state

        dt[threadIdx.x] = 1e-3; // needed for build_state
        symmetry_exploited[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.state[i];
        }
        build_state<id>(x_temp, 0, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, state, derivs, dt, srange, trange, zrange);

        for(int i=0; i<n; ++i){
            block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0;
        }
    } 
        
        // printf("calling interpolate for particle %d\n", threadIdx.x);
        int nphi = (trange[2]-1)/3;
        int nz = (zrange[2]-1)/3;

        __syncthreads();
        interpolate<n>(block_interpolants, quad_pts, index_i, index_j, index_k, r_shape, phi_shape, z_shape, nphi, nz, nparticles_blk);
        __syncthreads();
        // printf("returned from interpolate for particle %d\n", threadIdx.x);
        // interpolate(p, quad_pts, out_arr, srange, trange, zrange, n);
        if(is_valid){
            for(int i=0; i<n; ++i){
                out_arr[i] = block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x];

                // if(threadIdx.x==1){
                //     printf("interpolated value %d for particle %d: %.15e\n", i, idx, out_arr[i]);
                // }
            }

            account_for_symmetry<id>(out_arr, symmetry_exploited);
        }



}



extern "C" py::array_t<double> test_gpu_interpolation(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, std::string coordinates, int n_points){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);


    int n;
    if(coordinates == "cartesian"){
        n = 7;
        for(int i=0; i<n_points; ++i){
            double x = loc_arr[3*i] * cos(loc_arr[3*i + 1]);
            double y = loc_arr[3*i] * sin(loc_arr[3*i + 1]);
            
            loc_arr[3*i] = x;
            loc_arr[3*i+1] = y;
        }
    } else if(coordinates == "boozer"){
        n = 6;
        for(int i=0; i<n_points; ++i){
            double x1 = loc_arr[3*i] * cos(loc_arr[3*i + 1]);
            double x2 = loc_arr[3*i] * sin(loc_arr[3*i + 1]);
            
            loc_arr[3*i] = x1;
            loc_arr[3*i+1] = x2;
        }
    }



    // allocate and copy to device memory

    double* srange_d;
    cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* zrange_d;
    cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if(coordinates == "cartesian"){
        test_gpu_interpolation_kernel<RHS::GC_CartesianVacuum, 7><<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, out_d, n_points);
    } else if(coordinates == "boozer") {
        test_gpu_interpolation_kernel<RHS::GC_BoozerVacuum, 6><<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, out_d, n_points);
    }
    double out[n*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, n*n_points * sizeof(double), cudaMemcpyDeviceToHost) );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "interpolation kernel time (ms): " << milliseconds<< "\n";
    

    auto result = py::array_t<double>(n*n_points, out);
    return result;

}


template<RHS id, typename... Args>
__global__ void test_gpu_derivs_kernel(double* quad_pts, double* srange, double* trange, double* zrange, double* loc, double* vpar, double vtotal, double* out, double m, double q, int n_points, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;    
    double* loc_arr = loc + 3*idx;
    double* out_arr  =  out + 4*idx;

    __shared__ double x_temp[4 * PARTICLES_PER_BLOCK];
    __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
    __shared__ double dt[PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int index_i[PARTICLES_PER_BLOCK];
    __shared__ int index_j[PARTICLES_PER_BLOCK];
    __shared__ int index_k[PARTICLES_PER_BLOCK];
    __shared__ double r_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double phi_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double z_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double mu[PARTICLES_PER_BLOCK];
    __shared__ double t[PARTICLES_PER_BLOCK];
    __shared__ double dtmax[PARTICLES_PER_BLOCK];
    __shared__ double state[4 * PARTICLES_PER_BLOCK];
    particle_t p;

    bool is_valid = idx < n_points && threadIdx.x < PARTICLES_PER_BLOCK;
    int nparticles_blk = __syncthreads_count(is_valid);

    if(is_valid){
        double vpar_val = vpar[idx];
        double r = loc_arr[0];
        double phi = loc_arr[1];
        double z = loc_arr[2];

        p.state[0] = r*cos(phi);
        p.state[1] = r*sin(phi);
        p.state[2] = z;
        p.state[3] = vpar_val;
        p.v_total = vtotal;
        p.v_perp = sqrt(vtotal*vtotal -  vpar_val*vpar_val);

        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.state[i];
        }
        // printf("particle location: %.15e, %.15e, %.15e, %.15e\n", p.state[0], p.state[1], p.state[2], p.state[3]);
    }
    __syncthreads();

    setup_particle<id>(mu, t, dt, dtmax, x_temp, symmetry_exploited, index_i, index_j, index_k,
                        quad_pts, r_shape, phi_shape, z_shape, state, derivs,
                        srange, trange, zrange, p.v_total, 1e-2, m, q, nparticles_blk, args...);
    int nphi = (trange[2]-1)/3;
    int nz = (zrange[2]-1)/3;
    __syncthreads();

    calc_derivs<id>(derivs, 0, quad_pts, x_temp, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, mu, m, q, nphi, nz, nparticles_blk, args...);
    __syncthreads();

    if(is_valid){
        // copy back
        for(int i=0; i<4; ++i){
            p.derivs[i] = derivs[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }

        out_arr[0] = p.derivs[0];
        out_arr[1] = p.derivs[1];
        out_arr[2] = p.derivs[2];
        out_arr[3] = p.derivs[3];

    }
}

extern "C" py::array_t<double> test_derivatives_cartesian(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, int n_points){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);

    py::buffer_info vpar_buf = vpar.request();
    double* vpar_arr = static_cast<double*>(vpar_buf.ptr);
    

    double* srange_d;
    cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* zrange_d;
    cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(double));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* vpar_d;
    cudaMalloc((void**)&vpar_d, vpar.size() * sizeof(double));
    cudaMemcpy(vpar_d, vpar_arr, vpar.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    cudaMalloc((void**)&out_d, 4*n_points * sizeof(double));



    int nthreads = THREADS_PER_BLOCK;

    int nblks = n_points / PARTICLES_PER_BLOCK + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
    test_gpu_derivs_kernel<RHS::GC_CartesianVacuum><<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, vpar_d, v_total, out_d, m, q, n_points);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "derivatives kernel time (ms): " << milliseconds<< "\n";
    
    double out[4*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, 4*n_points * sizeof(double), cudaMemcpyDeviceToHost) );
    auto result = py::array_t<double>(4*n_points, out);
    return result;
}


extern "C" py::array_t<double> test_derivatives_boozer(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, double psi0, int n_points){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);

    py::buffer_info vpar_buf = vpar.request();
    double* vpar_arr = static_cast<double*>(vpar_buf.ptr);

    double* srange_d;
    cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* zrange_d;
    cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(double));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* vpar_d;
    cudaMalloc((void**)&vpar_d, vpar.size() * sizeof(double));
    cudaMemcpy(vpar_d, vpar_arr, vpar.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    cudaMalloc((void**)&out_d, 4*n_points * sizeof(double));



    int nthreads = THREADS_PER_BLOCK;

    int nblks = n_points / PARTICLES_PER_BLOCK + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
    test_gpu_derivs_kernel<RHS::GC_BoozerVacuum><<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, vpar_d, v_total, out_d, m, q, n_points, psi0);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "derivatives kernel time (ms): " << milliseconds<< "\n";
    
    double out[4*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, 4*n_points * sizeof(double), cudaMemcpyDeviceToHost) );
    auto result = py::array_t<double>(4*n_points, out);
    return result;
}


template<RHS id, typename... Args>
__global__ void test_gpu_timestep_kernel(particle_t* particles, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double m, double q, double atol, double rtol, int nparticles, Args... args){
    int idx = threadIdx.x + blockIdx.x*PARTICLES_PER_BLOCK;
    particle_t p;

    __shared__ double x_temp[4 * PARTICLES_PER_BLOCK];
    __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
    __shared__ double dt[PARTICLES_PER_BLOCK];
    __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
    __shared__ int index_i[PARTICLES_PER_BLOCK];
    __shared__ int index_j[PARTICLES_PER_BLOCK];
    __shared__ int index_k[PARTICLES_PER_BLOCK];
    __shared__ double r_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double phi_shape[4 * PARTICLES_PER_BLOCK];
    __shared__ double z_shape[4 * PARTICLES_PER_BLOCK];
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
        p = particles[idx];
        has_left[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.state[i];
        }
    }
    __syncthreads();

    // calculate the particle's magnetic moment mu, dt, dtmax
    setup_particle<id>(mu, t, dt, dtmax, x_temp, symmetry_exploited, index_i, index_j, index_k,
                        quadpts_arr, r_shape, phi_shape, z_shape, state, derivs,
                        srange_arr, trange_arr, zrange_arr, p.v_total, 1e-2, m, q, nparticles_blk, args...);
    int nphi = (trange_arr[2]-1)/3;
    int nz = (zrange_arr[2]-1)/3;
    __syncthreads();

    // if there exists a particle at t=0, which is a real particle, then keep tracing
    while(__syncthreads_count(t[threadIdx.x] == 0.0  && is_valid) > 0){
        // calculate the 7 Dormand-Prince 5 derivatives
        for(int k=0; k<7; ++k){
            // if the thread is responsible for a particle, compute the point at which the derivative will be computed
             if(is_valid){
                build_state<id>(x_temp, k, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, state, derivs, dt,
                            srange_arr, trange_arr, zrange_arr);
            }

            // ensure that all threads have updated x_temp before calculating derivatives, where a data race would occur
            __syncthreads();
            calc_derivs<id>(derivs, k, quadpts_arr, x_temp, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, mu, m, q,
                        nphi, nz, nparticles_blk, args...);

            // ensure all particles have derivative calculations before accepting/rejecting timestep
            __syncthreads();

        }
        __syncthreads();
        if(is_valid && t[threadIdx.x] == 0.0){
            adjust_time<id>(t, dt, state, derivs, x_temp, has_left, atol, rtol, 1e-2, dtmax);
        }
        __syncthreads();
    }
    __syncthreads();
    if(is_valid){
        // printf("tracing particle %d finished at t=%.15e\n", idx, particles[idx].t);
        particles[idx].dt = dt[threadIdx.x];
        particles[idx].t = t[threadIdx.x];
        particles[idx].has_left = has_left[threadIdx.x];
        for(int i=0; i<4; ++i){
            particles[idx].state[i] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }
    }
    return;
}


extern "C" vector<double> test_timestep_cartesian(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, int nparticles){

    //  read data in from python
    py::buffer_info stz_init_buf = stz_init.request();
    double* stz_init_arr = static_cast<double*>(stz_init_buf.ptr);

    py::buffer_info vtang_buf = vtang.request();
    double* vtang_arr = static_cast<double*>(vtang_buf.ptr);

    // contains b field
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);


    particle_t* particles =  new particle_t[nparticles];

    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        double r = stz_init_arr[start];
        double phi = stz_init_arr[start+1];
        
        // convert to cartesian coordinates
        particles[i].state[0] = r*cos(phi); // x
        particles[i].state[1] = r*sin(phi); // y
        particles[i].state[2] = stz_init_arr[start+2]; // z
        particles[i].state[3] = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  vtang_arr[i]*vtang_arr[i]);
        particles[i].v_total = vtotal;
        particles[i].has_left = false;
        particles[i].t = 0.0;
        
        particles[i].step_accept = 0;
        particles[i].step_attempt = 0;
        particles[i].id = i;
    }
    
    particle_t* particles_d;
    gpuErrchk( cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t)) );
    gpuErrchk( cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice) );

    double* srange_d;
    gpuErrchk( cudaMalloc((void**)&srange_d, 3 * sizeof(double)) );
    gpuErrchk( cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* zrange_d;
    gpuErrchk( cudaMalloc((void**)&zrange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* trange_d;
    gpuErrchk(cudaMalloc((void**)&trange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );


    double* quadpts_d;

    // std::cout << "quadpts.size() = " << quad_pts.size() << "\n";
    gpuErrchk( cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice) );

    int nthreads = THREADS_PER_BLOCK;

    int nblks = nparticles / PARTICLES_PER_BLOCK + 1;
    std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_timestep_kernel<RHS::GC_CartesianVacuum><<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, m, q, tol, tol, nparticles);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError(); 
    // CHECK_CUDA_ERROR(err);
    gpuErrchk( cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost) );


    // err = cudaGetLastError(); 
    // CHECK_CUDA_ERROR(err);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";

    
    vector<double> particle_output(7*nparticles);
    for(int i=0; i<nparticles; ++i){
        double y1 = particles[i].state[0];
        double y2 = particles[i].state[1];
        double z = particles[i].state[2];
        double v_par = particles[i].state[3];

        // double t = atan2(y2, y1);
        // t += 2*M_PI*(t < 0);
        
        // last location in Boozer coordinates
        particle_output[7*i] = y1;
        particle_output[7*i + 1] = y2;
        particle_output[7*i + 2] = z;
        particle_output[7*i + 3] = v_par;
        particle_output[7*i + 4] = particles[i].t;
        // std::cout << "copied back t=" << particles[i].t << "\n";
        particle_output[7*i + 5] = particles[i].step_accept;
        particle_output[7*i + 6] = particles[i].step_attempt;
    }


    delete[] particles;
    gpuErrchk( cudaFree(particles_d) );
    return particle_output;
}

extern "C" vector<double> test_timestep_boozer(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, double psi0, int nparticles){

    //  read data in from python
    py::buffer_info stz_init_buf = stz_init.request();
    double* stz_init_arr = static_cast<double*>(stz_init_buf.ptr);

    py::buffer_info vtang_buf = vtang.request();
    double* vtang_arr = static_cast<double*>(vtang_buf.ptr);

    // contains b field
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);


    particle_t* particles =  new particle_t[nparticles];

    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        double r = stz_init_arr[start];
        double phi = stz_init_arr[start+1];
        
        // convert to cartesian coordinates
        particles[i].state[0] = r*cos(phi); // x
        particles[i].state[1] = r*sin(phi); // y
        particles[i].state[2] = stz_init_arr[start+2]; // z
        particles[i].state[3] = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  vtang_arr[i]*vtang_arr[i]);
        particles[i].v_total = vtotal;
        particles[i].has_left = false;
        particles[i].t = 0.0;
        
        particles[i].step_accept = 0;
        particles[i].step_attempt = 0;
        particles[i].id = i;
    }
    
    particle_t* particles_d;
    gpuErrchk( cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t)) );
    gpuErrchk( cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice) );

    double* srange_d;
    gpuErrchk( cudaMalloc((void**)&srange_d, 3 * sizeof(double)) );
    gpuErrchk( cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* zrange_d;
    gpuErrchk( cudaMalloc((void**)&zrange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* trange_d;
    gpuErrchk(cudaMalloc((void**)&trange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );


    double* quadpts_d;

    // std::cout << "quadpts.size() = " << quad_pts.size() << "\n";
    gpuErrchk( cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice) );

    int nthreads = THREADS_PER_BLOCK;

    int nblks = nparticles / PARTICLES_PER_BLOCK + 1;
    std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_timestep_kernel<RHS::GC_BoozerVacuum><<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, m, q, tol, tol, nparticles, psi0);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError(); 
    // CHECK_CUDA_ERROR(err);
    gpuErrchk( cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost) );


    // err = cudaGetLastError(); 
    // CHECK_CUDA_ERROR(err);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";

    
    vector<double> particle_output(7*nparticles);
    for(int i=0; i<nparticles; ++i){
        double y1 = particles[i].state[0];
        double y2 = particles[i].state[1];
        double z = particles[i].state[2];
        double v_par = particles[i].state[3];

        // double t = atan2(y2, y1);
        // t += 2*M_PI*(t < 0);
        
        // last location in Boozer coordinates
        particle_output[7*i] = sqrt(y1*y1 + y2*y2);
        particle_output[7*i + 1] = atan2(y2, y1);
        particle_output[7*i + 2] = z;
        particle_output[7*i + 3] = v_par;
        particle_output[7*i + 4] = particles[i].t;
        // std::cout << "copied back t=" << particles[i].t << "\n";
        particle_output[7*i + 5] = particles[i].step_accept;
        particle_output[7*i + 6] = particles[i].step_attempt;
    }


    delete[] particles;
    gpuErrchk( cudaFree(particles_d) );
    return particle_output;
}