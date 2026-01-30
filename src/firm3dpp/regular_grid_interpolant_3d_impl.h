#include "regular_grid_interpolant_3d.h"
#include <xtensor/xarray.hpp>
#include "xtensor/xlayout.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/format.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#ifdef USE_MPI
#include <mpi.h>
#endif

#define _EPS_ 1e-13

static int find_span(const Vec& knots, int degree, double x) {
    int n = static_cast<int>(knots.size()) - degree - 2;
    if (x >= knots[n + 1]) {
        return n;
    }
    if (x <= knots[degree]) {
        return degree;
    }
    int low = degree;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (x < knots[mid] || x >= knots[mid + 1]) {
        if (x < knots[mid]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

static void basis_funs(int span, double x, int degree, const Vec& knots, Vec& N) {
    N.assign(degree + 1, 0.0);
    Vec left(degree + 1, 0.0);
    Vec right(degree + 1, 0.0);
    N[0] = 1.0;
    for (int j = 1; j <= degree; ++j) {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            double temp = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

static void basis_funs_der1(int span, double x, int degree, const Vec& knots, Vec& N, Vec& dN) {
    Vec ndu((degree + 1) * (degree + 1), 0.0);
    auto ndu_at = [&](int i, int j) -> double& { return ndu[i * (degree + 1) + j]; };
    Vec left(degree + 1, 0.0);
    Vec right(degree + 1, 0.0);
    ndu_at(0, 0) = 1.0;
    for (int j = 1; j <= degree; ++j) {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            ndu_at(j, r) = right[r + 1] + left[j - r];
            double temp = ndu_at(r, j - 1) / ndu_at(j, r);
            ndu_at(r, j) = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu_at(j, j) = saved;
    }
    N.assign(degree + 1, 0.0);
    for (int j = 0; j <= degree; ++j) {
        N[j] = ndu_at(j, degree);
    }

    Vec ders((degree + 1) * (degree + 1), 0.0);
    auto ders_at = [&](int i, int j) -> double& { return ders[i * (degree + 1) + j]; };
    Vec a((degree + 1) * 2, 0.0);
    auto a_at = [&](int i, int j) -> double& { return a[i * 2 + j]; };

    for (int r = 0; r <= degree; ++r) {
        int s1 = 0;
        int s2 = 1;
        a_at(0, 0) = 1.0;
        for (int k = 1; k <= 1; ++k) {
            double d = 0.0;
            int rk = r - k;
            int pk = degree - k;
            if (r >= k) {
                a_at(s2, 0) = a_at(s1, 0) / ndu_at(pk + 1, rk);
                d = a_at(s2, 0) * ndu_at(rk, pk);
            }
            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (r - 1 <= pk) ? k - 1 : degree - r;
            for (int j = j1; j <= j2; ++j) {
                a_at(s2, j) = (a_at(s1, j) - a_at(s1, j - 1)) / ndu_at(pk + 1, rk + j);
                d += a_at(s2, j) * ndu_at(rk + j, pk);
            }
            if (r <= pk) {
                a_at(s2, k) = -a_at(s1, k - 1) / ndu_at(pk + 1, r);
                d += a_at(s2, k) * ndu_at(r, pk);
            }
            ders_at(k, r) = d;
            std::swap(s1, s2);
        }
    }
    dN.assign(degree + 1, 0.0);
    for (int j = 0; j <= degree; ++j) {
        dN[j] = ders_at(1, j) * degree;
    }
}

static void cubic_bspline_basis(double u, Vec& B, Vec& dB) {
    double u2 = u * u;
    double u3 = u2 * u;
    B.resize(4);
    dB.resize(4);
    B[0] = (1.0 - u) * (1.0 - u) * (1.0 - u) / 6.0;
    B[1] = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
    B[2] = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
    B[3] = u3 / 6.0;

    dB[0] = -0.5 * (1.0 - u) * (1.0 - u);
    dB[1] = 0.5 * (3.0 * u2 - 4.0 * u);
    dB[2] = 0.5 * (-3.0 * u2 + 2.0 * u + 1.0);
    dB[3] = 0.5 * u2;
}

static Vec solve_tridiagonal(const Vec& a, const Vec& b, const Vec& c, const Vec& d) {
    int n = static_cast<int>(d.size());
    Vec cp(n, 0.0);
    Vec dp(n, 0.0);
    Vec x(n, 0.0);
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (int i = 1; i < n; ++i) {
        double denom = b[i] - a[i] * cp[i - 1];
        cp[i] = (i < n - 1) ? c[i] / denom : 0.0;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
    }
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    return x;
}

static Vec solve_cyclic_tridiagonal(const Vec& a, const Vec& b, const Vec& c, double alpha, double beta, const Vec& d) {
    int n = static_cast<int>(d.size());
    Vec bb = b;
    Vec u(n, 0.0);
    Vec z(n, 0.0);
    Vec x(n, 0.0);
    double gamma = -b[0];
    bb[0] = b[0] - gamma;
    bb[n - 1] = b[n - 1] - alpha * beta / gamma;
    u[0] = gamma;
    u[n - 1] = alpha;
    Vec a0 = a;
    Vec c0 = c;
    a0[0] = 0.0;
    c0[n - 1] = 0.0;
    x = solve_tridiagonal(a0, bb, c0, d);
    z = solve_tridiagonal(a0, bb, c0, u);
    double fact = (x[0] + beta * x[n - 1] / gamma) /
                  (1.0 + z[0] + beta * z[n - 1] / gamma);
    for (int i = 0; i < n; ++i) {
        x[i] -= fact * z[i];
    }
    return x;
}

static Vec solve_banded(int n, int bw, std::vector<Vec>& ab, Vec bvec) {
    for (int i = 0; i < n; ++i) {
        int j_start = std::max(0, i - bw);
        for (int j = j_start; j < i; ++j) {
            double denom = ab[j][bw];
            if (std::abs(denom) < 1e-14) {
                continue;
            }
            double factor = ab[i][bw + j - i] / denom;
            ab[i][bw + j - i] = 0.0;
            int col_end = std::min(n - 1, j + bw);
            for (int col = j + 1; col <= col_end; ++col) {
                ab[i][bw + col - i] -= factor * ab[j][bw + col - j];
            }
            bvec[i] -= factor * bvec[j];
        }
    }
    Vec x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = bvec[i];
        int col_end = std::min(n - 1, i + bw);
        for (int col = i + 1; col <= col_end; ++col) {
            sum -= ab[i][bw + col - i] * x[col];
        }
        x[i] = sum / ab[i][bw];
    }
    return x;
}

static Vec build_not_a_knot_knots(const Vec& x) {
    int n = static_cast<int>(x.size()) - 1;
    Vec knots;
    knots.reserve(n + 5);
    for (int i = 0; i < 4; ++i) {
        knots.push_back(x[0]);
    }
    for (int i = 2; i <= n - 2; ++i) {
        knots.push_back(x[i]);
    }
    for (int i = 0; i < 4; ++i) {
        knots.push_back(x[n]);
    }
    return knots;
}

static Vec compute_bspline_coeffs_not_a_knot(const Vec& x, const Vec& y) {
    int n = static_cast<int>(x.size()) - 1;
    int degree = 3;
    Vec knots = build_not_a_knot_knots(x);
    std::vector<Vec> ab(n + 1, Vec(2 * degree + 1, 0.0));
    for (int i = 0; i <= n; ++i) {
        double xi = x[i];
        int span = find_span(knots, degree, xi);
        Vec N;
        basis_funs(span, xi, degree, knots, N);
        for (int j = 0; j <= degree; ++j) {
            int col = span - degree + j;
            ab[i][degree + col - i] = N[j];
        }
    }
    return solve_banded(n + 1, degree, ab, y);
}

template<class Array>
const int RegularGridInterpolant3D<Array>::simdcount;

template<class Array>
void RegularGridInterpolant3D<Array>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f) {
    if (rule.rule_type == InterpolationRuleType::CubicBSpline) {
        const int degree = 3;
        nx_coeff = nx + 1;
        ny_coeff = periodic_y ? ny : (ny + 1);
        nz_coeff = periodic_z ? nz : (nz + 1);

        xknots = build_not_a_knot_knots(xmesh);
        xbase_cell.resize(nx, 0);
        ybase_cell.resize(ny, 0);
        zbase_cell.resize(nz, 0);
        for (int i = 0; i < nx; ++i) {
            double xmid = xmesh[i] + 0.5 * hx;
            int span = find_span(xknots, degree, xmid);
            xbase_cell[i] = span - degree;
        }
        for (int j = 0; j < ny; ++j) {
            int base = j - 1;
            if (periodic_y) {
                base = (base % ny_coeff + ny_coeff) % ny_coeff;
            }
            ybase_cell[j] = base;
        }
        for (int k = 0; k < nz; ++k) {
            int base = k - 1;
            if (periodic_z) {
                base = (base % nz_coeff + nz_coeff) % nz_coeff;
            }
            zbase_cell[k] = base;
        }

        uint32_t total_points = static_cast<uint32_t>(nx_coeff * ny_coeff * nz_coeff);
        Vec fvals(total_points * value_size, 0.0);

        int BATCH_SIZE = 16384;
        Vec xsub, ysub, zsub;
        std::vector<uint32_t> idxs;
        xsub.reserve(BATCH_SIZE);
        ysub.reserve(BATCH_SIZE);
        zsub.reserve(BATCH_SIZE);
        idxs.reserve(BATCH_SIZE);

        auto flush_batch = [&]() {
            if (xsub.empty()) {
                return;
            }
            Vec fxyzsub = f(xsub, ysub, zsub);
            for (size_t j = 0; j < idxs.size(); ++j) {
                uint32_t idx = idxs[j];
                for (int l = 0; l < value_size; ++l) {
                    fvals[idx * value_size + l] = fxyzsub[j * value_size + l];
                }
            }
            xsub.clear();
            ysub.clear();
            zsub.clear();
            idxs.clear();
        };

        for (int i = 0; i < nx_coeff; ++i) {
            double x = xmesh[i];
            for (int j = 0; j < ny_coeff; ++j) {
                double y = ymesh[j];
                for (int k = 0; k < nz_coeff; ++k) {
                    double z = zmesh[k];
                    uint32_t idx = (i * ny_coeff + j) * nz_coeff + k;
                    xsub.push_back(x);
                    ysub.push_back(y);
                    zsub.push_back(z);
                    idxs.push_back(idx);
                    if (static_cast<int>(xsub.size()) >= BATCH_SIZE) {
                        flush_batch();
                    }
                }
            }
        }
        flush_batch();

        Vec coeffs(total_points * value_size, 0.0);

        for (int j = 0; j < ny_coeff; ++j) {
            for (int k = 0; k < nz_coeff; ++k) {
                for (int l = 0; l < value_size; ++l) {
                    Vec line(nx_coeff, 0.0);
                    for (int i = 0; i < nx_coeff; ++i) {
                        uint32_t idx = (i * ny_coeff + j) * nz_coeff + k;
                        line[i] = fvals[idx * value_size + l];
                    }
                    Vec cline = compute_bspline_coeffs_not_a_knot(xmesh, line);
                    for (int i = 0; i < nx_coeff; ++i) {
                        uint32_t idx = (i * ny_coeff + j) * nz_coeff + k;
                        coeffs[idx * value_size + l] = cline[i];
                    }
                }
            }
        }

        if (periodic_y) {
            Vec a(ny_coeff, 0.0);
            Vec b(ny_coeff, 4.0 / 6.0);
            Vec c(ny_coeff, 0.0);
            for (int i = 1; i < ny_coeff; ++i) {
                a[i] = 1.0 / 6.0;
            }
            for (int i = 0; i < ny_coeff - 1; ++i) {
                c[i] = 1.0 / 6.0;
            }
            double alpha = 1.0 / 6.0;
            double beta = 1.0 / 6.0;
            for (int i = 0; i < nx_coeff; ++i) {
                for (int k = 0; k < nz_coeff; ++k) {
                    for (int l = 0; l < value_size; ++l) {
                        Vec line(ny_coeff, 0.0);
                        for (int j = 0; j < ny_coeff; ++j) {
                            uint32_t idx = (i * ny_coeff + j) * nz_coeff + k;
                            line[j] = coeffs[idx * value_size + l];
                        }
                        Vec cline = solve_cyclic_tridiagonal(a, b, c, alpha, beta, line);
                        for (int j = 0; j < ny_coeff; ++j) {
                            uint32_t idx = (i * ny_coeff + j) * nz_coeff + k;
                            coeffs[idx * value_size + l] = cline[j];
                        }
                    }
                }
            }
        }

        if (periodic_z) {
            Vec a(nz_coeff, 0.0);
            Vec b(nz_coeff, 4.0 / 6.0);
            Vec c(nz_coeff, 0.0);
            for (int i = 1; i < nz_coeff; ++i) {
                a[i] = 1.0 / 6.0;
            }
            for (int i = 0; i < nz_coeff - 1; ++i) {
                c[i] = 1.0 / 6.0;
            }
            double alpha = 1.0 / 6.0;
            double beta = 1.0 / 6.0;
            for (int i = 0; i < nx_coeff; ++i) {
                for (int j = 0; j < ny_coeff; ++j) {
                    for (int l = 0; l < value_size; ++l) {
                        Vec line(nz_coeff, 0.0);
                        for (int k = 0; k < nz_coeff; ++k) {
                            uint32_t idx = (i * ny_coeff + j) * nz_coeff + k;
                            line[k] = coeffs[idx * value_size + l];
                        }
                        Vec cline = solve_cyclic_tridiagonal(a, b, c, alpha, beta, line);
                        for (int k = 0; k < nz_coeff; ++k) {
                            uint32_t idx = (i * ny_coeff + j) * nz_coeff + k;
                            coeffs[idx * value_size + l] = cline[k];
                        }
                    }
                }
            }
        }

        all_local_vals_map = std::unordered_map<int, AlignedPaddedVec>();
        all_local_vals_map.reserve(cells_to_keep);
        for (int xidx = 0; xidx < nx; ++xidx) {
            for (int yidx = 0; yidx < ny; ++yidx) {
                for (int zidx = 0; zidx < nz; ++zidx) {
                    int meshidx = idx_cell(xidx, yidx, zidx);
                    if (skip_cell[meshidx]) {
                        continue;
                    }
                    AlignedPaddedVec local_vals(local_vals_size, 0.0);
                    int xbase = xbase_cell[xidx];
                    int ybase = ybase_cell[yidx];
                    int zbase = zbase_cell[zidx];
                    for (int i = 0; i < degree + 1; ++i) {
                        int xi = xbase + i;
                        for (int j = 0; j < degree + 1; ++j) {
                            int yi = ybase + j;
                            if (periodic_y) {
                                yi = (yi % ny_coeff + ny_coeff) % ny_coeff;
                            }
                            for (int k = 0; k < degree + 1; ++k) {
                                int zi = zbase + k;
                                if (periodic_z) {
                                    zi = (zi % nz_coeff + nz_coeff) % nz_coeff;
                                }
                                uint32_t idx = (xi * ny_coeff + yi) * nz_coeff + zi;
                                int offset_local = padded_value_size * idx_dof_local(i, j, k);
                                for (int l = 0; l < value_size; ++l) {
                                    local_vals[offset_local + l] = coeffs[idx * value_size + l];
                                }
                            }
                        }
                    }
                    all_local_vals_map.insert({meshidx, local_vals});
                }
            }
        }
        return;
    }
    int BATCH_SIZE = 16384;
    int NUM_BATCHES = dofs_to_keep/BATCH_SIZE + (dofs_to_keep % BATCH_SIZE != 0);
    
    auto batch_loop_start = std::chrono::high_resolution_clock::now();
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
    auto batch_loop_end = std::chrono::high_resolution_clock::now();
    double batch_seconds = std::chrono::duration<double>(batch_loop_end - batch_loop_start).count();
    
    std::cout << std::scientific
              << "[RegularGridInterpolant3D] serial interpolate_batch batch loop took "
              << batch_seconds << " s" << std::endl;

    int degree = rule.degree;
    all_local_vals_map = std::unordered_map<int, AlignedPaddedVec>();
    all_local_vals_map.reserve(cells_to_keep);

    auto grid_loop_start = std::chrono::high_resolution_clock::now();
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
    // auto grid_loop_end = std::chrono::high_resolution_clock::now();
    // grid_seconds += std::chrono::duration<double>(grid_loop_end - grid_loop_start).count();

    // std::cout << std::scientific
    //           << "[RegularGridInterpolant3D] serial interpolate_batch batch loop took "
    //           << batch_seconds << " s" << std::endl
    //           << "[RegularGridInterpolant3D] serial interpolate_batch grid loop took "
    //           << grid_seconds << " s" << std::endl;
}

#ifdef USE_MPI
template<class Array>
void RegularGridInterpolant3D<Array>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f, MPI_Comm comm) {
    if (rule.rule_type == InterpolationRuleType::CubicBSpline) {
        interpolate_batch(f);
        return;
    }
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int total_dofs = dofs_to_keep;
    int base = total_dofs / size;
    int rem = total_dofs % size;
    int local_dofs = base + (rank < rem ? 1 : 0);
    int local_first = rank * base + (rank < rem ? rank : rem);
    int local_last = local_first + local_dofs;

    int BATCH_SIZE = 16384;
    Vec local_vals(local_dofs * value_size, 0.);
    
    auto batch_loop_start = std::chrono::high_resolution_clock::now();
    for (int first = local_first; first < local_last; first += BATCH_SIZE) {
        int last = std::min(first + BATCH_SIZE, local_last);
        Vec xsub(xdoftensor_reduced.begin() + first, xdoftensor_reduced.begin() + last);
        Vec ysub(ydoftensor_reduced.begin() + first, ydoftensor_reduced.begin() + last);
        Vec zsub(zdoftensor_reduced.begin() + first, zdoftensor_reduced.begin() + last);
        Vec fxyzsub = f(xsub, ysub, zsub);

        int local_offset = (first - local_first) * value_size;
        for (int j = 0; j < last-first; ++j) {
            for (int l = 0; l < value_size; ++l) {
                local_vals[local_offset + j * value_size + l] = fxyzsub[j * value_size + l];
            }
        }
    }
    auto batch_loop_end = std::chrono::high_resolution_clock::now();
    double batch_seconds = std::chrono::duration<double>(batch_loop_end - batch_loop_start).count();

    std::vector<int> recv_counts(size, 0);
    std::vector<int> recv_displs(size, 0);
    for (int r = 0; r < size; ++r) {
        int r_dofs = base + (r < rem ? 1 : 0);
        recv_counts[r] = r_dofs * value_size;
        if (r > 0) {
            recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
        }
    }

    vals = Vec(total_dofs * value_size, 0.);
    auto comm_start = std::chrono::high_resolution_clock::now();
    MPI_Allgatherv(
        local_vals.data(),
        local_dofs * value_size,
        MPI_DOUBLE,
        vals.data(),
        recv_counts.data(),
        recv_displs.data(),
        MPI_DOUBLE,
        comm
    );
    auto comm_end = std::chrono::high_resolution_clock::now();
    double comm_seconds = std::chrono::duration<double>(comm_end - comm_start).count();

    if (rank == 0) {
        std::cout << std::scientific
                  << "[RegularGridInterpolant3D] MPI interpolate_batch batch loop took "
                  << batch_seconds << " s" << std::endl
                  << "[RegularGridInterpolant3D] MPI interpolate_batch communication took "
                  << comm_seconds << " s" << std::endl;
    }

    // Build all_local_vals_map (same on all ranks)
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
#endif

// #ifdef USE_MPI
// template<class Array>
// void RegularGridInterpolant3D<Array>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f, MPI_Comm comm) {
//     int BATCH_SIZE = 16384;
//     int rank, size;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &size);
    
//     // === Step 1: Parallel DOF evaluation (already good) ===
//     int total_dofs = dofs_to_keep;
//     int base = total_dofs / size;
//     int rem = total_dofs % size;
//     int local_dofs = base + (rank < rem ? 1 : 0);
//     int local_first = rank * base + (rank < rem ? rank : rem);
//     int local_last = local_first + local_dofs;

//     Vec local_vals(local_dofs * value_size, 0.);
//     for (int first = local_first; first < local_last; first += BATCH_SIZE) {
//         int last = std::min(first + BATCH_SIZE, local_last);
//         Vec xsub(xdoftensor_reduced.begin() + first, xdoftensor_reduced.begin() + last);
//         Vec ysub(ydoftensor_reduced.begin() + first, ydoftensor_reduced.begin() + last);
//         Vec zsub(zdoftensor_reduced.begin() + first, zdoftensor_reduced.begin() + last);
//         Vec fxyzsub = f(xsub, ysub, zsub);

//         int local_offset = (first - local_first) * value_size;
//         for (int j = 0; j < last-first; ++j) {
//             for (int l = 0; l < value_size; ++l) {
//                 local_vals[local_offset + j * value_size + l] = fxyzsub[j * value_size + l];
//             }
//         }
//     }

//     std::vector<int> recv_counts(size, 0);
//     std::vector<int> recv_displs(size, 0);
//     for (int r = 0; r < size; ++r) {
//         int r_dofs = base + (r < rem ? 1 : 0);
//         recv_counts[r] = r_dofs * value_size;
//         if (r > 0) {
//             recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
//         }
//     }

//     vals = Vec(total_dofs * value_size, 0.);
//     MPI_Allgatherv(
//         local_vals.data(),
//         local_dofs * value_size,
//         MPI_DOUBLE,
//         vals.data(),
//         recv_counts.data(),
//         recv_displs.data(),
//         MPI_DOUBLE,
//         comm
//     );

//     // === Step 2: Parallel grid loop construction (NEW) ===
//     int degree = rule.degree;
    
//     // Distribute cells across ranks
//     int total_cells_with_work = 0;
//     for (int xidx = 0; xidx < nx; ++xidx) {
//         for (int yidx = 0; yidx < ny; ++yidx) {
//             for (int zidx = 0; zidx < nz; ++zidx) {
//                 int meshidx = idx_cell(xidx, yidx, zidx);
//                 if (!skip_cell[meshidx])
//                     total_cells_with_work++;
//             }
//         }
//     }
    
//     int cells_base = total_cells_with_work / size;
//     int cells_rem = total_cells_with_work % size;
//     int my_cells_count = cells_base + (rank < cells_rem ? 1 : 0);
//     int my_cells_start = rank * cells_base + (rank < cells_rem ? rank : cells_rem);
//     int my_cells_end = my_cells_start + my_cells_count;
    
//     // Build only my portion of all_local_vals_map
//     std::unordered_map<int, AlignedPaddedVec> local_map;
//     local_map.reserve(my_cells_count);
    
//     int cell_counter = 0;
//     for (int xidx = 0; xidx < nx; ++xidx) {
//         for (int yidx = 0; yidx < ny; ++yidx) {
//             for (int zidx = 0; zidx < nz; ++zidx) {
//                 int meshidx = idx_cell(xidx, yidx, zidx);
//                 if (skip_cell[meshidx])
//                     continue;
                
//                 // Only process if this cell belongs to this rank
//                 if (cell_counter >= my_cells_start && cell_counter < my_cells_end) {
//                     AlignedPaddedVec local_vals(local_vals_size, 0.);
//                     for (int i = 0; i < degree+1; ++i) {
//                         for (int j = 0; j < degree+1; ++j) {
//                             for (int k = 0; k < degree+1; ++k) {
//                                 int offset = value_size*full_to_reduced_map[idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+k)];
//                                 int offset_local = padded_value_size * idx_dof_local(i, j, k);
//                                 for (int l = 0; l < value_size; ++l) {
//                                     local_vals[offset_local + l] = vals[offset + l];
//                                 }
//                             }
//                         }
//                     }
//                     local_map.insert({meshidx, local_vals});
//                 }
//                 cell_counter++;
//             }
//         }
//     }
    
//     // === Step 3: Gather all maps (if needed for evaluation) ===
//     // Option A: If all ranks need all data for evaluation
//     all_local_vals_map = gather_all_maps(local_map, comm);
    
//     // Option B: If evaluation is always local, just use:
//     // all_local_vals_map = std::move(local_map);
// }
// #endif

// #ifdef USE_MPI
// template<class Array>
// std::unordered_map<int, AlignedPaddedVec> 
// RegularGridInterpolant3D<Array>::gather_all_maps(
//     const std::unordered_map<int, AlignedPaddedVec>& local_map, 
//     MPI_Comm comm) 
// {
//     int rank, size;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &size);
    
//     // Serialize local map: [meshidx1, data1..., meshidx2, data2..., ...]
//     std::vector<int> local_keys;
//     std::vector<double> local_data;
//     for (const auto& [meshidx, vals] : local_map) {
//         local_keys.push_back(meshidx);
//         local_data.insert(local_data.end(), vals.begin(), vals.end());
//     }
    
//     int local_count = local_keys.size();
//     std::vector<int> all_counts(size);
//     MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);
    
//     std::vector<int> displs(size, 0);
//     int total_cells = 0;
//     for (int r = 0; r < size; ++r) {
//         if (r > 0) displs[r] = displs[r-1] + all_counts[r-1];
//         total_cells += all_counts[r];
//     }
    
//     std::vector<int> all_keys(total_cells);
//     MPI_Allgatherv(local_keys.data(), local_count, MPI_INT,
//                    all_keys.data(), all_counts.data(), displs.data(), MPI_INT, comm);
    
//     // Gather data (each cell has local_vals_size doubles)
//     for (int r = 0; r < size; ++r) {
//         all_counts[r] *= local_vals_size;
//         displs[r] *= local_vals_size;
//     }
    
//     std::vector<double> all_data(total_cells * local_vals_size);
//     MPI_Allgatherv(local_data.data(), local_count * local_vals_size, MPI_DOUBLE,
//                    all_data.data(), all_counts.data(), displs.data(), MPI_DOUBLE, comm);
    
//     // Reconstruct map
//     std::unordered_map<int, AlignedPaddedVec> result;
//     result.reserve(total_cells);
//     for (int i = 0; i < total_cells; ++i) {
//         AlignedPaddedVec vals(local_vals_size);
//         std::copy(all_data.begin() + i * local_vals_size,
//                   all_data.begin() + (i + 1) * local_vals_size,
//                   vals.begin());
//         result.insert({all_keys[i], vals});
//     }
    
//     return result;
// }
// #endif

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
void RegularGridInterpolant3D<Array>::evaluate_batch_derivs(
    Array& xyz, Array& fxyz, Array& dfdx, Array& dfdy, Array& dfdz
){
    if(fxyz.layout() != xt::layout_type::row_major
        || dfdx.layout() != xt::layout_type::row_major
        || dfdy.layout() != xt::layout_type::row_major
        || dfdz.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Outputs need to be in row-major storage order");
    int npoints = xyz.shape(0);
    for (int i = 0; i < npoints; ++i) {
        double x = xyz(i, 0);
        double y = xyz(i, 1);
        double z = xyz(i, 2);
        if(x >= xmax) x -= _EPS_;
        else if (x <= xmin) x += _EPS_;
        if(y >= ymax) y -= _EPS_;
        else if (y <= ymin) y += _EPS_;
        if(z >= zmax) z -= _EPS_;
        else if (z <= zmin) z += _EPS_;
        int xidx = int(nx*(x-xmin)/(xmax-xmin));
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
        evaluate_local_derivs(
            xlocal, ylocal, zlocal, idx_cell(xidx, yidx, zidx),
            fxyz.data() + value_size*i,
            dfdx.data() + value_size*i,
            dfdy.data() + value_size*i,
            dfdz.data() + value_size*i
        );
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
    if (rule.rule_type == InterpolationRuleType::CubicBSpline) {
        int xidx = cell_idx / (ny * nz);
        int yidx = (cell_idx / nz) % ny;
        int zidx = cell_idx % nz;
        double xglobal = xmesh[xidx] + x * hx;
        int span = find_span(xknots, degree, xglobal);
        Vec N;
        basis_funs(span, xglobal, degree, xknots, N);
        for (int k = 0; k < degree + 1; ++k) {
            pkxs[k] = N[k];
        }
        Vec By, dBy;
        Vec Bz, dBz;
        cubic_bspline_basis(y, By, dBy);
        cubic_bspline_basis(z, Bz, dBz);
        for (int k = 0; k < degree + 1; ++k) {
            pkys[k] = By[k];
            pkzs[k] = Bz[k];
        }
    } else {
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

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_local_derivs(
    double x, double y, double z, int cell_idx, double* res, double* dx, double* dy, double* dz
) {
    int degree = rule.degree;
    auto got = all_local_vals_map.find(cell_idx);
    if (got == all_local_vals_map.end()) {
        if(out_of_bounds_ok)
            return;
        else
            throw std::runtime_error((boost::format("cell_idx={} not in all_local_vals_map") % cell_idx).str());
    }
    if (rule.rule_type != InterpolationRuleType::CubicBSpline) {
        throw std::runtime_error("Derivative evaluation only implemented for cubic B-splines.");
    }

    int xidx = cell_idx / (ny * nz);
    int yidx = (cell_idx / nz) % ny;
    int zidx = cell_idx % nz;
    double xglobal = xmesh[xidx] + x * hx;
    int span = find_span(xknots, degree, xglobal);
    Vec N, dN;
    basis_funs_der1(span, xglobal, degree, xknots, N, dN);
    for (int k = 0; k < degree + 1; ++k) {
        pkxs[k] = N[k];
        dpkxs[k] = dN[k];
    }
    Vec By, dBy;
    Vec Bz, dBz;
    cubic_bspline_basis(y, By, dBy);
    cubic_bspline_basis(z, Bz, dBz);
    for (int k = 0; k < degree + 1; ++k) {
        pkys[k] = By[k];
        pkzs[k] = Bz[k];
        dpkys[k] = dBy[k] / hy;
        dpkzs[k] = dBz[k] / hz;
    }

    double* vals_local = got->second.data();
    for (int l = 0; l < value_size; ++l) {
        res[l] = 0.0;
        dx[l] = 0.0;
        dy[l] = 0.0;
        dz[l] = 0.0;
    }
    for (int i = 0; i < degree + 1; ++i) {
        for (int j = 0; j < degree + 1; ++j) {
            for (int k = 0; k < degree + 1; ++k) {
                int offset_local = padded_value_size * idx_dof_local(i, j, k);
                double wx = pkxs[i];
                double wy = pkys[j];
                double wz = pkzs[k];
                double wdx = dpkxs[i];
                double wdy = dpkys[j];
                double wdz = dpkzs[k];
                for (int l = 0; l < value_size; ++l) {
                    double coeff = vals_local[offset_local + l];
                    res[l] += coeff * wx * wy * wz;
                    dx[l] += coeff * wdx * wy * wz;
                    dy[l] += coeff * wx * wdy * wz;
                    dz[l] += coeff * wx * wy * wdz;
                }
            }
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
