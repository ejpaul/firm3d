import numpy as np

import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)
from firm3d.field.tracing import (
    IterationStoppingCriterion,
    trace_particles_boozer_perturbed,
)

# for SAW wave
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import ALPHA_PARTICLE_CHARGE as CHARGE
from firm3d.util.constants import ALPHA_PARTICLE_MASS as MASS
from firm3d.util.constants import FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
from firm3d.util.gpu_utils import boozer_saw_interpolant

np.random.seed(1800)


def test_derivs(field, saw_filename, nfp, n_metagrid_pts, n_test_pts):
    saw = ShearAlfvenWavesSuperposition.from_ae3d(
        eigenvector=AE3DEigenvector.load_from_numpy(
            filename=saw_filename,
        ),
        B0=field,
        max_dB_normal_by_B0=5e-3,
        minor_radius_meters=1.7,
    )
    # generate test points
    s = np.random.uniform(low=0, high=1, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
    stz = np.hstack((s, t, z))

    VELOCITY = np.sqrt(2 * ENERGY / MASS)
    vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))

    # print("computing simsopt derivatives")
    old_derivs = np.empty((n_test_pts, 4))
    for i in range(n_test_pts):
        old_derivs[i, :] = firm3dpp.simsopt_derivs_saw(
            saw, stz[i, :], MASS, CHARGE, VELOCITY, vpar_init[i], time[i], "nok_saw"
        )

    # print("firm3d saw derivs", old_derivs)
    ### NEW INTERPOLANT
    srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
        field, nfp, n_metagrid_pts
    )
    stz = np.ascontiguousarray(stz)

    psi0 = field.psi0
    saw_nharmonics = 5
    ## load saw data as arrays
    saw_data = np.load(saw_filename, allow_pickle=True)
    saw_data = saw_data[()]
    saw_omega = saw.get_wave(0).omega
    # print("omega=", saw_omega)
    s = saw.get_wave(0).phihat.get_s_basis()
    # print("s", s)
    # print(s[40:])
    saw_srange = (s[0], s[-1], len(s))

    saw_m = [saw.get_wave(i).Phim for i in range(saw_nharmonics)]
    saw_n = [saw.get_wave(i).Phin for i in range(saw_nharmonics)]
    saw_phihats = np.ascontiguousarray(
        np.column_stack(
            [
                np.array([saw.get_wave(i).phihat(s_val) for s_val in s])
                for i in range(saw_nharmonics)
            ]
        )
    )
    # print("saw_phihats", saw_phihats[40:, :])
    # print(saw_phihats.shape)

    # print(saw_data['harmonics'][0])
    # saw_m = np.ascontiguousarray([x[0] for x in saw_data['harmonics']])
    # saw_n = np.ascontiguousarray([x[1] for x in saw_data['harmonics']])
    # scale_factor = 21333.26539440892
    # saw_phihats = np.ascontiguousarray(np.column_stack([x[2].T for x in saw_data['harmonics']])) * scale_factor
    # saw_nharmonics = len(saw_m)

    # print("saw phihats shape: ", saw_phihats.shape)
    # print("stz: ", stz)
    # print("saw_srange: ", saw_srange)
    # print("saw_m", saw_m)
    # print("saw_n", saw_n)
    # print("saw_phihats", saw_phihats)
    # print("calculating new derivatives")
    new_derivs = firm3dpp.test_derivatives_saw_nok(
        quad_info,
        srange,
        trange,
        zrange,
        saw_omega,
        saw_srange,
        saw_m,
        saw_n,
        saw_phihats,
        saw_nharmonics,
        stz,
        vpar_init,
        time,
        VELOCITY,
        MASS,
        CHARGE,
        psi0,
        stz.shape[0],
    )
    new_derivs = np.reshape(new_derivs, (stz.shape[0], 4))
    # print("computed new derivatives", new_derivs)

    rel_err = np.abs((old_derivs - new_derivs) / old_derivs)
    diff = np.max(rel_err)
    # print(np.abs(old_derivs - new_derivs) / old_derivs)

    print(f"Maximum relative error in derivative values on {n_test_pts} points: {diff}")

    if diff > 1e-8:
        print("SAW RHS TEST FAILED")

        print("culprit particle:")
        row_index = np.argmax(rel_err) // rel_err.shape[1]
        print(stz[row_index, :])
        print(vpar_init[row_index])
        print(time[row_index])
        print("simsopt", old_derivs[row_index, :])
        print("new", new_derivs[row_index, :])
        print(rel_err[row_index, :])
    else:
        print("SAW RHS TEST SUCCESS")


def test_timestep(field, saw_filename, nfp, n_metagrid_pts, n_test_pts):
    # generate saw object
    saw = ShearAlfvenWavesSuperposition.from_ae3d(
        eigenvector=AE3DEigenvector.load_from_numpy(
            filename=saw_filename,
        ),
        B0=field,
        max_dB_normal_by_B0=5e-3,
        minor_radius_meters=1.7,
    )

    # generate test points
    s = np.random.uniform(low=0, high=0.95, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    time = (
        np.random.uniform(low=0, high=1e-3, size=(n_test_pts,)) * 0
    )  # for now just test t=0

    stz = np.hstack((s, t, z))

    VELOCITY = np.sqrt(2 * ENERGY / MASS)
    vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))

    # print("computing simsopt timestep")

    field.set_points(stz)
    mu_init = (VELOCITY**2 - vpar_init**2) / (2 * field.modB()[:, 0])
    gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
        saw,
        stz,
        vpar_init,
        mu_init,
        tmax=1e-2,
        mass=MASS,
        charge=CHARGE,
        Ekin=ENERGY,
        tol=1e-9,
        stopping_criteria=[IterationStoppingCriterion(0)],
        forget_exact_path=True,
    )

    final_positions = np.array([x[-1] for x in gc_tys])
    final_positions = np.array(
        [
            [x[0], x[1] * np.cos(x[2]), x[1] * np.sin(x[2]), x[3], x[4]]
            for x in final_positions
        ]
    )

    # print("computing new timesteps")
    srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
        field, nfp, n_metagrid_pts
    )
    stz = np.ascontiguousarray(stz)
    psi0 = field.psi0
    saw_nharmonics = 5
    ## load saw data as arrays
    saw_data = np.load(saw_filename, allow_pickle=True)
    saw_data = saw_data[()]
    saw_omega = saw.get_wave(0).omega
    # print("omega=", saw_omega)
    s = saw.get_wave(0).phihat.get_s_basis()
    # print("s", s)
    # print(s[40:])
    saw_srange = (s[0], s[-1], len(s))

    saw_m = [saw.get_wave(i).Phim for i in range(saw_nharmonics)]
    saw_n = [saw.get_wave(i).Phin for i in range(saw_nharmonics)]
    saw_phihats = np.ascontiguousarray(
        np.column_stack(
            [
                np.array([saw.get_wave(i).phihat(s_val) for s_val in s])
                for i in range(saw_nharmonics)
            ]
        )
    )

    last_time = firm3dpp.test_timestep_saw_nok(
        quad_pts=quad_info,
        srange=srange,
        trange=trange,
        zrange=zrange,
        saw_omega=saw_omega,
        saw_srange=saw_srange,
        saw_m=saw_m,
        saw_n=saw_n,
        saw_phihats=saw_phihats,
        saw_nharmonics=saw_nharmonics,
        stz_init=stz,
        m=MASS,
        q=CHARGE,
        vtotal=np.sqrt(2 * ENERGY / MASS),
        vtang=vpar_init,
        time=time,
        tol=1e-9,
        psi0=psi0,
        nparticles=n_test_pts,
    )

    last_time = np.reshape(last_time, (n_test_pts, 5))

    # map to pseudo-cylindrical coordinates
    new_final_positions = np.array(
        [
            [x[0], x[1] * np.cos(x[2]), x[1] * np.sin(x[2]), x[3], x[4]]
            for x in last_time
        ]
    )

    # normalize vparallel errors
    abs_err = np.abs(final_positions - new_final_positions)
    abs_err[:, 4] /= np.sqrt(2 * ENERGY / MASS)
    diff = np.max(abs_err)
    # print(np.abs(final_positions - new_final_positions) / final_positions)

    print(f"Maximum absolute error in final positions on {n_test_pts} points: {diff}")
    if diff > 1e-6:
        print("SAW TIMESTEP TEST FAILED")

        print("culprit particle:")
        row_index = np.argmax(abs_err) // abs_err.shape[1]
        print(stz[row_index, :])
        print(vpar_init[row_index])
        print("simsopt", final_positions[row_index, :])
        print("new", new_final_positions[row_index, :])
        print(abs_err[row_index, :])
    else:
        print("SAW TIMESTEP TEST SUCCESS")


if __name__ == "__main__":
    ### CREATE A FIELD FOR TRACING
    boozmn_filename = "examples/inputs/boozmn_aten_rescaled.nc"
    bri = BoozerRadialInterpolant(boozmn_filename, 3, no_K=True)

    nfp = bri.nfp
    degree = 3
    n_metagrid_pts = 15
    srange = (0, 1, n_metagrid_pts)
    thetarange = (0, np.pi, n_metagrid_pts)
    zetarange = (0, 2 * np.pi / nfp, n_metagrid_pts)
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )

    ### SET UP A PERTURBED B FIELD
    saw_filename = "./examples/tracing_with_AE/ae.npy"

    test_derivs(field, saw_filename, nfp, 15, 10000)

    test_timestep(field, saw_filename, nfp, 15, 10000)
