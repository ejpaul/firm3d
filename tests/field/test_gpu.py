# import time
import unittest

import numpy as np

import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)
from firm3d.field.tracing import (
    IterationStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
)
from firm3d.util.constants import (
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)
from firm3d.util.gpu_utils import boozer_interpolant, boozer_saw_interpolant

HAS_CUDA = hasattr(firm3dpp, "test_gpu_interpolation")


def get_field(boozmn_filename, n_metagrid_pts, vacuum):
    # start_time = time.time()
    bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=vacuum)
    nfp = bri.nfp
    degree = 3
    # start_time = time.time()
    # print("bri field type:", bri.field_type)
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )
    # Even though bri isn't used further in this script, we need to return it,
    # or else it is garbage-collected, resulting in an error.
    return bri, field, nfp


def construct_interpolant(field, nfp, saw_present=False):
    ns, ntheta, nzeta = 15, 15, 15

    if isinstance(field, ShearAlfvenWavesSuperposition):
        field = field.B0
        srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
            field, nfp, ns, ntheta, nzeta
        )
    else:  # the field is an InterpolatedBoozerField (unperturbed)
        # print(field.field_type)
        if field.field_type == "vac":
            srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
                field, nfp, ns, ntheta, nzeta, vacuum=True
            )
        elif field.field_type == "":  # implies finite beta
            srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
                field, nfp, ns, ntheta, nzeta, vacuum=False
            )

    return srange, trange, zrange, quad_info, maxJ


def sample_test_points(n_test_pts):
    np.random.seed(1865)
    # generate test points
    s = np.random.uniform(low=0, high=0.95, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    stz = np.hstack((s, t, z))
    return stz


def test_interpolant(field, nfp, stz, saw_present=False, tol=1e-8, quad_data=None):
    if quad_data is None:
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(
            field, nfp, saw_present=saw_present
        )
    else:
        srange, trange, zrange, quad_info, maxJ = quad_data

    # evaluate interpolants
    if isinstance(field, ShearAlfvenWavesSuperposition):
        field = field.B0
        field.set_points(stz)
        modB = field.modB()
        modB_derivs = field.modB_derivs()
        G = field.G()
        dGds = field.dGds()
        I = field.I()
        dIds = field.dIds()
        iota = field.iota()
        diotads = field.diotads()
        cpu_interpolation = np.hstack(
            (modB, modB_derivs, G, dGds, I, dIds, iota, diotads)
        )

        ## evaluate GPU interpolant
        stz = np.ascontiguousarray(stz)
        gpu_interpolation = firm3dpp.test_gpu_interpolation(
            quad_info,
            srange,
            trange,
            zrange,
            stz.copy(),
            "boozer_saw_vacuum",
            stz.shape[0],
        )
    else:
        if field.field_type == "vac":
            # print("saw not present")
            # evaluate CPU interpolant
            field.set_points(stz)
            modB = field.modB()
            modB_derivs = field.modB_derivs()
            G = field.G()
            iota = field.iota()
            cpu_interpolation = np.hstack((modB, modB_derivs, G, iota))

            ## evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            gpu_interpolation = firm3dpp.test_gpu_interpolation(
                quad_info,
                srange,
                trange,
                zrange,
                stz.copy(),
                "boozer_vacuum",
                stz.shape[0],
            )
        elif field.field_type == "":  # implies finite beta
            # evaluate CPU interpolant
            field.set_points(stz)
            modB = field.modB()
            modB_derivs = field.modB_derivs()
            G = field.G()
            dGds = field.dGds()
            I = field.I()
            dIds = field.dIds()
            iota = field.iota()
            K = field.K()
            K_derivs = field.K_derivs()
            cpu_interpolation = np.hstack(
                (modB, modB_derivs, G, dGds, I, dIds, iota, K, K_derivs)
            )

            # evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            gpu_interpolation = firm3dpp.test_gpu_interpolation(
                quad_info, srange, trange, zrange, stz.copy(), "boozer", stz.shape[0]
            )

    gpu_interpolation = np.reshape(gpu_interpolation, (stz.shape[0], -1))

    # compute error
    error_is_small = np.isclose(
        gpu_interpolation, cpu_interpolation, rtol=tol, atol=tol
    ).all()
    error = np.abs(cpu_interpolation - gpu_interpolation) / (
        np.abs(cpu_interpolation) + 1
    )

    if error.max() > tol:
        print("tolerance not satisfied in interpolant")
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_interpolation[row_idx, :])
        print("gpu:", gpu_interpolation[row_idx, :])
        print("error:", error[row_idx, :])

    return error_is_small


def test_derivatives(
    field,
    nfp,
    stz,
    vpar,
    vtotal,
    psi0,
    time=None,
    saw_present=False,
    saw_filename=None,
    tol=1e-8,
    quad_data=None,
):
    if quad_data is None:
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp)
    else:
        srange, trange, zrange, quad_info, maxJ = quad_data
    ## evaluate derivatives
    if isinstance(field, ShearAlfvenWavesSuperposition):
        assert time is not None, (
            "time array must be provided when testing derivatives with SAW"
        )
        assert saw_filename is not None, (
            "saw filename must be provided when testing derivatives with SAW"
        )
        # evaluate CPU derivatives
        # print("computing simsopt derivatives")
        cpu_derivs = np.empty((stz.shape[0], 4))

        if field.B0.field_type == "vac":
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_saw(
                    field,
                    stz[i, :],
                    MASS,
                    CHARGE,
                    vtotal,
                    vpar[i],
                    time[i],
                    "vacuum_saw",
                )
        elif field.B0.field_type == "nok":  # NoK tracing
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_saw(
                    field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], time[i], "nok_saw"
                )
        else:
            ValueError("Field type not recognized for SAW derivatives")

        saw_nharmonics = 5
        ## load saw data as arrays
        saw_data = np.load(saw_filename, allow_pickle=True)
        saw_data = saw_data[()]
        saw_omega = field.get_wave(0).omega
        # print("omega=", saw_omega)
        s = field.get_wave(0).phihat.get_s_basis()
        # print("s", s)
        # print(s[40:])
        saw_srange = (s[0], s[-1], len(s))

        saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
        saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
        saw_phihats = np.ascontiguousarray(
            np.column_stack(
                [
                    np.array([field.get_wave(i).phihat(s_val) for s_val in s])
                    for i in range(saw_nharmonics)
                ]
            )
        )
        ## evaluate GPU interpolant
        stz = np.ascontiguousarray(stz)
        vpar = np.ascontiguousarray(vpar)
        # print("calculating new derivatives")
        # start_time = time.time()

        if field.B0.field_type == "vac":
            gpu_derivs = firm3dpp.test_derivatives_saw(
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
                vpar,
                time,
                vtotal,
                MASS,
                CHARGE,
                psi0,
                stz.shape[0],
            )
        elif field.B0.field_type == "nok":
            gpu_derivs = firm3dpp.test_derivatives_saw_nok(
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
                vpar,
                time,
                vtotal,
                MASS,
                CHARGE,
                psi0,
                stz.shape[0],
            )
        else:
            ValueError("Field type not recognized for SAW derivatives")
    else:
        if field.field_type == "vac":
            # evaluate CPU derivatives
            # print("computing simsopt derivatives")
            cpu_derivs = np.empty((stz.shape[0], 4))
            # start_time = time.time()
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                    field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], vacuum=True
                )

            ## evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            vpar = np.ascontiguousarray(vpar)
            # print("calculating new derivatives")
            # start_time = time.time()
            gpu_derivs = firm3dpp.test_derivatives_boozer(
                quad_info,
                srange,
                trange,
                zrange,
                stz.copy(),
                vpar,
                vtotal,
                MASS,
                CHARGE,
                psi0,
                stz.shape[0],
                vacuum=True,
            )
        elif field.field_type == "":  # implies finite beta
            # evaluate CPU derivatives
            # print("computing simsopt derivatives")
            cpu_derivs = np.empty((stz.shape[0], 4))
            # start_time = time.time()
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                    field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], vacuum=False
                )

            ## evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            vpar = np.ascontiguousarray(vpar)
            # print("calculating new derivatives")
            # start_time = time.time()
            gpu_derivs = firm3dpp.test_derivatives_boozer(
                quad_info,
                srange,
                trange,
                zrange,
                stz.copy(),
                vpar,
                vtotal,
                MASS,
                CHARGE,
                psi0,
                stz.shape[0],
                vacuum=False,
            )
    gpu_derivs = np.reshape(gpu_derivs, (stz.shape[0], 4))

    error_is_small = np.isclose(gpu_derivs, cpu_derivs, rtol=tol, atol=tol).all()
    error = np.abs(cpu_derivs - gpu_derivs) / (np.abs(cpu_derivs) + 1)

    if not error_is_small:
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_derivs[row_idx, :])
        print("gpu:", gpu_derivs[row_idx, :])
        print("rel error:", error[row_idx, :])

    return error_is_small


def test_timestep(
    field, nfp, stz, vpar, vtotal, psi0, time=None, saw_filename=None, tol=1e-8, quad_data=None
):
    if quad_data is None:
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp)
    else:
        srange, trange, zrange, quad_info, maxJ = quad_data

    if isinstance(field, ShearAlfvenWavesSuperposition):
        assert saw_filename is not None, (
            "saw filename must be provided when testing timesteps with SAW"
        )
        # evaluate CPU timestep
        field.B0.set_points(stz)
        mu_init = (vtotal**2 - vpar**2) / (2 * field.B0.modB()[:, 0])

        gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
            field,
            stz,
            vpar,
            mu_init,
            tmax=1e-2,
            mass=MASS,
            charge=CHARGE,
            tol=1e-9,
            stopping_criteria=[IterationStoppingCriterion(1)],
            forget_exact_path=True,
        )

        saw_nharmonics = 5
        ## load saw data as arrays
        saw_data = np.load(saw_filename, allow_pickle=True)
        saw_data = saw_data[()]
        saw_omega = field.get_wave(0).omega
        # print("omega=", saw_omega)
        s = field.get_wave(0).phihat.get_s_basis()
        # print("s", s)
        # print(s[40:])
        saw_srange = (s[0], s[-1], len(s))

        saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
        saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
        saw_phihats = np.ascontiguousarray(
            np.column_stack(
                [
                    np.array([field.get_wave(i).phihat(s_val) for s_val in s])
                    for i in range(saw_nharmonics)
                ]
            )
        )
        stz = np.ascontiguousarray(stz)
        psi0 = field.B0.psi0

        if field.B0.field_type == "vac":
            last_time = firm3dpp.test_timestep_saw(
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
                vtang=vpar,
                time=time,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
            )
        elif field.B0.field_type == "nok":
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
                vtang=vpar,
                time=time,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
            )
        last_time = np.reshape(last_time, (stz.shape[0], 5))
    else:
        if field.field_type == "vac":
            gc_tys, gc_zeta_hits = trace_particles_boozer(
                field,
                stz,
                vpar,
                tmax=1e-2,
                mass=MASS,
                charge=CHARGE,
                Ekin=ENERGY,
                tol=1e-9,
                stopping_criteria=[IterationStoppingCriterion(1)],
                forget_exact_path=True,
            )

            stz = np.ascontiguousarray(stz)
            psi0 = field.psi0
            last_time = firm3dpp.test_timestep_boozer(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                stz_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=np.sqrt(2 * ENERGY / MASS),
                vtang=vpar,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
                vacuum=True,
            )
            last_time = np.reshape(last_time, (stz.shape[0], 5))
        elif field.field_type == "":  # implies finite beta
            gc_tys, gc_zeta_hits = trace_particles_boozer(
                field,
                stz,
                vpar,
                tmax=1e-2,
                mass=MASS,
                charge=CHARGE,
                Ekin=ENERGY,
                tol=1e-9,
                stopping_criteria=[IterationStoppingCriterion(1)],
                forget_exact_path=True,
            )

            stz = np.ascontiguousarray(stz)
            psi0 = field.psi0
            last_time = firm3dpp.test_timestep_boozer(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                stz_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=np.sqrt(2 * ENERGY / MASS),
                vtang=vpar,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
                vacuum=False,
            )
            last_time = np.reshape(last_time, (stz.shape[0], 5))

    # map to pseudo-cylindrical coordinates
    cpu_positions = np.array([x[-1] for x in gc_tys])
    cpu_positions = np.array(
        [
            [x[0], x[1] * np.cos(x[2]), x[1] * np.sin(x[2]), x[3], x[4]]
            for x in cpu_positions
        ]
    )
    gpu_final_positions = np.array(
        [
            [x[0], x[1] * np.cos(x[2]), x[1] * np.sin(x[2]), x[3], x[4]]
            for x in last_time
        ]
    )
    error_is_small = np.isclose(
        gpu_final_positions, cpu_positions, rtol=tol, atol=tol
    ).all()
    error = np.abs(cpu_positions - gpu_final_positions) / (np.abs(cpu_positions) + 1)

    if error.max() > tol:
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_positions[row_idx, :])
        print("gpu:", gpu_final_positions[row_idx, :])
        print("error:", error[row_idx, :])

    return error_is_small


@unittest.skipUnless(HAS_CUDA, "CUDA support not available")
class TestGPUTracing(unittest.TestCase):
    def test_boozer_vacuum(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)

        tol = 1e-8
        quad_data = construct_interpolant(field, nfp)

        ### test interpolant
        is_small = test_interpolant(field, nfp, stz, tol=tol, quad_data=quad_data)
        self.assertTrue(is_small)

        ### test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = test_derivatives(
            field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol, quad_data=quad_data
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(
            field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol, quad_data=quad_data
        )
        self.assertTrue(is_small)

    def test_boozer_finite_beta(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = False
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)

        tol = 1e-8
        quad_data = construct_interpolant(field, nfp)

        ### test interpolant
        is_small = test_interpolant(field, nfp, stz, tol=tol, quad_data=quad_data)
        self.assertTrue(is_small)

        ### test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = test_derivatives(
            field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol, quad_data=quad_data
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(
            field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol, quad_data=quad_data
        )
        self.assertTrue(is_small)

    def test_boozer_vacuum_saw(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        ### set up SAW
        saw_filename = "./examples/tracing_with_AE/ae.npy"
        saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=saw_filename,
            ),
            B0=field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)
        tol = 1e-8
        quad_data = construct_interpolant(saw, nfp)

        ### test interpolant
        is_small = test_interpolant(saw, nfp, stz, saw_present=True, tol=tol, quad_data=quad_data)
        self.assertTrue(is_small)

        ## test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
        is_small = test_derivatives(
            saw,
            nfp,
            stz,
            vpar_init,
            VELOCITY,
            field.psi0,
            time=time,
            saw_present=True,
            saw_filename=saw_filename,
            tol=tol,
            quad_data=quad_data,
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(
            saw,
            nfp,
            stz,
            vpar_init,
            VELOCITY,
            field.psi0,
            time=time,
            saw_filename=saw_filename,
            tol=tol,
            quad_data=quad_data,
        )
        self.assertTrue(is_small)

    def test_boozer_nok_saw(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        ### set up SAW
        saw_filename = "./examples/tracing_with_AE/ae.npy"
        saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=saw_filename,
            ),
            B0=field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)
        tol = 1e-8
        quad_data = construct_interpolant(saw, nfp)

        ### test interpolant
        is_small = test_interpolant(saw, nfp, stz, saw_present=True, tol=tol, quad_data=quad_data)
        self.assertTrue(is_small)

        ## test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
        is_small = test_derivatives(
            saw,
            nfp,
            stz,
            vpar_init,
            VELOCITY,
            field.psi0,
            time=time,
            saw_present=True,
            saw_filename=saw_filename,
            tol=tol,
            quad_data=quad_data,
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(
            saw,
            nfp,
            stz,
            vpar_init,
            VELOCITY,
            field.psi0,
            time=time,
            saw_filename=saw_filename,
            tol=tol,
            quad_data=quad_data,
        )
        self.assertTrue(is_small)


if __name__ == "__main__":
    print("Running GPU tracing tests...")
    unittest.main()
