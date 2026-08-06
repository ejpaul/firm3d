# import time
import unittest

import numpy as np

import firm3dpp

try:
    from simsopt.field.tracing import (
        IterationStoppingCriterion as SimsoptIterationStoppingCriterion,
    )
    from simsopt.geo import SurfaceRZFourier

    from simsopt.field import (
        BiotSavart,
        InterpolatedField,
        SurfaceClassifier,
        coils_via_symmetries,
        load_coils_from_makegrid_file,
        trace_particles,
    )

    HAS_SIMSOPT = True
except Exception:
    HAS_SIMSOPT = False
    InterpolatedField = type(None)
from firm3d.catapult.tracing import trace_particles_boozer_gpu
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
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

HAS_CUDA = hasattr(firm3dpp, "test_gpu_interpolation")


def get_field(boozmn_filename, n_metagrid_pts, vacuum):
    bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=vacuum)
    nfp = bri.nfp
    degree = 3
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
    s = np.random.uniform(low=0, high=1.1, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    stz = np.hstack((s, t, z))
    return stz


def cartesian_rhs(position, vpar, field, mass, charge, velocity):
    field.set_points_cyl(position.reshape(-1, 3))
    B = field.B()
    GradAbsB = field.GradAbsB()
    AbsB = np.linalg.norm(B[0])

    BcrossGradAbsB = [0] * 3
    BcrossGradAbsB[0] = B[0, 1] * GradAbsB[0, 2] - B[0, 2] * GradAbsB[0, 1]
    BcrossGradAbsB[1] = B[0, 2] * GradAbsB[0, 0] - B[0, 0] * GradAbsB[0, 2]
    BcrossGradAbsB[2] = B[0, 0] * GradAbsB[0, 1] - B[0, 1] * GradAbsB[0, 0]

    v_perp2 = velocity**2 - vpar**2
    mu = v_perp2 / (2 * AbsB)
    fak1 = vpar / AbsB
    fak2 = (mass / (charge * AbsB**3)) * (0.5 * v_perp2 + vpar**2)

    out = [0] * 4
    for i in range(3):
        out[i] = fak1 * B[0, i] + fak2 * BcrossGradAbsB[i]
    out[3] = -mu * np.sum([B[0, i] * GradAbsB[0, i] for i in range(3)]) / AbsB
    return out


def test_interpolant(
    field, nfp, stz, saw_present=False, surf_classifier=None, tol=1e-8
):
    # if in Cartesian coordinates
    if isinstance(field, InterpolatedField):
        rrange, phirange, zrange, quad_info = cartesian_interpolant(
            field, surf_classifier
        )
        field.set_points_cyl(stz)
        # Quantities to interpolate
        B = field.B_cyl()
        GradAbsB = field.GradAbsB_cyl()

        # Compare interpolation of B and GradAbsB
        cpu_interpolation = np.hstack((B, GradAbsB))
        gpu_interpolation = firm3dpp.test_gpu_interpolation(
            quad_info,
            rrange,
            phirange,
            zrange,
            stz.copy(),
            "cartesian_vacuum",
            stz.shape[0],
        )

        gpu_interpolation = np.reshape(gpu_interpolation, (stz.shape[0], -1))
        gpu_interpolation = gpu_interpolation[:, 0:6]

    else:  # Boozer coordinates
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(
            field, nfp, saw_present=saw_present
        )

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
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    stz.copy(),
                    "boozer",
                    stz.shape[0],
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
    psi0=0,
    time=None,
    saw_present=False,
    saw_filename=None,
    surf_classifier=None,
    tol=1e-8,
):
    if isinstance(field, InterpolatedField):  # Cartesian
        rrange, phirange, zrange, quad_info = cartesian_interpolant(
            field, surf_classifier
        )
        gpu_derivs = firm3dpp.test_derivatives_cartesian(
            quad_info,
            rrange,
            phirange,
            zrange,
            stz,
            vpar,
            vtotal,
            MASS,
            CHARGE,
            stz.shape[0],
        )
        gpu_derivs = np.reshape(gpu_derivs, (stz.shape[0], 4))
        cpu_derivs = np.empty((stz.shape[0], 4))
        for i in range(stz.shape[0]):
            cpu_derivs[i, :] = cartesian_rhs(
                stz[i, :], vpar[i], field, MASS, CHARGE, vtotal
            )

    else:
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp)
        ## evaluate derivatives
        if isinstance(field, ShearAlfvenWavesSuperposition):
            assert time is not None, (
                "time array must be provided when testing derivatives with SAW"
            )
            assert saw_filename is not None, (
                "saw filename must be provided when testing derivatives with SAW"
            )
            # evaluate CPU derivatives
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
                        field,
                        stz[i, :],
                        MASS,
                        CHARGE,
                        vtotal,
                        vpar[i],
                        time[i],
                        "nok_saw",
                    )
            else:
                ValueError("Field type not recognized for SAW derivatives")

            saw_nharmonics = 5
            ## load saw data as arrays
            saw_data = np.load(saw_filename, allow_pickle=True)
            saw_data = saw_data[()]
            saw_omega = field.get_wave(0).omega
            s = field.get_wave(0).phihat.get_s_basis()
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
                cpu_derivs = np.empty((stz.shape[0], 4))
                for i in range(stz.shape[0]):
                    cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                        field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], vacuum=True
                    )

                ## evaluate GPU interpolant
                stz = np.ascontiguousarray(stz)
                vpar = np.ascontiguousarray(vpar)
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
                cpu_derivs = np.empty((stz.shape[0], 4))
                # start_time = time.time()
                for i in range(stz.shape[0]):
                    cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                        field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], vacuum=False
                    )

                ## evaluate GPU interpolant
                stz = np.ascontiguousarray(stz)
                vpar = np.ascontiguousarray(vpar)
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
    field,
    nfp,
    stz,
    vpar,
    vtotal,
    psi0=0,
    time=None,
    saw_filename=None,
    tol=1e-8,
    surf_classifier=None,
):
    if isinstance(field, InterpolatedField):  # Cartesian
        rrange, phirange, zrange, quad_info = cartesian_interpolant(
            field, surf_classifier
        )
        last_time = firm3dpp.test_timestep_cartesian(
            quad_pts=quad_info,
            rrange=rrange,
            phirange=phirange,
            zrange=zrange,
            loc_init=stz,
            m=MASS,
            q=CHARGE,
            vtotal=np.sqrt(2 * ENERGY / MASS),
            vtang=vpar,
            tol=1e-9,
            nparticles=stz.shape[0],
        )
        gpu_final_positions = np.reshape(last_time, (stz.shape[0], 5))

        rphiz = stz
        r = rphiz[:, 0].reshape(-1, 1)
        phi = rphiz[:, 1].reshape(-1, 1)
        z = rphiz[:, 2].reshape(-1, 1)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xyz = np.hstack((x, y, z))
        gc_tys, gc_zeta_hits = trace_particles(
            field,
            xyz,
            vpar,
            tmax=1e-2,
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            tol=1e-9,
            stopping_criteria=[SimsoptIterationStoppingCriterion(1)],
            forget_exact_path=True,
        )
        cpu_positions = np.array([x[-1] for x in gc_tys])

    else:
        srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp)

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
            s = field.get_wave(0).phihat.get_s_basis()
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

        ### test interpolant
        is_small = test_interpolant(field, nfp, stz, tol=tol)
        self.assertTrue(is_small)

        ### test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = test_derivatives(
            field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol)
        self.assertTrue(is_small)

    def test_boozer_collisions(self):
        """
        Collisional GPU tracing against the collisionless path.

        The discriminating assertion is that the two differ: if the kick were
        a no-op -- the species count never reaching the device, the constant
        memory not populated, the RNG dead -- the collisional trace would
        reproduce the collisionless one exactly, and every other check here
        would still pass.

        Every call gets its own copy of the initial conditions.  The tracer
        rewrites stz_init in place, turning (s, theta) into
        (s cos theta, s sin theta) in the caller's buffer, so handing the same
        array to both calls would launch them from different particles and
        satisfy the comparison for a reason that has nothing to do with
        collisions.  Measured on an A100: sharing the array gives
        frac_moved = 1.00 whether or not the kick runs, while with copies a
        zero-density background gives 0.02 against 1.00 for a live one.
        """
        from firm3d.catapult.tracing import trace_particles_boozer_with_collisions_gpu
        from firm3d.field.collisions import ThermalBackground
        from firm3d.util.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS

        n_metagrid_pts = 15
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, True)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        n = 256
        rng = np.random.default_rng(0)
        stz = np.column_stack(
            [
                np.full(n, 0.3),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        )
        vpar = 0.5 * VELOCITY * np.ones(n)

        # Dense and cold, so the collision rates are fast enough to move the
        # ensemble measurably within a short trace.
        bg = ThermalBackground(
            n_profile=lambda s: 1e21,
            T_profile=lambda s: 1e3 * ONE_EV,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        kw = {
            "tmax": 2e-6,
            "mass": MASS,
            "charge": CHARGE,
            "vtotal": VELOCITY,
            "tol": 1e-8,
            "ns": 15,
            "ntheta": 15,
            "nzeta": 15,
        }

        # A zero-density background makes every coefficient identically zero,
        # so it is the control: it goes through the whole collisional entry
        # point -- upload, constant memory, RNG, kick -- and must still land
        # on the collisionless answer.  Without it, "the two runs differ" is
        # satisfied by any difference at all, including one the kick did not
        # cause.
        zero_bg = ThermalBackground(
            n_profile=lambda s: 0.0,
            T_profile=lambda s: 1e3 * ONE_EV,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

        without = trace_particles_boozer_gpu(field, stz.copy(), vpar, **kw)
        with_coll = trace_particles_boozer_with_collisions_gpu(
            field, stz.copy(), vpar, backgrounds=bg, rng_seed=0, **kw
        )
        no_kick = trace_particles_boozer_with_collisions_gpu(
            field, stz.copy(), vpar, backgrounds=zero_bg, rng_seed=0, **kw
        )

        self.assertEqual(with_coll.shape, (n, 6))
        self.assertTrue(np.all(np.isfinite(with_coll)), "non-finite GPU results")

        # The guard in adjust_time must hold with the kick active too: it
        # fires on every accepted step, so a particle running past tmax would
        # keep being scattered.
        np.testing.assert_allclose(with_coll[:, 0], kw["tmax"], rtol=1e-12)

        def frac_moved(a, b):
            return np.mean(np.abs(a[:, 4] - b[:, 4]) > 1e-6 * VELOCITY)

        moved = frac_moved(with_coll, without)
        self.assertGreater(
            moved,
            0.9,
            f"only {moved:.2f} of particles differ from the collisionless "
            f"run; the collision kick is not reaching the device",
        )

        unmoved = frac_moved(no_kick, without)
        self.assertLess(
            unmoved,
            0.1,
            f"{unmoved:.2f} of particles differ from the collisionless run "
            f"with a zero-density background, where every coefficient is "
            f"zero; the comparison above is not measuring the kick",
        )

        # Column 5 is the speed, which is what makes the energy and mu
        # recoverable.  Checked against two references, because a column that
        # merely looked plausible would satisfy either alone: the launch speed
        # (which the zero-density run must return) and the zero-density run
        # itself (which the collisional one must not match).
        # With every coefficient zero the kick changes nothing, so the speed
        # must come back as the launch speed, up to the energy drift of a
        # non-symplectic adaptive integrator.  Measured 3.4e-05 here; the
        # bound below is loose against that but still tight by orders of
        # magnitude against the ways this column could be wrong -- dt would
        # read 6e-08 and v_par half the launch speed.
        #
        # This assertion is what caught the first version of the column, which
        # rebuilt v at output time from stage 6 of the derivative buffer.
        # That stage is evaluated at state + dt*(...), which coincides with
        # the state only in the instant after a step is accepted, so for any
        # particle that finished before its block did the |B| was taken at the
        # wrong point: the drift was 1.0e-02, 300x worse than this.
        np.testing.assert_allclose(
            no_kick[:, 5],
            VELOCITY,
            rtol=1e-3,
            err_msg=(
                "zero-density run does not return the launch speed in column "
                "5; the speed column is not what is being written"
            ),
        )
        # The collisional speeds must differ from the zero-density ones, not
        # merely from the launch speed.  Comparing against VELOCITY instead
        # would be satisfied by the 3.4e-05 integrator drift measured above --
        # 68x any threshold worth setting -- so it would pass with the
        # coefficients zeroed.  The zero-density run carries that same drift,
        # so differencing against it cancels the drift and leaves the kick.
        moved_v = np.mean(np.abs(with_coll[:, 5] - no_kick[:, 5]) > 1e-6 * VELOCITY)
        self.assertGreater(
            moved_v,
            0.9,
            f"only {moved_v:.2f} of speeds differ from the zero-density run; "
            f"the kick is not changing the speed it reports",
        )
        # Not asserted: that the ensemble slows on average.  An alpha's
        # slowing-down time in this background is of order 0.1 s, so over
        # 2e-6 s drag moves the mean by ~1e-5 relative -- below the spread the
        # pitch-angle noise puts on it, and it can go either way on one seed.
        # v >= |v_par| is required for mu = (v^2 - v_par^2)/(2|B|) >= 0.
        self.assertTrue(
            np.all(with_coll[:, 5] >= np.abs(with_coll[:, 4]) - 1e-6 * VELOCITY),
            "speed is below |v_par|, so the recovered mu would be negative",
        )

    def test_boozer_finite_beta(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = False
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)

        tol = 1e-8

        ### test interpolant
        is_small = test_interpolant(field, nfp, stz, tol)
        self.assertTrue(is_small)

        ### test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = test_derivatives(
            field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol
        )
        self.assertTrue(is_small)

        ### test timesteps
        is_small = test_timestep(field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol)
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

        ### test interpolant
        is_small = test_interpolant(saw, nfp, stz, saw_present=True, tol=tol)
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

        ### test interpolant
        is_small = test_interpolant(saw, nfp, stz, saw_present=True, tol=tol)
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
        )
        self.assertTrue(is_small)

    @unittest.skipUnless(HAS_SIMSOPT, "simsopt not available")
    def test_cartesian_vacuum(self):
        np.random.seed(0)
        degree = 3  # degree of interpolant
        n = 16  # resolution of interpolant
        order = 12  # order of coil curves

        filename = "examples/inputs/coils.curves_22_7_21"
        wout_filename = "examples/inputs/wout_aten_rescaled.nc"

        surf = SurfaceRZFourier.from_wout(wout_filename)

        coils = load_coils_from_makegrid_file(filename, order, ppp=20, group_names=None)

        curves = []
        currents = []
        for _i, coil in enumerate(coils):
            curves.append(coil.curve)
            currents.append(coil.current)

        coils_full = coils_via_symmetries(curves, currents, surf.nfp, True)
        bs = BiotSavart(coils_full)

        sc_particle = SurfaceClassifier(surf, h=0.1, p=2)
        rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
        zs = surf.gamma()[:, :, 2]

        rrange = (np.min(rs), np.max(rs), n)
        phirange = (0, 2 * np.pi / surf.nfp, n * 2)
        # exploit stellarator symmetry and only consider positive z values:
        zrange = (0, np.max(zs), n // 2)
        bsh = InterpolatedField(
            bs, degree, rrange, phirange, zrange, True, nfp=surf.nfp, stellsym=True
        )

        # rejection sample points inside the surface uniformly
        nparticles = 10000
        rphiz = np.empty((nparticles, 3))
        for i in range(nparticles):
            pt = np.random.uniform(low=0, high=1, size=(1, 3))
            pt[0, 0] = pt[0, 0] * (rrange[1] - rrange[0]) + rrange[0]
            pt[0, 1] *= 2 * np.pi
            pt[0, 2] = (pt[0, 2] - 0.5) * 2 * zrange[1]

            # particle is outside the surface or too close to the surface
            max_iters = 1000
            for _ in range(max_iters):
                if sc_particle.evaluate_rphiz(pt) > 0.2:
                    break
                pt = np.random.uniform(low=0, high=1, size=(1, 3))
                pt[0, 0] = pt[0, 0] * (rrange[1] - rrange[0]) + rrange[0]
                pt[0, 1] *= 2 * np.pi
                pt[0, 2] = (pt[0, 2] - 0.5) * 2 * zrange[1]
            else:
                raise RuntimeError("Could not sample a valid point inside the surface")
            rphiz[i, :] = pt
        xyz = np.empty((nparticles, 3))
        xyz[:, 0] = rphiz[:, 0] * np.cos(rphiz[:, 1])
        xyz[:, 1] = rphiz[:, 0] * np.sin(rphiz[:, 1])
        xyz[:, 2] = rphiz[:, 2]

        # test interpolant
        is_small = test_interpolant(
            bsh, surf.nfp, rphiz, surf_classifier=sc_particle, tol=1e-8
        )
        self.assertTrue(is_small)

        # test rhs
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (nparticles,))
        is_small = test_derivatives(
            bsh,
            surf.nfp,
            rphiz,
            vpar_init,
            VELOCITY,
            surf_classifier=sc_particle,
            tol=1e-8,
        )
        self.assertTrue(is_small)

        # test timestep
        is_small = test_timestep(
            bsh,
            surf.nfp,
            rphiz,
            vpar_init,
            VELOCITY,
            surf_classifier=sc_particle,
            tol=1e-8,
        )
        self.assertTrue(is_small)


if __name__ == "__main__":
    print("Running GPU tracing tests...")
    unittest.main()
