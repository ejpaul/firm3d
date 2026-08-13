# import time
import functools
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


@functools.lru_cache(maxsize=1)
def build_wout_boozer_field():
    """
    Boozer field for the cross-coordinate tests, built from the same wout
    file as the coil surface; the boozmn files in examples/inputs are
    differently rescaled equilibria whose boundaries do not match the coils.
    """
    bri = BoozerRadialInterpolant(
        "examples/inputs/wout_aten_rescaled.nc", 3, enforce_vacuum=True
    )
    bfield = InterpolatedBoozerField(
        bri, 3, ns_interp=15, ntheta_interp=15, nzeta_interp=15
    )
    return bri, bfield, bri.nfp


@functools.lru_cache(maxsize=1)
def build_cartesian_field():
    """
    Coil field, boundary classifier, and grid ranges for the Cartesian tests.
    """
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
    return surf, sc_particle, bsh, rrange, phirange, zrange


def sample_rphiz_inside(nparticles, rrange, zrange, sc_particle, lo=0.2, hi=np.inf):
    """
    Rejection sample cylindrical points whose signed distance to the plasma
    boundary lies in (lo, hi); the interior is positive.  The band form lets
    the collisional tests place ensembles at chosen depths in the flux label.
    """
    rphiz = np.empty((nparticles, 3))
    for i in range(nparticles):
        pt = np.random.uniform(low=0, high=1, size=(1, 3))
        pt[0, 0] = pt[0, 0] * (rrange[1] - rrange[0]) + rrange[0]
        pt[0, 1] *= 2 * np.pi
        pt[0, 2] = (pt[0, 2] - 0.5) * 2 * zrange[1]

        # particle is outside the surface or too close to the surface
        max_iters = 1000
        for _ in range(max_iters):
            if lo < sc_particle.evaluate_rphiz(pt) < hi:
                break
            pt = np.random.uniform(low=0, high=1, size=(1, 3))
            pt[0, 0] = pt[0, 0] * (rrange[1] - rrange[0]) + rrange[0]
            pt[0, 1] *= 2 * np.pi
            pt[0, 2] = (pt[0, 2] - 0.5) * 2 * zrange[1]
        else:
            raise RuntimeError("Could not sample a valid point inside the surface")
        rphiz[i, :] = pt
    return rphiz


def rphiz_to_xyz(rphiz):
    xyz = np.empty_like(rphiz)
    xyz[:, 0] = rphiz[:, 0] * np.cos(rphiz[:, 1])
    xyz[:, 1] = rphiz[:, 0] * np.sin(rphiz[:, 1])
    xyz[:, 2] = rphiz[:, 2]
    return xyz


def distance_flux_label(sc_particle, scale=0.3):
    """
    Flux-label stand-in built from the signed boundary distance d:
    s = 1 - d/scale, clipped to [0, 2].  It is 1 on the boundary, falls
    toward the core, and exceeds 1 outside, mimicking a normalized flux
    without needing an equilibrium; the coil-field tests use it because a
    Biot-Savart field carries no flux surfaces of its own.
    """

    def label(points_rphiz):
        d = np.asarray(sc_particle.evaluate_rphiz(points_rphiz)).reshape(-1)
        return np.clip(1.0 - d / scale, 0.0, 2.0)

    return label


def equilibrium_flux_label(bfield, nfp, n_s=48, n_theta=48, n_zeta_per_period=48):
    """
    Flux label s(r, phi, z) built by forward-mapping a dense Boozer grid
    through the equilibrium and answering queries with the nearest mapped
    point.
    """
    from scipy.spatial import cKDTree

    from firm3d.field.coordinates import boozer_to_cylindrical

    s = np.linspace(0.02, 1.0, n_s)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, nfp * n_zeta_per_period, endpoint=False)
    grid = np.array(np.meshgrid(s, theta, zeta, indexing="ij")).reshape(3, -1).T
    samples_xyz = rphiz_to_xyz(boozer_to_cylindrical(bfield, grid))
    tree = cKDTree(samples_xyz)
    s_samples = grid[:, 0]

    def label(points_rphiz):
        _, idx = tree.query(rphiz_to_xyz(np.asarray(points_rphiz)))
        return s_samples[idx]

    return label


@functools.lru_cache(maxsize=1)
def build_equilibrium_label():
    """Flux label for the wout-built equilibrium, cached across tests."""
    bri, bfield, nfp = build_wout_boozer_field()
    return equilibrium_flux_label(bfield, nfp)


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

    def test_zero_density_reproduces_the_collisionless_path(self):
        """
        A zero-density background makes every collision coefficient zero, so
        the collisional entry point must return the collisionless answer.
        """
        from firm3d.catapult.tracing import trace_particles_boozer_with_collisions_gpu
        from firm3d.field.collisions import ThermalBackground
        from firm3d.util.constants import ELEMENTARY_CHARGE, PROTON_MASS

        bri, field, nfp = get_field(
            "examples/inputs/boozmn_aten_rescaled_low_res.nc", 15, True
        )
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
        zero_bg = ThermalBackground(
            n_profile=lambda s: 0.0,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        without = trace_particles_boozer_gpu(field, stz.copy(), vpar, **kw)
        no_kick = trace_particles_boozer_with_collisions_gpu(
            field, stz.copy(), vpar, backgrounds=zero_bg, rng_seed=0, **kw
        )

        self.assertEqual(no_kick.shape, (n, 7))
        differing = np.mean(np.abs(no_kick[:, 4] - without[:, 4]) > 1e-6 * VELOCITY)
        self.assertLess(
            differing,
            0.1,
            f"{differing:.2f} of particles differ from the collisionless run "
            f"with every coefficient zero; the kick is not a no-op there",
        )
        np.testing.assert_allclose(
            no_kick[:, 5],
            VELOCITY,
            rtol=1e-3,
            err_msg=(
                "zero-density run does not return the launch speed in column "
                "5; the speed column is not what is being written"
            ),
        )

    def _collisional_ensemble(self, background, vtotal, xi0, tmax, n=256, seed=0):
        """
        Trace n particles with collisions on the GPU; return their final
        (xi, v).  Shared by the two physics tests below, which differ only in
        the background and the launch pitch.
        """
        from firm3d.catapult.tracing import trace_particles_boozer_with_collisions_gpu

        bri, field, nfp = get_field(
            "examples/inputs/boozmn_aten_rescaled_low_res.nc", 15, True
        )
        rng = np.random.default_rng(seed)
        stz = np.column_stack(
            [
                np.full(n, 0.3),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        )
        out = trace_particles_boozer_with_collisions_gpu(
            field,
            stz,
            xi0 * vtotal * np.ones(n),
            backgrounds=background,
            tmax=tmax,
            mass=MASS,
            charge=CHARGE,
            vtotal=vtotal,
            tol=1e-8,
            ns=15,
            ntheta=15,
            nzeta=15,
            rng_seed=seed,
        )
        self.assertTrue(np.all(np.isfinite(out)), "non-finite GPU results")
        np.testing.assert_allclose(out[:, 0], tmax, rtol=1e-12)
        v, vpar = out[:, 5], out[:, 4]
        self.assertTrue(
            np.all(v >= np.abs(vpar) - 1e-6 * vtotal),
            "speed is below |v_par|, so the recovered mu would be negative",
        )
        return vpar / v, v

    def test_collisions_isotropize_the_pitch(self):
        """
        An ion background scatters pitch: an ensemble launched at xi = 0.9
        must relax toward isotropy, <xi> -> 0 and <xi^2> -> 1/3.
        """
        from firm3d.field.collisions import ThermalBackground
        from firm3d.util.constants import ELEMENTARY_CHARGE, PROTON_MASS

        deuterium = ThermalBackground(
            n_profile=lambda s: 1e25,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        xi, _ = self._collisional_ensemble(
            deuterium, 0.1 * np.sqrt(2 * ENERGY / MASS), xi0=0.9, tmax=2e-6
        )
        mean_xi, mean_xi2 = np.mean(xi), np.mean(xi**2)
        # Targets are the stationary values of the pitch SDE, whose
        # distribution is xi ~ U(-1, 1): <xi> = 0 and <xi^2> = 1/3
        self.assertLess(abs(mean_xi), 0.15, f"<xi> = {mean_xi:.3f}, launched at 0.9")
        self.assertGreater(mean_xi2, 0.24, f"<xi^2> = {mean_xi2:.3f}, isotropic is 1/3")
        self.assertLess(mean_xi2, 0.43, f"<xi^2> = {mean_xi2:.3f}, isotropic is 1/3")

    def test_electron_drag_slows_without_scattering_pitch(self):
        """
        An electron background drags but barely scatters: <v> must fall while
        <xi> stays where it was launched.
        """
        from firm3d.field.collisions import ThermalBackground
        from firm3d.util.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

        electrons = ThermalBackground(
            n_profile=lambda s: 1e25,
            T_profile=lambda s: 10e3,
            mass=ELECTRON_MASS,
            charge=-ELEMENTARY_CHARGE,
        )
        v0 = np.sqrt(2 * ENERGY / MASS)
        xi, v = self._collisional_ensemble(electrons, v0, xi0=0.5, tmax=1e-6)
        ratio = np.mean(v) / v0
        # Integrating dv/dt = K(v) over tmax gives v/v0 = 0.825
        self.assertGreater(ratio, 0.80, f"<v>/v0 = {ratio:.3f}; too little drag")
        self.assertLess(ratio, 0.87, f"<v>/v0 = {ratio:.3f}; too much drag")
        self.assertAlmostEqual(
            np.mean(xi),
            0.5,
            delta=0.05,
            msg=f"<xi> = {np.mean(xi):.3f}; electrons should barely scatter pitch",
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
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()

        # rejection sample points inside the surface uniformly
        nparticles = 10000
        rphiz = sample_rphiz_inside(nparticles, rrange, zrange, sc_particle)

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

    @unittest.skipUnless(HAS_SIMSOPT, "simsopt not available")
    def test_cartesian_collision_interpolant(self):
        """
        The 8-column quad-point layout must reproduce the field columns of
        the 7-column layout and return the flux label in the last column.
        The label is quadratic per axis, so the tricubic interpolant
        represents it exactly; it is phi-independent and even in z, so the
        symmetry fold must return it unchanged.
        """
        np.random.seed(0)
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()

        def label(points_rphiz):
            r, z = points_rphiz[:, 0], points_rphiz[:, 2]
            return 0.3 + 0.1 * r + 0.2 * z**2

        rrange_i, phirange_i, zrange_i, quad_info = cartesian_interpolant(
            bsh, sc_particle, flux_label=label
        )
        self.assertEqual(quad_info.shape[1], 8)

        n_pts = 500
        rphiz = sample_rphiz_inside(n_pts, rrange, zrange, sc_particle)
        gpu = firm3dpp.test_gpu_interpolation(
            quad_info,
            rrange_i,
            phirange_i,
            zrange_i,
            rphiz.copy(),
            "cartesian_vacuum_coll",
            n_pts,
        )
        gpu = np.reshape(gpu, (n_pts, -1))

        bsh.set_points_cyl(rphiz)
        cpu = np.hstack((bsh.B_cyl(), bsh.GradAbsB_cyl()))
        np.testing.assert_allclose(gpu[:, 0:6], cpu, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(gpu[:, 7], label(rphiz), rtol=1e-8, atol=1e-8)

    @unittest.skipUnless(HAS_SIMSOPT, "simsopt not available")
    def test_cartesian_collisions(self):
        """
        Collisional Cartesian GPU tracing against the collisionless path,
        mirroring test_boozer_collisions: the kick must reach the device,
        and a zero-density background must reproduce the collisionless run.
        """
        from firm3d.catapult.tracing import (
            trace_particles_cartesian_gpu,
            trace_particles_cartesian_with_collisions_gpu,
        )
        from firm3d.field.collisions import ThermalBackground
        from firm3d.util.constants import ELEMENTARY_CHARGE, PROTON_MASS

        np.random.seed(0)
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()

        n = 128
        # Launch deep so the ensemble reaches tmax rather than the boundary.
        rphiz = sample_rphiz_inside(n, rrange, zrange, sc_particle, lo=0.2)
        xyz = rphiz_to_xyz(rphiz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar = 0.5 * VELOCITY * np.ones(n)
        label = distance_flux_label(sc_particle)

        # Dense and cold, so the collision rates are fast enough to move the
        # ensemble measurably within a short trace.
        bg = ThermalBackground(
            n_profile=lambda s: 1e21,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        # A zero-density background makes every coefficient identically zero
        zero_bg = ThermalBackground(
            n_profile=lambda s: 0.0,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        kw = {
            "tmax": 1e-6,
            "mass": MASS,
            "charge": CHARGE,
            "vtotal": VELOCITY,
            "tol": 1e-8,
        }

        without = trace_particles_cartesian_gpu(
            bsh, sc_particle, xyz.copy(), vpar, **kw
        )
        with_coll = trace_particles_cartesian_with_collisions_gpu(
            bsh, sc_particle, label, xyz.copy(), vpar, backgrounds=bg, rng_seed=0, **kw
        )
        no_kick = trace_particles_cartesian_with_collisions_gpu(
            bsh,
            sc_particle,
            label,
            xyz.copy(),
            vpar,
            backgrounds=zero_bg,
            rng_seed=0,
            **kw,
        )

        self.assertEqual(with_coll.shape, (n, 7))
        self.assertTrue(np.all(np.isfinite(with_coll)), "non-finite GPU results")

        # Restrict the comparison to particles every run traced to tmax, so
        # a boundary loss cannot masquerade as a collisional change.
        done = (
            np.isclose(without[:, 0], kw["tmax"], rtol=1e-12)
            & np.isclose(with_coll[:, 0], kw["tmax"], rtol=1e-12)
            & np.isclose(no_kick[:, 0], kw["tmax"], rtol=1e-12)
        )
        self.assertGreater(
            np.mean(done), 0.9, "too many particles lost for the statistics below"
        )

        def frac_moved(a, b):
            return np.mean(np.abs(a[done, 4] - b[done, 4]) > 1e-6 * VELOCITY)

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

        np.testing.assert_allclose(
            no_kick[done, 5],
            VELOCITY,
            rtol=1e-3,
            err_msg=(
                "zero-density run does not return the launch speed in column "
                "5; the speed column is not what is being written"
            ),
        )
        moved_v = np.mean(
            np.abs(with_coll[done, 5] - no_kick[done, 5]) > 1e-6 * VELOCITY
        )
        self.assertGreater(
            moved_v,
            0.9,
            f"only {moved_v:.2f} of speeds differ from the zero-density run; "
            f"the kick is not changing the speed it reports",
        )
        # v >= |v_par| is required for mu = (v^2 - v_par^2)/(2|B|) >= 0.
        self.assertTrue(
            np.all(with_coll[:, 5] >= np.abs(with_coll[:, 4]) - 1e-6 * VELOCITY),
            "speed is below |v_par|, so the recovered mu would be negative",
        )

    @unittest.skipUnless(HAS_SIMSOPT, "simsopt not available")
    def test_cartesian_collisions_read_the_flux_label(self):
        """
        A background confined to small flux-label values must leave particles
        at large label values collisionless: the end-to-end check that the
        label is interpolated at the particle position and fed to the kick.

        Launching at 0.1 v_alpha keeps drift orbits within millimeters of
        their surface; at full energy the shallow ensemble grazes the dense
        zone.  The shallow group is compared against a zero-density control
        run at the same seed.
        """
        from firm3d.catapult.tracing import (
            trace_particles_cartesian_with_collisions_gpu,
        )
        from firm3d.field.collisions import ThermalBackground
        from firm3d.util.constants import ELEMENTARY_CHARGE, PROTON_MASS

        np.random.seed(0)
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()

        # The density cut at s = 0.5 sits at boundary distance d = 0.15,
        # between the deep band (collides) and the shallow band (must not).
        label = distance_flux_label(sc_particle, scale=0.3)
        n_deep, n_shallow = 64, 64
        deep = sample_rphiz_inside(n_deep, rrange, zrange, sc_particle, lo=0.2)
        shallow = sample_rphiz_inside(
            n_shallow, rrange, zrange, sc_particle, lo=0.04, hi=0.10
        )
        xyz = rphiz_to_xyz(np.vstack((deep, shallow)))
        n = n_deep + n_shallow

        bg = ThermalBackground(
            n_profile=lambda s: 1e21 if s < 0.5 else 0.0,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )
        zero_bg = ThermalBackground(
            n_profile=lambda s: 0.0,
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

        vtotal = 0.1 * np.sqrt(2 * ENERGY / MASS)
        kw = {
            "xyz_inits": xyz,
            "parallel_speeds": 0.5 * vtotal * np.ones(n),
            "tmax": 1e-6,
            "mass": MASS,
            "charge": CHARGE,
            "vtotal": vtotal,
            "tol": 1e-8,
            "rng_seed": 0,
        }
        out = trace_particles_cartesian_with_collisions_gpu(
            bsh, sc_particle, label, backgrounds=bg, **kw
        )
        control = trace_particles_cartesian_with_collisions_gpu(
            bsh, sc_particle, label, backgrounds=zero_bg, **kw
        )
        self.assertTrue(np.all(np.isfinite(out)), "non-finite GPU results")

        moved = np.mean(np.abs(out[:n_deep, 5] - vtotal) > 1e-6 * vtotal)
        self.assertGreater(
            moved,
            0.9,
            f"only {moved:.2f} of the deep ensemble changed speed; the kick "
            f"is not seeing the dense region of the profile",
        )
        row_identical = np.all(
            np.isclose(out[n_deep:, :], control[n_deep:, :], rtol=1e-12, atol=0.0),
            axis=1,
        )
        self.assertGreater(
            np.mean(row_identical),
            0.9,
            f"only {np.mean(row_identical):.2f} of the zero-density ensemble "
            f"matches the zero-density control run; the flux label reaching "
            f"the kick is wrong",
        )

    @unittest.skipUnless(HAS_SIMSOPT, "simsopt not available")
    def test_cross_coordinate_collision_consistency(self):
        """
        The same alpha ensemble in the same ATEN configuration must produce
        the same collision statistics whether traced in Boozer coordinates
        (equilibrium field; the state carries s) or in Cartesian coordinates
        (coil field; s arrives through the interpolated flux label).

        The density varies with s, so the drag each ensemble feels is set by
        the label its tracer reads.  A third run with a deliberately wrong
        (constant) label is the control: it must disagree by much more than
        the two coordinate systems disagree with each other, which is what
        makes the agreement above evidence rather than coincidence.

        Drag is the discriminator, not pitch: at this launch speed an ion
        background barely scatters pitch (<xi^2> moves 0.49 -> 0.48 over
        tmax), while <v>/v0 falls to 0.87 and shifts by 0.12 per decade of
        density.  tmax is chosen to stop well short of thermalization, since
        a relaxed ensemble reaches the same Maxwellian at any rate and would
        pass with any label at all.
        """
        from firm3d.catapult.tracing import (
            trace_particles_boozer_with_collisions_gpu,
            trace_particles_cartesian_with_collisions_gpu,
        )
        from firm3d.field.collisions import ThermalBackground
        from firm3d.field.coordinates import boozer_to_cylindrical
        from firm3d.util.constants import ELEMENTARY_CHARGE, PROTON_MASS

        bri, bfield, nfp = build_wout_boozer_field()
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()
        label = build_equilibrium_label()

        n = 256
        s_launch = 0.35
        rng = np.random.default_rng(3)
        stz = np.column_stack(
            [
                np.full(n, s_launch),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        )
        xyz = rphiz_to_xyz(boozer_to_cylindrical(bfield, stz.copy()))

        deuterium = ThermalBackground(
            n_profile=lambda s: 1e23 * (1.0 - 0.9 * s),
            T_profile=lambda s: 1e3,
            mass=2 * PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )

        # 0.1 v_alpha keeps the drift orbits tight: the ensemble holds to
        # s = 0.3500 +/- 0.0005, so the field difference between the two
        # tracers shifts the density they see by ~0.1%, while a label error
        # is order 1 in s and so order 10 in this profile.
        vtotal = 0.1 * np.sqrt(2 * ENERGY / MASS)
        vpar = 0.7 * vtotal * np.ones(n)
        kw = {
            "backgrounds": deuterium,
            "tmax": 2e-6,
            "mass": MASS,
            "charge": CHARGE,
            "vtotal": vtotal,
            "tol": 1e-8,
            "rng_seed": 0,
        }

        boozer_out = trace_particles_boozer_with_collisions_gpu(
            bfield, stz.copy(), vpar.copy(), ns=15, ntheta=15, nzeta=15, **kw
        )
        cart_out = trace_particles_cartesian_with_collisions_gpu(
            bsh, sc_particle, label, xyz.copy(), vpar.copy(), **kw
        )
        wrong_out = trace_particles_cartesian_with_collisions_gpu(
            bsh,
            sc_particle,
            lambda pts: np.full(len(pts), 0.9),
            xyz.copy(),
            vpar.copy(),
            **kw,
        )

        self.assertTrue(np.all(np.isfinite(boozer_out)), "non-finite Boozer results")
        self.assertTrue(np.all(np.isfinite(cart_out)), "non-finite Cartesian results")

        done_b = np.isclose(boozer_out[:, 0], kw["tmax"], rtol=1e-12)
        done_c = np.isclose(cart_out[:, 0], kw["tmax"], rtol=1e-12)
        done_w = np.isclose(wrong_out[:, 0], kw["tmax"], rtol=1e-12)
        self.assertGreater(np.mean(done_b), 0.95, "Boozer ensemble lost particles")
        self.assertGreater(np.mean(done_c), 0.95, "Cartesian ensemble lost particles")
        self.assertGreater(np.mean(done_w), 0.95, "control ensemble lost particles")

        drag_b = np.mean(boozer_out[done_b, 5]) / vtotal
        drag_c = np.mean(cart_out[done_c, 5]) / vtotal
        drag_w = np.mean(wrong_out[done_w, 5]) / vtotal

        # Measured: Boozer 0.8679, Cartesian 0.8677.  The bound is set by the
        # seed-to-seed spread of the drag (~0.007), not by that agreement,
        # which is closer than the statistics warrant.  The control at
        # s = 0.9 lands 0.097 away, and a label wrong by 0.25 in s would
        # still land 0.045 away.
        self.assertLess(
            abs(drag_b - drag_c),
            0.02,
            f"<v>/v0 disagrees across coordinates: Boozer {drag_b:.4f}, "
            f"Cartesian {drag_c:.4f}",
        )
        self.assertGreater(
            abs(drag_b - drag_w),
            0.04,
            f"a constant flux label of 0.9 gives <v>/v0 = {drag_w:.4f} against "
            f"the Boozer {drag_b:.4f}; the agreement above is insensitive to "
            f"the label and proves nothing",
        )

        # The two ensembles must also sit on the same surface, so that the
        # drag agreement is not two errors cancelling.
        final_rphiz = np.column_stack(
            [
                np.hypot(cart_out[done_c, 1], cart_out[done_c, 2]),
                np.arctan2(cart_out[done_c, 2], cart_out[done_c, 1]),
                cart_out[done_c, 3],
            ]
        )
        s_cart = np.mean(label(final_rphiz))
        s_booz = np.mean(boozer_out[done_b, 1])
        self.assertLess(
            abs(s_booz - s_cart),
            0.05,
            f"mean flux label disagrees: Boozer {s_booz:.3f}, Cartesian {s_cart:.3f}",
        )

    @unittest.skipUnless(HAS_SIMSOPT, "simsopt not available")
    def test_cross_coordinate_alpha_relaxation_stages(self):
        """
        Full-energy alphas in a DT + electron background relax in two
        stages: electron drag slows them with almost no pitch scattering,
        then ion scattering isotropizes the pitch as the speed approaches
        the critical velocity.  Both stages must appear at the same times
        in Boozer and Cartesian tracing, and the confined speed
        distributions must agree as distributions (KS), not just in mean.

        This complements test_cross_coordinate_collision_consistency: that
        test pins the flux label with a wrong-label control at reduced
        speed; this one pins the multi-species physics -- the pitch channel
        exercises the kick's (v_par, mu) writeback, and the late-time decay
        depends on the ion Chandrasekhar functions at the local label.

        Bounds sit ~3x above the differences measured at these settings
        (energy 0.002, pitch 0.008 over a six-point time ladder).
        """
        from scipy.stats import ks_2samp

        from firm3d.catapult.tracing import (
            trace_particles_boozer_with_collisions_gpu,
            trace_particles_cartesian_with_collisions_gpu,
        )
        from firm3d.field.collisions import ThermalBackground
        from firm3d.field.coordinates import boozer_to_cylindrical
        from firm3d.util.constants import (
            ELECTRON_MASS,
            ELEMENTARY_CHARGE,
            PROTON_MASS,
        )

        bri, bfield, nfp = build_wout_boozer_field()
        surf, sc_particle, bsh, rrange, phirange, zrange = build_cartesian_field()
        label = build_equilibrium_label()

        # 50/50 DT with electrons, reactor profile shapes, density boosted
        # 500x so full slowing down fits in a sub-millisecond trace.
        def ne(s):
            return 5e22 * (1.0 - 0.8 * s**2)

        def Te(s):
            return 10e3 * (1.0 - 0.8 * s) + 1e3

        bgs = [
            ThermalBackground(
                n_profile=lambda s: 0.5 * ne(s),
                T_profile=Te,
                mass=2 * PROTON_MASS,
                charge=ELEMENTARY_CHARGE,
            ),
            ThermalBackground(
                n_profile=lambda s: 0.5 * ne(s),
                T_profile=Te,
                mass=3 * PROTON_MASS,
                charge=ELEMENTARY_CHARGE,
            ),
            ThermalBackground(
                n_profile=ne,
                T_profile=Te,
                mass=ELECTRON_MASS,
                charge=-ELEMENTARY_CHARGE,
            ),
        ]

        n = 512
        xi0 = 0.9
        v0 = np.sqrt(2 * ENERGY / MASS)
        rng = np.random.default_rng(0)
        stz = np.column_stack(
            [
                rng.uniform(0.05, 0.6, n),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        )
        xyz = rphiz_to_xyz(boozer_to_cylindrical(bfield, stz.copy()))
        vpar = xi0 * v0 * np.ones(n)
        kw = {
            "backgrounds": bgs,
            "mass": MASS,
            "charge": CHARGE,
            "vtotal": v0,
            "tol": 1e-8,
            "rng_seed": 0,
        }

        def stats(tmax):
            booz = trace_particles_boozer_with_collisions_gpu(
                bfield,
                stz.copy(),
                vpar.copy(),
                tmax=tmax,
                ns=15,
                ntheta=15,
                nzeta=15,
                **kw,
            )
            cart = trace_particles_cartesian_with_collisions_gpu(
                bsh, sc_particle, label, xyz.copy(), vpar.copy(), tmax=tmax, **kw
            )
            conf_b = np.isclose(booz[:, 0], tmax, rtol=1e-12)
            conf_c = np.isclose(cart[:, 0], tmax, rtol=1e-12)
            self.assertGreater(np.mean(conf_b), 0.95, f"Boozer losses at {tmax:g}")
            self.assertGreater(np.mean(conf_c), 0.95, f"Cartesian losses at {tmax:g}")
            v_b, v_c = booz[conf_b, 5], cart[conf_c, 5]
            return {
                "E_b": np.mean(v_b**2) / v0**2,
                "E_c": np.mean(v_c**2) / v0**2,
                "xi_b": np.mean(booz[conf_b, 4] / v_b),
                "xi_c": np.mean(cart[conf_c, 4] / v_c),
                "ks": ks_2samp(v_b, v_c).statistic,
            }

        # Drag stage: a quarter of the energy is gone, the pitch is not.
        early = stats(1e-4)
        self.assertLess(early["E_b"], 0.85, "drag has not started; regime is wrong")
        self.assertLess(
            abs(early["E_b"] - early["E_c"]),
            0.02,
            f"<E>/E0 disagrees in the drag stage: {early['E_b']:.4f} vs "
            f"{early['E_c']:.4f}",
        )
        for name in ("xi_b", "xi_c"):
            self.assertGreater(
                early[name],
                xi0 - 0.03,
                f"pitch scattered during the drag stage ({name}="
                f"{early[name]:.3f}); electron drag must not scatter pitch",
            )

        # Scattering stage: near the critical energy the pitch must have
        # decayed, by the same amount in both tracers.
        late = stats(6e-4)
        self.assertLess(
            abs(late["E_b"] - late["E_c"]),
            0.02,
            f"<E>/E0 disagrees in the scattering stage: {late['E_b']:.4f} vs "
            f"{late['E_c']:.4f}",
        )
        for name in ("xi_b", "xi_c"):
            self.assertLess(
                late[name],
                xi0 - 0.03,
                f"pitch did not decay near the critical energy ({name}="
                f"{late[name]:.3f})",
            )
        self.assertLess(
            abs(late["xi_b"] - late["xi_c"]),
            0.05,
            f"isotropization rates disagree: <xi> {late['xi_b']:.3f} vs "
            f"{late['xi_c']:.3f}",
        )
        self.assertLess(
            late["ks"],
            0.10,
            f"confined speed distributions disagree (KS {late['ks']:.3f})",
        )


if __name__ == "__main__":
    print("Running GPU tracing tests...")
    unittest.main()
