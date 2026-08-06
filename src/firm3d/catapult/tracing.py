__all__ = [
    "trace_particles_boozer_gpu",
    "trace_particles_boozer_with_collisions_gpu",
    "trace_particles_cartesian_gpu",
]
import numpy as np

import firm3dpp
from firm3d.catapult.utils import (
    boozer_interpolant,
    boozer_saw_interpolant,
    cartesian_interpolant,
)
from firm3d.field.boozermagneticfield import ShearAlfvenWavesSuperposition


def trace_particles_boozer_gpu(
    field,
    stz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    ns,
    ntheta,
    nzeta,
    dt=None,
):
    """
    Trace particles in Boozer coordinates using CATAPULT
    field: a magnetic field object representing the field in Boozer coordinates
    stz_inits: initial conditions for particles in (s, theta, zeta) coordinates
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles
    mass: mass of each particle
    charge: charge of each particle
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional)
    """
    nparticles = stz_inits.shape[0]

    if isinstance(field, ShearAlfvenWavesSuperposition):
        B0 = field.B0
        srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
            B0, B0.nfp, ns, ntheta, nzeta
        )
        saw_nharmonics = len(field)
        saw_omega = field.get_wave(0).omega
        saw_s = field.get_wave(0).phihat.get_s_basis()
        saw_srange = (saw_s[0], saw_s[-1], len(saw_s))
        saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
        saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
        saw_phihats = np.ascontiguousarray(
            np.column_stack(
                [
                    np.array([field.get_wave(i).phihat(s_val) for s_val in saw_s])
                    for i in range(saw_nharmonics)
                ]
            )
        )

        if B0.field_type == "vac":
            last_time = firm3dpp.boozer_saw_gpu_tracing(
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
                stz_init=stz_inits,
                m=mass,
                q=charge,
                vtotal=vtotal,
                vtang=parallel_speeds,
                tmax=tmax,
                tol=tol,
                dt_in=dt if dt is not None else -np.ones(nparticles),
                psi0=B0.psi0,
                nparticles=nparticles,
            )
        elif B0.field_type == "nok":
            last_time = firm3dpp.boozer_saw_nok_gpu_tracing(
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
                stz_init=stz_inits,
                m=mass,
                q=charge,
                vtotal=vtotal,
                vtang=parallel_speeds,
                tmax=tmax,
                tol=tol,
                dt_in=dt if dt is not None else -np.ones(nparticles),
                psi0=B0.psi0,
                nparticles=nparticles,
            )
        else:
            raise ValueError(f"Unsupported field type {B0.field_type} for SAW tracing")
    else:
        if field.field_type not in ["vac", ""]:
            raise ValueError(
                f"Unsupported field type {field.field_type} for Boozer tracing, \
                     expected 'vac' or ''"
            )
        vacuum = field.field_type == "vac"  # true if vacuum, false if finite beta
        srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
            field, field.nfp, ns, ntheta, nzeta, vacuum=vacuum
        )
        psi0 = field.psi0
        last_time = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits,
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds,
            tmax=tmax,
            tol=tol,
            dt_in=-np.ones(nparticles),
            psi0=psi0,
            nparticles=nparticles,
            vacuum=vacuum,
        )

    last_time = np.reshape(last_time, (nparticles, 6))
    return last_time


def trace_particles_cartesian_gpu(
    field,
    surface_classifier,
    xyz_inits,
    parallel_speeds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    dt=None,
):
    """
    Trace particles in Cartesian coordinates using CATAPULT
    field: a magnetic field object representing the field in Cartesian coordinates
    surface_classifier: a simsopt surface classifier object for detecting a surface
    xyz_inits: initial conditions for particles in (x, y, z) coordinates
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles
    mass: mass of each particle
    charge: charge of each particle
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional)
    """
    nparticles = xyz_inits.shape[0]
    r_range, phi_range, z_range, quad_info = cartesian_interpolant(
        field, surface_classifier
    )
    last_time = firm3dpp.cartesian_gpu_tracing(
        quad_pts=quad_info,
        rrange=r_range,
        phirange=phi_range,
        zrange=z_range,
        xyz_init=xyz_inits,
        m=mass,
        q=charge,
        vtotal=vtotal,
        vtang=parallel_speeds,
        tmax=tmax,
        tol=tol,
        dt_in=dt if dt is not None else -np.ones(nparticles),
        nparticles=nparticles,
    )
    last_time = np.reshape(last_time, (nparticles, 6))
    return last_time


def trace_particles_boozer_with_collisions_gpu(
    field,
    stz_inits,
    parallel_speeds,
    backgrounds,
    tmax,
    mass,
    charge,
    vtotal,
    tol,
    ns,
    ntheta,
    nzeta,
    dt=None,
    rng_seed=0,
    validate_profiles=True,
):
    """
    Trace particles in Boozer coordinates on the GPU, including Monte Carlo
    Coulomb collisions.

    The collision operator is the one
    :func:`~firm3d.field.collisions.trace_particles_boozer_with_collisions`
    uses, compiled for device from the same source, so the coefficients are
    not a separate implementation.  It is applied as a kick after each
    accepted orbit step, sub-cycled when the collision rates are fast relative
    to that step.

    Differences from the CPU entry point, which are properties of the GPU
    tracer rather than of collisions:

    * a single ``vtotal`` sets the velocity normalisation for every particle,
      as in :func:`trace_particles_boozer_gpu`, rather than per-particle
      ``(v_par, mu)``;
    * only the final state is returned, not the trajectory;
    * stopping criteria are not available.

    Results are not comparable element-by-element with the CPU tracer even at
    a fixed seed: the two draw from different generators (Philox against
    mt19937_64) and take different step sequences, so agreement is
    statistical.

    Args:
        field: A :class:`~firm3d.field.boozermagneticfield.BoozerMagneticField`
            with ``field_type`` ``"vac"`` or ``""``.
        stz_inits: ``(nparticles, 3)`` array of initial ``(s, theta, zeta)``.
        parallel_speeds: ``(nparticles,)`` array of initial ``v_par`` in m/s.
        backgrounds: A :class:`~firm3d.field.collisions.ThermalBackground` or a
            list of them; coefficients are summed over species.
        tmax: Integration time in seconds.
        mass: EP mass in kg.
        charge: EP charge in C.
        vtotal: Speed setting the velocity normalisation, in m/s.
        tol: Tolerance for the adaptive step control.
        ns, ntheta, nzeta: Interpolant resolution.
        dt: Initial step size; ``None`` lets the tracer choose.
        rng_seed: Base seed.  Each particle uses a Philox stream keyed on its
            global index, so results do not depend on the block layout.
        validate_profiles: Check up front that the profiles give a usable
            Coulomb logarithm.  The device cannot raise, so this is the only
            place an unphysical profile is reported.

    Returns:
        ``(nparticles, 6)`` array of final states,
        ``[t, s, theta, zeta, v_par, dt]``, as for
        :func:`trace_particles_boozer_gpu`.
    """
    from firm3d.field.collisions import ThermalBackground, _validate_coulomb_log

    nparticles = stz_inits.shape[0]
    if field.field_type not in ["vac", ""]:
        raise ValueError(
            f"Unsupported field type {field.field_type} for Boozer tracing, "
            f"expected 'vac' or ''"
        )
    vacuum = field.field_type == "vac"

    if isinstance(backgrounds, ThermalBackground):
        backgrounds = [backgrounds]
    cpp_backgrounds = [b._to_cpp() for b in backgrounds]
    if validate_profiles:
        _validate_coulomb_log(cpp_backgrounds, float(mass), float(charge))

    srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
        field, field.nfp, ns, ntheta, nzeta, vacuum=vacuum
    )
    last_time = firm3dpp.boozer_collision_gpu_tracing(
        quad_pts=quad_info,
        srange=srange,
        trange=trange,
        zrange=zrange,
        stz_init=stz_inits,
        m=mass,
        q=charge,
        vtotal=vtotal,
        vtang=parallel_speeds,
        tmax=tmax,
        tol=tol,
        dt_in=-np.ones(nparticles) if dt is None else dt * np.ones(nparticles),
        psi0=field.psi0,
        nparticles=nparticles,
        backgrounds=cpp_backgrounds,
        vacuum=vacuum,
        rng_seed=int(rng_seed),
    )
    return np.reshape(last_time, (nparticles, 6))
