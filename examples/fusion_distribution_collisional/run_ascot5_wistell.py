"""
ASCOT5 companion for the reactor-scale collisional alpha run: the same
Wistell-A (ARIES-CS scale) equilibrium, plasma, and birth distribution,
traced with ASCOT5's GC-adaptive scheme in a B_STS spline field built
directly from the VMEC wout file via a5py's vmec_field importer.

Differences from the firm3d run that are inherent to the comparison:
  * field representation: real-space (R,phi,z) splines of the VMEC
    field vs firm3d's Boozer-coordinate splines of the booz_xform field;
  * birth positions: radial distribution ~ reactivity x dV/ds with
    uniform VMEC surface angles, statistically equivalent (not
    marker-identical) to firm3d's Jacobian-weighted Boozer sampling;
  * losses: ASCOT5 rho_max = 1 end condition vs firm3d s = 1.

Required ASCOT5 source patches (this file cannot fix these -- ascot5.h
and B_field.c live in the separate, non-vendored ascot5 repo):

  1. WIENERSLOTS 20 -> 200 in src/ascot5.h.  The collision operator's
     fixed-size Wiener process array (one slot per rejected/subdivided
     step within a macro step) overflows for a meaningful fraction of
     markers at the default size, aborting them with ERR_WIENER_ARRAY.

  2. B_field_eval_rho in src/B_field.c: when (psi - psi0)/delta < 0 it
     currently raises ERR_INPUT_UNPHYSICAL unconditionally.  Near the
     magnetic axis psi is a smooth extremum equal to psi0, so floating-
     point/interpolation error in the upstream psi(R,z) evaluation can
     push this fractionally negative for a location that is genuinely
     at or very near rho = 0 -- NOT a real physical/domain problem.
     B_STS_eval_rho_drho (src/Bfield/B_STS.c) already has an opt-in
     clamp for exactly this case (B_STS_CLAMP_RHO_NONNEGATIVE), but
     B_field.c's generic wrapper -- which is what actually fires here,
     for EVERY rho evaluation including plasma-profile lookups, not
     just B_STS's own -- has no such escape hatch.  Diagnosed from
     production data: 229/1024 markers aborted with this error in an
     unpatched run, and the abort rate fell monotonically from 98% for
     markers born at rho < 0.1 down to 0% for rho > 0.7 -- unambiguously
     a near-axis effect, not (as first suspected) an artifact of the
     (R,z) grid padding below.  Patch, mirroring B_STS's own fix:
         real delta = (psi1 - psi0);
         if( (psi - psi0) / delta < 0 ) {
             rho[0] = 0.0;
             rho[1] = 0.0;
         } else {
             rho[0] = sqrt( (psi - psi0) / delta );
             rho[1] = 1.0 / (2*delta*rho[0]);
         }
     (rho[1] = drho/dpsi also has to go to zero, not just rho[0]: it
     diverges as rho -> 0 by construction, so it cannot be assigned any
     finite nonzero value exactly at the axis either.)

Usage (in the environment with a5py):
    python run_ascot5_wistell.py --outdir /path/to/workdir [--smoke]
    (--nphi/--nr/--nz override the B_STS import grid; see the comment
    at their point of use for the cost/accuracy tradeoff.)
Then compare with compare_wistell.py.
"""

import argparse
import os
import subprocess

import numpy as np
from a5py import Ascot
from a5py.ascot5io.marker import Marker
from a5py.ascot5io.options import Opt
from a5py.templates.importdata import ImportData
from netCDF4 import Dataset

# Case parameters: keep in sync with fusion_distribution_collisional.py
N0 = 1e20  # core D (and T) density [m^-3]
T0_KEV = 11.5
T_EDGE_KEV = 0.1
E0_EV = 3.52e6
TMAX = 1.5e-1
SEED = 20260706


def nD(s):
    return 1 - np.minimum(s, 1.0) ** 5


def T_keV(s):
    return T0_KEV * (1 - np.minimum(s, 1.0)) + T_EDGE_KEV


def sigmav(T):
    return np.where(T > 0, T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3)), 0.0)


def sample_birth(wout, n, rng):
    """Sample (R, phi, z) from reactivity(s) x dV/ds with uniform VMEC
    surface angles, using the wout Fourier geometry."""
    nc = Dataset(wout)
    ns = int(nc.variables["ns"][:])
    xm = np.asarray(nc.variables["xm"][:])
    xn = np.asarray(nc.variables["xn"][:])
    rmnc = np.asarray(nc.variables["rmnc"][:])  # (ns, mn), full mesh
    zmns = np.asarray(nc.variables["zmns"][:])
    vp = np.asarray(nc.variables["vp"][:])  # dV/ds (half mesh, ~4pi^2)
    nc.close()

    s_full = np.linspace(0, 1, ns)
    s_half = s_full[:-1] + 0.5 / (ns - 1)
    dVds = np.interp(s_full, s_half, vp[1:])

    # radial pdf ~ n_D n_T <sigma v> dV/ds; inverse-CDF sampling
    pdf = nD(s_full) ** 2 * sigmav(T_keV(s_full) - T_EDGE_KEV) * dVds
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    s_k = np.interp(rng.random(n), cdf, s_full)

    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)

    # interpolate Fourier coefficients to the sampled surfaces
    R = np.empty(n)
    Z = np.empty(n)
    for k in range(n):
        rc = np.array([np.interp(s_k[k], s_full, rmnc[:, j]) for j in range(len(xm))])
        zs = np.array([np.interp(s_k[k], s_full, zmns[:, j]) for j in range(len(xm))])
        ang = xm * u[k] - xn * v[k]
        R[k] = np.sum(rc * np.cos(ang))
        Z[k] = np.sum(zs * np.sin(ang))
    return R, v, Z, s_k


def import_padded_vmec_bfield(wout, nphi, nr, nz, pad_frac=0.4, psifill_factor=1.2):
    """
    Build a B_STS input dict from the VMEC equilibrium, then pad the
    (R, z) grid outward by pad_frac on each side.

    a5py's vmec_field/vmec_sts importers set the (R, z) rectangular
    spline grid to EXACTLY the bounding box of the last closed flux
    surface -- there is no buffer region outside the plasma at all.
    A marker that scatters radially outward (pitch-angle collisions
    push guiding centers off their unperturbed orbits) can reach the
    edge of this box, and hence leave the spline's valid domain,
    before the rho > rho_max end condition is registered on the
    intervening adaptive step.  When that happens B_STS's spline
    evaluation fails with ERR_INPUT_UNPHYSICAL and the marker aborts
    with no clean end condition, silently biasing the loss statistics.

    The padding is filled by repeating the outermost (R, z) ring of
    a5py's own returned arrays outward (np.pad mode="edge"), rather than
    inventing a new constant, purely to avoid adding any discontinuity
    beyond what a5py's own fill (a constant psi = psi1 * psifill_factor
    outside the LCFS, giving rho > rho_max there, plus a nearest-
    neighbor extrapolation of B) already has.  Note this padding is NOT
    what was behind the 229/1024 (22%) abort rate seen in early testing
    -- that turned out to be an unrelated near-axis floating-point edge
    case in ASCOT5 itself (see the module docstring's B_field.c patch
    for the diagnosis and fix); it was mistakenly suspected to be a
    padding-seam artifact at first, before the per-marker abort-vs-rho
    data made the real, near-axis cause unambiguous.
    """
    raw = ImportData.vmec_field(
        wout,
        phimin=0,
        phimax=360,
        nphi=nphi,
        ntheta=120,
        nr=nr,
        nz=nz,
        extrapolate=True,
        psifill_factor=psifill_factor,
    )
    br, bphi, bz, psi = raw["br"], raw["bphi"], raw["bz"], raw["psi"]  # (nr,nphi,nz)

    rmin, rmax, nr0 = raw["b_rmin"], raw["b_rmax"], raw["b_nr"]
    zmin, zmax, nz0 = raw["b_zmin"], raw["b_zmax"], raw["b_nz"]
    dr = (rmax - rmin) / (nr0 - 1)
    dz = (zmax - zmin) / (nz0 - 1)
    pad_r = max(2, int(pad_frac * (rmax - rmin) / dr))
    pad_z = max(2, int(pad_frac * (zmax - zmin) / dz))

    new_rmin, new_rmax = rmin - pad_r * dr, rmax + pad_r * dr
    new_zmin, new_zmax = zmin - pad_z * dz, zmax + pad_z * dz

    pad_width = ((pad_r, pad_r), (0, 0), (pad_z, pad_z))
    new_br = np.pad(br, pad_width, mode="edge")
    new_bphi = np.pad(bphi, pad_width, mode="edge")
    new_bz = np.pad(bz, pad_width, mode="edge")
    new_psi = np.pad(psi, pad_width, mode="edge")
    new_nr, new_nz = new_psi.shape[0], new_psi.shape[2]

    raw.update(
        {
            "br": new_br,
            "bphi": new_bphi,
            "bz": new_bz,
            "psi": new_psi,
            "b_rmin": new_rmin,
            "b_rmax": new_rmax,
            "b_nr": new_nr,
            "b_zmin": new_zmin,
            "b_zmax": new_zmax,
            "b_nz": new_nz,
            "psi_rmin": new_rmin,
            "psi_rmax": new_rmax,
            "psi_nr": new_nr,
            "psi_zmin": new_zmin,
            "psi_zmax": new_zmax,
            "psi_nz": new_nz,
        }
    )
    raw.pop("rlcfs", None)
    raw.pop("zlcfs", None)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wout", default="../inputs/wout_aten_rescaled.nc")
    parser.add_argument(
        "--ascot5-main",
        default="/Users/elizabethpaul/Documents/Research/ascot5/build/ascot5_main",
    )
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--nmarkers", type=int, default=1024)
    parser.add_argument("--tmax", type=float, default=TMAX)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--nphi", type=int, default=None)
    parser.add_argument("--nr", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument(
        "--collisionless",
        action="store_true",
        help="disable Coulomb collisions (orbit-following-only companion, "
        "matching fusion_distribution_collisional.py's res_free run)",
    )
    args = parser.parse_args()

    n = 8 if args.smoke else args.nmarkers
    tmax = 2e-3 if args.smoke else args.tmax
    # nphi, nr, nz for the B_STS import.  Cost is dominated by nphi, NOT
    # by nr/nz: a5py's vmec_field runs 4 scipy.interpolate.griddata calls
    # (Delaunay triangulation + barycentric interpolation) PER TOROIDAL
    # SLICE to map the VMEC (theta, phi) flux-surface mesh onto the
    # rectangular (R, Z) grid, so import time is ~linear in nphi
    # (measured ~50 min at nphi = 120 on this machine) and roughly
    # independent of nr, nz.  Coarsening nphi is tempting for cost, but
    # nphi = 24 was tried and made things much WORSE physically: an
    # under-resolved toroidal ripple forces the adaptive orbit stepper
    # to take many tiny steps, overflowing ASCOT5's fixed-size Wiener
    # process array (ERR_WIENER_ARRAY) for most markers.  Keep nphi high.
    res = (60, 64, 64) if args.smoke else (180, 120, 120)
    res = (args.nphi or res[0], args.nr or res[1], args.nz or res[2])

    suffix = "_collisionless" if args.collisionless else ""
    fn = os.path.join(args.outdir, f"wistell_ascot5{suffix}.h5")
    if os.path.exists(fn):
        os.remove(fn)
    a5 = Ascot(fn, create=True)
    init = a5.data.create_input

    for tpl in [
        ("opt",),
        ("gc",),
        ("B_TC",),
        ("E_TC",),
        ("wall_2D",),
        ("plasma_1D",),
        ("N0_1D",),
        ("Boozer",),
        ("MHD_STAT",),
        ("asigma_loc",),
    ]:
        init(*tpl, desc="DUMMY")

    # Field: B_STS splines from the VMEC equilibrium, padded outside the
    # LCFS bounding box (see import_padded_vmec_bfield's docstring).
    print(f"Importing VMEC field from {args.wout} ...")
    padded = import_padded_vmec_bfield(args.wout, nphi=res[0], nr=res[1], nz=res[2])
    init("B_STS", **padded, desc="WISTELL")

    # Plasma: D + T + electrons, same profiles as the firm3d run.
    # ASCOT5 rho = sqrt(normalized toroidal flux), so s = rho^2.
    nrho = 202
    rho = np.concatenate([np.linspace(0, 1, 200), [1.001, 10.0]])
    s = np.minimum(rho, 1.0) ** 2
    prof = nD(s)
    prof[rho > 1] = 0.0
    edens = np.maximum(2 * N0 * prof, 1.0)  # electron density, floor 1 m^-3
    idens = np.column_stack([np.maximum(N0 * prof, 0.5), np.maximum(N0 * prof, 0.5)])
    temp = T_keV(s) * 1e3  # eV, finite (T_EDGE) outside the LCFS
    init(
        "plasma_1D",
        **{
            "nrho": nrho,
            "nion": 2,
            "anum": np.array([2, 3]),
            "znum": np.array([1, 1]),
            "mass": np.array([2.0141, 3.0160]),
            "charge": np.array([1, 1]),
            "rho": rho,
            "vtor": np.zeros((nrho, 1)),
            "edensity": edens.reshape(-1, 1),
            "etemperature": temp.reshape(-1, 1),
            "idensity": idens,
            "itemperature": temp.reshape(-1, 1),
        },
        desc="WISTELL",
    )

    # Markers
    rng = np.random.default_rng(SEED)
    print("Sampling birth distribution ...")
    R, phi, Z, s_k = sample_birth(args.wout, n, rng)
    mrk = Marker.generate("gc", n=n, species="alpha")
    mrk["r"][:] = R
    mrk["phi"][:] = np.degrees(phi)
    mrk["z"][:] = Z
    mrk["zeta"][:] = 2 * np.pi * rng.random(n)
    mrk["energy"][:] = E0_EV
    mrk["pitch"][:] = 1.0 - 2.0 * rng.random(n)
    init("gc", **mrk, desc="WISTELL")

    # Options: GC adaptive, collisions, rho_max = 1 loss boundary
    opt = Opt.get_default()
    opt.update(
        {
            "SIM_MODE": 2,
            "ENABLE_ADAPTIVE": 1,
            "ADAPTIVE_TOL_ORBIT": 1e-6,
            "ADAPTIVE_TOL_CCOL": 1e-2,
            "ADAPTIVE_MAX_DRHO": 0.1,
            "ADAPTIVE_MAX_DPHI": 10,
            "FIXEDSTEP_USERDEFINED": 1e-8,
            "ENDCOND_SIMTIMELIM": 1,
            "ENDCOND_LIM_SIMTIME": tmax,
            "ENDCOND_RHOLIM": 1,
            "ENDCOND_MAX_RHO": 1.02,
            "ENABLE_ORBIT_FOLLOWING": 1,
            "ENABLE_COULOMB_COLLISIONS": 0 if args.collisionless else 1,
        }
    )
    init("opt", **opt, desc="WISTELL")

    data = a5.data
    for grp in [
        "bfield",
        "efield",
        "wall",
        "plasma",
        "neutral",
        "boozer",
        "mhd",
        "asigma",
        "options",
        "marker",
    ]:
        node = getattr(data, grp)
        tag = "WISTELL" if hasattr(node, "WISTELL") else "DUMMY"
        node[tag].activate()

    print(f"Running ascot5_main: {n} alphas, tmax = {tmax * 1e3:.1f} ms ...")
    subprocess.run(
        [args.ascot5_main, f"--in={fn[:-3]}", "--d=WISTELL"],
        check=True,
    )

    from a5py.ascot5io.state import State

    a5 = Ascot(fn)
    run = a5.data.active
    t_end = np.asarray(run.getstate("time", state="end"))
    e_end = np.asarray(run.getstate("ekin", state="end").to("eV"))
    endcond = np.asarray(run.getstate("endcond", state="end"), dtype=int)
    errormsg = np.asarray(run.getstate("errormsg", state="end"), dtype=int)
    errormod = np.asarray(run.getstate("errormod", state="end"), dtype=int)

    # Use errormsg (nonzero a5err "type" code) directly to identify
    # aborted markers.  a5py's own endcond decoding
    # (State.read -> "endcond": item[err>0] = item[err>0] & _ABORTED)
    # ANDs the raw endcond bits with the _ABORTED bit rather than
    # OR-ing it in, so an aborted marker with e.g. a RHOMAX bit already
    # set is zeroed out and relabeled _NONE instead of _ABORTED -- do
    # not rely on the _ABORTED bit of endcond to find these markers.
    aborted = errormsg > 0
    lost = ((endcond & (State._RHOMAX | State._WALL)) > 0) & ~aborted
    unended = (endcond == State._NONE) & ~aborted
    if unended.any():
        print(f"WARNING: {unended.sum()} markers ended with NO end condition")
    if aborted.any():
        mods, cnts = np.unique(errormod[aborted], return_counts=True)
        msgs, mcnts = np.unique(errormsg[aborted], return_counts=True)
        print(f"aborted (errormsg > 0): {aborted.sum()}/{n}")
        print(f"  by module: {dict(zip(mods.tolist(), cnts.tolist()))}")
        print(f"  by errtype: {dict(zip(msgs.tolist(), mcnts.tolist()))}")

    out = os.path.join(args.outdir, f"ascot5_wistell{suffix}.npz")
    np.savez(
        out,
        t_end=t_end,
        e_end_MeV=e_end / 1e6,
        lost=lost,
        aborted=aborted,
        unended=unended,
        endcond=endcond,
        s_birth=s_k,
    )
    print(f"saved {out}")
    print(
        f"lost (rho > 1, excl. aborted): {lost.sum()}/{n} ({100 * lost.mean():.1f} %)"
    )
    print(
        f"energy loss fraction: "
        f"{100 * e_end[lost].sum() / (n * E0_EV):.1f} % of total birth energy"
    )


if __name__ == "__main__":
    main()
