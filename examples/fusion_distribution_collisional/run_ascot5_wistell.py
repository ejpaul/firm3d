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

Usage (in the environment with a5py):
    python run_ascot5_wistell.py --outdir /path/to/workdir [--smoke]
Then compare with compare_wistell.py.
"""

import argparse
import os
import subprocess

import numpy as np
from a5py import Ascot
from a5py.ascot5io.marker import Marker
from a5py.ascot5io.options import Opt
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
    args = parser.parse_args()

    n = 8 if args.smoke else args.nmarkers
    tmax = 2e-3 if args.smoke else args.tmax
    # nphi, nr, nz for the B_STS import.  The a5py importer does a pure-
    # Python inverse-VMEC mapping per grid point, so cost scales directly
    # with grid size: (180, 120, 120) imports in tens of minutes and gives
    # 45 toroidal points per field period (nfp = 4), comparable to the
    # firm3d Boozer grid (48 per period).
    res = (120, 64, 64) if args.smoke else (180, 120, 120)

    fn = os.path.join(args.outdir, "wistell_ascot5.h5")
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

    # Field: B_STS splines from the VMEC equilibrium
    print(f"Importing VMEC field from {args.wout} ...")
    init(
        "vmec_sts",
        ncfile=args.wout,
        nphi=res[0],
        ntheta=120,
        nr=res[1],
        nz=res[2],
        extrapolate=True,
        desc="WISTELL",
    )

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
            "ENDCOND_MAX_RHO": 1.0,
            "ENABLE_ORBIT_FOLLOWING": 1,
            "ENABLE_COULOMB_COLLISIONS": 1,
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

    a5 = Ascot(fn)
    run = a5.data.active
    t_end = np.asarray(run.getstate("time", state="end"))
    e_end = np.asarray(run.getstate("ekin", state="end").to("eV"))
    endcond = np.asarray(run.getstate("endcond", state="end"), dtype=str)
    lost = np.array(["rho" in ec.lower() or "wall" in ec.lower() for ec in endcond])
    aborted = np.array(["abort" in ec.lower() for ec in endcond])

    out = os.path.join(args.outdir, "ascot5_wistell.npz")
    np.savez(
        out,
        t_end=t_end,
        e_end_MeV=e_end / 1e6,
        lost=lost,
        aborted=aborted,
        endcond=endcond,
        s_birth=s_k,
    )
    print(f"saved {out}")
    print(f"aborted: {aborted.sum()}/{n}")
    print(f"lost (rho > 1): {lost.sum()}/{n} ({100 * lost.mean():.1f} %)")
    print(
        f"energy loss fraction: "
        f"{100 * e_end[lost].sum() / (n * E0_EV):.1f} % of total birth energy"
    )


if __name__ == "__main__":
    main()
