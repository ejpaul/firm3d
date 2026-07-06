"""
ASCOT5 side of the collision validation: generate input, run
ascot5_main (GC adaptive mode), extract per-marker E(t) and xi(t).

Usage:
    python run_ascot5.py --ascot5-main /path/to/build/ascot5_main \
                         --outdir /path/to/workdir

Requires a5py (pip install -e . in the ascot5 repo) and the compiled
ascot5_main binary.  Writes ascot5_moments.npz into outdir.
"""

import argparse
import os
import subprocess

import numpy as np
import params
from a5py import Ascot
from a5py.ascot5io.marker import Marker
from a5py.ascot5io.options import Opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ascot5-main",
        default="/Users/elizabethpaul/Documents/Research/ascot5/build/ascot5_main",
    )
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    fn = os.path.join(args.outdir, "validation_ascot5.h5")
    if os.path.exists(fn):
        os.remove(fn)

    a5 = Ascot(fn, create=True)
    init = a5.data.create_input

    # Dummy inputs for the modules we do not use
    init("opt", desc="DUMMY")
    init("gc", desc="DUMMY")
    init("B_TC", desc="DUMMY")
    init("E_TC", desc="DUMMY")
    init("wall_2D", desc="DUMMY")
    init("plasma_1D", desc="DUMMY")
    init("N0_1D", desc="DUMMY")
    init("Boozer", desc="DUMMY")
    init("MHD_STAT", desc="DUMMY")
    init("asigma_loc", desc="DUMMY")

    # Field: ITER-like circular tokamak; plasma: uniform hydrogen
    init("bfield_analytical_iter_circular", desc="VALIDATION")
    init(
        "plasma_flat",
        density=params.DENSITY,
        temperature=params.TEMPERATURE_EV,
        desc="VALIDATION",
    )

    # Options: GC adaptive with collisions, orbit output on a fixed
    # interval, energy end condition at EMIN (as in the ccoll test)
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
            "ENDCOND_LIM_SIMTIME": params.TMAX,
            "ENDCOND_ENERGYLIM": 1,
            "ENDCOND_MIN_ENERGY": params.EMIN_EV,
            "ENDCOND_MIN_THERMAL": 0,
            "ENABLE_ORBIT_FOLLOWING": 1,
            "ENABLE_COULOMB_COLLISIONS": 1,
            "ENABLE_ORBITWRITE": 1,
            "ORBITWRITE_MODE": 1,
            "ORBITWRITE_INTERVAL": params.DT_SAVE,
            "ORBITWRITE_NPOINT": params.N_SAVE + 10,
        }
    )
    init("opt", **opt, desc="VALIDATION")

    # Markers: alphas on a mid-radius ring, shared pitch samples
    rng = np.random.default_rng(params.SEED + 1)
    n = params.N_MARKERS
    mrk = Marker.generate("gc", n=n, species="alpha")
    pol = 2 * np.pi * rng.random(n)
    mrk["r"][:] = 6.2 + 0.8 * np.cos(pol)
    mrk["phi"][:] = 90
    mrk["z"][:] = 0.8 * np.sin(pol)
    mrk["zeta"][:] = 2 * np.pi * rng.random(n)
    mrk["energy"][:] = params.E0_EV
    mrk["pitch"][:] = params.initial_pitches()
    init("gc", **mrk, desc="VALIDATION")

    # Activate the validation inputs (dummies stay active elsewhere)
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
        tag = "VALIDATION" if hasattr(node, "VALIDATION") else "DUMMY"
        node[tag].activate()

    print(f"Running ascot5_main (tmax = {params.TMAX * 1e3:.2f} ms) ...")
    subprocess.run(
        [args.ascot5_main, f"--in={fn[:-3]}", "--d=VALIDATION"],
        check=True,
    )

    # Post-process: per-marker E(t), xi(t) on the common grid
    a5 = Ascot(fn)
    run = a5.data.active
    ids, time, ekin, pitch = run.getorbit("ids", "time", "ekin", "pitch")
    ids = np.asarray(ids)
    time = np.asarray(time)
    ekin = np.asarray(ekin.to("eV"))
    pitch = np.asarray(pitch)

    tgrid = params.DT_SAVE * np.arange(1, params.N_SAVE + 1)
    E = np.full((n, tgrid.size), np.nan)
    XI = np.full((n, tgrid.size), np.nan)
    uid = np.unique(ids)
    for k, i in enumerate(uid):
        sel = ids == i
        t_i, e_i, x_i = time[sel], ekin[sel], pitch[sel]
        order = np.argsort(t_i)
        t_i, e_i, x_i = t_i[order], e_i[order], x_i[order]
        alive = tgrid <= t_i[-1] + params.DT_SAVE
        E[k, alive] = np.interp(tgrid[alive], t_i, e_i)
        XI[k, alive] = np.interp(tgrid[alive], t_i, x_i)

    # Slowing-down time from the end state (ENERGYLIM end condition)
    t_end = np.asarray(run.getstate("time", state="end"))
    e_end = np.asarray(run.getstate("ekin", state="end").to("eV"))
    endcond = run.getstate("endcond", state="end")
    slowed = e_end < 1.5 * params.EMIN_EV

    out = os.path.join(args.outdir, "ascot5_moments.npz")
    np.savez(
        out,
        tgrid=tgrid,
        E=E,
        XI=XI,
        t_end=t_end,
        e_end=e_end,
        slowed=slowed,
        endcond=np.asarray(endcond, dtype=str),
    )
    print(f"saved {out}")
    print(f"slowed to Emin: {slowed.sum()}/{n}")
    if slowed.sum():
        print(f"mean slowing time: {t_end[slowed].mean() * 1e3:.3f} ms")
    print(f"analytic 0.5 ts ln(E0/Emin): {params.analytic_slowing_time() * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
