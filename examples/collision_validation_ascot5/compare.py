"""
Compare the firm3d and ASCOT5 slowing-down runs.

Usage:
    python compare.py --outdir /path/to/workdir

Reads {firm3d,ascot5}_moments.npz, prints a comparison table, and saves
validation_comparison.png.  Exits nonzero if the codes disagree beyond
the combined statistical tolerance.
"""

import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=".")
    params.add_case_arguments(parser)
    args = parser.parse_args()
    params.set_case(density=args.density, n_markers=args.nmarkers)

    f3d = np.load(os.path.join(args.outdir, "firm3d_moments.npz"))
    a5 = np.load(os.path.join(args.outdir, "ascot5_moments.npz"))
    tgrid = f3d["tgrid"]
    assert np.allclose(tgrid, a5["tgrid"])

    # Mean energy and pitch over markers alive in BOTH datasets at each
    # time (ASCOT5 stops markers at EMIN; firm3d keeps tracing them).
    both = np.isfinite(f3d["E"]) & np.isfinite(a5["E"])
    nboth = both.sum(axis=0)
    ok = nboth >= 0.5 * params.N_MARKERS

    def mean_where(A, mask):
        A = np.where(mask, A, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(A, axis=0)

    # <E(t)> is the collision-operator observable (geometry-insensitive
    # for a uniform plasma).  <xi(t)> is plotted for qualitative
    # comparison only: pitch is sampled along orbits, and the two codes'
    # fields have different mirror ratios and trapped fractions, so a
    # persistent O(0.1) offset between the curves is geometric, not
    # collisional.
    E_f = mean_where(f3d["E"], both)
    E_a = mean_where(a5["E"], both)
    XI_f = mean_where(f3d["XI"], both)
    XI_a = mean_where(a5["XI"], both)

    # Slowing-down times (markers that reached EMIN in both codes)
    ts_f = f3d["t_slow"][f3d["slowed"]]
    ts_a = a5["t_end"][a5["slowed"]]
    t_ana = params.analytic_slowing_time()

    print("=" * 64)
    print("firm3d <-> ASCOT5 collision validation: alpha slowing-down")
    print(
        f"n = {params.DENSITY:.1e} m^-3, T = {params.TEMPERATURE_EV / 1e3:.1f} keV, "
        f"N = {params.N_MARKERS}, lnL(alpha-e) = "
        f"{params.coulomb_log_alpha_electron():.2f}"
    )
    print("=" * 64)
    print(f"{'':28s} {'firm3d':>10s} {'ASCOT5':>10s} {'analytic':>10s}")
    print(
        f"{'slowed to Emin':28s} {f3d['slowed'].sum():>10d} "
        f"{a5['slowed'].sum():>10d} {'--':>10s}"
    )
    m_f, m_a = ts_f.mean(), ts_a.mean()
    s_f = ts_f.std() / np.sqrt(ts_f.size)
    s_a = ts_a.std() / np.sqrt(ts_a.size)
    print(
        f"{'mean slowing time [ms]':28s} {m_f * 1e3:>10.3f} "
        f"{m_a * 1e3:>10.3f} {t_ana * 1e3:>10.3f}"
    )
    print(f"{'  standard error [ms]':28s} {s_f * 1e3:>10.3f} {s_a * 1e3:>10.3f}")
    rel = m_f / m_a - 1
    sig = abs(m_f - m_a) / np.hypot(s_f, s_a)
    print(f"{'firm3d/ASCOT5 - 1':28s} {rel * 100:>9.1f}%  ({sig:.1f} sigma)")

    # <E(t)> agreement over the slowing window
    resid = (E_f[ok] - E_a[ok]) / params.E0_EV
    print(f"{'max |<E>_f3d - <E>_a5|/E0':28s} {np.nanmax(np.abs(resid)) * 100:>9.1f}%")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(tgrid * 1e3, E_f / 1e6, label="firm3d")
    axes[0].plot(tgrid * 1e3, E_a / 1e6, "--", label="ASCOT5")
    # analytic v^3 decay: E(t) = (E0^1.5 + Ec^1.5) exp(-3t/ts) - Ec^1.5 ...
    ts = params.spitzer_ts()
    v0 = np.sqrt(2 * params.E0_EV * params.EV / params.M_ALPHA)
    # electron drag only (Ec << E over most of the window): v = v0 e^{-t/ts}
    axes[0].plot(
        tgrid * 1e3,
        0.5 * params.M_ALPHA * (v0 * np.exp(-tgrid / ts)) ** 2 / params.EV / 1e6,
        ":",
        color="k",
        label="analytic (e-drag)",
    )
    axes[0].set_xlabel("t [ms]")
    axes[0].set_ylabel(r"$\langle E \rangle$ [MeV]")
    axes[0].legend()

    axes[1].plot(tgrid * 1e3, XI_f, label="firm3d")
    axes[1].plot(tgrid * 1e3, XI_a, "--", label="ASCOT5")
    axes[1].set_xlabel("t [ms]")
    axes[1].set_ylabel(r"$\langle \xi \rangle$")
    axes[1].legend()

    bins = np.linspace(0, params.TMAX * 1e3, 20)
    axes[2].hist(ts_f * 1e3, bins=bins, alpha=0.5, label="firm3d")
    axes[2].hist(ts_a * 1e3, bins=bins, alpha=0.5, label="ASCOT5")
    axes[2].axvline(t_ana * 1e3, color="k", ls=":", label="analytic")
    axes[2].set_xlabel("slowing time to 50 keV [ms]")
    axes[2].set_ylabel("markers")
    axes[2].legend()

    fig.tight_layout()
    out = os.path.join(args.outdir, "validation_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    # Pass/fail: slowing times agree within 3 sigma and 10 %
    passed = sig < 3.0 and abs(rel) < 0.10
    print("PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
