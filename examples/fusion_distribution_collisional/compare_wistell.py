"""
Compare the firm3d and ASCOT5 reactor-scale Wistell-A collisional runs.

Usage:
    python compare_wistell.py --outdir /path/to/workdir

Reads results_collisional.npz (firm3d, from
fusion_distribution_collisional.py) and ascot5_wistell.npz (from
run_ascot5_wistell.py), overlays the loss-fraction curves, and prints
the particle- and energy-loss comparisons.  The two codes use different
field representations and statistically-equivalent (not
marker-identical) birth samples, so agreement is expected at the
ensemble level with sqrt(N) statistics.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def loss_curve(t_lost, n_total, tmin, tmax):
    t = np.sort(t_lost)
    frac = np.arange(1, t.size + 1) / n_total
    t = np.concatenate([[tmin], t, [tmax]])
    frac = np.concatenate([[0], frac, [frac[-1] if frac.size else 0]])
    return t, frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--e0-mev", type=float, default=3.52)
    args = parser.parse_args()

    f3d = np.load(os.path.join(args.outdir, "results_collisional.npz"))
    a5 = np.load(os.path.join(args.outdir, "ascot5_wistell.npz"))

    n_f = f3d["lost"].size
    n_a = a5["lost"].size
    tmax = max(f3d["t_end"].max(), a5["t_end"].max())

    lost_f = f3d["lost"]
    lost_a = a5["lost"]
    pf, pa = lost_f.mean(), lost_a.mean()
    sf = np.sqrt(pf * (1 - pf) / n_f)
    sa = np.sqrt(pa * (1 - pa) / n_a)

    ef = f3d["E_end_MeV"][lost_f].sum() / (n_f * args.e0_mev)
    ea = a5["e_end_MeV"][lost_a].sum() / (n_a * args.e0_mev)

    print("=" * 64)
    print("Wistell-A (ARIES-CS scale) collisional alphas: firm3d vs ASCOT5")
    print(f"N = {n_f} (firm3d) / {n_a} (ASCOT5)")
    print("=" * 64)
    print(f"{'':30s} {'firm3d':>10s} {'ASCOT5':>10s}")
    print(
        f"{'particle loss fraction':30s} "
        f"{100 * pf:>9.1f}% {100 * pa:>9.1f}%   "
        f"(+/- {100 * np.hypot(sf, sa):.1f}% combined)"
    )
    print(f"{'energy loss fraction':30s} {100 * ef:>9.1f}% {100 * ea:>9.1f}%")
    sig = abs(pf - pa) / np.hypot(sf, sa)
    print(f"{'particle-loss difference':30s} {sig:>9.1f} sigma")

    plt.figure()
    tf, ff = loss_curve(f3d["t_end"][lost_f], n_f, 1e-5, tmax)
    ta, fa = loss_curve(a5["t_end"][lost_a], n_a, 1e-5, tmax)
    plt.loglog(tf, ff, label="firm3d")
    plt.loglog(ta, fa, "--", label="ASCOT5")
    plt.xlabel("Time [s]")
    plt.ylabel("Fraction of lost alphas")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(args.outdir, "wistell_loss_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"saved {out}")

    plt.figure()
    bins = np.linspace(0, args.e0_mev, 30)
    plt.hist(f3d["E_end_MeV"][lost_f], bins=bins, alpha=0.5, label="firm3d")
    plt.hist(a5["e_end_MeV"][lost_a], bins=bins, alpha=0.5, label="ASCOT5")
    plt.xlabel("E at loss [MeV]")
    plt.ylabel("lost alphas")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(args.outdir, "wistell_lost_energy_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
