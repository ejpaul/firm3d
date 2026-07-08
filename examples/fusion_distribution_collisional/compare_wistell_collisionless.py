"""
Compare firm3d and ASCOT5 COLLISIONLESS (orbit-following only) losses
for the Wistell-A reactor-scale case.

Isolates the field-representation/geometry contribution to losses from
the collision operator: the collisional comparison (compare_wistell.py)
showed similar particle-loss counts between the codes but a large
discrepancy in energy-loss fraction, traced to ASCOT5's lost population
skewing much more toward near-birth-energy ("prompt") losses than
firm3d's.  Since collisionless particles carry their full birth energy
until lost, particle-loss fraction IS energy-loss fraction here, so
this comparison uses only the loss-fraction-vs-time curves.

firm3d's collisionless companion (fusion_distribution_collisional.py's
res_free run) only saves the aggregate loss-fraction curve
(loss_times_free, loss_fraction_free), not per-marker states, so that
is what this script compares against -- no per-marker histogram is
possible on the firm3d side for this case.

Usage:
    python compare_wistell_collisionless.py --outdir /path/to/workdir

Reads results_collisional.npz (firm3d; the "_free" fields) and
ascot5_wistell_collisionless.npz (from
`run_ascot5_wistell.py --collisionless`) in --outdir.
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
    args = parser.parse_args()

    f3d = np.load(os.path.join(args.outdir, "results_collisional.npz"))
    a5 = np.load(os.path.join(args.outdir, "ascot5_wistell_collisionless.npz"))

    tf, ff = f3d["loss_times_free"], f3d["loss_fraction_free"]

    n_a = a5["lost"].size
    aborted_a = a5["aborted"] if "aborted" in a5.files else np.zeros(n_a, dtype=bool)
    n_a_valid = n_a - aborted_a.sum()
    t_end_a_valid = a5["t_end"][~aborted_a]
    lost_a = a5["lost"][~aborted_a]

    tmax = max(tf.max(), t_end_a_valid.max())
    ta, fa = loss_curve(t_end_a_valid[lost_a], n_a_valid, 1e-5, tmax)

    n_f = f3d["lost"].size  # firm3d marker count (shared by both companions)
    pf, pa = ff[-1], fa[-1]
    sf = np.sqrt(pf * (1 - pf) / n_f)
    sa = np.sqrt(pa * (1 - pa) / n_a_valid)

    print("=" * 64)
    print("Wistell-A COLLISIONLESS (orbit-following only): firm3d vs ASCOT5")
    print(
        f"N = {n_f} (firm3d) / {n_a} (ASCOT5, {aborted_a.sum()} aborted -> "
        f"{n_a_valid} valid)"
    )
    print("=" * 64)
    print(f"{'':30s} {'firm3d':>10s} {'ASCOT5':>10s}")
    print(
        f"{'final loss fraction':30s} {100 * pf:>9.1f}% {100 * pa:>9.1f}%   "
        f"(+/- {100 * np.hypot(sf, sa):.1f}% combined)"
    )
    sig = abs(pf - pa) / np.hypot(sf, sa)
    print(f"{'loss-fraction difference':30s} {sig:>9.1f} sigma")

    plt.figure()
    plt.loglog(tf, ff, label="firm3d")
    plt.loglog(ta, fa, "--", label="ASCOT5")
    plt.xlabel("Time [s]")
    plt.ylabel("Fraction of lost alphas (collisionless)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(args.outdir, "wistell_loss_comparison_collisionless.png")
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
