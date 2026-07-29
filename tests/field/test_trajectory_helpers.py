import os
import tempfile
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from firm3d.field.trajectory_helpers import (  # noqa: E402
    PassingPerturbedPoincare,
    PassingPoincare,
    TrappedPoincare,
)

NTRAJ = 5
NPTS = 20


def _trajectories(seed):
    """Synthetic gathered-trajectory data, in the shape plot_poincare expects."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(0, 1, NPTS) for _ in range(NTRAJ)]


def _make(cls, **attrs):
    """Build a Poincare object without tracing.

    ``plot_poincare`` only reads the gathered trajectory lists, so the plotting
    behaviour can be exercised without constructing a field or running a trace.
    """
    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _passing():
    return _make(PassingPoincare, s_all=_trajectories(1), thetas_all=_trajectories(2))


def _trapped():
    return _make(TrappedPoincare, s_all=_trajectories(1), etas_all=_trajectories(2))


def _perturbed(DA_poinc):
    npoints = 3 if DA_poinc else 1
    rng = np.random.default_rng(3)
    return _make(
        PassingPerturbedPoincare,
        s_all=_trajectories(1),
        chis_all=_trajectories(2),
        DA_poinc=DA_poinc,
        nconvergence_points=npoints,
        DA_all=[rng.uniform(0, 7, npoints) for _ in range(NTRAJ)],
        DA_times=[np.arange(npoints) for _ in range(NTRAJ)],
    )


FACTORIES = {
    "PassingPoincare": _passing,
    "TrappedPoincare": _trapped,
    "PassingPerturbedPoincare": lambda: _perturbed(False),
    "PassingPerturbedPoincare_DA": lambda: _perturbed(True),
}


class TestPlotPoincareFigureHandling(unittest.TestCase):
    """plot_poincare must not accumulate figures in pyplot's global registry.

    Poincare maps are often plotted inside a scan over lam / Bcrit / beta. Each
    figure holds one PathCollection per trajectory, so a figure that is never
    closed keeps the whole map alive and rank 0 grows without bound.
    """

    def setUp(self):
        plt.close("all")
        self._tmpdir = tempfile.TemporaryDirectory()
        # plot_poincare derives sibling filenames by prefixing (e.g.
        # "convergence_" + filename), so run in the temp directory and pass
        # bare filenames, the way the examples call it.
        self._cwd = os.getcwd()
        os.chdir(self._tmpdir.name)

    def tearDown(self):
        plt.close("all")
        os.chdir(self._cwd)
        self._tmpdir.cleanup()

    def test_no_figures_leaked_over_repeated_calls(self):
        for name, factory in FACTORIES.items():
            with self.subTest(cls=name):
                plt.close("all")
                filename = f"{name}.png"
                for _ in range(5):
                    factory().plot_poincare(filename=filename)
                self.assertEqual(
                    plt.get_fignums(),
                    [],
                    f"{name}.plot_poincare left figures open after 5 calls",
                )
                self.assertTrue(os.path.exists(filename))
                self.assertGreater(os.path.getsize(filename), 0)

    def test_supplied_ax_is_returned_and_left_open(self):
        """A caller-supplied ax must stay composable after the call."""
        for name, factory in FACTORIES.items():
            with self.subTest(cls=name):
                plt.close("all")
                fig, ax = plt.subplots()
                returned = factory().plot_poincare(
                    ax=ax, filename=f"{name}_supplied.png"
                )

                self.assertIs(returned, ax)
                self.assertIn(
                    fig.number,
                    plt.get_fignums(),
                    f"{name}.plot_poincare closed a caller-supplied figure",
                )
                # Composition on the returned ax still works.
                returned.set_title("composed")
                fig.savefig(f"{name}_composed.png")


if __name__ == "__main__":
    unittest.main()
