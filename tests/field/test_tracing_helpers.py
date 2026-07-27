"""
Tests for the position/velocity initialization helpers in tracing_helpers.

Verifies that:
- A given seed reproduces the same sample, and different seeds do not
- seed=None leaves the caller's global numpy RNG state alone
- Sampled positions lie in the expected domain
- Under MPI, every rank receives the identical sample, the sample contains
  no duplicated particles, and the sample does not depend on the number of
  ranks

The MPI assertions are the regression guard for the case where positions were
sampled from a per-rank slice of a commonly seeded RNG: every rank generated
the same points, so the gathered array held only nparticles/comm.size unique
positions, each repeated comm.size times. That failure is silent -- the array
shape is correct and tracing runs normally -- so it is asserted on explicitly
here rather than being left to surface as noisy statistics downstream.

Note that test_sample_independent_of_comm_size and
test_no_duplicate_particles_under_mpi only have force when the suite is run
under more than one rank (e.g. mpirun -n 2 python -m pytest ...); they skip
otherwise, since the bug they cover cannot occur in a single process.
"""

import unittest
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_position_uniform_surf,
    initialize_position_uniform_vol,
)

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
TEST_FILE = str((TEST_DIR / "boozmn_LandremanPaul2021_QA_lowres.nc").resolve())

# Small grids everywhere: these tests exercise the sampling/communication
# structure, not interpolation accuracy.
NPARTICLES = 64
GRID = {"ns_max": 10, "ntheta_max": 10, "nzeta_max": 10}


def reactivity(s):
    return (1 - s**5) ** 2


class TracingHelpersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(TEST_FILE).exists():
            raise unittest.SkipTest(f"Test file not found: {TEST_FILE}")
        bri = BoozerRadialInterpolant(TEST_FILE, 3, comm=comm)
        cls.field = InterpolatedBoozerField(
            bri, degree=3, ns_interp=24, ntheta_interp=24, nzeta_interp=24
        )

    def _sample(self, seed, nparticles=NPARTICLES):
        return initialize_position_profile(
            self.field, nparticles, reactivity, comm=comm, seed=seed, **GRID
        )

    def test_seed_is_reproducible(self):
        np.testing.assert_array_equal(self._sample(0), self._sample(0))

    def test_different_seeds_differ(self):
        self.assertFalse(np.array_equal(self._sample(0), self._sample(1)))

    def test_none_seed_preserves_rng_state(self):
        """
        seed=None must not reseed the global RNG: a caller that seeded numpy
        itself should still get its own reproducible stream afterwards.
        """
        np.random.seed(12345)
        self._sample(None)
        after_first = np.random.uniform(0, 1, 5)

        np.random.seed(12345)
        self._sample(None)
        after_second = np.random.uniform(0, 1, 5)

        np.testing.assert_array_equal(after_first, after_second)

    def test_positions_in_domain(self):
        points = self._sample(0)
        self.assertEqual(points.shape, (NPARTICLES, 3))
        self.assertTrue(np.all((points[:, 0] >= 0) & (points[:, 0] <= 1)))
        self.assertTrue(np.all((points[:, 1] >= 0) & (points[:, 1] <= 2 * np.pi)))
        nfp = self.field.nfp
        self.assertTrue(np.all((points[:, 2] >= 0) & (points[:, 2] <= 2 * np.pi / nfp)))

    def test_uniform_surf_stays_on_surface(self):
        s = 0.3
        points = initialize_position_uniform_surf(
            self.field, NPARTICLES, s, ntheta_max=10, nzeta_max=10, comm=comm, seed=0
        )
        np.testing.assert_allclose(points[:, 0], s)

    def test_uniform_vol_matches_flat_profile(self):
        vol = initialize_position_uniform_vol(
            self.field, NPARTICLES, comm=comm, seed=0, **GRID
        )
        flat = initialize_position_profile(
            self.field, NPARTICLES, lambda s: 1.0, comm=comm, seed=0, **GRID
        )
        np.testing.assert_array_equal(vol, flat)

    @unittest.skipIf(comm is None, "mpi4py not available")
    def test_all_ranks_receive_same_sample(self):
        points = self._sample(0)
        gathered = comm.allgather(points)
        for other in gathered[1:]:
            np.testing.assert_array_equal(gathered[0], other)

    @unittest.skipIf(comm is None or comm.size == 1, "requires >1 MPI rank")
    def test_no_duplicate_particles_under_mpi(self):
        """
        Regression guard: sampling a per-rank slice from a commonly seeded RNG
        yielded only nparticles/comm.size unique positions.
        """
        for seed in (0, 42):
            points = self._sample(seed)
            unique = np.unique(points, axis=0)
            self.assertEqual(
                len(unique),
                NPARTICLES,
                f"seed={seed}: {len(unique)}/{NPARTICLES} unique positions on "
                f"{comm.size} ranks -- sample is degenerate",
            )

    @unittest.skipIf(comm is None or comm.size == 1, "requires >1 MPI rank")
    def test_sample_independent_of_comm_size(self):
        """
        A seeded sample must not depend on how many ranks the run uses. The
        serial result is recomputed here with comm=None on every rank and
        compared against the parallel result.
        """
        parallel = self._sample(0)
        serial = initialize_position_profile(
            self.field, NPARTICLES, reactivity, comm=None, seed=0, **GRID
        )
        np.testing.assert_array_equal(parallel, serial)


if __name__ == "__main__":
    unittest.main()
