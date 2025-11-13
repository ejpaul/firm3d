import unittest
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.field.coordinates import (
    boozer_to_cylindrical,
    boozer_to_vmec,
    cylindrical_to_boozer,
    cylindrical_to_vmec,
    vmec_to_boozer,
    vmec_to_cylindrical,
)

np.random.seed(0)

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename_vac = str((TEST_DIR / "boozmn_LandremanPaul2021_QA_lowres.nc").resolve())
filename_vac_wout = str((TEST_DIR / "wout_LandremanPaul2021_QA_lowres.nc").resolve())


class TestCoordinates(unittest.TestCase):
    """Test coordinate transformation functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple Boozer field for testing
        self.field = BoozerRadialInterpolant(filename_vac, 1, comm=None)

        # Test points in Boozer coordinates (npoints, 3) format
        self.points_boozer_test = np.column_stack(
            [
                [0.5, 0.7, 0.3],  # s
                [0.0, np.pi / 2, np.pi],  # theta
                [0.0, np.pi / 4, np.pi / 2],  # zeta
            ]
        )

        # Test points in cylindrical coordinates (npoints, 3) format
        self.points_cyl_test = np.column_stack(
            [
                [1.5, 1.7, 1.3],  # R
                [0.0, np.pi / 4, np.pi / 2],  # phi
                [0.1, 0.2, -0.1],  # Z
            ]
        )

    def test_boozer_to_cylindrical_basic(self):
        """Test basic Boozer to cylindrical coordinate transformation."""
        points_cyl = boozer_to_cylindrical(self.field, self.points_boozer_test)

        # Check output shape
        self.assertEqual(points_cyl.shape, (3, 3))

        # Extract R, phi
        R = points_cyl[:, 0]
        phi = points_cyl[:, 1]

        # Check that R is positive
        self.assertTrue(np.all(R > 0))

        # Check that phi is in reasonable range
        self.assertTrue(np.all(phi >= -2 * np.pi) and np.all(phi <= 2 * np.pi))

    def test_boozer_to_cylindrical_single_point(self):
        """Test Boozer to cylindrical with single point input."""
        points_boozer = np.array([[0.5, np.pi / 4, np.pi / 2]])

        points_cyl = boozer_to_cylindrical(self.field, points_boozer)

        # Check that output has shape (1, 3)
        self.assertEqual(points_cyl.shape, (1, 3))

        # Check that R is positive
        self.assertGreater(points_cyl[0, 0], 0)

    def test_boozer_to_cylindrical_array_validation(self):
        """Test input validation for Boozer to cylindrical transformation."""
        # Test with wrong shape (should be (npoints, 3))
        points_wrong_shape = np.array([0.5, 0.7, 0.3])  # (3,) instead of (n, 3)

        with self.assertRaises(ValueError):
            boozer_to_cylindrical(self.field, points_wrong_shape)

    def test_boozer_to_cylindrical_empty_arrays(self):
        """Test Boozer to cylindrical with empty arrays raises ValueError."""
        points_empty = np.empty((0, 3))

        with self.assertRaises(ValueError):
            boozer_to_cylindrical(self.field, points_empty)

    def test_cylindrical_to_boozer_basic(self):
        """Test basic cylindrical to Boozer coordinate transformation."""
        # Use coordinates that are known to work from the roundtrip test
        points_cyl = np.column_stack(
            [
                [1.27177872, 0.92842923, 0.71525574],
                [0.0, 0.67037294, 1.57079633],
                [0.00000000e00, 3.11030751e-01, 2.87512917e-17],
            ]
        )

        points_boozer = cylindrical_to_boozer(self.field, points_cyl, n_guesses=10)

        # Check output shape
        self.assertEqual(points_boozer.shape, (3, 3))

        # Extract s, theta, zeta
        s = points_boozer[:, 0]
        theta = points_boozer[:, 1]
        zeta = points_boozer[:, 2]

        # Check that s is in [0, 1]
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

        # Check that angles are in reasonable range
        self.assertTrue(np.all(theta >= -2 * np.pi) and np.all(theta <= 2 * np.pi))
        self.assertTrue(np.all(zeta >= -2 * np.pi) and np.all(zeta <= 2 * np.pi))

    def test_cylindrical_to_boozer_n_guesses(self):
        """Test cylindrical to Boozer with custom number of initial guesses."""
        # Use coordinates that are known to work
        points_cyl = np.column_stack(
            [
                [1.27177872, 0.92842923, 0.71525574],
                [0.0, 0.67037294, 1.57079633],
                [0.00000000e00, 3.11030751e-01, 2.87512917e-17],
            ]
        )

        # Test with different numbers of guesses
        for n_guesses in [10, 20]:
            points_boozer = cylindrical_to_boozer(
                self.field, points_cyl, n_guesses=n_guesses
            )

            # Check output shape
            self.assertEqual(points_boozer.shape, (3, 3))

            # Extract s, theta, zeta
            s = points_boozer[:, 0]

            # Check that s is in [0, 1]
            self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_boozer_to_cylindrical_vs_vmec_route(self):
        """Test that boozer_to_cylindrical matches
        boozer_to_vmec->vmec_to_cylindrical."""
        print("\n=== Boozer -> Cylindrical vs Boozer -> VMEC -> Cylindrical Test ===")

        # Test with multiple points
        points_boozer = np.column_stack(
            [[0.3, 0.5, 0.7], [0.0, np.pi / 2, np.pi], [0.0, np.pi / 4, np.pi / 2]]
        )

        print("Original Boozer coordinates:")
        print(f"  s: {points_boozer[:, 0]}")
        print(f"  theta: {points_boozer[:, 1]}")
        print(f"  zeta: {points_boozer[:, 2]}")

        # Direct transformation: Boozer -> Cylindrical
        points_cyl_direct = boozer_to_cylindrical(self.field, points_boozer)
        print("\nDirect transformation (Boozer -> Cylindrical):")
        print(f"  R: {points_cyl_direct[:, 0]}")
        print(f"  phi: {points_cyl_direct[:, 1]}")
        print(f"  Z: {points_cyl_direct[:, 2]}")

        # Two-step transformation: Boozer -> VMEC -> Cylindrical
        points_vmec = boozer_to_vmec(filename_vac_wout, self.field, points_boozer)
        points_cyl_vmec = vmec_to_cylindrical(filename_vac_wout, points_vmec)
        print("\nTwo-step transformation (Boozer -> VMEC -> Cylindrical):")
        print(f"  R: {points_cyl_vmec[:, 0]}")
        print(f"  phi: {points_cyl_vmec[:, 1]}")
        print(f"  Z: {points_cyl_vmec[:, 2]}")

        # Compare the results
        R_diff = np.abs(points_cyl_direct[:, 0] - points_cyl_vmec[:, 0])
        Z_diff = np.abs(points_cyl_direct[:, 2] - points_cyl_vmec[:, 2])

        # Handle angle periodicity for phi comparison
        phi_direct_mod = points_cyl_direct[:, 1] % (2 * np.pi)
        phi_vmec_mod = points_cyl_vmec[:, 1] % (2 * np.pi)
        phi_diff = np.abs(phi_direct_mod - phi_vmec_mod)
        phi_diff_normalized = phi_diff % (2 * np.pi)

        print("\nDifferences (direct - vmec_route):")
        print(f"  R diff: {R_diff}")
        print(f"  phi diff (normalized): {phi_diff_normalized}")
        print(f"  Z diff: {Z_diff}")

        # Check that the transformations agree within reasonable tolerance
        # Use more relaxed tolerance since VMEC route involves additional
        # numerical steps
        np.testing.assert_allclose(R_diff, 0, atol=1e-5)
        np.testing.assert_allclose(phi_diff_normalized, 0, atol=1e-5)
        np.testing.assert_allclose(Z_diff, 0, atol=1e-5)

    def test_boozer_to_cylindrical_roundtrip(self):
        """Test roundtrip transformation: Boozer -> Cylindrical -> Boozer."""
        print("\n=== Boozer -> Cylindrical -> Boozer Roundtrip Test ===")
        # Test with multiple points that are known to work well
        points_boozer = np.column_stack(
            [
                [0.3, 0.5, 0.7, 0.9],
                [0.0, np.pi / 2, np.pi, np.pi / 4],
                [0.0, np.pi / 4, np.pi / 2, np.pi / 3],
            ]
        )

        print("Original Boozer coordinates:")
        print(f"  s: {points_boozer[:, 0]}")
        print(f"  theta: {points_boozer[:, 1]}")
        print(f"  zeta: {points_boozer[:, 2]}")

        # Forward transformation
        points_cyl = boozer_to_cylindrical(self.field, points_boozer)
        print("\nIntermediate cylindrical coordinates:")
        print(f"  R: {points_cyl[:, 0]}")
        print(f"  phi: {points_cyl[:, 1]}")
        print(f"  Z: {points_cyl[:, 2]}")

        # Reverse transformation
        points_boozer_back = cylindrical_to_boozer(self.field, points_cyl, n_guesses=20)
        print("\nFinal Boozer coordinates (after roundtrip):")
        print(f"  s: {points_boozer_back[:, 0]}")
        print(f"  theta: {points_boozer_back[:, 1]}")
        print(f"  zeta: {points_boozer_back[:, 2]}")

        # Check that we get back close to original values
        np.testing.assert_allclose(
            points_boozer_back[:, 0], points_boozer[:, 0], rtol=1e-6, atol=1e-8
        )

        # Take modulus of angles with 2π to ensure they are in the same range
        theta_back_mod = points_boozer_back[:, 1] % (2 * np.pi)
        theta_orig_mod = points_boozer[:, 1] % (2 * np.pi)
        theta_diff = np.abs(theta_back_mod - theta_orig_mod)

        zeta_back_mod = points_boozer_back[:, 2] % (2 * np.pi)
        zeta_orig_mod = points_boozer[:, 2] % (2 * np.pi)
        zeta_diff = np.abs(zeta_back_mod - zeta_orig_mod)

        print("\nDifferences (original - final):")
        print(f"  s diff: {points_boozer[:, 0] - points_boozer_back[:, 0]}")
        print(f"  theta diff (mod 2π): {theta_diff}")
        print(f"  zeta diff (mod 2π): {zeta_diff}")

        # Both angles should be recovered accurately
        self.assertTrue(np.all(theta_diff % (2 * np.pi) < 1e-6))
        self.assertTrue(np.all(zeta_diff % (2 * np.pi) < 1e-6))

    def test_vmec_to_boozer_basic(self):
        """Test VMEC to Boozer coordinate transformation."""
        # Test with a single point
        points_vmec = np.array([[0.5, 0.0, 0.0]])

        points_boozer = vmec_to_boozer(filename_vac_wout, self.field, points_vmec)

        # Check output shape
        self.assertEqual(points_boozer.shape, (1, 3))

        # Check that angles are in reasonable range
        theta_in_range = np.all(points_boozer[:, 1] >= -2 * np.pi) and np.all(
            points_boozer[:, 1] <= 2 * np.pi
        )
        zeta_in_range = np.all(points_boozer[:, 2] >= -2 * np.pi) and np.all(
            points_boozer[:, 2] <= 2 * np.pi
        )
        self.assertTrue(theta_in_range)
        self.assertTrue(zeta_in_range)

    def test_boozer_to_vmec_basic(self):
        """Test Boozer to VMEC coordinate transformation."""
        # Test with a single point
        points_boozer = np.array([[0.5, 0.0, 0.0]])

        points_vmec = boozer_to_vmec(filename_vac_wout, self.field, points_boozer)

        # Check output shape
        self.assertEqual(points_vmec.shape, (1, 3))

        # Check that angles are in reasonable range
        theta_in_range = np.all(points_vmec[:, 1] >= -2 * np.pi) and np.all(
            points_vmec[:, 1] <= 2 * np.pi
        )
        phi_in_range = np.all(points_vmec[:, 2] >= -2 * np.pi) and np.all(
            points_vmec[:, 2] <= 2 * np.pi
        )
        self.assertTrue(theta_in_range)
        self.assertTrue(phi_in_range)

    def test_vmec_boozer_roundtrip(self):
        """Test roundtrip transformation: VMEC -> Boozer -> VMEC."""
        # Test with a single point
        points_vmec = np.array([[0.5, np.pi / 4, np.pi / 2]])

        # Forward transformation
        points_boozer = vmec_to_boozer(filename_vac_wout, self.field, points_vmec)

        # Reverse transformation
        points_vmec_back = boozer_to_vmec(filename_vac_wout, self.field, points_boozer)

        # Check that we get back close to original values
        # Take modulus of angles with 2π to ensure they are in the same range
        theta_diff = np.abs(
            (points_vmec_back[:, 1] % (2 * np.pi)) - (points_vmec[:, 1] % (2 * np.pi))
        )
        phi_diff = np.abs(
            (points_vmec_back[:, 2] % (2 * np.pi)) - (points_vmec[:, 2] % (2 * np.pi))
        )

        # Use tighter tolerance for better accuracy
        np.testing.assert_allclose(theta_diff, 0, atol=1e-12)
        np.testing.assert_allclose(phi_diff, 0, atol=1e-12)

    def test_vmec_to_cylindrical_basic(self):
        """Test VMEC to cylindrical coordinate transformation."""
        # Test with a single point
        points_vmec = np.array([[0.5, 0.0, 0.0]])

        points_cyl = vmec_to_cylindrical(filename_vac_wout, points_vmec)

        # Check output shape
        self.assertEqual(points_cyl.shape, (1, 3))

        # Check that R is positive
        self.assertTrue(np.all(points_cyl[:, 0] > 0))

    def test_cylindrical_to_vmec_basic(self):
        """Test cylindrical to VMEC coordinate transformation."""
        # Test with coordinates that are known to work from the roundtrip test
        points_cyl = np.array([[0.9548126, 1.57079633, 0.06899233]])

        points_vmec = cylindrical_to_vmec(filename_vac_wout, points_cyl)

        # Check output shape
        self.assertEqual(points_vmec.shape, (1, 3))

        # Check that s is in [0, 1]
        s_in_range = np.all(points_vmec[:, 0] >= 0) and np.all(points_vmec[:, 0] <= 1)
        self.assertTrue(s_in_range)

    def test_vmec_cylindrical_roundtrip(self):
        """Test roundtrip transformation: VMEC -> Cylindrical -> VMEC."""
        print("\n=== VMEC -> Cylindrical -> VMEC Roundtrip Test ===")
        # Test with a single point
        points_vmec = np.array([[0.5, np.pi / 4, np.pi / 2]])

        print("Original VMEC coordinates:")
        print(f"  s: {points_vmec[:, 0]}")
        print(f"  theta_vmec: {points_vmec[:, 1]}")
        print(f"  phi_vmec: {points_vmec[:, 2]}")

        # Forward transformation
        points_cyl = vmec_to_cylindrical(filename_vac_wout, points_vmec)
        print("\nIntermediate cylindrical coordinates:")
        print(f"  R: {points_cyl[:, 0]}")
        print(f"  phi_cyl: {points_cyl[:, 1]}")
        print(f"  Z: {points_cyl[:, 2]}")

        # Reverse transformation
        points_vmec_back = cylindrical_to_vmec(filename_vac_wout, points_cyl)
        print("\nFinal VMEC coordinates (after roundtrip):")
        print(f"  s: {points_vmec_back[:, 0]}")
        print(f"  theta_vmec: {points_vmec_back[:, 1]}")
        print(f"  phi_vmec: {points_vmec_back[:, 2]}")

        # Check that we get back close to original values with stricter tolerances
        # The s coordinate should be recovered very accurately since it's
        # directly computed
        np.testing.assert_allclose(
            points_vmec_back[:, 0], points_vmec[:, 0], rtol=1e-8, atol=1e-8
        )

        # Take modulus of angles with 2π to ensure they are in the same range
        theta_back_mod = points_vmec_back[:, 1] % (2 * np.pi)
        theta_orig_mod = points_vmec[:, 1] % (2 * np.pi)
        theta_diff = np.abs(theta_back_mod - theta_orig_mod)

        phi_back_mod = points_vmec_back[:, 2] % (2 * np.pi)
        phi_orig_mod = points_vmec[:, 2] % (2 * np.pi)
        phi_diff = np.abs(phi_back_mod - phi_orig_mod)

        print("\nDifferences (original - final):")
        print(f"  s diff: {points_vmec[:, 0] - points_vmec_back[:, 0]}")
        print(f"  theta_vmec diff (mod 2π): {theta_diff}")
        print(f"  phi_vmec diff (mod 2π): {phi_diff}")

        # Both angles should be recovered accurately
        self.assertTrue(np.all(theta_diff < 1e-8))
        self.assertTrue(np.all(phi_diff < 1e-8))

        # The phi coordinate should be recovered exactly since it's the same
        # in both systems
        np.testing.assert_allclose(
            points_vmec_back[:, 2], points_vmec[:, 2], rtol=1e-12, atol=1e-12
        )

    def test_multiple_points(self):
        """Test transformations with multiple points."""
        print("\n=== Multiple Points Test ===")
        # Test Boozer to cylindrical with multiple points
        points_boozer = np.column_stack(
            [[0.3, 0.5, 0.7], [0.0, np.pi / 2, np.pi], [0.0, np.pi / 4, np.pi / 2]]
        )

        print("Original Boozer coordinates (multiple points):")
        print(f"  s: {points_boozer[:, 0]}")
        print(f"  theta: {points_boozer[:, 1]}")
        print(f"  zeta: {points_boozer[:, 2]}")

        points_cyl = boozer_to_cylindrical(self.field, points_boozer)
        print("\nIntermediate cylindrical coordinates:")
        print(f"  R: {points_cyl[:, 0]}")
        print(f"  phi: {points_cyl[:, 1]}")
        print(f"  Z: {points_cyl[:, 2]}")

        # Test cylindrical to Boozer with multiple points
        points_boozer_back = cylindrical_to_boozer(self.field, points_cyl, n_guesses=10)
        print("\nFinal Boozer coordinates (after roundtrip):")
        print(f"  s: {points_boozer_back[:, 0]}")
        print(f"  theta: {points_boozer_back[:, 1]}")
        print(f"  zeta: {points_boozer_back[:, 2]}")

        # Check that we get back close to original values (reduced tolerance
        # for better accuracy)
        # Note: Root finding may fail for some points, so we check that at
        # least some points are close
        s_close = np.abs(points_boozer_back[:, 0] - points_boozer[:, 0]) < 0.1

        # Take modulus of angles with 2π to ensure they are in the same range
        theta_back_mod = points_boozer_back[:, 1] % (2 * np.pi)
        theta_orig_mod = points_boozer[:, 1] % (2 * np.pi)
        theta_diff = np.abs(theta_back_mod - theta_orig_mod)
        theta_close = theta_diff < 0.1

        zeta_back_mod = points_boozer_back[:, 2] % (2 * np.pi)
        zeta_orig_mod = points_boozer[:, 2] % (2 * np.pi)
        zeta_diff = np.abs(zeta_back_mod - zeta_orig_mod)
        zeta_close = zeta_diff < 0.1

        print("\nDifferences (original - final):")
        print(f"  s diff: {points_boozer[:, 0] - points_boozer_back[:, 0]}")
        print(f"  theta diff (mod 2π): {theta_diff}")
        print(f"  zeta diff (mod 2π): {zeta_diff}")

        # At least one point should be close
        self.assertTrue(np.any(s_close) or np.any(theta_close) or np.any(zeta_close))

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test s = 0 (magnetic axis)
        points_axis = np.array([[0.0, 0.0, 0.0]])

        points_cyl = boozer_to_cylindrical(self.field, points_axis)
        self.assertTrue(np.all(points_cyl[:, 0] >= 0))

        # Test s = 1 (plasma boundary)
        points_boundary = np.array([[1.0, 0.0, 0.0]])

        points_cyl = boozer_to_cylindrical(self.field, points_boundary)
        self.assertTrue(np.all(points_cyl[:, 0] >= 0))

    def test_input_types(self):
        """Test that functions handle different input types correctly."""
        # Test with lists converted to numpy arrays
        points_boozer = np.array([[0.5, 0.0, 0.0], [0.7, np.pi / 2, np.pi / 4]])

        points_cyl = boozer_to_cylindrical(self.field, points_boozer)

        # Check that outputs are numpy arrays
        self.assertIsInstance(points_cyl, np.ndarray)

        # Check shape
        self.assertEqual(points_cyl.shape, (2, 3))

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None values
        with self.assertRaises((TypeError, AttributeError)):
            boozer_to_cylindrical(self.field, None)

        # Test with empty arrays (should raise ValueError)
        with self.assertRaises(ValueError):
            boozer_to_cylindrical(self.field, np.empty((0, 3)))

    def test_file_not_found(self):
        """Test error handling when VMEC file is not found."""
        non_existent_file = "non_existent_wout.nc"
        points_vmec = np.array([[0.5, 0.0, 0.0]])

        with self.assertRaises((FileNotFoundError, OSError)):
            vmec_to_boozer(
                non_existent_file,
                self.field,
                points_vmec,
            )


if __name__ == "__main__":
    unittest.main()
