"""
Unit tests for InterpolatedBoozerField save/load functionality.

This test suite verifies that InterpolatedBoozerField objects can be correctly
saved to JSON and loaded back with identical attributes and evaluation results.
Tests are performed on all available equilibrium configurations:
- Vacuum equilibria (QA)
- Finite-beta stellarator symmetric equilibria
- Finite-beta asymmetric equilibria

NOTE ON MEMORY MANAGEMENT:
This test suite uses aggressive garbage collection to prevent C++ state accumulation
that can cause segfaults when running multiple tests together. The booz_xform C++
library maintains internal state that can accumulate if Python objects aren't
properly cleaned up between tests. We address this by:
1. Force garbage collection in setUp() and tearDown() methods
2. Setting objects to None in finally blocks (safer than del for pybind11)
3. Periodic deep garbage collection every N tests
"""

import unittest
import tempfile
import os
import gc  # For aggressive garbage collection to prevent C++ state accumulation
import sys
from pathlib import Path

import numpy as np

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)

# Path to test_files directory containing equilibrium data files
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

# Test file configurations - ALL 10 files in test_files directory
# These define the equilibrium configurations used across all tests:
# - 3 main boozmn files (vac_qa, mhd_sym, mhd_asym)
# - 3 corresponding wout files
# - 4 error-testing files (reduced/reordered variants)
TEST_CONFIGS = {
    # Main boozmn files (3 files)
    "vac_qa_boozmn": {
        "file": str((TEST_DIR / "boozmn_LandremanPaul2021_QA_lowres.nc").resolve()),
        "wout_file": str((TEST_DIR / "wout_LandremanPaul2021_QA_lowres.nc").resolve()),
        "nfp": 4,
        "stellsym": True,
        "order": 3,
    },
    "mhd_sym_boozmn": {
        "file": str((TEST_DIR / "boozmn_n3are_R7.75B5.7.nc").resolve()),
        "wout_file": str((TEST_DIR / "wout_n3are_R7.75B5.7.nc").resolve()),
        "reduced_file": str((TEST_DIR / "boozmn_n3are_R7.75B5.7_reduced.nc").resolve()),
        "reordered_file": str((TEST_DIR / "boozmn_n3are_R7.75B5.7_reordered.nc").resolve()),
        "nfp": 3,
        "stellsym": True,
        "order": 3,
    },
    "mhd_asym_boozmn": {
        "file": str((TEST_DIR / "boozmn_ITERModel_reference.nc").resolve()),
        "wout_file": str((TEST_DIR / "wout_ITERModel_reference.nc").resolve()),
        "reduced_file": str((TEST_DIR / "boozmn_ITERModel_reference_reduced.nc").resolve()),
        "reordered_file": str((TEST_DIR / "boozmn_ITERModel_reference_reordered.nc").resolve()),
        "nfp": 3,
        # Note: Despite "asym" in name, this file is actually symmetric (stellsym=True)
        # The stellsym value will be inferred from the file itself, matching passing_map.py behavior
        "stellsym": True,  # File is actually symmetric (verified in passing_map.py output)
        "order": 3,
    },
}

# MPI communicator for parallel processing (if available)
# BoozerRadialInterpolant uses this for distributed computation across MPI processes
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None  # Run in serial mode if MPI is not available


class TestInterpolatedBoozerFieldSaveLoad(unittest.TestCase):
    """
    Test save/load functionality for InterpolatedBoozerField.
    
    This class provides comprehensive testing of the JSON serialization/deserialization
    workflow for InterpolatedBoozerField objects, ensuring data integrity across
    save/load cycles.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test class - run once before all tests.
        
        Initializes class-level counters for tracking test execution and
        managing periodic deep garbage collection cycles.
        """
        cls._test_count = 0  # Track number of tests executed
        # Force aggressive garbage collection every N tests to prevent C++ state accumulation
        cls._max_tests_before_gc = 3
    
    def setUp(self):
        """
        Set up before each test - force garbage collection.
        
        Performs aggressive garbage collection before each test to ensure
        any C++ objects from previous tests are fully cleaned up. This prevents
        state accumulation in the booz_xform library that can cause segfaults.
        """
        # Multiple rounds of gc.collect() ensure all generations are cleaned
        # Single calls may not catch all references due to circular dependencies
        gc.collect()
        gc.collect()
        gc.collect()
        
    def tearDown(self):
        """
        Clean up after each test - force garbage collection.
        
        Performs cleanup after each test and triggers periodic deep garbage
        collection to prevent C++ state accumulation. This is critical for
        preventing segfaults when running all 10 tests together.
        """
        # Increment test counter for periodic deep cleanup
        TestInterpolatedBoozerFieldSaveLoad._test_count += 1
        
        # Standard cleanup after each test - multiple rounds ensure thorough cleanup
        gc.collect()
        gc.collect()
        gc.collect()
        
        # Periodic deep cleanup every N tests to handle any accumulated state
        # This is particularly important for C++ objects with internal state
        if TestInterpolatedBoozerFieldSaveLoad._test_count % self._max_tests_before_gc == 0:
            # Force multiple rounds of garbage collection to break cycles
            for _ in range(5):
                gc.collect()
            
            # Full generation-2 collection (oldest objects) for Python 3.4+
            # This collects objects that have survived multiple garbage collection cycles
            if sys.version_info >= (3, 4):
                gc.collect(generation=2)

    def _verify_attributes(self, field1, field2, config_name):
        """
        Verify that all attributes match between original and loaded field.
        
        This method performs comprehensive attribute-by-attribute comparison,
        including special handling for nested objects (like rule) and methods
        that need to be called (like get_extrapolate).
        
        Parameters
        ----------
        field1 : InterpolatedBoozerField
            Original field before save/load
        field2 : InterpolatedBoozerField
            Loaded field after save/load cycle
        config_name : str
            Name of the configuration being tested (for error messages)
        """
        # Get all attributes from both fields and find common ones
        # This ensures we verify all attributes present in both objects
        attrs1 = set(dir(field1))
        attrs2 = set(dir(field2))
        common_attrs = attrs1.intersection(attrs2)
        
        # Track verification statistics for reporting
        verified_count = 0
        failed_attrs = []
        
        # Verify each common attribute
        for attr in sorted(common_attrs):
            # Skip system/dunder attributes (e.g., __class__, __dict__)
            # These are Python internals and don't need verification
            if attr.startswith('__') and attr.endswith('__'):
                continue
                
            try:
                v1 = getattr(field1, attr)
                v2 = getattr(field2, attr)
                
                # Skip methods and functions - we verify these separately via
                # evaluation tests, not attribute comparison
                if callable(v1) or callable(v2):
                    continue
                
                # Special handling for rule object (InterpolationRule)
                # This is a nested object that requires component-wise comparison
                if attr == 'rule':
                    self.assertEqual(v1.degree, v2.degree, 
                                   f"{config_name}: rule.degree mismatch")
                    # Use strict tolerance for rule nodes and scalings
                    np.testing.assert_allclose(v1.nodes, v2.nodes, rtol=1e-12, atol=1e-14,
                                             err_msg=f"{config_name}: rule.nodes mismatch")
                    np.testing.assert_allclose(v1.scalings, v2.scalings, rtol=1e-12, atol=1e-14,
                                             err_msg=f"{config_name}: rule.scalings mismatch")
                    verified_count += 1
                    continue
                
                # Compare NumPy arrays with strict tolerance
                # Arrays require numerical comparison, not identity comparison
                if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                    np.testing.assert_allclose(v1, v2, rtol=1e-12, atol=1e-14,
                                             err_msg=f"{config_name}: {attr} mismatch")
                else:
                    # Scalar values use equality comparison
                    self.assertEqual(v1, v2, f"{config_name}: {attr} mismatch: {v1} != {v2}")
                
                verified_count += 1
                
            except Exception as e:
                # Track failures but continue checking other attributes
                failed_attrs.append((attr, str(e)))
        
        # Special case: extrapolate is accessed via getter method, not direct attribute
        # Verify this separately since it's a method call rather than an attribute
        try:
            self.assertEqual(field1.get_extrapolate(), field2.get_extrapolate(),
                           f"{config_name}: extrapolate mismatch")
            verified_count += 1
        except Exception as e:
            failed_attrs.append(("extrapolate", str(e)))
        
        # Assert that we verified enough attributes (sanity check)
        # Should verify at least 25 attributes (grid parameters, status flags, etc.)
        self.assertGreater(verified_count, 25, 
                          f"{config_name}: Only {verified_count} attributes verified")
        # Ensure no attributes failed verification
        self.assertEqual(len(failed_attrs), 0,
                        f"{config_name}: Failed attributes: {failed_attrs}")

    def _verify_python_attributes(self, field1, field2, config_name):
        """
        Verify Python-specific attributes match.
        
        This method focuses on high-level Python attributes that define the
        field configuration, as opposed to low-level C++ status flags.
        
        Parameters
        ----------
        field1 : InterpolatedBoozerField
            Original field
        field2 : InterpolatedBoozerField
            Loaded field
        config_name : str
            Name of the configuration being tested
        """
        # Core scalar attributes that define the field configuration
        # nfp: number of field periods (rotational symmetry)
        # stellsym: stellarator symmetry flag
        # psi0: normalization flux (typically 2*pi*phi_edge)
        python_attrs = ['nfp', 'stellsym', 'psi0']
        
        for attr in python_attrs:
            v1 = getattr(field1, attr)
            v2 = getattr(field2, attr)
            self.assertEqual(v1, v2, 
                           f"{config_name}: Python attribute {attr} mismatch: {v1} != {v2}")
        
        # Grid ranges define the interpolation domain in (s, theta, zeta) coordinates
        # These are stored as NumPy arrays [min, max] and must match exactly
        np.testing.assert_allclose(field1.s_range, field2.s_range, rtol=1e-12, atol=1e-14,
                                 err_msg=f"{config_name}: s_range mismatch")
        np.testing.assert_allclose(field1.theta_range, field2.theta_range, rtol=1e-12, atol=1e-14,
                                 err_msg=f"{config_name}: theta_range mismatch")
        np.testing.assert_allclose(field1.zeta_range, field2.zeta_range, rtol=1e-12, atol=1e-14,
                                 err_msg=f"{config_name}: zeta_range mismatch")

    def _verify_cpp_attributes(self, field1, field2, config_name):
        """
        Verify C++-specific attributes match.
        
        This method verifies the status flags that indicate which quantities
        are available for interpolation. These flags are set by the C++ layer
        and determine which methods can be safely called.
        
        Parameters
        ----------
        field1 : InterpolatedBoozerField
            Original field
        field2 : InterpolatedBoozerField
            Loaded field
        config_name : str
            Name of the configuration being tested
        """
        # Status flags (C++ bool attributes exposed to Python via pybind11)
        # Each flag indicates whether the corresponding quantity is available
        # for interpolation. These must match exactly between save and load.
        status_attrs = [
            # Magnetic field magnitude and derivatives
            'status_modB', 'status_dmodBdtheta', 'status_dmodBdzeta', 'status_dmodBds',
            # Flux functions (G, I, iota and their derivatives)
            'status_G', 'status_I', 'status_iota', 'status_dGds', 'status_dIds', 'status_diotads',
            # Poloidal flux
            'status_psip',
            # Geometric quantities (R, Z, nu) and derivatives
            'status_R', 'status_Z', 'status_nu',
            'status_dRdtheta', 'status_dRdzeta', 'status_dRds',
            'status_dZdtheta', 'status_dZdzeta', 'status_dZds',
            'status_dnudtheta', 'status_dnudzeta', 'status_dnuds',
            # Magnetic differential equation quantity
            'status_K', 'status_dKdtheta', 'status_dKdzeta',
            # Combined derivative methods
            'status_K_derivs', 'status_nu_derivs', 'status_R_derivs', 
            'status_Z_derivs', 'status_modB_derivs'
        ]
        
        for attr in status_attrs:
            v1 = getattr(field1, attr)
            v2 = getattr(field2, attr)
            # Status flags are boolean - use exact equality
            self.assertEqual(v1, v2, 
                           f"{config_name}: C++ status flag {attr} mismatch: {v1} != {v2}")

    def _verify_evaluations(self, field1, field2, config_name, num_points=1000):
        """
        Verify that evaluations at random points match between fields.
        
        This method tests the actual interpolation functionality by evaluating
        both fields at the same random points and comparing results. This is
        the most critical test as it verifies the saved/loaded field produces
        identical results to the original.
        
        Parameters
        ----------
        field1 : InterpolatedBoozerField
            Original field
        field2 : InterpolatedBoozerField
            Loaded field
        config_name : str
            Name of the configuration being tested
        num_points : int, optional
            Number of random test points (default: 1000)
        """
        # Generate random test points within the interpolation domain
        # Fixed seed ensures reproducibility across test runs
        np.random.seed(42)
        points = np.random.uniform(size=(num_points, 3))
        
        # Scale random points [0, 1] to actual domain coordinates
        # Points are in (s, theta, zeta) format
        s_range = field1.s_range
        theta_range = field1.theta_range
        zeta_range = field1.zeta_range
        
        points[:, 0] = points[:, 0] * (s_range[1] - s_range[0]) + s_range[0]  # s coordinate
        points[:, 1] = points[:, 1] * (theta_range[1] - theta_range[0]) + theta_range[0]  # theta
        points[:, 2] = points[:, 2] * (zeta_range[1] - zeta_range[0]) + zeta_range[0]  # zeta
        
        # Set the same evaluation points for both fields
        # This ensures we compare identical queries
        field1.set_points(points)
        field2.set_points(points)
        
        # Track which quantities are actually tested (for reporting)
        quantities_to_test = []
        
        # Scalar quantities (return arrays of shape [num_points, 1])
        # These represent single-valued functions at each point
        scalar_quantities = [
            ('modB', 'status_modB'),
            ('dmodBdtheta', 'status_dmodBdtheta'),
            ('dmodBdzeta', 'status_dmodBdzeta'),
            ('dmodBds', 'status_dmodBds'),
            ('G', 'status_G'),
            ('I', 'status_I'),
            ('iota', 'status_iota'),
            ('dGds', 'status_dGds'),
            ('dIds', 'status_dIds'),
            ('diotads', 'status_diotads'),
            ('psip', 'status_psip'),
            ('R', 'status_R'),
            ('Z', 'status_Z'),
            ('nu', 'status_nu'),
            ('K', 'status_K'),
            ('dRdtheta', 'status_dRdtheta'),
            ('dRdzeta', 'status_dRdzeta'),
            ('dRds', 'status_dRds'),
            ('dZdtheta', 'status_dZdtheta'),
            ('dZdzeta', 'status_dZdzeta'),
            ('dZds', 'status_dZds'),
            ('dnudtheta', 'status_dnudtheta'),
            ('dnudzeta', 'status_dnudzeta'),
            ('dnuds', 'status_dnuds'),
            ('dKdtheta', 'status_dKdtheta'),
            ('dKdzeta', 'status_dKdzeta'),
        ]
        
        # Multi-component quantities (return arrays of shape [num_points, N])
        # These represent vector-valued functions returning multiple components
        # Format: (method_name, status_flag, number_of_components)
        multi_quantities = [
            ('K_derivs', 'status_K_derivs', 2),        # [dK/dtheta, dK/dzeta]
            ('modB_derivs', 'status_modB_derivs', 3),  # [dmodB/ds, dmodB/dtheta, dmodB/dzeta]
            ('R_derivs', 'status_R_derivs', 3),        # [dR/ds, dR/dtheta, dR/dzeta]
            ('Z_derivs', 'status_Z_derivs', 3),        # [dZ/ds, dZ/dtheta, dZ/dzeta]
            ('nu_derivs', 'status_nu_derivs', 3),      # [dnu/ds, dnu/dtheta, dnu/dzeta]
        ]
        
        # Test scalar quantities - only test those with status_flag = True
        # This ensures we don't test quantities that weren't computed/loaded
        for qty_name, status_flag in scalar_quantities:
            if getattr(field1, status_flag):
                # Evaluate the quantity at all test points
                qty1 = getattr(field1, qty_name)()
                qty2 = getattr(field2, qty_name)()
                
                # Strict tolerance ensures numerical precision is maintained
                # rtol=1e-12, atol=1e-14 matches machine precision for double precision
                np.testing.assert_allclose(
                    qty1, qty2, rtol=1e-12, atol=1e-14,
                    err_msg=f"{config_name}: {qty_name} evaluation mismatch at {num_points} points"
                )
                quantities_to_test.append(qty_name)
        
        # Test multi-component quantities
        for qty_name, status_flag, components in multi_quantities:
            if getattr(field1, status_flag):
                qty1 = getattr(field1, qty_name)()
                qty2 = getattr(field2, qty_name)()
                
                # Verify correct number of components
                self.assertEqual(qty1.shape[1], components,
                               f"{config_name}: {qty_name} has wrong number of components")
                
                # Compare all components with strict tolerance
                np.testing.assert_allclose(
                    qty1, qty2, rtol=1e-12, atol=1e-14,
                    err_msg=f"{config_name}: {qty_name} evaluation mismatch at {num_points} points"
                )
                quantities_to_test.append(qty_name)
        
        # Sanity check: ensure we tested at least some quantities
        # This catches cases where status flags are incorrectly set
        self.assertGreater(len(quantities_to_test), 0,
                          f"{config_name}: No quantities were tested!")

    def _test_config(self, config_name, config_params):
        """
        Test save/load for a specific configuration.
        
        This is the main test workflow:
        1. Create BoozerRadialInterpolant from equilibrium file
        2. Create InterpolatedBoozerField with interpolation grid
        3. Save field to temporary JSON file
        4. Load field back from JSON
        5. Verify attributes and evaluations match
        
        Parameters
        ----------
        config_name : str
            Name of the configuration (for error messages)
        config_params : dict
            Configuration parameters containing:
            - file: path to equilibrium file (boozmn or wout)
            - order: spline order for radial interpolation
            - nfp: number of field periods (for reference, may be inferred)
            - stellsym: stellarator symmetry flag (for reference, may be inferred)
        """
        # Initialize all variables to None to ensure they exist in finally block
        # This prevents NameError if an exception occurs early
        bri = None
        field = None
        field2 = None
        json_path = None
        
        try:
            # Step 1: Create BoozerRadialInterpolant from equilibrium file
            # This reads the equilibrium data and prepares it for interpolation
            bri = BoozerRadialInterpolant(
                config_params["file"],
                config_params["order"],
                comm=comm
            )
            
            # Step 2: Create InterpolatedBoozerField with interpolation grid
            # Match passing_map.py behavior: let stellsym be inferred from bri.stellsym
            # This ensures consistency with typical usage patterns where symmetry
            # is determined from the equilibrium file itself
            n = 8  # Base grid resolution
            smin = 0.3
            smax = 0.7
            ssteps = n
            thetamin = 0
            # Use bri.stellsym (inferred from file) rather than explicit config value
            # For stellarator symmetric fields, theta range is [0, pi]
            # For asymmetric fields, theta range is [0, 2*pi]
            thetamax = np.pi if bri.stellsym else 2 * np.pi
            thetasteps = n
            zetamin = 0
            # Use bri.nfp to determine zeta range (one field period)
            zetamax = 2 * np.pi / bri.nfp
            zetasteps = n * 2  # Higher resolution in zeta due to periodicity
            
            field = InterpolatedBoozerField(
                bri,
                4,  # Interpolation degree (Chebyshev polynomial order)
                [smin, smax, ssteps],  # Radial grid: [min, max, num_points]
                [thetamin, thetamax, thetasteps],  # Poloidal grid
                [zetamin, zetamax, zetasteps],  # Toroidal grid
                True,  # extrapolate: allow evaluation outside grid
                # Don't pass stellsym/nfp explicitly - let them be inferred from bri
                # This matches passing_map.py behavior exactly and ensures correct
                # symmetry handling based on the actual equilibrium data
            )
            
            # Step 3: Save field to temporary JSON file
            # Use NamedTemporaryFile with delete=False so we can load it back
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json_path = tmp_file.name
            
            # Serialize the field to JSON
            field.to_json(json_path)
            
            # Step 4: Load field back from JSON
            # This tests the JSON deserialization constructor
            field2 = InterpolatedBoozerField(json_path)
            
            # Step 5: Comprehensive verification
            # Verify all attributes match (general attribute comparison)
            self._verify_attributes(field, field2, config_name)
            
            # Verify Python-specific attributes (high-level config)
            self._verify_python_attributes(field, field2, config_name)
            
            # Verify C++ status flags (low-level availability flags)
            self._verify_cpp_attributes(field, field2, config_name)
            
            # Verify evaluations at many points (functional correctness)
            self._verify_evaluations(field, field2, config_name, num_points=1000)
            
        finally:
            # Cleanup: ensure all resources are properly released
            # This is critical for preventing C++ state accumulation
            
            # Remove temporary JSON file
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            
            # Explicitly set objects to None to help garbage collection
            # Using None assignment rather than del is safer with pybind11 bindings,
            # as it properly decrements reference counts without interfering with
            # Python's internal reference counting mechanisms
            field2 = None
            field = None
            bri = None
            
            # Force garbage collection to ensure C++ objects are released immediately
            # This prevents state accumulation that can cause segfaults in subsequent tests
            gc.collect()

    def test_saveload_vac_qa_boozmn(self):
        """Test save/load for vacuum QA equilibrium (boozmn file #1)."""
        self._test_config("vac_qa_boozmn", TEST_CONFIGS["vac_qa_boozmn"])

    def test_saveload_vac_qa_wout(self):
        """Test save/load for vacuum QA equilibrium (wout file #2)."""
        config = TEST_CONFIGS["vac_qa_boozmn"].copy()
        config["file"] = config["wout_file"]
        self._test_config("vac_qa_wout", config)

    def test_saveload_mhd_sym_boozmn(self):
        """Test save/load for finite-beta stellarator symmetric equilibrium (boozmn file #3)."""
        self._test_config("mhd_sym_boozmn", TEST_CONFIGS["mhd_sym_boozmn"])

    def test_saveload_mhd_sym_wout(self):
        """Test save/load for finite-beta stellarator symmetric equilibrium (wout file #4)."""
        config = TEST_CONFIGS["mhd_sym_boozmn"].copy()
        config["file"] = config["wout_file"]
        self._test_config("mhd_sym_wout", config)

    def test_saveload_mhd_asym_boozmn(self):
        """Test save/load for finite-beta asymmetric equilibrium (boozmn file #5)."""
        self._test_config("mhd_asym_boozmn", TEST_CONFIGS["mhd_asym_boozmn"])

    def test_saveload_mhd_asym_wout(self):
        """Test save/load for finite-beta asymmetric equilibrium (wout file #6)."""
        config = TEST_CONFIGS["mhd_asym_boozmn"].copy()
        config["file"] = config["wout_file"]
        self._test_config("mhd_asym_wout", config)

    def test_invalid_reduced_reordered_files(self):
        """
        Test that reduced/reordered files fail to create BoozerRadialInterpolant.
        These are files #7-10 in test_files directory.
        
        These files have incorrect radial grids or ordering, so BoozerRadialInterpolant
        should raise ValueError when trying to load them.
        """
        # Test reduced files (#7 and #9)
        for config_name in ["mhd_sym_boozmn", "mhd_asym_boozmn"]:
            config = TEST_CONFIGS[config_name]
            if "reduced_file" in config:
                with self.assertRaises(ValueError, msg=f"{config_name} reduced file should fail"):
                    BoozerRadialInterpolant(config["reduced_file"], config["order"], comm=comm)
        
        # Test reordered files (#8 and #10)
        for config_name in ["mhd_sym_boozmn", "mhd_asym_boozmn"]:
            config = TEST_CONFIGS[config_name]
            if "reordered_file" in config:
                with self.assertRaises(ValueError, msg=f"{config_name} reordered file should fail"):
                    BoozerRadialInterpolant(config["reordered_file"], config["order"], comm=comm)

    def test_saveload_coarse_resolution(self):
        """Test save/load with coarse grid resolution (4x4x8)."""
        config = TEST_CONFIGS["vac_qa_boozmn"]
        self._test_config_simple(config, [0.3, 0.7, 4], [0, np.pi, 4], [0, 2 * np.pi / config["nfp"], 8], 
                                 "coarse_resolution", num_points=50)

    def test_saveload_medium_resolution(self):
        """Test save/load with medium grid resolution (8x8x16)."""
        config = TEST_CONFIGS["vac_qa_boozmn"]
        self._test_config_simple(config, [0.3, 0.7, 8], [0, np.pi, 8], [0, 2 * np.pi / config["nfp"], 16],
                                 "medium_resolution", num_points=50)

    def test_saveload_small_domain(self):
        """Test save/load with small spatial domain."""
        config = TEST_CONFIGS["vac_qa_boozmn"]
        self._test_config_simple(config, [0.2, 0.4, 6], [0, np.pi/2, 6], [0, np.pi / config["nfp"], 12],
                                 "small_domain", num_points=50)

    def _test_config_simple(self, config, s_range, theta_range, zeta_range, test_name, num_points=100):
        """
        Helper method to test save/load with a specific configuration.
        
        This is a simplified version of _test_config() used for resolution and
        domain tests. It avoids loops to prevent segfaults from repeated
        BoozerRadialInterpolant creation. Only performs essential verification
        (Python attributes and evaluations) to keep test runtime manageable.
        
        Parameters
        ----------
        config : dict
            Configuration parameters (file, order, stellsym, nfp)
        s_range : list
            Radial grid [min, max, num_points]
        theta_range : list
            Poloidal grid [min, max, num_points]
        zeta_range : list
            Toroidal grid [min, max, num_points]
        test_name : str
            Name of the test (for error messages)
        num_points : int
            Number of random points for evaluation testing
        """
        # Initialize all variables to None for safe cleanup
        bri = None
        field = None
        field2 = None
        json_path = None
        
        try:
            # Create BoozerRadialInterpolant from equilibrium file
            bri = BoozerRadialInterpolant(config["file"], config["order"], comm=comm)
            
            # Create InterpolatedBoozerField with custom grid ranges
            # Note: Explicitly pass stellsym and nfp since this method is used
            # for testing different grid configurations, not equilibrium types
            field = InterpolatedBoozerField(
                bri,
                3,  # Lower degree for faster computation in resolution tests
                s_range,
                theta_range,
                zeta_range,
                True,  # extrapolate
                stellsym=config["stellsym"],
                nfp=config["nfp"],
            )
            
            # Create temporary JSON file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json_path = tmp.name
            
            # Save and load cycle
            field.to_json(json_path)
            field2 = InterpolatedBoozerField(json_path)
            
            # Quick verification (skip full attribute comparison for speed)
            # Only verify essential attributes and functional correctness
            self._verify_python_attributes(field, field2, test_name)
            self._verify_evaluations(field, field2, test_name, num_points=num_points)
            
        finally:
            # Cleanup: remove temporary file and release objects
            if json_path and os.path.exists(json_path):
                os.remove(json_path)
            
            # Explicitly set objects to None to help garbage collection
            # This is critical for preventing C++ state accumulation
            field2 = None
            field = None
            bri = None
            
            # Force garbage collection to ensure immediate cleanup
            gc.collect()


if __name__ == "__main__":
    unittest.main()

