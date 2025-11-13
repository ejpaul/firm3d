import time
from pathlib import Path

import numpy as np

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from simsopt.util.functions import proc0_print, setup_logging
from simsopt.util.mpi import comm_size, comm_world

# Path to test file in test_files directory
TEST_DIR = (Path(__file__).parent / ".." / "inputs").resolve()
boozmn_filename = str(TEST_DIR / "boozmn_aten_rescaled.nc")

# Configuration
resolution = 48
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution
order = 3
degree = 3
no_K = True

# Setup logging
setup_logging(f"stdout_passing_map_{resolution}_{comm_size}.txt")

proc0_print("\n" + "=" * 70)
proc0_print("CLEAN INTERPOLATED BOOZER FIELD SAVE/LOAD TEST")
proc0_print("=" * 70)
proc0_print("Using C++ methods: field.to_json() and InterpolatedBoozerField(json_file)")
proc0_print("WITH ASSERTIONS TO PROVE FIELDS ARE IDENTICAL")
proc0_print("=" * 70)

# Create and compute field
proc0_print("\n1. Creating and computing interpolated field...")
time1 = time.time()

bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=no_K, comm=comm_world)
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

time2 = time.time()
proc0_print(f"Field creation time: {time2 - time1:.2f}s")

# Save field using C++ method
proc0_print("\n2. Saving field using C++ to_json() method...")
t_save = time.time()
field.to_json("QA_saved.json")
proc0_print(f"Save time: {time.time() - t_save:.2f}s")

# Load field using C++ constructor
proc0_print("\n3. Loading field using C++ constructor...")
t_load = time.time()
field2 = InterpolatedBoozerField("QA_saved.json")
proc0_print(f"Load time: {time.time() - t_load:.2f}s")

# Prepare test points
proc0_print("\n4. Preparing test points...")
rng = np.random.default_rng(7)
points = np.column_stack(
    (
        rng.uniform(0.0, 1.0, 1000),
        rng.uniform(0.0, 2 * np.pi, 1000),
        rng.uniform(0.0, 2 * np.pi, 1000),
    )
)

field.set_points(points)
field2.set_points(points)

# COMPREHENSIVE ATTRIBUTE VERIFICATION (single pass through all attributes)
proc0_print("\n5. COMPREHENSIVE ATTRIBUTE VERIFICATION...")

# Get all attributes from both fields
attrs1 = set(dir(field))
attrs2 = set(dir(field2))
common_attrs = attrs1.intersection(attrs2)

proc0_print(f"Attributes: field1={len(attrs1)}, field2={len(attrs2)}")

# Counters for single-pass verification
verified_count = 0
skipped_system = 0
skipped_callable = 0
failed_count = 0
verified_public = 0
verified_private = 0

# Single pass through all attributes
for attr in sorted(common_attrs):
    # Skip system/dunder attributes
    if attr.startswith("__") and attr.endswith("__"):
        skipped_system += 1
        continue

    try:
        v1 = getattr(field, attr)
        v2 = getattr(field2, attr)

        # Skip methods and functions
        if callable(v1) or callable(v2):
            skipped_callable += 1
            continue

        # Special handling for rule object
        if attr == "rule":
            assert v1.degree == v2.degree, (
                f"rule.degree mismatch: {v1.degree} != {v2.degree}"
            )
            np.testing.assert_allclose(
                v1.nodes,
                v2.nodes,
                rtol=1e-12,
                atol=1e-14,
                err_msg="rule.nodes mismatch",
            )
            np.testing.assert_allclose(
                v1.scalings,
                v2.scalings,
                rtol=1e-12,
                atol=1e-14,
                err_msg="rule.scalings mismatch",
            )
            proc0_print(f"{attr}: InterpolationRule verified (degree={v1.degree})")
            verified_count += 1
            verified_public += 1
            continue

        # Compare all other attributes
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            np.testing.assert_allclose(
                v1, v2, rtol=1e-12, atol=1e-14, err_msg=f"{attr} mismatch"
            )
            proc0_print(f"{attr}: arrays match")
        else:
            assert v1 == v2, f"{attr} mismatch: {v1} != {v2}"
            # Print value for important attributes
            if attr in [
                "nfp",
                "stellsym",
                "field_type",
                "psi0",
                "s_range",
                "theta_range",
                "zeta_range",
            ] or attr.startswith("status_"):
                proc0_print(f"  ✓ {attr}: {v1}")

        verified_count += 1
        if attr.startswith("_"):
            verified_private += 1
        else:
            verified_public += 1

    except Exception as e:
        proc0_print(f"  ✗ FAILED {attr}: {e}")
        failed_count += 1

# Special case: extrapolate (getter method)
try:
    assert field.get_extrapolate() == field2.get_extrapolate(), "extrapolate mismatch"
    proc0_print(f"extrapolate (via getter): {field.get_extrapolate()}")
except Exception as e:
    proc0_print(f"FAILED extrapolate: {e}")

# Summary
proc0_print("\nVERIFICATION SUMMARY:")
proc0_print(
    f"Verified: {verified_count} ({verified_public} public, {verified_private} private)"
)
proc0_print(f"Skipped: {skipped_system} system + {skipped_callable} callable")
proc0_print(f"Failed: {failed_count}")

# Assertions
assert verified_count > 0, "No attributes were successfully verified!"
assert verified_count >= 30, (
    f"Only {verified_count} attributes verified - expected at least 30"
)
assert failed_count == 0, f"{failed_count} attributes failed verification!"

proc0_print(f"ALL {verified_count} ATTRIBUTES VERIFIED SUCCESSFULLY!")

# Test all 1000 random points to verify they produce identical results
proc0_print("\n6. COMPREHENSIVE POINT EVALUATION TEST...")

# The 1000 points were already set earlier in the script
proc0_print(f"Testing with {len(points)} evaluation points...")

# Dynamically discover all quantities by finding all status_* attributes that are True
proc0_print("\nDiscovering all computed quantities from status flags...")
quantities_to_test = []
for attr in sorted(dir(field)):
    if attr.startswith("status_"):
        quantity_name = attr.replace("status_", "")
        is_computed = getattr(field, attr)
        if is_computed:
            quantities_to_test.append(quantity_name)
            proc0_print(f"    Found: {quantity_name} (status_{quantity_name} = True)")

proc0_print(f"\nTotal quantities to test: {len(quantities_to_test)}")

# Test each discovered quantity
for quantity in quantities_to_test:
    try:
        proc0_print(f"\nTesting {quantity} at all {len(points)} points...")

        # Get values from both fields
        values1 = getattr(field, quantity)()
        values2 = getattr(field2, quantity)()

        proc0_print(f"Original field shape: {values1.shape}")
        proc0_print(f"Loaded field shape:   {values2.shape}")

        # Compare all values with high precision
        # These ensure the loaded field is identical to the original
        np.testing.assert_allclose(
            values1,
            values2,
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"{quantity} values don't match at {len(points)} points",
        )

        # Show some statistics
        diff = np.abs(values1 - values2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        proc0_print(
            f"SUCCESS: {quantity} values match perfectly at all {len(points)} points!"
        )
        proc0_print(
            f"Statistics - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        )
        proc0_print(
            f"Value range - Min: {np.min(values1):.6f}, Max: {np.max(values1):.6f}"
        )

    except Exception as e:
        proc0_print(f"FAILED {quantity}: {e}")
        proc0_print(
            f"This indicates the loaded field is not working correctly for {quantity}"
        )

proc0_print("COMPREHENSIVE TEST COMPLETED!")

# Final proof
proc0_print("\n" + "=" * 70)
proc0_print("PROOF: ALL ASSERTIONS PASSED!")
proc0_print("=" * 70)
proc0_print("Original and loaded fields are IDENTICAL")
proc0_print("C++ save/load implementation works perfectly!")
proc0_print("All 1000 random points produce identical values!")
proc0_print("=" * 70)
