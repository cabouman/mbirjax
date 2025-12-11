"""
Test script for export_recon_hdf5 and apply_cylindrical_mask functions.
Tests the newly introduced parameters: batch_size, slice_start, and total_slices.
"""

import os
import numpy as np
import time
import jax
import mbirjax as mj
import warnings

# Volume shape parameters
factor = 3
num_rows = 1024 * factor
num_cols = 1024 * factor
num_slices = 1024 * factor

# Define batch size for testing apply_cylindrical_mask
batch_size = 64

# Create test volume
test_volume = np.ones((num_rows, num_cols, num_slices)).astype(np.float32)
print(f"Test volume: {test_volume.shape}, {test_volume.nbytes / (1024 ** 3):.3f} GB\n")

# Output directory
output_path = './test_export/'
os.makedirs(output_path, exist_ok=True)

# ============================================================================
# Test apply_cylindrical_mask with slice_start and total_slices parameters
# ============================================================================
print("=" * 70)
print("Testing apply_cylindrical_mask with slice_start and total_slices")
print("=" * 70)


# Test without slice_start and total_slices (full volume processing)
print("\nTest 1: Full volume (slice_start=None, total_slices=None)")
try:
    import time
    start_time = time.time()
    masked_full = mj.preprocess.apply_cylindrical_mask(
        test_volume,
        radial_margin=10,
        top_margin=10,
        bottom_margin=10
    )
    masked_full = masked_full.block_until_ready()
    elapsed_time = time.time() - start_time
    masked_full = jax.device_get(masked_full)
    print("✓ Full volume processing completed successfully in {} seconds".format(elapsed_time))
    run_comparison = True
except Exception as e:
    if "out of memory" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
        warnings.warn(
            f"GPU out of memory. To pass this test, use a smaller test volume. "
            f"Current volume size: {test_volume.shape}, {test_volume.nbytes / (1024 ** 3):.3f} GB. "
            f"Skipping full volume test and comparison.",
            RuntimeWarning
        )
    else:
        warnings.warn(f"Test 1 failed: {type(e).__name__}: {e}. Skipping comparison.", RuntimeWarning)
    run_comparison = False

mj.slice_viewer(test_volume, masked_full)
# Test with slice_start and total_slices (batch processing)
print("\nTest 2: Batch processing with slice_start and total_slices")
batch_result = np.zeros_like(test_volume)
batch_size = batch_size

for start in range(0, num_slices, batch_size):
    end = min(start + batch_size, num_slices)
    batch = test_volume[:, :, start:end]

    print(f"  Processing slices {start}-{end - 1} (slice_start={start}, total_slices={num_slices})")

    masked_batch = mj.preprocess.apply_cylindrical_mask(
        batch,
        radial_margin=10,
        top_margin=10,
        bottom_margin=10,
        slice_start=start,
        total_slices=num_slices
    )

    batch_result[:, :, start:end] = masked_batch

# Compare results
if run_comparison:
    max_diff = np.max(np.abs(masked_full - batch_result))
    print(f"\nMax difference between full and batch: {max_diff}")
    if max_diff < 1e-6:
        print("✓ Batch processing matches full volume processing!\n")
    else:
        print("✗ WARNING: Batch processing does NOT match full volume processing!\n")
else:
    print("\n Comparison skipped (full volume test did not complete)\n")


# Test error handling: only slice_start provided
print("\nTest 3: Error handling - only slice_start provided (should raise ValueError)")
try:
    batch = test_volume[:, :, :batch_size]
    masked_batch = mj.preprocess.apply_cylindrical_mask(
        batch,
        radial_margin=10,
        top_margin=10,
        bottom_margin=10,
        slice_start=0,
        total_slices=None
    )
    print("✗ ERROR: Should have raised ValueError but did not!\n")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}\n")
except Exception as e:
    print(f"✗ ERROR: Raised unexpected exception: {type(e).__name__}: {e}\n")


# Test error handling: only total_slices provided
print("\nTest 4: Error handling - only total_slices provided (should raise ValueError)")
try:
    batch = test_volume[:, :, :batch_size]
    masked_batch = mj.preprocess.apply_cylindrical_mask(
        batch,
        radial_margin=10,
        top_margin=10,
        bottom_margin=10,
        slice_start=None,
        total_slices=num_slices
    )
    print("✗ ERROR: Should have raised ValueError but did not!\n")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}\n")
except Exception as e:
    print(f"✗ ERROR: Raised unexpected exception: {type(e).__name__}: {e}\n")


# ============================================================================
# Test export_recon_hdf5 with batch_size parameter
# ============================================================================
print("=" * 70)
print("Testing export_recon_hdf5 with batch_size parameter")
print("=" * 70)

# Test cases with different batch_size values
batch_sizes = [16, 32, 64, 128, 256, 512, 1024]

print(f"\nRunning {len(batch_sizes)} test cases with different batch_size:\n")

for i, batch_size in enumerate(batch_sizes, 1):
    print(f"Test {i}/{len(batch_sizes)}: batch_size={batch_size}")

    file_path = os.path.join(output_path, f"test_batch_{batch_size}.h5")

    t0 = time.time()
    mj.export_recon_hdf5(
        file_path,
        test_volume,
        recon_dict=None,
        remove_flash=True,
        radial_margin=10,
        top_margin=10,
        bottom_margin=10,
        batch_size=batch_size
    )
    t1 = time.time()
    print(f"  Elapsed: {t1 - t0:.1f}s\n")

print("=" * 70)
print("All tests completed!")
print("=" * 70)
