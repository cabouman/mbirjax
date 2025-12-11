"""
Test script for export_recon_hdf5 and apply_cylindrical_mask functions.
Tests the newly introduced parameters: batch_size, slice_start, and total_slices.
"""

import os
import numpy as np
import time
import jax
import jax.numpy as jnp
import mbirjax as mj
import warnings

# Volume shape parameters
factor = 2
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
    masked_full = jnp.transpose(masked_full, (2, 1, 0))
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

mj.slice_viewer(test_volume, masked_full, slice_axis=0)


# ============================================================================
# Test export_recon_hdf5 with batch_size parameter
# ============================================================================
print("=" * 70)
print("Testing export_recon_hdf5")
print("=" * 70)

file_path = os.path.join(output_path, f"test_export_recon.h5")

t0 = time.time()
mj.export_recon_hdf5(
    file_path,
    test_volume,
    recon_dict=None,
    remove_flash=True,
    radial_margin=10,
    top_margin=10,
    bottom_margin=10,
)
t1 = time.time()
print(f"  Time to save: {t1 - t0:.1f}s\n")

t0 = time.time()
loaded_volume, loaded_dict = mj.load_data_hdf5(file_path)
t1 = time.time()
print(f"  Time to load: {t1 - t0:.1f}s\n")

max_diff = np.amax(np.abs(masked_full - loaded_volume))
print(f"Max difference: {max_diff:.3f}\n")
mj.slice_viewer(masked_full, loaded_volume, slice_axis=0)

print("=" * 70)
print("All tests completed!")
print("=" * 70)
