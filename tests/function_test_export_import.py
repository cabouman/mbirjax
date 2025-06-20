# dev_scripts/test_export_hdf5.py
import os
import h5py
import numpy as np
import mbirjax as mj
from mbirjax.utilities import export_recon_hdf5, import_recon_hdf5

# 1. Generate synthetic Shepp-Logan phantom

phantom, _, _ = mj.generate_demo_data(object_type='shepp-logan', model_type='parallel', num_views=64, num_det_rows=40, num_det_channels=128)

# 2. Export to fixed output path
output_dir = ".output"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "test_shepp_logan_export.h5")

export_recon_hdf5(file_path=file_path, recon=phantom,
                  recon_dict={"phantom": "shepp-logan", "test": "roundtrip"}, remove_flash=True,
                  radial_margin=10, top_margin=8, bottom_margin=6)

# 3. Import from HDF5
exported_recon, _ = mj.utilities.load_data_hdf5(file_path) # Load using the standard function

imported_recon, recon_dict = import_recon_hdf5(file_path)

# 4. Verify equivalence

# Expected behaviour: The phantom on the right is flash-removed and flipped to right hand coordinates.
mj.slice_viewer(
    phantom,
    exported_recon,
    slice_axis=0,
    slice_label=["Original Phantom (row, col, slice)", "Exported Phantom: Flash Removed & Flipped (slice, row, col)"],
    title="Shepp-Logan: Original vs Exported"
)

# Expected behaviour: Two phantoms should look similar, except the right one is flash-removed.

mj.slice_viewer(
    phantom,
    imported_recon,
    slice_axis=2,
    slice_label=["Original Phantom", "Imported Phantom: Flash Removed & Flipped Back"],
    title="Shepp-Logan: Original vs Imported"
)
