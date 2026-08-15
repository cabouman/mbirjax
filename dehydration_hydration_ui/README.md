# Dehydration / Hydration Correction

Native GUI (Rust, [egui](https://github.com/emilk/egui)) that reproduces the
VENUS **dehydration_hydration** notebook
(`python_notebooks/notebooks/dehydration_hydration.ipynb`): denoise a stack of
neutron images with the NMF **dehydrate / rehydrate** algorithm of
`mbirjax.hsnt.hyper_denoise`, compare the corrected and raw data, and export
the corrected stack as 32-bit float TIFFs.

The app lives inside the mbirjax repository and runs the correction (and the
"Auto" material-count estimation) through the **local mbirjax library**: the
data goes to `python/run_hyper_denoise.py` as a `.npy` file, that script
imports `mbirjax/hsnt.py` from this checkout, and the denoised stack comes
back the same way. Whatever version is checked out is what runs — no port to
keep in sync (the bundled Rust port in `src/hsnt.rs`/`src/nmf.rs` remains
only for unit tests and the `cross_check` example).

Algorithm reference: M. S. N. Chowdhury, D. Yang, S. Tang,
S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman,
"Fast Hyperspectral Neutron Tomography," *IEEE Transactions on Computational
Imaging*, vol. 11, pp. 663–677, 2025.
[doi:10.1109/TCI.2025.3567854](https://doi.org/10.1109/TCI.2025.3567854) —
[mbirjax documentation](https://mbirjax.readthedocs.io/en/latest/usr_hsnt.html).

## Workflow (same as the notebook)

1. **Open Folder…** — select the folder containing the TIFF images to correct
   (when the folder has none, its subfolders are searched, like the notebook).
   Files load in parallel; NaN/Inf pixels are zeroed (and counted in the
   Data set panel). The **🕒 Recent** menu reopens one of the last 5 dataset
   folders (persisted in `~/.config/venus_rust_tools/dehydration_hydration_recent`).
2. **Raw data** view — slide through the images next to the integrated (sum)
   image. The **Data set** panel shows the folder, image count/size, and
   memory footprint.
3. **Correction parameters** (left panel):
   - **Dataset type** — `attenuation` or `transmission`, where
     attenuation = −log(transmission). Default `attenuation`.
   - **Number of materials** — how many different materials the data set
     contains (1–10, default 2). The NMF subspace dimension is
     2 × this number (safety factor 2). **Auto** estimates it from the data
     (log-linear noise fit to the singular values of sampled pixel spectra —
     the `_estimate_subspace_dimension` algorithm of mbirjax).
   - **Beta loss** — `frobenius` (coordinate-descent solver) or
     `kullback-leibler` (multiplicative-update solver). Default `frobenius`.
   - **Max iterations** — NMF solver cap (50–1000, default 300).
4. **▶ Perform correction** — runs on a background thread with progress and a
   cancel button; the image index is treated as the spectral axis, every
   pixel spectrum is projected onto a low-dimensional non-negative subspace
   (dehydration) and multiplied back (rehydration), discarding the noise
   outside the subspace. **⚡ Preview** runs the same correction on
   2×2-binned pixels (~4× faster) for parameter tuning; preview results are
   labeled everywhere and cannot be exported.
5. **Corrected vs raw** view — side-by-side comparison with shared contrast.
   The right pane can switch to **Difference** (corrected − raw, symmetric
   color range): structure there is what the correction removed.
6. **Profiles** view — drag a region on the integrated corrected image and
   compare the mean-intensity profiles of the corrected and uncorrected
   stacks. The region can be **moved** (drag inside it) and **resized**
   (drag one of its 8 handles); **clicking a pixel** adds that single
   pixel's spectrum to the plot. When a `*_Spectra.txt` sits next to the
   images, the x-axis can switch from image index to **TOF (µs)** or
   **wavelength (Å)** (λ = h·t/(mₙ·L), source–detector distance editable,
   default 25 m). Linear/log y-axis toggle, cursor read-out in the plot
   corner, and **📄 Save CSV…** writes the plotted profiles (with TOF/λ
   columns when available).
7. **💾 Export corrected images…** — pick an output folder; the corrected
   stack is written as 32-bit float TIFFs (input file names kept) into a new
   subfolder `<input-folder>_dehydration_hydration_corrected` (suffixed `_1`,
   `_2`, … when it already exists), together with a
   **`correction_config.json`** provenance file recording the input folder,
   parameters, versions, and timestamp.

## Headless batch mode

```bash
dehydration_hydration /SNS/VENUS/IPTS-XXXX/.../Run_YYYY \
    --run --output /path/to/output \
    --materials 2 --dataset-type attenuation --beta-loss frobenius --max-iter 300
```

Runs the same load → correct → export pipeline without a window (progress on
stderr, the created folder printed on stdout) — for scripting many runs or
pipeline integration. `--bin N` runs spatially binned.

The **ℹ mbirjax** button (top-right) shows the algorithm provenance: the
version reported by the local mbirjax library (queried on every run) and the
paper reference. The exported `correction_config.json` records the same
version.

## Build & run

```bash
cargo build --release
# binary: target/release/dehydration_hydration

# or, rebuild-if-needed and run (needs a graphical session, e.g. ThinLinc):
./launch_dehydration_hydration.sh [folder-or-files...]
```

```bash
cargo test    # IO + native-port unit tests, no display or Python needed
```

### Python requirements

The correction needs a Python interpreter that can import the denoising
dependencies of `mbirjax/hsnt.py`: numpy, scipy, scikit-learn, h5py,
matplotlib (jax is **not** needed — the bridge loads `hsnt.py` directly).
Resolution order:

1. `$MBIRJAX_PYTHON`, when set (e.g. the repository's pixi environment:
   `MBIRJAX_PYTHON=$(pixi run which python)`);
2. `python3`, then `python`, from `$PATH`.

`$MBIRJAX_HSNT_BRIDGE` can point to an alternative bridge script; by default
the one under `python/` next to this crate is used.

## Implementation notes

- The denoising runs in the **local mbirjax library** (`mbirjax/hsnt.py` of
  this repository) through `src/py_bridge.rs` + `python/run_hyper_denoise.py`:
  matrices cross as `.npy` files in a per-run temp folder, progress streams
  back as `##PROGRESS` stdout lines, and Cancel kills the interpreter.
- `src/hsnt.rs`/`src/nmf.rs`/`src/linalg.rs` are a native Rust port of the
  same algorithm, kept for the unit tests (`run_correction_native`) and for
  cross-checking the library (`examples/cross_check.rs`) — the app itself
  never uses them.
- TIFF frames are transposed on load for display (VENUS detector
  orientation, same convention as rust_roi_selector / rust_tiff_viewer) and
  transposed back on export, so exported files align with the input files on
  disk.
- Light/dark theme preference is shared with the other VENUS rust tools
  (`~/.config/venus_rust_tools/theme`).
