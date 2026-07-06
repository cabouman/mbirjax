# Larger multi-GPU reconstructions, host-safe utilities, and a two-stage workflow

This PR builds on the multi-GPU reconstruction support merged previously.  It focuses on scaling to very
large cone-beam volumes, removing memory pitfalls in the utilities around reconstruction, and improving
the production workflow.  Everything runs with no change to existing scripts.

## Import mbirjax before jax

mbirjax configures JAX at import time -- it sets the XLA host-device-count flag (so multi-CPU runs see
all cores) and raises the C++ log level (`TF_CPP_MIN_LOG_LEVEL`) to silence the benign multi-GPU
allocator warnings (`cuda_vmm_allocator ... FABRIC+POSIX_FD ... NOT_PERMITTED`).  Both are read once when
jaxlib first initializes, so they only take effect if set first.  **Therefore `import mbirjax` (or
`import mbirjax.preprocess`) should come before any `import jax` / `import jax.numpy` in a script or
module.**  If jax is imported first, mbirjax still works correctly, but those allocator warnings reappear
and, on CPU-only systems, JAX may expose fewer (virtual) devices.  To make this independent of import
order, set the variables in the environment before launching Python, e.g. `export TF_CPP_MIN_LOG_LEVEL=2`.

## Larger reconstructions

- **Large cone-beam reconstructions now fit across multiple GPUs.**  A full 2048³ cone-beam volume
  reconstructs on an 8-GPU node, where it previously ran out of memory.  This comes from using a 
  partition sequence [4, 6, 7, 7, ...], together with lower peak memory use.
- **Lower peak memory.**  Reconstruction releases large inputs as soon as they are consumed, so each
  device holds less at the peak.
- **`split_sino_recon` hardened.**  Reconstructing a cone-beam volume in two overlapping halves now
  reliably extends capacity when fewer GPUs are available, uses the model's configured devices, and
  assembles the result on the host so the full volume is never rebuilt on one GPU.

## More robust memory behavior

- **Default reconstruction and projection results are assembled on the host.**  When you do not explicitly
  ask for the sharded device form, `recon`, `forward_project`, `back_project`, the FBP/FDK direct
  reconstructions, and the denoiser now return an ordinary NumPy array gathered directly to host memory.
  Previously a large result could be re-collected onto a single GPU on the way out and run out of memory;
  now the full volume is never rebuilt on one device, so these calls are safe at any size.
- **Utilities that touch the whole volume were reworked to avoid accidental single-GPU blowups.**  Weight
  generation, demo-data generation, the cylindrical-mask / HDF5 export, and the half-volume stitch now
  keep host data on the host and only use the GPU when appropriate.  Previously these could copy an
  entire large volume onto a single GPU and run out of memory.  In particular, **saving a large
  reconstruction to HDF5 no longer fails.**
- **Clearer out-of-memory guidance** is printed when a reconstruction exceeds GPU memory, including how to
  reduce peak memory.
- **The GPU memory reservation is now 94% and user-overridable.**  mbirjax previously hard-set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.98`, which left almost no room for allocations that live outside
  XLA's pool (GPU library workspaces such as cuSolver, and the NCCL buffers used by multi-GPU
  reductions) -- these could fail with out-of-memory errors even when the pool itself was nearly idle.
  The default is now `0.94`, set with `setdefault` so setting the environment variable before importing
  mbirjax overrides it.  Per-iteration reconstruction statistics are also computed in a single fused
  pass (rather than several separate full-array operations), which both removes sinogram-sized
  temporaries and reduces the number of separate multi-GPU collective allocations.

## Two-stage preprocessing / reconstruction

- New helpers (`save_preprocessing` / `load_preprocessing`) and example scripts let you preprocess a scan
  to disk once and then reconstruct from it in a separate run.  This is useful for debugging (inspect or
  reuse the preprocessed sinogram) and lets a large, memory-tight reconstruction start in a clean process.

## Simpler data generation

- `generate_demo_data` and the Shepp-Logan phantom generator now take a single `devices` argument, build
  large phantoms across the available devices automatically, and always return NumPy arrays.  This
  removes a confusing pair of arguments and a case where generating large demo data could exceed one
  GPU's memory.  A new "Data Generation" section in the docs describes `generate_demo_data`.

## Other

- FBP/FDK reconstructions keep intermediate data on-device across all geometries, avoiding an
  unnecessary host round-trip during the filter step.
- Dependency pins were tightened so installation always selects compatible jax/jaxlib versions.
- A deprecated phantom-generation method and obsolete diagnostic tooling were removed.

## Behavior changes

- Reconstruction and projection calls with the default output now return a NumPy array (previously a
  single-GPU device array).  Pass `output_sharded=True` to keep the sharded device form for further
  on-device work.
- The data-generation functions now always return NumPy arrays (some previously returned device arrays).
- `TomographyModel.gen_modified_3d_sl_phantom` (deprecated) was removed; use
  `mbirjax.generate_3d_shepp_logan_low_dynamic_range` instead.
- Installation now requires a compatible jax/jaxlib pair (a version floor was added to the dependencies).

## Validation

- Full test suite green on CPU (multi-device) and on GPU.
- The 2048³ cone-beam reconstruction was confirmed on an 8-GPU node, and the two-overlapping-halves path
  was confirmed on a 4-GPU node where the single-pass reconstruction does not fit.
