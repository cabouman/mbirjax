# Multi-GPU reconstruction (parallel beam) + usability updates

This PR adds automatic multi-GPU reconstruction for `ParallelBeamModel`, with supporting
controls and several usability changes.  Existing scripts run unchanged.

## Multi-GPU reconstruction

On a machine with multiple GPUs, `recon()` (and `fbp_recon`, `prox_map`, and the
projectors) now divides the work across all GPUs automatically — nothing in the script
changes.  The primary benefit is **capacity**: per-GPU memory drops by roughly 1/N, so
larger volumes fit; time improves too (≈2.2× on 2 GPUs at 1024×1023×1024, matching the
single-GPU result to float precision; validated up to 8 GPUs at 2048³).  Any number of
views and slices works with any GPU count.

With `model = ParallelBeamModel(sinogram_shape, angles)`, the following features are available:

- `model.device_summary` — reports what was chosen, e.g. `'2 x GPU (sharded) (slices padded 1023->1024)'`
- `model.configure_devices(n)` — explicit control when you don't want the automatic choice
- `model.prepare_sino_for_devices(sino)` — optional: pay the host→GPU transfer once when
  running several reconstructions on the same large sinogram
- `output_sharded=True` in `model.recon()` and other reconstruction/projection methods keeps results on the GPUs
  for on-device pipelines (e.g. PnP); the default returns plain arrays exactly as before

## Other features

- **`set_view_parameters(...)`**: change the view angles (or other per-view parameters)
  without rebuilding the projectors — milliseconds instead of a recompile.  Enables
  view-sweeping algorithms and motion-correction experiments; `vcls` now uses it (and now
  works with `ParallelBeamModel`, fixing a crash).
- **Bounded multi-GPU memory**: peak memory no longer grows with iteration count.
- **Clear errors instead of silent failures**: incompatible device configurations, stale
  prepared arrays, and multi-GPU requests on unsupported geometries all fail fast with
  instructions.

## Behavior changes relative to prerelease

- **Multi-GPU machines: reconstruction parallelizes by default.**  Results agree with the
  single-GPU result within the usual float tolerance (~1e-4 for iterative recon).
- **CPU: 2-way parallel by default** — faster above ~200³, a few seconds slower for very
  small reconstructions; results shift within the same float tolerance.
- `use_gpu='sinograms'` (the hybrid CPU/GPU mode) is removed; remaining values are
  `'automatic'`/`'full'`/`'none'`.  For oversized single-GPU problems, use
  `split_sino_recon` or more GPUs.
- Log files move from `./logs/` (scattered into whatever directory a script ran from) to
  `~/.mbirjax/logs/`.
- The JIT compilation cache moves from `/tmp/jax_cache` to `~/.mbirjax/jax_cache`, and is
  only set if you haven't configured your own.

## Scope and known limits

- Multi-GPU is **parallel beam only** for now; cone/translation/multiaxis behave exactly as
  in prerelease (single device), and the port is the next phase of work.
- Single machine only (no multi-node).
- The user-guide page for multi-GPU usage lands with the geometry port.

## For testers

The most valuable things to try: run your existing scripts unchanged on 1/2/4 GPUs and
compare results and times; use odd view/slice counts; check `model.device_summary`; and
report anything where time or memory surprises you (please include the `device_summary`
output in reports).

Test suite: `pytest tests/` (~2.5 min on CPU); multi-device specifics:
`MBIRJAX_NUM_CPU_DEVICES=4 pytest tests/sharding/`.
