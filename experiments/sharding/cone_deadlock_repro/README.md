# Cone-beam multi-GPU deadlock repro

Diagnoses the NCCL "Acquire clique … not all threads arrived" hang seen running a **cone-beam**
sharded recon on 8 GPUs at 2048³ (parallel beam at the same config works).  The hang appears at
the **first VCD subset update**, after the FDK init, error-sino init, and Hessian all complete.

## Background (why this is the right test)

MBIRJAX projection is collective-free by design (thread pool + `make_array_from_single_device_arrays`).
The only multi-device collectives come from XLA's GSPMD auto-inserting them for the VCD line-search
**scalar reductions** over the view-sharded sinogram (`get_forward_lin_quad`, the `alpha` sums). Those
reductions are shared with parallel beam, so the cone-only failure points to either a **cone-specific
divergence** at the first subset update or a **scale/resource** problem (cone's whole-cylinder kernels
are much heavier). The failing 2048³ case divides evenly across 8 (2048/8 = 256), so it is *not* a
padding-shape issue.

## Files

- `repro_recon.py`  — one (geometry, size) recon per process, with timestamped phase markers; device
  count comes from `CUDA_VISIBLE_DEVICES` (auto-sharding, same path as a real run).
- `run_sweep.sh`    — sweeps geometry × size × device-count, each in its own `timeout`-wrapped process
  (a deadlock is killed and recorded, not allowed to block the rest); dumps HLO for selected device
  counts; prints a summary table.
- `cone_deadlock.slurm` — batch submit (8 × H100 on the `ai` partition); sources the node preamble and
  activates the conda env, mirroring the regression harness.

## Run it

```bash
# from your cluster checkout of mbirjax:
sbatch experiments/sharding/cone_deadlock_repro/cone_deadlock.slurm
```

If this branch isn't installed in the env yet, point at the repo:
```bash
REPRO_MBIRJAX_DIR=$PWD sbatch experiments/sharding/cone_deadlock_repro/cone_deadlock.slurm
```

Defaults: geometries `parallel cone`, sizes `256 512`, device counts `1 2 4 8`, 3 iterations, 420 s
per-config timeout, HLO dumped for the 8-device runs. Override via env, e.g. to probe scale:
```bash
SIZES="256 512 1024" sbatch experiments/sharding/cone_deadlock_repro/cone_deadlock.slurm
```

## Reading the result

- **`summary.txt`** — the config table. `TIMEOUT` = killed on the per-config timeout (likely
  deadlock/hang); `ERR(n)` = exited with an error; `OK` = completed.
- The **decisive comparison**: does `cone s256 n8` `TIMEOUT` while `parallel s256 n8` is `OK`?
  - cone hangs at small size too → **structural** collective bug (then reproduce cheaply and fix).
  - cone only hangs at large size → **scale/resource** (memory/compile); go up in `SIZES` and watch
    `nvidia-smi`.
- **`logs/<config>.log`** — the last phase marker before a `TIMEOUT` pins where it hung (e.g. at
  "starting recon" = the first subset update).
- **HLO** — find the compiled collectives and compare cone vs parallel:
  ```bash
  grep -lE 'all-reduce|collective-permute|all-gather|reduce-scatter' cone_deadlock_out_*/hlo/*/*.txt
  ```

Send `summary.txt`, the hanging config's log, and the HLO dir back for analysis.
