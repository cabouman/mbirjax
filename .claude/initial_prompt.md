Post-compaction refresher for the GPU-headroom KERNEL session (rewritten 2026-07-13;
the original campaign-opening prompt is in this file's git history).

We are on branch `greg/gpu_headroom`, working ONLY in the dedicated worktrees
(local `…/Research/mbirjax_headroom`, gautschi `~/PycharmProjects/mbirjax_headroom`,
env `mbirjax_headroom`).  **Claude may commit and push on this branch ONLY** (all
other branches stage-only).  The local `mbirjax` conda env resolves the MAIN worktree
— only pytest (via conftest) and the cluster env run the headroom code.

## Current task

Increment 6: the CONE FORWARD kernel.  Entry point =
`plans/projector_kernels/gpu_headroom_findings.md`, section
"Cone forward kernel — DESIGN OPENING" (three candidate architectures; the design
pass must sharpen the traffic models and pick one — a Greg checkpoint — then the
established pattern: design → adversarial panel → GPU spike → integrate behind
TilePolicy flags → gates).

## State (all shipped + gated on this branch, 2026-07-13)

Pallas kernels in `mbirjax/_pallas_kernels.py` (its module docstring = the dispatch
map + the hard constraints; READ IT).  Increments 1–5 complete: parallel back
(9.1× n=1; band-adopted 7.5×/7.4× at n=2/4), parallel forward (all sizes, no guard;
band-adopted 3.2×/3.1×), cone fused-vfan back (3.9×/9.9×/6.0× at n=1/2/4 — the
campaign-opening cone anti-scaling is DEAD), both coeff powers everywhere
(`_PALLAS_BACK_COEFF_POWERS`).  Parallel VCD n=2: 97.9 → 34.9 s.  Value doctrine:
per-call gates at rel 1e-5 (gradient) / 1e-4 (Hessian); **trajectory max-norm
comparisons only pass bitwise-identical code** (intrinsic edge-voxel conditioning,
proven by an XLA-vs-XLA control) — recon-path equivalence is gated OCCASIONALLY
(never nightly) by `w2_inc5_convergence.py`: real data, depot parity refs,
control-calibrated NRMSE band.

## Concurrent sessions (shared branch + shared local worktree)

- Docs session (Opus): owns `docs/source/dev_projector_kernels.rst` + the new-GPU
  section; I stopped editing the pallas docs.
- Soak session (Opus, chip task_6e8f84ac): repeated-run stability of the gate
  harnesses + demo sanity; stops and reports on reproducible anomalies.
- Discipline: `git pull --rebase --autostash` before every commit; stage ONLY my
  files; never `git add -A`; small frequent commits.

## Workflow reminders

Discussion-first for designs and library changes (experiment scripts in
`plans/experiments/projector_kernels/` are pre-approved once a direction is agreed);
1–2-line self-contained progress notes during every long stretch; "variants" never
"arms"; delegation policy = routine/settled work → Opus sessions or model='opus'
subagents, Fable for design/diagnosis/adversarial review (Greg's quota).

Measurement rules that have bitten: seed np.random before ANY off/on VCD comparison;
peak memory = GPU entries of get_memory_stats only (trailing CPU entry is RSS); one
config per subprocess; MBIRJAX_DISABLE_PALLAS=1 is the off config; never construct
jit/pallas_call per call; launch shapes from array shapes only; no host syncs in
per-call paths; bare pallas_call on Hopper = the WRONG (Mosaic) backend — Triton
compiler_params are backend selection; verify tokens (`(pallas: …)` on the device
line / `get_compute_config()`) before trusting any timing — gates have been vacuous
twice.

## Standing cluster context

gautschi: sbatch `-A bouman -p ai -q normal`, **14 CPUs per GPU enforced**; slurm
preamble = `set -u; source ~/load_conda_cuda.sh; source "$(conda info --base)/etc/profile.d/conda.sh";
conda activate mbirjax_headroom`; logs `/home/buzzard/headroom/logs/`; big outputs →
`/scratch/gautschi/buzzard/` (HOME quota 25 GB fails jobs SILENTLY — empty-log
failure ⇒ check myquota); depot refs
`/depot/bouman/data/mbirjax_metrics/slice_parity/refs/`.  Monitor jobs with
background `squeue` loops; profiler traces crash on cone-1024³ n=1 windows.
