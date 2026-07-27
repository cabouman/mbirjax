#!/bin/bash
# A3 full-res BGA on gautschi (1x H100; resubmit with --gpus-per-node=2 if OOM).
# Submit from real_bga/:  sbatch run_a3_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH -t 04:00:00
#SBATCH -J a3_fullres
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/%x_%j.log
# 2 GPUs: the single-GPU host-memory share (~125G) OOM-killed job 14202362 at
# MaxRSS 130G, and this scheduler forbids --mem (proportional allocation: memory
# scales with GPUs, 14 CPUs required per GPU).  The code now also streams
# snapshots to disk and does volume metrics in f32; with 2 GPUs the recon shards
# (the driver and gathers are sharding-safe) and host memory doubles to ~257G.

# Library: the sharpness_main venv (regression jax/CUDA stack + editable
# origin/main worktree); on-GPU assert per the silent-CPU lesson.
source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== A3 full-res BGA ==="
$PY -u a3_fullres.py
