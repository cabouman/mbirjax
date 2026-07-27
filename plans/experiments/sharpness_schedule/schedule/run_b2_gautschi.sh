#!/bin/bash
# b2: Phase B full-resolution confirmation (fresh baselines + winner, 2 GPUs).
# Set WINNER in b2_fullres.py first.  Submit from schedule/:
#   sbatch run_b2_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH -t 04:00:00
#SBATCH -J sharp_b2
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/%x_%j.log
# 2 GPUs: host-memory headroom (the ai partition forbids --mem; memory scales with
# GPU count) and the recon shards; the run_io hook is memory-lean (disk snapshots,
# f32 metrics, z_step=3).

source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== b2 full-res confirmation ==="
$PY -u b2_fullres.py
