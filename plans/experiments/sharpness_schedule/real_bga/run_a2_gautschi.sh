#!/bin/bash
# A2 downsampled-BGA runs on gautschi (1x H100).  Submit from real_bga/:
#   sbatch run_a2_gautschi.sh
# Outputs to /scratch/gautschi/buzzard/sharpness_schedule/a2_bga; log lands here.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J a2_bga
#SBATCH -o a2_bga_%j.log

# Same library discipline as run_a1_gautschi.sh: the sharpness_main venv =
# regression env's jax/CUDA stack + editable origin/main worktree.
source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== A2 BGA ==="
$PY -u a2_bga.py
