#!/bin/bash
# Mechanism probes (coarse_late + q2_control) on gautschi (1x H100).
# Submit from mechanism/:  sbatch run_probes_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J sharp_probes
#SBATCH -o probes_%j.log

# Same library discipline as run_a1_gautschi.sh (sharpness_main venv = regression
# jax/CUDA stack + editable origin/main worktree).
source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== mechanism probes ==="
$PY -u probes.py
