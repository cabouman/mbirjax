#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J headroom-e0b
#SBATCH -o /home/buzzard/headroom/logs/e0b-%j.log
#
# E0b: per-kernel time shares (fwd sort share, back kernel composition) via jax trace --
# the nsys replacement (nsys is unavailable in batch shells here; `module` is not
# initialized under #!/bin/bash).
#
#     sbatch plans/experiments/projector_kernels/e0b_kernel_share.slurm

set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom

echo "=== provenance ==="
git log --oneline -1
python -c "import mbirjax, jax; print('mbirjax:', mbirjax.__file__); print('jax:', jax.__version__)"
nvidia-smi -L

python -u plans/experiments/projector_kernels/e0b_kernel_share.py
echo "=== E0b done ==="
