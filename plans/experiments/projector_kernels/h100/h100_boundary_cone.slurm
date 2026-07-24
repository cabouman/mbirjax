#!/bin/bash
# H100 BOUNDARY RE-RUN (gautschi).  The A100 campaign closed every boundary case on
# sm80, but each of those cells is still UNTESTED on the architecture that actually
# ships the constants.  This is the cone half: CONE_LC 256/512 were never measured on
# H100 (its sweep stopped at 128 with the curve still improving).
#
# PREDICTION (pre-registered): lc=256 is WORSE on H100 too.  The cone-back L2 working
# set is ~view_chunk * channels * (lc + 2*bp) * 4 B -- about 34.5 MB at lc=128 and
# ~68 MB at lc=256.  H100 has 50 MB of L2, so 256 should overflow it just as it does on
# A100's 40 MB (measured there: +6.7%).  If lc=256 instead WINS on H100, the L2 model is
# wrong and a real speedup has been sitting unclaimed on the primary platform.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J h100_cone
#SBATCH -o /scratch/gautschi/buzzard/h100_tuning/logs/h100_cone_%j.out
#SBATCH -e /scratch/gautschi/buzzard/h100_tuning/logs/h100_cone_%j.err
set -u
PY=$HOME/.conda/envs/mbirjax/bin/python
export A100_OUT_DIR=/scratch/gautschi/buzzard/h100_tuning
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git -C "$HOME/PycharmProjects/mbirjax" log --oneline -1
$PY -u $A100_OUT_DIR/scripts/a100_cone_sweep.py
echo "exit: $?"
