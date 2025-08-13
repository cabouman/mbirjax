#!/bin/bash

DATA_OUTPUT_FILEPATH="../output/back_project.csv"
LOG_OUTPUT_FILEPATH="../output/back_project.log"

module load proxy
module load modtree/gpu
module load cuda/12.9.0
module load conda

# load cudnn
source ~/.bashrc
module load cudnn/9.11.0

conda activate mbirjax

echo "Saving log to $LOG_OUTPUT_FILEPATH"

for size in 128 256 512 1024; do
  for num_gpus in 1 2 4 8; do
    echo "Running with size=$size and gpus=$num_gpus"
    python3 back_projection.py "$size" "$num_gpus" "$DATA_OUTPUT_FILEPATH"
  done
done