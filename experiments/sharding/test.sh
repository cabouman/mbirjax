#!/bin/bash

OUTPUT_FILEPATH="../output/output.csv"

rm -f "$OUTPUT_FILEPATH"
echo "size,num gpus,elapsed time,gpu0 peak bytes,gpu1 peak bytes,gpu2 peak bytes,gpu3 peak bytes,gpu4 peak bytes,gpu5 peak bytes,gpu6 peak bytes,gpu7 peak bytes" >> "$OUTPUT_FILEPATH"

module load proxy
module load conda
module load cuda
conda activate mbirjax

for size in 128 256 512 1024; do
  for gpus in 1 2 4 8; do
    echo "Running with size=$size and gpus=$gpus"
    python3 back_projection.py "$size" "$gpus" "$OUTPUT_FILEPATH"
  done
done