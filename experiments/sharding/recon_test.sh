#!/bin/bash

DATA_OUTPUT_FILEPATH="../output/recon.csv"
LOG_OUTPUT_FILEPATH="../output/recon.log"

echo "Saving log to $LOG_OUTPUT_FILEPATH"

for size in 128 256 512 1024; do
  for num_gpus in 8 4 2 1; do
    for module_type in 'cone' 'parallel'; do
      echo "Running with module_type=$module_type, size=$size, and gpus=$num_gpus"
      python3 recon.py "$size" "$num_gpus" "$module_type" "$DATA_OUTPUT_FILEPATH"
    done
  done
done