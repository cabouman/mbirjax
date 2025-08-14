#!/bin/bash

DATA_OUTPUT_FILEPATH="../output/forward_project.csv"
LOG_OUTPUT_FILEPATH="../output/forward_project.log"

echo "Saving log to $LOG_OUTPUT_FILEPATH"

for size in 128 256 512 1024; do
  for num_gpus in 1 2 4 8; do
    for module_type in 'cone' 'parallel'; do
      echo "Running with module_type=$module_type, size=$size, and gpus=$num_gpus"
      python3 forward_projection.py "$size" "$num_gpus" "$module_type" "$DATA_OUTPUT_FILEPATH"
    done
  done
done
