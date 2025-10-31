#!/bin/bash

DATA_OUTPUT_FILEPATH="../output/recon_mem.csv"
LOG_OUTPUT_FILEPATH="../output/recon_mem.log"

echo "Saving log to $LOG_OUTPUT_FILEPATH"

for size in 128 256 512 1024; do
  echo "Running with size=$size"
  python3 recon_memory.py "$size" "$DATA_OUTPUT_FILEPATH"
done