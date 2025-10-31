#!/bin/bash

DATA_OUTPUT_FILEPATH="logs/recon_memory.txt"

for view in 256; do
  for row in 256; do
    for channel in 256; do
      echo "Running with shape=($view, $row, $channel)"
      python3 recon_memory.py "$view" "$row" "$channel" "$DATA_OUTPUT_FILEPATH"
    done
  done
done