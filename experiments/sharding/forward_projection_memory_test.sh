#!/bin/bash

DATA_OUTPUT_FILEPATH="logs/forward_projection_memory.txt"

for view in 256; do
  for row in 256; do
    for channel in 256; do
      echo "Running with shape=($view, $row, $channel)"
      python3 forward_projection_memory.py "$view" "$row" "$channel" "$DATA_OUTPUT_FILEPATH"
    done
  done
done