#!/bin/bash
# Run all IQ evaluation cases at the per-case downsampling in dataset_registry.py
# under the 'low_res' tag, for rapid turnaround. Extra args are passed to
# run_recons.py, e.g. ./run_low_res.sh --overwrite
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
python run_recons.py --all --tag low_res "$@"
