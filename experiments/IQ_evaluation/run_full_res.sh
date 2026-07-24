#!/bin/bash
# Run all IQ evaluation cases at full resolution (no detector or view
# downsampling) under the 'full_res' tag. Long runtime. Extra args are
# passed to run_recons.py, e.g. ./run_full_res.sh --overwrite
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
python run_recons.py --all --full-res --tag full_res "$@"
