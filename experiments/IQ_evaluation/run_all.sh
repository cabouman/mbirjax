#!/bin/bash
# Rebuild all IQ evaluation baselines. Extra args are passed to run_recon.py,
# e.g. ./run_all.sh --overwrite
cd "$(dirname "$0")"
python run_recon.py --all "$@"
