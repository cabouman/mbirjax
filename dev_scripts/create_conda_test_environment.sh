#!/bin/bash
# Create and activate new conda environment
# First check if the target environment is active and deactivate if so
NEW_NAME="test"

if [ "$CONDA_DEFAULT_ENV" = "$NEW_NAME" ]; then
    conda deactivate
fi

conda env remove --name "$NEW_NAME" -y
conda create --name "$NEW_NAME" python=3.10 -y
conda activate "$NEW_NAME"

echo
echo "Use 'conda activate $NEW_NAME' to activate the test environment."
echo
