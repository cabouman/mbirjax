#!/bin/bash
# This script destroys the conda environment for this pacakge and reinstalls it.
cd ..
/bin/rm -r docs/build
/bin/rm -r dist
/bin/rm -r mbirjax.egg-info
/bin/rm -r build
cd dev_scripts

# Create and activate new conda environment
# First check if the target environment is active and deactivate if so
NAME=mbirjax
if [ "$CONDA_DEFAULT_ENV"==$NAME ]; then
    conda deactivate
fi
yes | conda env remove --name $NAME
yes | conda env create --name $NAME --file ../environment.yml
conda activate $NAME

yes | conda env update --file ../demo/environment.yml --prune

echo
echo "Use 'conda activate" $NAME "' to activate this environment."
echo

