#!/bin/bash
# This script installs mbirjax from scratch, but there may still be cached packaages.
# To do a really clean install, first run
#  source deep_clean_conda.sh

#####
# Update the cluster host names, modules, and jax installation as needed, here and in
# get_demo_data_server.sh
#####
NAME="mbirjax"
GILBRETH="gilbreth"
NEGISHI="negishi"
GAUTSCHI="gautschi"
PYTHON_VERSION="3.12"

# Remove any previous builds
cd ..
/bin/rm -r docs/build
/bin/rm -r dist
/bin/rm -r "$NAME.egg-info"
/bin/rm -r build
cd dev_scripts

# Create and activate new conda environment
# First check if the target environment is active and deactivate if so

# Deactivate all conda environments
while [ ${#CONDA_DEFAULT_ENV} -gt 0 ]; do
    conda deactivate
done
output=$(conda remove --name $NAME --all 2>&1)
if echo "$output" | grep -q "DirectoryNotACondaEnvironmentError:"; then
  conda activate $NAME
  CUR_ENV_PATH=$CONDA_PREFIX
  conda deactivate
  rm -rf $CUR_ENV_PATH
fi

# Install based on the host
# Gilbreth (gpu)
if [[ "$HOSTNAME" == *"$GILBRETH"* ]]; then
  echo "Installing on Gilbreth"
  module --force purge
  # The following two lines are required in late 2024 to interface jax to an older version of cuda in use on gilbreth.
  # After gilbreth/cuda is updated, then the pattern for gautschi could be used here.
  module load jax/0.4.31
  yes | conda create -n $NAME python=3.11.7
  conda activate $NAME
  pip install -e ..[cuda12]
# Gautschi (gpu)
elif [[ "$HOSTNAME" == *"$GAUTSCHI"* ]]; then
  module load anaconda
  yes | conda create -n $NAME python="$PYTHON_VERSION"
  conda activate $NAME
  pip install -e ..[cuda12]
# Negishi (cpu)
elif [[ "$HOSTNAME" == *"$NEGISHI"* ]]; then
  module load anaconda
  yes | conda create -n $NAME python="$PYTHON_VERSION"
  conda activate $NAME
  pip install -e ..
# Other (cpu)
else
  yes | conda create -n $NAME python="$PYTHON_VERSION"
  conda activate $NAME
  pip install -e ..
fi

pip install ..[test]
pip install ..[docs]
source build_docs.sh

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

echo " "
echo "Use"
echo "${red}   conda activate mbirjax   ${reset}"
echo "to activate the conda environment."
echo " "
