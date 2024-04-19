#!/bin/bash
# This script installs everything from scratch

# Steps to recover from jax/cuda mismatch
# 1. Delete env from .conda directory.  This should be achieved with `conda remove env --name $NAME --all` but check
# in /scratch/gilbreth/$USER/.conda/envs and ~/.conda/envs
# 2. Delete all nvidia and jax related library dirs from /home/$USER/.local
# 3. Delete numpy, scipy, importlib_metadata and other related dirs downloaded with jax installation command from .local
# 4. Delete ~/.cache
#####
# Update the cluster host names, modules, and jax installation as needed, here and in
# get_demo_data_server.sh
#####
GPUCLUSTER="gilbreth"
CPUCLUSTER="negishi"

if [[ "$HOSTNAME" == *"$GPUCLUSTER"* ]]; then
  module load  anaconda/2020.11-py38
  echo "$GPUCLUSTER setting"
  conda config --add pkgs_dirs /scratch/$GPUCLUSTER/$USER/.conda/pkgs
  CONDA_ENVS_PATH="/scratch/$GPUCLUSTER/$USER/.conda/envs"
  conda config --add envs_dirs /scratch/$GPUCLUSTER/$USER/.conda/envs
fi
if [[ "$HOSTNAME" == *"$CPUCLUSTER"* ]]; then
  module load  anaconda/2020.11-py38
  echo "$CPUCLUSTER setting"
  conda config --add pkgs_dirs /scratch/$CPUCLUSTER/$USER/.conda/pkgs
  CONDA_ENVS_PATH="/scratch/$CPUCLUSTER/$USER/.conda/envs"
  conda config --add envs_dirs /scratch/$CPUCLUSTER/$USER/.conda/envs
fi

source install_conda_environment.sh

if [[ "$HOSTNAME" == *"gilbreth"* ]]; then
  #  pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  # To install lower version of jax (say v0.4.13) incase of XLA parallel compilation warnings use the following
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#  pip install --upgrade "jax[cuda12]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#  pip install jaxlib==0.4.13+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  # Ref: https://github.com/google/jax/issues/18027
  echo " "
  echo "To run with jax on Gilbreth, first load the cuda module using "
  echo "    module load cudnn/cuda-12.1_8.9"
  echo " "
else
  pip install --upgrade "jax[cpu]"
fi

source install_package.sh
source build_docs.sh

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

echo " "
echo "Use"
echo "${red}   conda activate mbirjax   ${reset}"
echo "to activate the conda environment."
echo " "

if [[ "$HOSTNAME" == *"gilbreth"* ]]; then
  echo " "
  echo "Verify the versions of anaconda, jax, and cuda as specified in clean_install_all.sh"
  echo " "
fi
