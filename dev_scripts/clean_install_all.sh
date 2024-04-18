#!/bin/bash
# This script installs everything from scratch

#####
# Update the cluster host names, modules, and jax installation as needed, here and in
# get_demo_data_server.sh
#####
GPUCLUSTER="gilbreth"
CPUCLUSTER="negishi"

if [[ "$HOSTNAME" == *"$GPUCLUSTER"* ]]; then
  module load  anaconda/2020.11-py38
  echo "$GPUCLUSTER setting"
  conda config --add pkgs_dirs /scratch/$GPUCLUSTER/$USERNAME/.conda/pkgs
  CONDA_ENVS_PATH="/scratch/$GPUCLUSTER/$USERNAME/.conda/envs"
  conda config --add envs_dirs /scratch/$GPUCLUSTER/$USERNAME/.conda/envs
fi
if [[ "$HOSTNAME" == *"$CPUCLUSTER"* ]]; then
  module load  anaconda/2020.11-py38
  echo "$CPUCLUSTER setting"
  conda config --add pkgs_dirs /scratch/$CPUCLUSTER/$USERNAME/.conda/pkgs
  CONDA_ENVS_PATH="/scratch/$CPUCLUSTER/$USERNAME/.conda/envs"
  conda config --add envs_dirs /scratch/$CPUCLUSTER/$USERNAME/.conda/envs
fi

source install_conda_environment.sh

if [[ "$HOSTNAME" == *"gilbreth"* ]]; then
  #  pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  # To install lower version of jax (say v0.4.13) incase of XLA parallel compilation warnings use the following
   pip install --upgrade "jax[cuda12_local]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
