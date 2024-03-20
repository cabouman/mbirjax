#!/bin/bash
# This script installs everything from scratch

#####
# Update the cluster host names, modules, and jax installation as needed, here and in
# get_demo_data_server.sh
#####
GPUCLUSTER="gilbreth"
CPUCLUSTER="brown"

if [[ "$HOSTNAME" == *"$GPUCLUSTER"* ]]; then
  module load  anaconda/2020.11-py38
  echo "$GPUCLUSTER setting"
  gcc --version
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

source remove_package.sh
source install_conda_environment.sh

if [[ "$HOSTNAME" == *"gilbreth"* ]]; then
  pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
echo "${red}   source get_demo_data_server.sh   ${reset}"
echo "to download data needed for demos"
echo " "
echo "Use"
echo "${red}   conda activate mbirjax_sandbox   ${reset}"
echo "to activate the conda environment."
echo " "
