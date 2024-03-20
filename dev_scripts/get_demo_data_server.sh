#!/bin/bash

#####
# Change the cluster, source, and target info as needed, then uncomment the cp and scp lines below
#####
GPUCLUSTER="gilbreth"
CPUCLUSTER="brown"

source1="/depot/bouman/users/nhalavic/pcd_data"
source2="/depot/bouman/users/mnagare/calibration_module_demo_data"

# Note that the target directories are relative to dev_scripts
target_dir1="../demo/data"
target_dir2="../data"

mkdir -p "$target_dir1"
mkdir -p "$target_dir2"

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

if [[ "$HOSTNAME" == *"$GPUCLUSTER"* || "$HOSTNAME" == *"$CPUCLUSTER"* ]]; then
  account_name=""
  command="cp"
else
  echo "Enter your Purdue Career Account user name:"
  read cluster_user_name
  account_name="$cluster_user_name@gilbreth.rcac.purdue.edu:"
  command="scp"
fi

echo "${red}   Copying $source1 to $target_dir1 ${reset}"
$command -r "$account_name$source1" "$target_dir1"
echo "${red}   Copying $source2 to $target_dir2 ${reset}"
$command -r "$account_name$source2" "$target_dir2"

