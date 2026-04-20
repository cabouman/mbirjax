#!/bin/bash
# This script  cleans out the repository, but there may still be cached packaages.
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
PYTHON_VERSION="3.11"

# Remove any previous builds
cd ..
/bin/rm -r docs/build &> /dev/null
/bin/rm -r dist &> /dev/null
/bin/rm -r "$NAME.egg-info" &> /dev/null
/bin/rm -r build &> /dev/null
cd dev_scripts

