#!/bin/bash
# This script purges the docs and environment

cd ..
/bin/rm -r docs/build
/bin/rm -r dist
/bin/rm -r mbirjax.egg-info
/bin/rm -r build

pip uninstall mbirjax

cd dev_scripts
