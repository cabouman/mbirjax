#!/bin/bash
# This script just installs mbirjax_sandbox along with all requirements
# for the package, demos, and documentation.
# However, it does not remove the existing installation of mbirjax_sandbox.

conda activate mbirjax_sandbox
cd ..
pip install -r requirements.txt
pip install -e .
pip install -r demo/requirements.txt
pip install -r docs/requirements.txt 
cd dev_scripts

