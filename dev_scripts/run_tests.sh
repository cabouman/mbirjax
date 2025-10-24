#!/bin/bash
# This script runs all the pytests to be used before deployment

cd ..

pytest 
pytest -m data_dependent

cd dev_scripts
