#!/bin/bash

while [ ${#CONDA_DEFAULT_ENV} -gt 0 ]; do
    conda deactivate
done

rm -rf ~/.conda/* ~/.cache/conda/* ~/.cache/pip/* ~/.local/lib/python*