#!/bin/bash

while [ ${#CONDA_DEFAULT_ENV} -gt 0 ]; do
    conda deactivate
done

rm -rf ~/.conda/* ~/.cache/conda/* ~/.cache/pip/* ~/.local/lib/python*

# Also remove pip entry-point shims in ~/.local/bin whose packages (~/.local/lib/python*) were
# just deleted above; a leftover shim (e.g. pytest) shadows the conda env's copy with a stale
# interpreter. Only python-shebang files are removed, keeping non-python tools (claude, micro).
find ~/.local/bin -maxdepth 1 -type f -exec grep -l '^#!.*python' {} + 2>/dev/null | xargs -r rm --