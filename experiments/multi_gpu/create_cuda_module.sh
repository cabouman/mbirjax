#!/bin/bash

# Exit on error
set -e

# Full and short versioning
CUDA_VERSION_FULL="12.9.0"
CUDA_VERSION_SHORT="12.9"

# Define paths
INSTALL_DIR="/depot/bouman/apps/cuda/${CUDA_VERSION_SHORT}"
MODULE_DIR="/depot/bouman/apps/modules/cuda"
MODULEFILE="${MODULE_DIR}/${CUDA_VERSION_SHORT}"


echo "=== Cleaning previous installation in $INSTALL_DIR"
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# List of CUDA components to download and extract
COMPONENTS=(
  "linux/x86_64/cuda-toolkit-linux-x86_64-12.9.1-archive.tar.xz"
  "linux/x86_64/cuda-compiler-linux-x86_64-12.9.1-archive.tar.xz"
  "linux/x86_64/cuda-cudart-linux-x86_64-12.9.1-archive.tar.xz"
)
BASE_URL="https://developer.download.nvidia.com/compute/cuda/redist"

for comp in "${COMPONENTS[@]}"; do
    echo "=== Downloading $comp"
    wget "${BASE_URL}/${comp}"
    echo "=== Extracting $(basename "$comp")"
    tar -xJf "$(basename "$comp")" --strip-components=1 -C "$INSTALL_DIR"
    rm -f "$(basename "$comp")"
done

echo "=== Creating modulefile directory at $MODULE_DIR"
mkdir -p "$MODULE_DIR"

echo "=== Writing modulefile to $MODULEFILE"
cat << EOF > "$MODULEFILE"
#%Module1.0
proc ModulesHelp { } {
    puts stderr "CUDA ${CUDA_VERSION_FULL} (local install)"
}
module-whatis "CUDA ${CUDA_VERSION_FULL} (user local)"

set root ${INSTALL_DIR}
prepend-path PATH          \$root/bin
prepend-path LD_LIBRARY_PATH \$root/lib64
prepend-path CPATH          \$root/include
EOF

echo -e "\n=== ✓ Done!"

echo -e "\nTo make the module available run"
echo "    module use /depot/bouman/apps/modules"
echo -e "\nTo load CUDA ${CUDA_VERSION_SHORT}, use:\n    module load cuda/${CUDA_VERSION_SHORT}"
