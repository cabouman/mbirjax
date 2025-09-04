#!/bin/bash

# Exit on error
set -e

# Versions for URL determined at this link
# https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/
CUDNN_VERSION="9.12.0"
CUDNN_FULL_VERSION="9.12.0.46"
CUDA_VERSION="cuda13"

# Define paths
INSTALL_DIR="/depot/bouman/apps/cudnn/${CUDNN_VERSION}"
MODULE_DIR="/depot/bouman/apps/modules/cudnn"
MODULEFILE="${MODULE_DIR}/${CUDNN_VERSION}"

# Construct names/URL from components
ARCHIVE_BASE="cudnn-linux-x86_64-${CUDNN_FULL_VERSION}_${CUDA_VERSION}-archive"
ARCHIVE_EXT=".tar.xz"
ARCHIVE_FILE="${ARCHIVE_BASE}${ARCHIVE_EXT}"
CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${ARCHIVE_FILE}"

echo "=== Cleaning previous installation in $INSTALL_DIR"
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "=== Downloading cuDNN ${CUDNN_VERSION}"
wget "$CUDNN_URL"

echo "=== Extracting archive"
tar -xf "$ARCHIVE_FILE"

echo "=== Organizing files"
mv "${ARCHIVE_BASE}/include" include
mv "${ARCHIVE_BASE}/lib" lib
rm -rf "${ARCHIVE_BASE}"*
rm -f "$ARCHIVE_FILE"

echo "=== Creating modulefile directory at $MODULE_DIR"
mkdir -p "$MODULE_DIR"

echo "=== Writing modulefile to $MODULEFILE"
cat << EOF > "$MODULEFILE"
#%Module1.0
proc ModulesHelp { } {
    puts stderr "cuDNN ${CUDNN_VERSION} (local install)"
}
module-whatis "cuDNN ${CUDNN_VERSION} for ${CUDA_VERSION^^} (user local)"

set root ${INSTALL_DIR}
prepend-path LD_LIBRARY_PATH \$root/lib
prepend-path CPATH          \$root/include
EOF

echo -e "\n=== ✅ Done!"

echo -e "\nTo make the module available run"
echo "    module use /depot/bouman/apps/modules"
echo -e "\nTo load cuDNN ${CUDNN_VERSION}, use:\n    module load cudnn/${CUDNN_VERSION}"
