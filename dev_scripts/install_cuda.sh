#!/bin/bash
# Exit on error
set -e

# =========================
# Config
# =========================

# Versions for URL determined at this link
# https://developer.nvidia.com/cuda-downloads/?target_os=Linux&target_arch=x86_64&Distribution=Rocky&target_version=9&target_type=runfile_local
CUDA_VERSION="13.0.0"
CUDA_FULL_VERSION="13.0.0_580.65.06"

CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_FULL_VERSION}_linux.run"

# Install & module paths
INSTALL_DIR="/depot/bouman/apps/cuda/${CUDA_VERSION}"
MODULE_DIR="/depot/bouman/apps/modules/cuda"
MODULEFILE="${MODULE_DIR}/${CUDA_VERSION}"

# =========================
# Prep
# =========================
echo "=== Cleaning previous installation in $INSTALL_DIR"
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# =========================
# Download
# =========================
echo "=== Downloading CUDA ${CUDA_VERSION}"
RUNFILE_NAME=$(basename "$CUDA_RUNFILE_URL")
wget "$CUDA_RUNFILE_URL" -O "$RUNFILE_NAME"

# =========================
# Install (user prefix)
# =========================
echo "=== Installing CUDA Toolkit into ${INSTALL_DIR}"
chmod +x "$RUNFILE_NAME"

# Try direct install into prefix; if NVIDIA changes flags, we fall back to extract + local installer.
if ./"$RUNFILE_NAME" --silent --toolkit --override --installpath="$INSTALL_DIR"; then
    echo "=== CUDA installed to $INSTALL_DIR"
else
    echo "=== Direct install flags failed; trying extractor route"
    EXTRACT_DIR="${INSTALL_DIR}/.extract"
    mkdir -p "$EXTRACT_DIR"
    ./"$RUNFILE_NAME" --extract="$EXTRACT_DIR"
    bash "${EXTRACT_DIR}/cuda-installer.sh" --silent --toolkit --override --installpath="$INSTALL_DIR"
    rm -rf "$EXTRACT_DIR"
fi

# Some installers install into ${INSTALL_DIR}/cuda-<ver>. Normalize to $INSTALL_DIR if needed.
if [ -d "${INSTALL_DIR}/cuda-${CUDA_VERSION}" ]; then
    echo "=== Normalizing directory layout"
    rsync -a "${INSTALL_DIR}/cuda-${CUDA_VERSION}/" "${INSTALL_DIR}/"
    rm -rf "${INSTALL_DIR}/cuda-${CUDA_VERSION}"
fi

# Cleanup installer
rm -f "$RUNFILE_NAME"

# =========================
# Modulefile
# =========================
echo "=== Creating modulefile directory at $MODULE_DIR"
mkdir -p "$MODULE_DIR"

echo "=== Writing modulefile to $MODULEFILE"
cat << 'EOF' > "$MODULEFILE"
#%Module1.0
# CUDA module
proc ModulesHelp { } {
    puts stderr "CUDA Toolkit (local install)"
}
module-whatis "CUDA Toolkit (user-local)"

# Resolve this modulefile's versioned root from its path
# Expect layout: /.../modules/cuda/<version>
set me [file normalize [info script]]
set version [file tail $me]
set root [file normalize [file join [file dirname $me] ../.. cuda $version]]

# Common CUDA env
setenv CUDA_HOME $root
setenv CUDA_ROOT $root

# Paths
prepend-path PATH            $root/bin
prepend-path LD_LIBRARY_PATH $root/lib64
prepend-path LIBRARY_PATH    $root/lib64
prepend-path CPATH           $root/include
EOF

echo -e "\n=== ✅ Done!"

echo -e "\nTo make the module available run"
echo "    module use /depot/bouman/apps/modules"
echo -e "\nTo load CUDA ${CUDA_VERSION}, use:\n    module load cuda/${CUDA_VERSION}"
