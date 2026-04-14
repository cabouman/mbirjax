#!/bin/bash
# check_cuda13_compat.sh
# Usage: source check_cuda13_compat.sh   (sets CUDA13_COMPATIBLE=0 or 1)
#        ./check_cuda13_compat.sh         (exits with code 0 or 1)

_cuda13_check_finish() {
    local code=$1
    # If sourced, use 'return' to avoid closing the terminal
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        CUDA13_COMPATIBLE=$code
        return $code
    else
        exit $code
    fi
}

echo "=== CUDA 13 Compatibility Check ==="
echo ""

# Check nvidia-smi exists
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is a GPU driver installed?"
    _cuda13_check_finish 1
fi

# Get driver version string
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "Driver version:        $DRIVER_VERSION"

# Get driver's reported max CUDA version via nvidia-smi
CUDA_VERSION_SMI=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -1)
echo "Max CUDA (nvidia-smi): $CUDA_VERSION_SMI"

# Get driver's actual reported version via libcuda
DRIVER_CUDA_INT=$(python3 -c "
import ctypes, sys
try:
    libcuda = ctypes.CDLL('libcuda.so.1')
    version = ctypes.c_int(0)
    ret = libcuda.cuDriverGetVersion(ctypes.byref(version))
    if ret != 0:
        print('ERROR')
        sys.exit(1)
    print(version.value)
except Exception as e:
    print('ERROR')
    sys.exit(1)
" 2>/dev/null)

if [[ "$DRIVER_CUDA_INT" == "ERROR" || -z "$DRIVER_CUDA_INT" ]]; then
    echo "ERROR: Could not query libcuda.so.1 — is the driver installed and libcuda in ldconfig?"
    _cuda13_check_finish 1
fi

DRIVER_MAJOR=$(( DRIVER_CUDA_INT / 1000 ))
DRIVER_MINOR=$(( (DRIVER_CUDA_INT % 1000) / 10 ))
echo "Driver CUDA version:   ${DRIVER_MAJOR}.${DRIVER_MINOR} (raw: $DRIVER_CUDA_INT)"

REQUIRED=13000
echo ""
echo "Required for CUDA 13:  >= ${REQUIRED} (i.e., driver CUDA version >= 13.0)"
echo ""

if (( DRIVER_CUDA_INT >= REQUIRED )); then
    echo "✓ COMPATIBLE — driver supports CUDA 13.0"
    _cuda13_check_finish 0
else
    echo "✗ NOT COMPATIBLE — driver only supports up to CUDA ${DRIVER_MAJOR}.${DRIVER_MINOR}"
    echo ""
    echo "  To use JAX with this node, install the cuda12 variant:"
    echo "    pip install --force-reinstall 'jax[cuda12]'"
    _cuda13_check_finish 1
fi
