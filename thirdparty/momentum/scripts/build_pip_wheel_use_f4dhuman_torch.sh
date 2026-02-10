#!/usr/bin/env bash

# Build Momentum wheel using PyTorch from the existing f4dhuman env.
# Build deps are installed into a separate build env to avoid altering f4dhuman.

set -eo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Please install Miniconda/Anaconda and re-run." >&2
  exit 1
fi

# Parameters
TORCH_ENV_NAME="${MOMENTUM_TORCH_ENV:-f4dhuman}"
BUILD_ENV_NAME="${MOMENTUM_BUILD_ENV:-momentum_f4dhuman_build}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
TORCH_MIN_PY312="${MOMENTUM_TORCH_MIN_PY312:-2.5.1}"
TORCH_MAX_PY312="${MOMENTUM_TORCH_MAX_PY312:-2.6}"
CUDA_VERSION="${MOMENTUM_CUDA_VERSION:-12.1}"
SKIP_CUDA_DEV="${MOMENTUM_SKIP_CUDA_DEV:-0}"

eval "$(conda shell.bash hook)"

TORCH_PREFIX="$(conda env list | awk -v env="${TORCH_ENV_NAME}" '($1==env){print $2} ($1=="*" && $2==env){print $3}' | head -n 1)"
if [[ -z "${TORCH_PREFIX}" ]] || [[ ! -d "${TORCH_PREFIX}" ]]; then
  echo "Unable to locate torch env '${TORCH_ENV_NAME}'. Set MOMENTUM_TORCH_ENV." >&2
  exit 1
fi

TORCH_PY="${TORCH_PREFIX}/bin/python"
if [[ ! -x "${TORCH_PY}" ]]; then
  echo "Torch env python not found at ${TORCH_PY}" >&2
  exit 1
fi

TORCH_VERSION="$("${TORCH_PY}" -c "import torch; print(torch.__version__)")"
TORCH_CUDA_VERSION="$("${TORCH_PY}" -c "import torch; print(torch.version.cuda or 'None')")"
TORCH_CUDA_AVAILABLE="$("${TORCH_PY}" -c "import torch; print(torch.cuda.is_available())")"
echo "Using torch from '${TORCH_ENV_NAME}': ${TORCH_VERSION} (CUDA ${TORCH_CUDA_VERSION}, available=${TORCH_CUDA_AVAILABLE})"
if [[ "${TORCH_CUDA_VERSION}" == "None" ]] || [[ "${TORCH_CUDA_AVAILABLE}" != "True" ]]; then
  echo "ERROR: f4dhuman PyTorch is CPU-only. Install CUDA PyTorch in that env first." >&2
  exit 1
fi

PY_VER="$("${TORCH_PY}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"

# Use CUDA version from parameters (defaults to 12.1, same as the working new_env script)
echo "Using CUDA version: ${CUDA_VERSION} (override with MOMENTUM_CUDA_VERSION)"

# Create/Update build env
if [[ "$FORCE_RECREATE" == "1" ]] && conda env list | awk '{print $1}' | grep -qx "${BUILD_ENV_NAME}"; then
  echo "Removing existing conda environment '${BUILD_ENV_NAME}'..."
  conda env remove -n "${BUILD_ENV_NAME}" -y
fi

if ! conda env list | awk '{print $1}' | grep -qx "${BUILD_ENV_NAME}"; then
  echo "Creating conda env '${BUILD_ENV_NAME}' with python=${PY_VER}..."
  conda create -y -n "${BUILD_ENV_NAME}" "python=${PY_VER}"
fi

echo "Activating build env '${BUILD_ENV_NAME}'..."
conda activate "${BUILD_ENV_NAME}"

# CRITICAL: Prevent Python from using packages from ~/.local/lib/pythonX.Y/site-packages
# This ensures we use ONLY the conda environment's packages
export PYTHONNOUSERSITE=1
echo "Set PYTHONNOUSERSITE=1 to isolate conda environment from user site-packages"

# Determine install command (prefer mamba if available)
if command -v mamba &> /dev/null; then
    INSTALL_CMD="mamba"
else
    echo "mamba not found, falling back to conda (this might be slower)..."
    INSTALL_CMD="conda"
fi

INSTALL_CMD="conda"

# Install PyTorch with CUDA support into build env (use 2.5.1 like the working script)
echo "Installing PyTorch 2.5.1 with CUDA ${CUDA_VERSION} support into build env..."
"$INSTALL_CMD" install -y -c pytorch -c nvidia \
    "pytorch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1" \
    "pytorch-cuda=${CUDA_VERSION}"

echo "First verification of PyTorch CUDA..."
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

echo "PyTorch version: ${TORCH_VERSION}"
echo "CUDA version: ${TORCH_CUDA_VERSION}"
echo "CUDA available: ${CUDA_AVAILABLE}"

if [[ "${CUDA_AVAILABLE}" != "True" ]]; then
  echo "ERROR: PyTorch CUDA was replaced/broken during dependency installation!" >&2
  echo "Please investigate the conda dependency resolution." >&2
  exit 1
fi
echo "PyTorch CUDA verification passed!"


# Update CUDA_VERSION based on what was actually installed in the build env
CUDA_VERSION="${TORCH_CUDA_VERSION}"

if [[ "${SKIP_CUDA_DEV}" != "1" ]]; then
  # Install CUDA dev tools from nvidia channel (match torch CUDA version by default).
  echo "Installing CUDA ${CUDA_VERSION} development tools from nvidia channel..."
  "$INSTALL_CMD" install -y -c nvidia -c conda-forge \
    "cuda-cudart-dev=${CUDA_VERSION}.*" \
    "cuda-cudart-static=${CUDA_VERSION}.*" \
    "cuda-nvcc=${CUDA_VERSION}.*" \
    "cuda-nvrtc-dev=${CUDA_VERSION}.*" \
    "libcublas-dev=${CUDA_VERSION}.*" \
    "cuda-cccl=${CUDA_VERSION}.*"
else
  echo "Skipping CUDA dev tool install (MOMENTUM_SKIP_CUDA_DEV=1)."
fi

echo "Installing build tools into '${BUILD_ENV_NAME}'..."
"$INSTALL_CMD" install -y -c conda-forge \
  cmake \
  ninja \
  pybind11 \
  scikit-build-core \
  "gxx_linux-64=12.*" \
  "gcc_linux-64=12.*" \
  ceres-solver \
  cli11 \
  dispenso \
  drjit-cpp \
  fx-gltf \
  openfbx \
  ezc3d \
  eigen \
  fmt \
  nlohmann_json \
  indicators \
  re2 \
  "librerun-sdk=0.23.3" \
  spdlog \
  urdfdom \
  kokkos \
  "rerun-sdk=0.23.3" \
  ms-gsl \
  gflags \
  glog \
  boost-cpp \
  zlib \
  openssl \
  libnvjitlink

echo "Installing Python packaging tools via pip..."
pip install jinja2 patchelf auditwheel setuptools-scm setuptools

echo "Second verification of PyTorch CUDA..."
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

echo "PyTorch version: ${TORCH_VERSION}"
echo "CUDA version: ${CUDA_VERSION}"
echo "CUDA available: ${CUDA_AVAILABLE}"

if [[ "${CUDA_AVAILABLE}" != "True" ]]; then
  echo "ERROR: PyTorch CUDA was replaced/broken during dependency installation!" >&2
  echo "Please investigate the conda dependency resolution." >&2
  exit 1
fi
echo "PyTorch CUDA verification passed!"

# Remove any pip-installed NVIDIA CUDA wheels that can conflict with conda CUDA libs.
REMOVE_PIP_NVIDIA="${MOMENTUM_REMOVE_PIP_NVIDIA:-1}"
if [[ "${REMOVE_PIP_NVIDIA}" == "1" ]]; then
  python - <<'PY'
import importlib.metadata as md
import subprocess
import sys

pkgs = [d.metadata["Name"] for d in md.distributions() if d.metadata["Name"].lower().startswith("nvidia-")]
if pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *pkgs])
PY
fi

# nvcc expects NVVM under targets/x86_64-linux; conda places it at $CONDA_PREFIX/nvvm.
if [[ ! -e "${CONDA_PREFIX}/targets/x86_64-linux/nvvm" ]] && [[ -d "${CONDA_PREFIX}/nvvm" ]]; then
  ln -s "${CONDA_PREFIX}/nvvm" "${CONDA_PREFIX}/targets/x86_64-linux/nvvm"
fi

# Setup build environment variables
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_TOOLKIT_ROOT_DIR="${CONDA_PREFIX}"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
# Set CUDAHOSTCXX to ensure nvcc uses the correct host compiler from the build env
export CUDAHOSTCXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
if [[ ! -f "${CUDAHOSTCXX}" ]]; then
  # Fall back to plain g++ if wrapper doesn't exist
  export CUDAHOSTCXX="$(which g++)"
fi
echo "Using CUDAHOSTCXX: ${CUDAHOSTCXX}"

if [[ -d "${CONDA_PREFIX}/targets/x86_64-linux" ]]; then
  export CUDAToolkit_ROOT="${CONDA_PREFIX}/targets/x86_64-linux"
  export CUDA_HOME="${CONDA_PREFIX}/targets/x86_64-linux"
  export CUDA_TOOLKIT_ROOT_DIR="${CONDA_PREFIX}/targets/x86_64-linux"
fi
if [[ -f "${CONDA_PREFIX}/lib/libcudart.so" ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDA_CUDART_LIBRARY=${CONDA_PREFIX}/lib/libcudart.so -DCUDAToolkit_ROOT=${CUDA_TOOLKIT_ROOT_DIR}"
fi
# Explicitly set CUDA host compiler for CMake to ensure nvcc uses the correct g++
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_CUDA_HOST_COMPILER=${CUDAHOSTCXX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# CRITICAL: Set LIBRARY_PATH for the linker to find CUDA static libraries during compilation
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/stubs:${LIBRARY_PATH:-}"
# Also set CPATH for header files
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"

# Point CMake to torch from the build env (PyTorch is now installed here)
TORCH_CMAKE_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${TORCH_CMAKE_PATH}"
export Torch_DIR="${CONDA_PREFIX}/share/cmake/Torch"
if [[ ! -f "${Torch_DIR}/TorchConfig.cmake" ]] && [[ -d "${TORCH_CMAKE_PATH}/Torch" ]]; then
  export Torch_DIR="${TORCH_CMAKE_PATH}/Torch"
fi

# Match Torch C++ string ABI for all targets (pybind modules and core libs).
TORCH_ABI="$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
export MOMENTUM_GLIBCXX_ABI="${TORCH_ABI}"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI} ${CXXFLAGS:-}"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DMOMENTUM_GLIBCXX_ABI=${TORCH_ABI} -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}"

# Build C++ dependencies from source with matching ABI if Torch uses the old C++ string ABI.
if [[ "${TORCH_ABI}" == "0" ]]; then
  DEPS_PREFIX="${PWD}/build/deps-install"
  mkdir -p "${DEPS_PREFIX}"

  # Clear any cached compiler paths that might point to wrong env
  unset CMAKE_C_COMPILER CMAKE_CXX_COMPILER CC CXX

  # Clean deps build directories if FORCE_RECREATE is set (they may have cached wrong compiler paths)
  if [[ "${FORCE_RECREATE}" == "1" ]]; then
    echo "Cleaning old deps build directories..."
    rm -rf "${PWD}/build/ezc3d-src/build" 2>/dev/null || true
    rm -rf "${PWD}/build/console_bridge-src/build" 2>/dev/null || true
    rm -rf "${PWD}/build/urdfdom_headers-src/build" 2>/dev/null || true
    rm -rf "${PWD}/build/urdfdom-src/build" 2>/dev/null || true
    rm -rf "${PWD}/build/abseil-cpp-src/build" 2>/dev/null || true
    rm -rf "${PWD}/build/re2-src/build" 2>/dev/null || true
    rm -rf "${PWD}/build/dispenso-src/build" 2>/dev/null || true
  fi

  # Common CMake flags for all dependencies - explicitly use gcc/g++ from PATH
  COMMON_CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${DEPS_PREFIX} -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI} -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}"

  # Build ezc3d
  EZC3D_SRC="${PWD}/build/ezc3d-src"
  if [[ ! -d "${EZC3D_SRC}" ]]; then
    echo "Cloning ezc3d source..."
    git clone --depth 1 https://github.com/pyomeca/ezc3d.git "${EZC3D_SRC}"
  fi
  echo "Building ezc3d (ABI=${TORCH_ABI})..."
  cmake -S "${EZC3D_SRC}" -B "${EZC3D_SRC}/build" ${COMMON_CMAKE_FLAGS}
  cmake --build "${EZC3D_SRC}/build" --target install -j"$(nproc)"

  # Build console_bridge (urdfdom dependency)
  CONSOLE_BRIDGE_SRC="${PWD}/build/console_bridge-src"
  if [[ ! -d "${CONSOLE_BRIDGE_SRC}" ]]; then
    echo "Cloning console_bridge source..."
    git clone --depth 1 https://github.com/ros/console_bridge.git "${CONSOLE_BRIDGE_SRC}"
  fi
  echo "Building console_bridge (ABI=${TORCH_ABI})..."
  cmake -S "${CONSOLE_BRIDGE_SRC}" -B "${CONSOLE_BRIDGE_SRC}/build" ${COMMON_CMAKE_FLAGS}
  cmake --build "${CONSOLE_BRIDGE_SRC}/build" --target install -j"$(nproc)"

  # Build urdfdom_headers (urdfdom dependency)
  URDFDOM_HEADERS_SRC="${PWD}/build/urdfdom_headers-src"
  if [[ ! -d "${URDFDOM_HEADERS_SRC}" ]]; then
    echo "Cloning urdfdom_headers source..."
    git clone --depth 1 https://github.com/ros/urdfdom_headers.git "${URDFDOM_HEADERS_SRC}"
  fi
  echo "Building urdfdom_headers (ABI=${TORCH_ABI})..."
  cmake -S "${URDFDOM_HEADERS_SRC}" -B "${URDFDOM_HEADERS_SRC}/build" ${COMMON_CMAKE_FLAGS}
  cmake --build "${URDFDOM_HEADERS_SRC}/build" --target install -j"$(nproc)"

  # Build urdfdom
  URDFDOM_SRC="${PWD}/build/urdfdom-src"
  if [[ ! -d "${URDFDOM_SRC}" ]]; then
    echo "Cloning urdfdom source..."
    git clone --depth 1 https://github.com/ros/urdfdom.git "${URDFDOM_SRC}"
  fi
  echo "Building urdfdom (ABI=${TORCH_ABI})..."
  cmake -S "${URDFDOM_SRC}" -B "${URDFDOM_SRC}/build" ${COMMON_CMAKE_FLAGS} \
    -DCMAKE_PREFIX_PATH="${DEPS_PREFIX}" \
    -Dconsole_bridge_DIR="${DEPS_PREFIX}/lib/cmake/console_bridge" \
    -Durdfdom_headers_DIR="${DEPS_PREFIX}/lib/cmake/urdfdom_headers"
  cmake --build "${URDFDOM_SRC}/build" --target install -j"$(nproc)"

  # Build abseil-cpp (dependency of re2) with old ABI
  ABSL_SRC="${PWD}/build/abseil-cpp-src"
  if [[ ! -d "${ABSL_SRC}" ]]; then
    echo "Cloning abseil-cpp source..."
    git clone --depth 1 --branch 20250512.0 https://github.com/abseil/abseil-cpp.git "${ABSL_SRC}"
  fi
  echo "Building abseil-cpp (ABI=${TORCH_ABI})..."
  cmake -S "${ABSL_SRC}" -B "${ABSL_SRC}/build" ${COMMON_CMAKE_FLAGS} \
    -DABSL_BUILD_TESTING=OFF -DABSL_PROPAGATE_CXX_STD=ON
  cmake --build "${ABSL_SRC}/build" --target install -j"$(nproc)"

  # Build re2 with old ABI (using our absl)
  RE2_SRC="${PWD}/build/re2-src"
  if [[ ! -d "${RE2_SRC}" ]]; then
    echo "Cloning re2 source..."
    git clone --depth 1 https://github.com/google/re2.git "${RE2_SRC}"
  fi
  echo "Building re2 (ABI=${TORCH_ABI})..."
  cmake -S "${RE2_SRC}" -B "${RE2_SRC}/build" ${COMMON_CMAKE_FLAGS} -DRE2_BUILD_TESTING=OFF \
    -DCMAKE_PREFIX_PATH="${DEPS_PREFIX}"
  cmake --build "${RE2_SRC}/build" --target install -j"$(nproc)"

  # Build dispenso with old ABI
  DISPENSO_SRC="${PWD}/build/dispenso-src"
  if [[ ! -d "${DISPENSO_SRC}" ]]; then
    echo "Cloning dispenso source..."
    git clone --depth 1 https://github.com/facebookincubator/dispenso.git "${DISPENSO_SRC}"
  fi
  echo "Building dispenso (ABI=${TORCH_ABI})..."
  cmake -S "${DISPENSO_SRC}" -B "${DISPENSO_SRC}/build" ${COMMON_CMAKE_FLAGS} -DDISPENSO_BUILD_TESTS=OFF
  cmake --build "${DISPENSO_SRC}/build" --target install -j"$(nproc)"

  # Update paths for all dependencies - deps-install MUST come first to override conda packages
  export CMAKE_PREFIX_PATH="${DEPS_PREFIX}:${CMAKE_PREFIX_PATH}"
  export LD_LIBRARY_PATH="${DEPS_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${DEPS_PREFIX}/lib:${LIBRARY_PATH:-}"
  # Also add to CPATH to ensure headers are found
  export CPATH="${DEPS_PREFIX}/include:${DEPS_PREFIX}/include/dispenso/third-party/moodycamel:${CPATH:-}"

  # Patch rpaths for deps shared objects to avoid conda env conflicts
  if command -v patchelf >/dev/null 2>&1; then
    echo "Patching deps rpaths to avoid conda conflicts..."
    find "${DEPS_PREFIX}/lib" -name 'libabsl_*.so*' -print0 2>/dev/null | xargs -0 -P 8 -n 1 patchelf --set-rpath "${DEPS_PREFIX}/lib" 2>/dev/null || true
    if [[ -f "${DEPS_PREFIX}/lib/libre2.so.11" ]]; then
      patchelf --set-rpath "${DEPS_PREFIX}/lib" "${DEPS_PREFIX}/lib/libre2.so.11" 2>/dev/null || true
    fi
  fi
fi

# Generate pyproject.toml variants
echo "Generating pyproject.toml variants..."
python scripts/generate_pyproject.py \
  --torch-min-py312 "${TORCH_MIN_PY312}" \
  --torch-max-py312 "${TORCH_MAX_PY312}"

# Determine variant (CPU or GPU)
VARIANT="gpu"
PY_SUFFIX=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

echo "Building ${VARIANT} wheel for Python ${PY_VER} using torch from '${TORCH_ENV_NAME}'..."

cp pyproject.toml pyproject.toml.bak
cp "pyproject-pypi-${VARIANT}.toml" pyproject.toml 2>/dev/null || cp "pyproject-pypi-${VARIANT}-py${PY_SUFFIX}.toml" pyproject.toml

# Clean old build artifacts but preserve deps-install
rm -rf dist
# Only remove pymomentum build artifacts, keep deps
rm -rf build/cp*
mkdir -p dist

# Build CMAKE_ARGS with deps prefix if we built deps from source
# CRITICAL: Always include ABI flags in CMAKE_ARGS - these MUST be present for all builds
ABI_FLAGS="-DMOMENTUM_GLIBCXX_ABI=${TORCH_ABI} -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}"

if [[ -d "${PWD}/build/deps-install" ]]; then
  DEPS_PREFIX="${PWD}/build/deps-install"
  # Put DEPS_PREFIX first in CMAKE_PREFIX_PATH to override conda packages
  export CMAKE_PREFIX_PATH="${DEPS_PREFIX}:${CMAKE_PREFIX_PATH}"
  # Add explicit include paths for concurrentqueue.h and other third-party headers
  export CMAKE_INCLUDE_PATH="${DEPS_PREFIX}/include:${DEPS_PREFIX}/include/dispenso/third-party/moodycamel:${CMAKE_INCLUDE_PATH:-}"

  # CRITICAL: Hide conda's dispenso to force CMake to use our old-ABI version
  # The conda dispenso is missing concurrentqueue.h and uses new ABI
  if [[ -d "${CONDA_PREFIX}/include/dispenso" ]]; then
    echo "Temporarily hiding conda dispenso headers..."
    mv "${CONDA_PREFIX}/include/dispenso" "${CONDA_PREFIX}/include/dispenso.conda.bak" 2>/dev/null || true
  fi

  export CMAKE_ARGS="${ABI_FLAGS} -DMOMENTUM_ENABLE_FBX_SAVING=OFF -DMOMENTUM_ENABLE_SIMD=OFF -DMOMENTUM_USE_SYSTEM_GOOGLETEST=ON -DMOMENTUM_USE_SYSTEM_PYBIND11=OFF -DMOMENTUM_USE_SYSTEM_RERUN_CPP_SDK=ON -DBUILD_SHARED_LIBS=OFF -DMOMENTUM_BUILD_RENDERER=OFF -Ddrjit_DIR=${CONDA_PREFIX}/share/cmake/drjit -DCMAKE_PREFIX_PATH=${DEPS_PREFIX}\;${CONDA_PREFIX} -Durdfdom_DIR=${DEPS_PREFIX}/lib/cmake/urdfdom -Dezc3d_DIR=${DEPS_PREFIX}/lib/cmake/ezc3d -Dre2_DIR=${DEPS_PREFIX}/lib/cmake/re2 -DDispenso_DIR=${DEPS_PREFIX}/lib/cmake/Dispenso-1.4.0 -DCMAKE_INSTALL_RPATH=${DEPS_PREFIX}/lib:${CONDA_PREFIX}/lib:\\\$ORIGIN/../torch/lib -DCMAKE_BUILD_RPATH=${DEPS_PREFIX}/lib:${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF"
else
  export CMAKE_ARGS="${ABI_FLAGS} -DMOMENTUM_ENABLE_FBX_SAVING=OFF -DMOMENTUM_ENABLE_SIMD=OFF -DMOMENTUM_USE_SYSTEM_GOOGLETEST=ON -DMOMENTUM_USE_SYSTEM_PYBIND11=OFF -DMOMENTUM_USE_SYSTEM_RERUN_CPP_SDK=ON -DBUILD_SHARED_LIBS=OFF -DMOMENTUM_BUILD_RENDERER=OFF -Ddrjit_DIR=${CONDA_PREFIX}/share/cmake/drjit -DCMAKE_INSTALL_RPATH=${CONDA_PREFIX}/lib:\\\$ORIGIN/../torch/lib -DCMAKE_BUILD_RPATH=${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF"
fi

# Function to restore conda dispenso headers
restore_conda_dispenso() {
  if [[ -d "${CONDA_PREFIX}/include/dispenso.conda.bak" ]]; then
    echo "Restoring conda dispenso headers..."
    mv "${CONDA_PREFIX}/include/dispenso.conda.bak" "${CONDA_PREFIX}/include/dispenso" 2>/dev/null || true
  fi
}

# Set trap to restore on exit
trap restore_conda_dispenso EXIT

echo "Running pip wheel..."
pip wheel . --no-deps --no-build-isolation --wheel-dir=dist

mv pyproject.toml.bak pyproject.toml

# Restore conda dispenso headers
restore_conda_dispenso
trap - EXIT

echo "Repairing wheel with auditwheel..."
# Add deps lib path to LD_LIBRARY_PATH for auditwheel to find them
if [[ -d "${PWD}/build/deps-install/lib" ]]; then
  export LD_LIBRARY_PATH="${PWD}/build/deps-install/lib:${LD_LIBRARY_PATH:-}"
fi
auditwheel repair \
    --exclude 'libtorch*.so' --exclude 'libc10*.so' \
    --exclude 'libcu*.so*' --exclude 'libnv*.so*' --exclude 'libmkl*.so' \
    --exclude 'libgomp*.so*' --exclude 'libstdc++*.so*' --exclude 'libgcc_s*.so*' \
    dist/pymomentum_*.whl -w dist/repaired

echo "Done. Wheel is in dist/repaired/"

WHEEL_FILE=$(find dist -maxdepth 1 -name "*.whl" | head -n 1)
echo "Generated wheel: ${WHEEL_FILE}"

# Optional: install wheel into the build env and patch rpaths for local testing.
INSTALL_WHEEL="${MOMENTUM_INSTALL_WHEEL:-1}"
if [[ "${INSTALL_WHEEL}" == "1" ]]; then
  echo "Installing wheel into '${BUILD_ENV_NAME}'..."
  pip install --force-reinstall --no-deps "${WHEEL_FILE}"

  # Patch pymomentum rpaths to prefer local deps over conda packages
  DEPS_PREFIX="${PWD}/build/deps-install"
  if command -v patchelf >/dev/null 2>&1; then
    echo "Patching pymomentum rpaths to prefer local deps..."
    for so in "${CONDA_PREFIX}"/lib/python*/site-packages/pymomentum/*.so; do
      if [[ -f "${so}" ]]; then
        if [[ -d "${DEPS_PREFIX}" ]]; then
          patchelf --set-rpath "${DEPS_PREFIX}/lib:${CONDA_PREFIX}/lib:\$ORIGIN/../torch/lib" "${so}" 2>/dev/null || true
        else
          patchelf --set-rpath "${CONDA_PREFIX}/lib:\$ORIGIN/../torch/lib" "${so}" 2>/dev/null || true
        fi
      fi
    done
  fi
fi

# Final summary
echo ""
echo "============================================"
echo "Build Summary:"
echo "  Build Environment: ${BUILD_ENV_NAME}"
echo "  Torch Environment: ${TORCH_ENV_NAME}"
echo "  PyTorch: ${TORCH_VERSION}"
echo "  CUDA: ${CUDA_VERSION}"
echo "  Wheel: ${WHEEL_FILE}"
echo "============================================"
