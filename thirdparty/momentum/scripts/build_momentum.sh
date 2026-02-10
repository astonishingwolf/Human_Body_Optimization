#!/bin/bash
# Build pymomentum with old ABI and custom ezc3d

set -e

EZC3D_PREFIX="/data1/users/jianjinx/F4DHuman-priv/thirdparty/momentum/build/ezc3d-install"
TORCH_CMAKE_PATH="/home/jianjinx/data/miniconda3/envs/f4dhuman/lib/python3.12/site-packages/torch/share/cmake"
CONDA_PREFIX="${CONDA_PREFIX:-/home/jianjinx/data/miniconda3/envs/f4dhuman}"

export CMAKE_PREFIX_PATH="${EZC3D_PREFIX}:${CONDA_PREFIX}:${TORCH_CMAKE_PATH}"
export LD_LIBRARY_PATH="${EZC3D_PREFIX}/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CMAKE_ARGS="-DMOMENTUM_ENABLE_FBX_SAVING=OFF -DMOMENTUM_ENABLE_SIMD=OFF -DMOMENTUM_USE_SYSTEM_GOOGLETEST=ON -DMOMENTUM_USE_SYSTEM_PYBIND11=OFF -DMOMENTUM_USE_SYSTEM_RERUN_CPP_SDK=ON -DBUILD_SHARED_LIBS=OFF -DMOMENTUM_BUILD_RENDERER=OFF -Ddrjit_DIR=${CONDA_PREFIX}/share/cmake/drjit -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"

echo "Building pymomentum with:"
echo "  CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
echo "  CXXFLAGS=$CXXFLAGS"

rm -rf build/cp312*
pip install -e . --no-deps --no-build-isolation
