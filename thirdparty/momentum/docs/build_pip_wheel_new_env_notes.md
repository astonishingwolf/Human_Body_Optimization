# Momentum Build Script - New Environment with PyTorch 2.5.1 CUDA 12.1

## Task Description

Create a build script (`scripts/build_pip_wheel_new_env.sh`) that:
1. Creates a **new** conda environment (not reusing existing one)
2. Installs PyTorch 2.5.1 with CUDA 12.1 support
3. Installs all momentum build dependencies
4. **Ensures CUDA-enabled PyTorch is NOT replaced with CPU version** during dependency installation
5. Builds the pymomentum wheel

## Key Challenges Solved

### 1. PyTorch CUDA Being Replaced
**Problem**: When installing conda packages, some dependencies could pull in CPU-only PyTorch.

**Solution**:
- Install PyTorch **first** before any other packages
- Use `PYTHONNOUSERSITE=1` to prevent Python from using `~/.local` packages
- Pin CUDA dev tools to exact version 12.1 to match pytorch-cuda=12.1

### 2. `--freeze-installed` Causing Conflicts
**Problem**: Using `--freeze-installed` flag caused CUDA package version conflicts.

**Solution**: Removed `--freeze-installed` and instead:
- Used nvidia channel for CUDA 12.1 dev tools (compatible with pytorch-cuda=12.1)
- C++ dependencies don't have PyTorch dependencies so they're safe to install normally

### 3. Missing CUDA Static Libraries
**Problem**: CMake/linker couldn't find `libcudart_static.a` and `libcudadevrt.a`.

**Solution**: Added packages:
- `cuda-cudart-static=12.1.*` - provides `libcudart_static.a`
- `cuda-cccl=12.1.*` - CUDA C++ Core Libraries

Also added environment variables:
```bash
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/stubs:${LIBRARY_PATH:-}"
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"
```

### 4. Python Version Requirement
**Problem**: pymomentum requires Python >= 3.12.

**Solution**: Changed default `PYTHON_VERSION` from 3.10 to 3.12.

## Current Progress

| Step | Status |
|------|--------|
| Create conda environment | ✅ Done |
| Install PyTorch 2.5.1 + CUDA 12.1 | ✅ Done |
| Verify PyTorch CUDA | ✅ Passed (`torch.cuda.is_available() = True`) |
| Install CUDA dev tools (12.1) | ✅ Done |
| Install build tools | ✅ Done |
| Install C++ dependencies | ✅ Done |
| Install Python packaging tools | ✅ Done |
| Final PyTorch CUDA verification | ✅ Passed |
| Build wheel (`pip wheel`) | 🔄 In Progress (was running, interrupted) |
| Repair wheel (`auditwheel`) | ⏳ Pending |

## How to Resume

### Option 1: Continue Build in Existing Environment

```bash
cd /data1/users/jianjinx/F4DHuman-priv/thirdparty/momentum

source /home/jianjinx/data/miniconda3/etc/profile.d/conda.sh
conda activate momentum_build

export PYTHONNOUSERSITE=1
export CONDA_PREFIX="/home/jianjinx/data/miniconda3/envs/momentum_build"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_TOOLKIT_ROOT_DIR="${CONDA_PREFIX}"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/stubs:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/stubs:${LIBRARY_PATH:-}"
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"

TORCH_CMAKE_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${TORCH_CMAKE_PATH}"

# Copy GPU pyproject
cp pyproject-pypi-gpu.toml pyproject.toml

# Clean and build
rm -rf dist build
mkdir -p dist

export CMAKE_ARGS="-DMOMENTUM_ENABLE_FBX_SAVING=OFF -DMOMENTUM_ENABLE_SIMD=OFF -DMOMENTUM_USE_SYSTEM_GOOGLETEST=ON -DMOMENTUM_USE_SYSTEM_PYBIND11=OFF -DMOMENTUM_USE_SYSTEM_RERUN_CPP_SDK=ON -DBUILD_SHARED_LIBS=OFF -DMOMENTUM_BUILD_RENDERER=OFF -Ddrjit_DIR=${CONDA_PREFIX}/share/cmake/drjit"

pip wheel . --no-deps --no-build-isolation --wheel-dir=dist

# After build succeeds, repair wheel
auditwheel repair \
    --exclude 'libtorch*.so' --exclude 'libc10*.so' \
    --exclude 'libcu*.so*' --exclude 'libnv*.so*' --exclude 'libmkl*.so' \
    dist/pymomentum_*.whl -w dist/repaired
```

### Option 2: Run Full Script from Scratch

```bash
cd /data1/users/jianjinx/F4DHuman-priv/thirdparty/momentum

# Remove existing environment and start fresh
FORCE_RECREATE=1 ./scripts/build_pip_wheel_new_env.sh
```

### Option 3: Use Different Environment Name

```bash
MOMENTUM_CONDA_ENV=momentum_build2 ./scripts/build_pip_wheel_new_env.sh
```

## Script Location

`/data1/users/jianjinx/F4DHuman-priv/thirdparty/momentum/scripts/build_pip_wheel_new_env.sh`

## Environment Details

- **Environment name**: `momentum_build`
- **Location**: `/home/jianjinx/data/miniconda3/envs/momentum_build`
- **Python**: 3.12
- **PyTorch**: 2.5.1
- **CUDA**: 12.1
- **PyTorch CUDA available**: Yes (verified)

## Verification Commands

```bash
# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Available: {torch.cuda.is_available()}')"

# Verify static libraries exist
ls -la $CONDA_PREFIX/lib/libcudart_static.a $CONDA_PREFIX/lib/libcudadevrt.a

# Verify nvcc
nvcc --version
```

## Files Modified

1. `scripts/build_pip_wheel_new_env.sh` - New script created
2. `scripts/generate_pyproject.py` - Used to generate `pyproject-pypi-gpu.toml`

## Known Issues

1. Build takes a long time (~10-20 minutes) due to C++ compilation
2. The script uses `mamba` if available (faster), otherwise falls back to `conda`
