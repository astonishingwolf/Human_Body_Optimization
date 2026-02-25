# PyMomentum Build Progress Summary

## Goal



Install full pymomentum (NOT TorchScript) by improving `build_pip_wheel_use_f4dhuman_torch.sh` and pass `test_mhr.py` without using TorchScript.

## Core Problem
**C++ ABI Incompatibility**: PyTorch 2.5.1+cu121 uses the **old C++ ABI** (`_GLIBCXX_USE_CXX11_ABI=0`), but conda-forge packages use the **new ABI** (`=1`). This causes `undefined symbol` errors at runtime when pymomentum tries to load.

### Symbol Mangling Differences
- Old ABI: `_ZN4urdf13parseURDFFileERKSs` (uses `RKSs` for std::string)
- New ABI: `_ZN4urdf13parseURDFFileERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE`

## Progress Made

### 1. Built Dependencies from Source with Old ABI ✅
Modified `scripts/build_pip_wheel_use_f4dhuman_torch.sh` to build these dependencies with `-D_GLIBCXX_USE_CXX11_ABI=0`:
- ezc3d
- console_bridge
- urdfdom_headers
- urdfdom
- re2
- dispenso

All dependencies built successfully and installed to `build/deps-install/`.

### 2. Fixed Momentum Core Library ABI ✅
Added ABI flag handling to `CMakeLists.txt`:
```cmake
# Handle C++ ABI compatibility when building with PyTorch
if(DEFINED MOMENTUM_GLIBCXX_ABI)
  message(STATUS "Setting _GLIBCXX_USE_CXX11_ABI=${MOMENTUM_GLIBCXX_ABI}")
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${MOMENTUM_GLIBCXX_ABI})
endif()
```

Verified `libmomentum_character.a` now has old ABI symbols:
```
000000000000e2a0 T _ZN8momentum24replaceSkeletonHierarchyERKNS_10CharacterTIfEES3_RKSsS5_
```

### 3. Build Script Improvements ✅
- Added `ABI_FLAGS` variable to ensure ABI flags are always included in CMAKE_ARGS
- Added code to temporarily hide conda headers that conflict with custom-built deps
- Added explicit CMAKE_PREFIX_PATH ordering to prioritize custom-built deps

### 4. Wheel Build Successful ✅
- pymomentum wheel builds successfully
- auditwheel repair creates manylinux wheel with bundled libraries

## Current Blocker

### Runtime Library Loading Issue
The built wheel has a **cascading dependency problem**:

1. **urdfdom** - Built with old ABI ✅
2. **re2** - Built with old ABI ✅, but depends on **abseil-cpp**
3. **abseil-cpp** - Still using conda's new ABI version ❌

Error when importing:
```
ImportError: libre2.so.11: undefined symbol: _ZN4absl12lts_2025051219str_format_internal13FormatArgImpl8DispatchISsEEbNS2_4DataENS1_24FormatConversionSpecImplEPv
```

### Attempted Fixes
1. **LD_LIBRARY_PATH** - Didn't work because RPATH in .so files points to conda lib first
2. **Patched RPATH** - Changed to `$ORIGIN:...` but re2 still needs old-ABI abseil
3. **Copied libs to pymomentum package** - Same issue with abseil dependency

## Next Steps

### Option A: Build abseil-cpp from Source (In Progress)
Already added to build script but build was interrupted:
```bash
# Build abseil-cpp (dependency of re2) with old ABI
ABSL_SRC="${PWD}/build/abseil-cpp-src"
git clone --depth 1 --branch 20250512.0 https://github.com/abseil/abseil-cpp.git "${ABSL_SRC}"
cmake -S "${ABSL_SRC}" -B "${ABSL_SRC}/build" ${COMMON_CMAKE_FLAGS} \
  -DABSL_BUILD_TESTING=OFF -DABSL_PROPAGATE_CXX_STD=ON
cmake --build "${ABSL_SRC}/build" --target install -j"$(nproc)"
```

**To continue:**
```bash
cd /data1/users/jianjinx/F4DHuman-priv/thirdparty/momentum
rm -rf build/cp* build/abseil-cpp-src build/re2-src dist
MOMENTUM_SKIP_CUDA_DEV=1 bash scripts/build_pip_wheel_use_f4dhuman_torch.sh
```

### Option B: Static Linking
Modify build to statically link all C++ dependencies into pymomentum modules, avoiding runtime library loading issues.

### Option C: Use Conda's PyTorch with New ABI
Install PyTorch from conda-forge (uses new ABI) instead of pip. This would avoid all ABI issues but may require changes to other parts of f4dhuman.

## Files Modified

1. **`scripts/build_pip_wheel_use_f4dhuman_torch.sh`**
   - Added building of ezc3d, console_bridge, urdfdom_headers, urdfdom, re2, dispenso, abseil-cpp from source
   - Added ABI_FLAGS variable
   - Added conda header hiding/restore logic

2. **`CMakeLists.txt`**
   - Added MOMENTUM_GLIBCXX_ABI handling with add_compile_definitions()

## Verification Commands

```bash
# Check if momentum library uses old ABI
nm build/cp312-cp312-linux_x86_64/libmomentum_character.a | grep replaceSkeletonHierarchy
# Should show: RKSs (old ABI), NOT NSt7__cxx11 (new ABI)

# Check torch ABI
python -c "import torch; print('ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
# Should print: ABI: False (meaning old ABI = 0)

# Test pymomentum import
python -c "import pymomentum.geometry as pym_geometry; print('Success')"
```

## Build Artifacts Location
- Custom deps: `build/deps-install/`
- Wheel: `dist/pymomentum_gpu-*.whl`
- Repaired wheel: `dist/repaired/pymomentum_gpu-*-manylinux*.whl`
- Build logs: `build_log*.txt`
