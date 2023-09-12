---
title: "Getting Started"
linkTitle: "Getting Started"
weight: 2
keywords: ["MLIR", "LLVM", "Model Compilation", "Getting Started", "Guidelines"]
description: "This document covers the preparation of the development environment, quick start and basic tutorials of ByteIR."
---
## Quick Start

ByteIR has three major independent components, and three major frontends. 
Each has a different config, and different dependency. 
This chapter will get you started with ByteIR using a simple PyTorch executable example.

### Prerequisites
ByteIR PyTorch frontend requires a specific version of PyTorch, and other tools to build from the source code. 
Please refer this [link](https://github.com/bytedance/byteir/blob/main/frontends/torch-frontend/requirements.txt).

Similarly, ByteIR compiler also requires specific LLVM and Python versions to build from the source code.
Please refer the [page](https://github.com/bytedance/byteir/tree/main/compiler). 

ByteIR runtime requires a LLVM version to build from the source code. The LLVM version is typically the same as the compiler. 

### Build the ByteIR PyTorch frontend

```bash
git clone https://github.com/bytedance/byteir.git
cd byteir/frontends/torch-frontend

# prepare python environment and torch-mlir dependency
bash scripts/prepare.sh

cmake -S . \
      -B ./build \
      -GNinja \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DPython3_EXECUTABLE=$(which python3)

cmake --build ./build --target all
# torch_frontend-*.whl in ./build/torch-frontend/python/dist/
```

### Run the ByteIR PyTorch frontend example
```bash
PYTHONPATH=./build/python_packages/ python3 examples/inference/infer_resnet.py
```

### Build the ByteIR compiler 

Make sure to apply possible patches for submodules
```bash
bash /path_to_byteir/scripts/apply_patches.sh
```

Then build the compiler if using cmake with Ninja
```bash
mkdir /path_to_byteir/build
cd /path_to_byteir/build/

# build ByteIR
cmake ../cmake/ -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DLLVM_EXTERNAL_LIT=lit_executatble_location # or using $(which lit), this is optional for external lit 

cmake --build . --target all
```

Or build the compiler if using Visual Studio.
Here, we use Visual Studio 2019 as an example.

```bash
mkdir /path_to_byteir/build
cd /path_to_byteir/build/

# build ByteIR
cmake ../cmake/ -G "Visual Studio 16 2019" -A x64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DLLVM_EXTERNAL_LIT=lit_location # this is optional for external lit 

cmake --build . --target all
```

## Test the ByteIR compiler  
This command runs all ByteIR compiler unit tests:
```bash
cmake --build . --target check-byteir
```

### Build the ByteIR runtime 

Build the runtime if using cmake with Ninja
```bash
mkdir /path_to_runtime/build
cd /path_to_runtime/build/

# build runtime
cmake ../cmake/ -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
                -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
                -Dbrt_USE_CUDA=On

cmake --build . --target all --target install
```

Or build the runtime if using Visual Studio.
```bash
mkdir /path_to_runtime/build
cd /path_to_runtime/build/

# build runtime
cmake ..\cmake -G "Visual Studio 16 2019" -A x64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
                -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
                -Dbrt_USE_CUDA=On

cmake --build . --target all --target install
```

### Test the ByteIR runtime

Test the runtime with the runtime unit tests
```bash
cd /path_to_runtime/build
./bin/brt_test_all
```
