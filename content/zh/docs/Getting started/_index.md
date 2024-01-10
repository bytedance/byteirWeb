---
title: "快速开始"
linkTitle: "快速开始"
weight: 2
keywords: ["MLIR", "LLVM", "模型编译", "快速开始", "指南"]
description: "本文档涵盖了 ByteIR 的开发环境的准备、快速入门和基础教程。"
---

## 快速上手

ByteIR 有三大独立组件，以及三大前端。
每个都有不同的配置和不同的依赖关系。
本章将使用一个简单的 PyTorch 可执行示例开始使用 ByteIR。

### 前提
ByteIR PyTorch 前端需要特定版本的 PyTorch，以及从源代码构建的其他工具。
请参考此[链接](https://github.com/bytedance/byteir/blob/main/frontends/torch-frontend/requirements.txt)。

同样，ByteIR 编译器也需要特定的 LLVM 和 Python 版本来从源代码构建。
请参考此[页面](https://github.com/bytedance/byteir/tree/main/compiler)。

ByteIR 运行时需要从源代码构建的 LLVM 版本。LLVM 版本通常与编译器相同。

### 构建 ByteIR PyTorch 前端

```bash
git clone https://github.com/bytedance/byteir.git
cd byteir/frontends/torch-frontend

# 准备 python 环境和 torch-mlir 依赖
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

### 运行 ByteIR PyTorch 前段示例
```bash
PYTHONPATH=./build/python_packages/ python3 examples/inference/infer_resnet.py
```

### 构建 ByteIR 编译器

确保应用子模块中可能存在的补丁文件
```bash
bash /path_to_byteir/scripts/apply_patches.sh
```

然后构建编译器。如果使用 cmake 和 Ninja
```bash
mkdir /path_to_byteir/build
cd /path_to_byteir/build/

# 构建 ByteIR
cmake ../cmake/ -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DLLVM_EXTERNAL_LIT=lit_executatble_location # or using $(which lit), this is optional for external lit

cmake --build . --target all
```

或者使用 Visual Studio。
这里我们使用 Visual Studio 2019 示范。

```bash
mkdir /path_to_byteir/build
cd /path_to_byteir/build/

# 构建 ByteIR
cmake ../cmake/ -G "Visual Studio 16 2019" -A x64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DLLVM_EXTERNAL_LIT=lit_location # this is optional for external lit

cmake --build . --target all
```

## 测试 ByteIR 编译器
这条指令执行所有 ByteIR 编译器的单元测试:
```bash
cmake --build . --target check-byteir
```

### 构建 ByteIR 运行时

构建 ByteIR 运行时。 如果使用 cmake 和 Ninja
```bash
mkdir /path_to_runtime/build
cd /path_to_runtime/build/

# 构建 runtime
cmake ../cmake/ -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
                -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
                -Dbrt_USE_CUDA=On

cmake --build . --target all --target install
```

或者使用 Visual Studio。
```bash
mkdir /path_to_runtime/build
cd /path_to_runtime/build/

# 构建 runtime
cmake ..\cmake -G "Visual Studio 16 2019" -A x64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLVM_INSTALL_PATH=path_to_LLVM_installed_or_built_directory \
                -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
                -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
                -Dbrt_USE_CUDA=On

cmake --build . --target all --target install
```

### 测试 ByteIR 运行时

使用运行时的单元测试测试运行时
```bash
cd /path_to_runtime/build
./bin/brt_test_all
```
