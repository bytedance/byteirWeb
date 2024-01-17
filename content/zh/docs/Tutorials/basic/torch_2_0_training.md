---
title: "PyTorch 2.0 训练"
date: 2023-09-04
weight: 6
keywords: ["GPU"]
description:
---

ByteIR 对于 PyTorch 2.0 是高度优化的编译器。本文档将介绍我们通过编译技术在 NVGPU 上加速 PyTorch 2.0 的实践。

## 概述
我们的编译流水线的概述：
```
         PyTorch2.0
             |
       Dynamo/FuncTorch
             |
           FX 图
             |
         Torch 前端
             |
             |(输出 mhlo)
             |
        ByteIR 编译器
             |
  +----------+----------+
  |          |          |
Codegen  AITemplate  运行时库
  |          |
  |(linalg)  |(cutlass)
  |          |
 PTX     调优过的核
```

## 准备材料
### 环境
我们推荐以下的环境来构建：
* cuda>=11.8
* python>=3.9
* gcc>=8.5 or clang>=7

或者使用我们的 **[Dockerfile](https://github.com/bytedance/byteir/tree/main/docker/Dockerfile)** 来构建一个 docker 镜像。
### 构建ByteIR组件
查看每个组件的自述文件来构建ByteIR组件:
* [Torch 前端](https://github.com/bytedance/byteir/blob/main/frontends/torch-frontend/README.md)
* [ByteIR 编译器](https://github.com/bytedance/byteir/blob/main/compiler/README.md)
* [ByteIR 运行时](https://github.com/bytedance/byteir/blob/main/runtime/README.md)

构建完成后，会产生**三个** python wheel 包：`torch_frontend*.whl`, `byteir*.whl` 和 `brt*.whl`

### 安装 PyTorch 和 ByteIR 组件
安装 torch-nightly:
* `cd byteir/frontends/torch-frontend`
* `python3 -m pip install -r ./torch-requirements.txt`

安装 ByteIR 组件:
* `python3 -m pip install /path_to/torch_frontend*.whl /path_to/byteir*.whl /path_to/brt*.whl`

## 训练加速实例

### 基于 ByteIR 后端的 Torch2.0
参见 [byteir_backend.py](https://github.com/bytedance/byteir/tree/main/frontends/torch-frontend/examples/training/byteir_backend.py).
### MLP 训练示例
参见 [mlp.py](https://github.com/bytedance/byteir/tree/main/frontends/torch-frontend/examples/training/mlp.py).
