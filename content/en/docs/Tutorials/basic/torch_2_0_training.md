---
title: "PyTorch 2.0 Training"
date: 2023-09-04
weight: 6
keywords: ["GPU"]
description: 
---

ByteIR is a well optimized compiler for PyTorch 2.0. This doc will introduce our practice on accelerating PyTorch 2.0 on NVGPU by compilation technology.

## Overview
Overview of our compilation pipeline:
```
         PyTorch2.0
             |
       Dynamo/FuncTorch
             |
          FX Graph
             |
       Torch Frontend
             |
             |(emit mhlo)
             |
       ByteIR Compiler
             |
  +----------+----------+
  |          |          |
Codegen  AITemplate  Runtime Library
  |          |
  |(linalg)  |(cutlass)
  |          |
 PTX    Tuned Kernel
```

## Prepare Materials
### Environment
We recommond the following environment for building:  
* cuda>=11.8
* python>=3.9
* gcc>=8.5 or clang>=7  

Or use our **[Dockerfile](https://github.com/bytedance/byteir/tree/main/docker/Dockerfile)** to build an docker image.
### Build ByteIR Components
See each components' README to build themself:  
* [Torch Frontend](https://github.com/bytedance/byteir/blob/main/frontends/torch-frontend/README.md)
* [ByteIR Compiler](https://github.com/bytedance/byteir/blob/main/compiler/README.md)
* [ByteIR Runtime](https://github.com/bytedance/byteir/blob/main/runtime/README.md)

When building is completed, it will produce **three** python wheel packages: torch_frontend*.whl, byteir*.whl and brt*.whl

### Install PyTorch and ByteIR Components
Install torch-nightly:
* cd byteir/frontends/torch-frontend
* python3 -m pip install -r ./torch-requirements.txt

Install ByteIR Components:
* python3 -m pip install /path_to/torch_frontend*.whl /path_to/byteir*.whl /path_to/brt*.whl

## Training Accelerating Examples

### ByteIR Backend for Torch2.0
See [byteir_backend.py](https://github.com/bytedance/byteir/tree/main/frontends/torch-frontend/examples/training/byteir_backend.py).
### MLP Training Demo
See [mlp.py](https://github.com/bytedance/byteir/tree/main/frontends/torch-frontend/examples/training/mlp.py).
