---
title: "GPU 支持"
date: 2023-09-04
weight: 6
keywords: ["GPU"]
description:
---

ByteIR 编译器目前对 NVIDIA GPU 提供有限支持。

## Passes

从 mhlo 到 GPU 后端的主要途径是通过 mhlo 方言、linalg-tenor 方言、linalg-memref 方言、affine/scf 方言、gpu 方言。
前半部分，从 mhlo 到 affine/scf 方言，在不同后端之间是相似的。
因此，我们只讨论后半部分，从 affine/scf 方言到 gpu 方言。
ByteIR 编译器开发了多种 Pass 来支持 GPU 后端。

### InsertTrivialSCFLoop Pass

这个 pass 只是插入一个简单的 scf ForOp 用于标量计算。
它通常用于简化 pass 流程，而无需稍后检查是否进行标量计算。
注意，在 scf 规范化之后，此 pass 的效果将被移除。

```
// input.mlir
func.func @scalar_func(%arg0: memref<f32>) -> memref<f32> {
  %cst = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() : memref<f32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = memref.load %arg0[] : memref<f32>
  %1 = arith.cmpf une, %0, %cst_0 : f32
  %2 = arith.select %1, %0, %cst : f32
  memref.store %2, %alloc[] : memref<f32>
  return %alloc : memref<f32>
}

// result after -insert-trivial-scf-loop
func.func @scalar_func(%arg0: memref<f32>) -> memref<f32> {
  %cst = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() : memref<f32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg1 = %c0 to %c1 step %c1 {
    %0 = memref.load %arg0[] : memref<f32>
    %1 = arith.cmpf une, %0, %cst_0 : f32
    %2 = arith.select %1, %0, %cst : f32
    memref.store %2, %alloc[] : memref<f32>
  }
  return %alloc : memref<f32>
}

```

### ConvertFuncToGPU Pass

这个 Pass 将循环形式的 FuncOp 转换为 SIMT 形式的 GPUFuncOp。
源 FuncOp 中带有注解的一些循环被转换为 SIMT 语句。

```
// input.mlir
#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
func.func private @matmul_tiled(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) attributes {__byteir_to_gpu__} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<8x64xf32, 1>
  %alloc_0 = memref.alloc() : memref<64x64xf32, 2>
  %alloc_1 = memref.alloc() : memref<8x64xf32, 3>
  scf.for %arg3 = %c0 to %c128 step %c8 {
    %subview = memref.subview %arg0[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    %subview_2 = memref.subview %arg2[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    linalg.copy ins(%subview : memref<8x64xf32, #map>) outs(%alloc : memref<8x64xf32, 1>)
    linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%alloc_0 : memref<64x64xf32, 2>)
    linalg.matmul ins(%alloc, %alloc_0 : memref<8x64xf32, 1>, memref<64x64xf32, 2>) outs(%alloc_1 : memref<8x64xf32, 3>)
    linalg.copy ins(%alloc_1 : memref<8x64xf32, 3>) outs(%subview_2 : memref<8x64xf32, #map>)
  } {__byteir_loop_to_simt__ = "block_id.x"}
  return
}

// result after convert-func-to-gpu
gpu.module @unified {
gpu.func @matmul_tiled(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) workgroup(%arg3 : memref<8x64xf32, 3>, %arg4 : memref<64x64xf32, 3>, %arg5 : memref<8x64xf32, 3>) kernel {
  %c128 = arith.constant 128 : index
  %c8 = arith.constant 8 : index
  %0 = gpu.block_id  x
  %alloc = memref.alloc() : memref<8x64xf32, 3>
  %alloc_0 = memref.alloc() : memref<64x64xf32, 2>
  %1 = arith.muli %0, %c8 : index
  %2 = arith.cmpi slt, %1, %c128 : index
  scf.if %2 {
    %subview = memref.subview %arg0[%1, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    %subview_1 = memref.subview %arg2[%1, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    linalg.copy ins(%subview : memref<8x64xf32, #map>) outs(%arg5 : memref<8x64xf32, 3>)
    linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%alloc_0 : memref<64x64xf32, 2>)
    linalg.matmul ins(%arg5, %alloc_0 : memref<8x64xf32, 3>, memref<64x64xf32, 2>) outs(%alloc : memref<8x64xf32, 3>)
    linalg.copy ins(%alloc : memref<8x64xf32, 3>) outs(%subview_1 : memref<8x64xf32, #map>)
  }
  gpu.return
}
}
func.func private @matmul_tiled(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  gpu.launch_func  @unified::@matmul_tiled blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x64xf32>, %arg1 : memref<64x64xf32>, %arg2 : memref<128x64xf32>)
  return
}
```

## 后端目标

ByteIR 编译器要么通过 LLVM PTX 代码生成，要么通过 CUDA C 源代码触发器支持 NVIDIA GPU 后端。

其他的后端目标, 例如 cubin 代码生成, 将是我们未来的工作。

### LLVM PTX 代码生成

LLVM PTX 代码生成将 GPU 方言递降为 LLVM/NVVM 方言, 然后将 LLVM/NVVM 方言通过 LLVM PTX 后端翻译为 PTX。
第一步依赖于一个通用的流水线 `NVVMCodegenPipeline`, 第二步使用带有 `gen-ptx` 选项的 `byteir-translate`。

### CUDA 触发器

CUDA 触发器直接通过带有 `emit-cuda` 选项的 `byteir-translate` 将 GPU 方言翻译为 CUDA C 源码。

