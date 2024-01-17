---
title: "Linalg 扩展"
date: 2022-06-09
weight: 5
keywords: ["Linalg", "MLIR"]
description:
---

ByteIR 编译器扩展了 MLIR linalg 方言，以支持多种非平凡的模式。
ByteIR 以在现有 linalg 方言之上引入 linalg-ext 方言的方式实现。
linalg-ext 中的操作和转换是期望能够与 linalg 中的现有操作和转换互换工作，并有望最终上传到 LLVM。


## 理由

### 对于 linalg 非平凡模式的需要

上游的 linalg 仍然没有很好地覆盖几个性能关键的模式。
有些模式可能无法通过通用算子或仅依赖现有的 linalg 接口在 linalg 方言中轻松表达。Top-k 和 Scan（cumsum）可能属于这一类。
有些可能通过组合几个通用算子来表达，但由于缺乏适当的接口，可能会阻碍所需的转换。Softmax 属于这一类。
有些旨在成为现有上游版本的更通用的替代品，`linalg_ext.batch_matmul` 属于这一类。

### 引入 linalg-ext 的实现

引入 linalg-ext 可以提供以下几个好处，

- 它清楚地将算子或转换的扩展与现有的 linalg 分开，避免了误用。
- 可以直观地解决需要引入接口的模式。

## 变换（Transformation）扩展

ByteIR linalg-ext 中增强或引入了几种变换。

引入了 **_合并维度变换_**

- 来合并 linalg.generic 算子的维数。

引入了 **_消去单位长度维数变换_**

- 来移除 Linalg 算子中单位长度的维数。

引入了 **_递降为循环变换_**

- 来将算子递降为循环。

引入了 **_Linalg 概括变换_**

- 将 linalg 算子概括为命名函数，其中 `libcall` 用于外部库调用。如果 `libcall` 设置为 False，则每个概述的函数都将具有唯一的名称，在这种情况下 `func_name` 只提供命名提示。否则，所有转换的函数调用都引用名为 `func_name` 的相同的外部函数。

引入了 **_分块标记变换_**

- 来通过属性表示循环类别 (并行或规约)。

注意这个分块标记变换也能够作用在现有的 linalg 分块和融合变换。

引入了 **_共享输出的形式到分布式形式变换_**

- 将并行分块由共享输出的形式转到分布式形式。

增强了 **_分块变换_**

- 以支持 linalg-ext 算子。

增强了 **_融合变换_**

- 以支持 linalg-ext 算子，
- 来正确地支持沿着规约轴的分块，
- 以支持在融合中将中间结果作为输出，
- 以支持中间结果的张量的维度简化，
- 以支持钻石型结构（if-else），
- 以支持可选的 stop 属性，
- 以支持与 tensor dialect 的融合。

引入了 **_融合操作数变换_**

- 以支持融合中多个根节点的情况，
- 以支持检查 func 算子中的运算是否都被融合了。

注意，这个变换将会和融合变换合并起来。

这里我们展示了沿着规约轴分块的不同。

```
// input.mlir
func.func @tiled_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %2 : tensor<128x128xf32>
}

// result after transform.structured.fuse, wrong tiling result
func.func @tile_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = scf.for %arg2 = %c0 to %c128 step %c8 iter_args(%arg3 = %0) -> (tensor<128x128xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[0, %arg2] [128, 8] [1, 1] : tensor<128x128xf32> to tensor<128x8xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, 0] [8, 128] [1, 1] : tensor<128x128xf32> to tensor<8x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%arg3 : tensor<128x128xf32>) -> tensor<128x128xf32>   // shouldn't fill to zero every step
    %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<128x8xf32>, tensor<8x128xf32>) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %3 : tensor<128x128xf32>
  }
  return %1 : tensor<128x128xf32>
}

// result after transform.structured.fuse_ext
func.func @tile_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %2 = scf.for %arg2 = %c0 to %c128 step %c8 iter_args(%arg3 = %1) -> (tensor<128x128xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[0, %arg2] [128, 8] [1, 1] : tensor<128x128xf32> to tensor<128x8xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, 0] [8, 128] [1, 1] : tensor<128x128xf32> to tensor<8x128xf32>
    %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<128x8xf32>, tensor<8x128xf32>) outs(%arg3 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %3 : tensor<128x128xf32>
  }
  return %2 : tensor<128x128xf32>
}

```

这里我们展示了当有一个中间结果作为输出时的不同。

```
// input.mlir
func.func @fuse_element(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<512x128xf32>)
                             outs(%arg1: tensor<512x128xf32>) -> tensor<512x128xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<512x128xf32>, tensor<512x128xf32>)
                             outs(%arg1: tensor<512x128xf32>) -> tensor<512x128xf32>
  return %0, %1 : tensor<512x128xf32>, tensor<512x128xf32>
}

// result after transform.structured.fuse
func.func @fuse_element_static(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %0 = linalg.elemwise_unary ...  // duplicate producer elemwise_unary
  %1 = scf.for %arg2 = %c0 to %c512 step %c32 iter_args(%arg3 = %arg1) -> (tensor<512x128xf32>) {
    %2 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %arg3) -> (tensor<512x128xf32>) {
      ...
      %3 = linalg.elemwise_unary ...  // producer fusion
      ...
      %4 = linalg.elemwise_binary ...
      %inserted_slice = tensor.insert_slice ...
      scf.yield %inserted_slice : tensor<512x128xf32>
    }
    scf.yield %2 : tensor<512x128xf32>
  }
  return %0, %1 : tensor<512x128xf32>, tensor<512x128xf32>
}

// result after transform.structured.fuse_ext
func.func @fuse_element_static(%arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %0:2 = scf.for %arg2 = %c0 to %c512 step %c32 iter_args(%arg3 = %arg1, %arg4 = %arg1) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
    %1:2 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %arg1, %arg7 = %arg1) -> (tensor<512x128xf32>, tensor<512x128xf32>) {
      ...
      %2 = linalg.elemwise_unary ...  // producer fusion
      ...
      %3 = linalg.elemwise_binary ...
      %inserted_slice = tensor.insert_slice ...
      %inserted_slice_3 = tensor.insert_slice ...
      scf.yield %inserted_slice, %inserted_slice_3 : tensor<512x128xf32>, tensor<512x128xf32>
    }
    scf.yield %1#0, %1#1 : tensor<512x128xf32>, tensor<512x128xf32>
  }
  return %0#1, %0#0 : tensor<512x128xf32>, tensor<512x128xf32>
}
```

这里展示了分块和融合残差块时的不同。
顺便说一句，上游版本在一系列残差块上执行非常慢，因为一些节点被访问了 2^N 次，N 是残差块的数量。

```
// input.mlir
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @resnet_block(%arg0: tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16> {
  %cst = arith.constant dense_resource<__elided__> : tensor<256x1x1x64xf32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<64x3x3x64xf32>
  %cst_1 = arith.constant dense_resource<__elided__> : tensor<64x1x1x256xf32>
  %cst_2 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1x56x56x256xf16>
  %1 = linalg.fill ins(%cst_2 : f16) outs(%0 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %2 = linalg.elemwise_unary {__revisited__} ins(%arg0 : tensor<1x56x56x256xf16>) outs(%1 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %3 = tensor.empty() : tensor<1x56x56x64xf16>
  %4 = linalg.fill ins(%cst_2 : f16) outs(%3 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %5 = linalg.conv_2d_nhwc_fhwc {__conv_0__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%2, %cst_1 : tensor<1x56x56x256xf16>, tensor<64x1x1x256xf32>) outs(%4 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %padded = tensor.pad %5 nofold low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_2 : f16
  } : tensor<1x56x56x64xf16> to tensor<1x58x58x64xf16>
  %6 = tensor.empty() : tensor<1x56x56x64xf16>
  %7 = linalg.fill ins(%cst_2 : f16) outs(%6 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %8 = linalg.conv_2d_nhwc_fhwc {__conv_1__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %cst_0 : tensor<1x58x58x64xf16>, tensor<64x3x3x64xf32>) outs(%7 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %9 = tensor.empty() : tensor<1x56x56x256xf16>
  %10 = linalg.fill ins(%cst_2 : f16) outs(%9 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %11 = linalg.conv_2d_nhwc_fhwc {__conv_2__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%8, %cst : tensor<1x56x56x64xf16>, tensor<256x1x1x64xf32>) outs(%10 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %12 = tensor.empty() : tensor<1x56x56x256xf16>
  %13 = linalg.generic {__root__, indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %11 : tensor<1x56x56x256xf16>, tensor<1x56x56x256xf16>) outs(%12 : tensor<1x56x56x256xf16>) {
  ^bb0(%in: f16, %in_3: f16, %out: f16):
    %14 = arith.addf %in, %in_3 : f16
    linalg.yield %14 : f16
  } -> tensor<1x56x56x256xf16>
  return %13 : tensor<1x56x56x256xf16>
}
transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [0, 8, 0, 32]}
  cleanup
}

// result after transform.structured.fuse_ext, `linalg.elemwise_unary {__revisited__}` is tiled only once
// and its tile size is calculated by getting the maximum of two paths

#map = affine_map<(d0) -> (-d0 + 1, 0)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (56, d0)>
#map3 = affine_map<(d0) -> (56, d0 + 9)>
#map4 = affine_map<(d0, d1) -> (d0 - d1)>
#map5 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 10)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @resnet_block(%arg0: tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16> {
  %c256 = arith.constant 256 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<64x1x1x256xf32>
  %cst_1 = arith.constant dense_resource<__elided__> : tensor<64x3x3x64xf32>
  %cst_2 = arith.constant dense_resource<__elided__> : tensor<256x1x1x64xf32>
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %0 = tensor.empty() : tensor<1x56x56x256xf16>
  %1 = tensor.empty() : tensor<1x56x56x64xf16>
  %2:7 = scf.for %arg1 = %c0 to %c56 step %c8 iter_args(%arg2 = %0, %arg3 = %1, %arg4 = %1, %arg5 = %1, %arg6 = %0, %arg7 = %1, %arg8 = %0) -> (tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>) {
    %3:7 = scf.for %arg9 = %c0 to %c256 step %c32 iter_args(%arg10 = %arg2, %arg11 = %arg3, %arg12 = %arg4, %arg13 = %arg5, %arg14 = %arg6, %arg15 = %arg7, %arg16 = %arg8) -> (tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>) {
      %4 = arith.addi %arg1, %c8 : index
      %5 = arith.addi %arg9, %c32 : index
      %6 = arith.minsi %arg9, %c0 : index
      %7 = arith.maxsi %5, %c256 : index
      %8 = arith.subi %7, %6 : index
      %9 = arith.subi %arg9, %6 : index
      %10 = arith.subi %c0, %6 : index
      %11 = affine.max #map(%arg1)
      %12 = affine.max #map1(%arg1)
      %13 = affine.min #map2(%12)
      %14 = affine.min #map3(%arg1)
      %15 = affine.apply #map4(%14, %13)
      %16 = affine.apply #map5(%11, %14, %13)
      %extracted_slice = tensor.extract_slice %arg13[0, %13, 0, 0] [1, %15, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf16> to tensor<1x?x56x64xf16>
      %17 = linalg.fill ins(%cst : f16) outs(%extracted_slice : tensor<1x?x56x64xf16>) -> tensor<1x?x56x64xf16>
      %extracted_slice_3 = tensor.extract_slice %arg11[0, %arg1, 0, 0] [1, 8, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf16> to tensor<1x8x56x64xf16>
      %18 = linalg.fill ins(%cst : f16) outs(%extracted_slice_3 : tensor<1x8x56x64xf16>) -> tensor<1x8x56x64xf16>
      %extracted_slice_4 = tensor.extract_slice %cst_2[%arg9, 0, 0, 0] [32, 1, 1, 64] [1, 1, 1, 1] : tensor<256x1x1x64xf32> to tensor<32x1x1x64xf32>
      %19 = tensor.empty() : tensor<1x8x56x32xf16>
      %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<1x8x56x32xf16>) -> tensor<1x8x56x32xf16>
      %inserted_slice = tensor.insert_slice %18 into %arg12[0, %arg1, 0, 0] [1, 8, 56, 64] [1, 1, 1, 1] : tensor<1x8x56x64xf16> into tensor<1x56x56x64xf16>
      %inserted_slice_5 = tensor.insert_slice %17 into %arg15[0, %13, 0, 0] [1, %15, 56, 64] [1, 1, 1, 1] : tensor<1x?x56x64xf16> into tensor<1x56x56x64xf16>
      %21 = arith.minsi %arg1, %13 : index
      %22 = arith.addi %13, %15 : index
      %23 = arith.maxsi %4, %22 : index
      %24 = arith.subi %23, %21 : index
      %extracted_slice_6 = tensor.extract_slice %arg0[0, %21, 0, %6] [1, %24, 56, %8] [1, 1, 1, 1] : tensor<1x56x56x256xf16> to tensor<1x?x56x?xf16>
      %extracted_slice_7 = tensor.extract_slice %arg14[0, %21, 0, %6] [1, %24, 56, %8] [1, 1, 1, 1] : tensor<1x56x56x256xf16> to tensor<1x?x56x?xf16>
      %25 = linalg.fill ins(%cst : f16) outs(%extracted_slice_7 : tensor<1x?x56x?xf16>) -> tensor<1x?x56x?xf16>
      %26 = linalg.elemwise_unary {__revisited__} ins(%extracted_slice_6 : tensor<1x?x56x?xf16>) outs(%25 : tensor<1x?x56x?xf16>) -> tensor<1x?x56x?xf16>
      %27 = arith.subi %arg1, %21 : index
      %extracted_slice_8 = tensor.extract_slice %26[0, %27, 0, %9] [1, 8, 56, 32] [1, 1, 1, 1] : tensor<1x?x56x?xf16> to tensor<1x8x56x32xf16>
      %28 = arith.subi %13, %21 : index
      %extracted_slice_9 = tensor.extract_slice %26[0, %28, 0, %10] [1, %15, 56, 256] [1, 1, 1, 1] : tensor<1x?x56x?xf16> to tensor<1x?x56x256xf16>
      %29 = linalg.conv_2d_nhwc_fhwc {__conv_0__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%extracted_slice_9, %cst_0 : tensor<1x?x56x256xf16>, tensor<64x1x1x256xf32>) outs(%17 : tensor<1x?x56x64xf16>) -> tensor<1x?x56x64xf16>
      %padded = tensor.pad %29 nofold low[0, %11, 1, 0] high[0, %16, 1, 0] {
      ^bb0(%arg17: index, %arg18: index, %arg19: index, %arg20: index):
        tensor.yield %cst : f16
      } : tensor<1x?x56x64xf16> to tensor<1x10x58x64xf16>
      %30 = linalg.conv_2d_nhwc_fhwc {__conv_1__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %cst_1 : tensor<1x10x58x64xf16>, tensor<64x3x3x64xf32>) outs(%18 : tensor<1x8x56x64xf16>) -> tensor<1x8x56x64xf16>
      %31 = linalg.conv_2d_nhwc_fhwc {__conv_2__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%30, %extracted_slice_4 : tensor<1x8x56x64xf16>, tensor<32x1x1x64xf32>) outs(%20 : tensor<1x8x56x32xf16>) -> tensor<1x8x56x32xf16>
      %32 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_8, %31 : tensor<1x8x56x32xf16>, tensor<1x8x56x32xf16>) outs(%19 : tensor<1x8x56x32xf16>) attrs =  {__root__} {
      ^bb0(%in: f16, %in_15: f16, %out: f16):
        %33 = arith.addf %in, %in_15 : f16
        linalg.yield %33 : f16
      } -> tensor<1x8x56x32xf16>
      %inserted_slice_10 = tensor.insert_slice %32 into %arg10[0, %arg1, 0, %arg9] [1, 8, 56, 32] [1, 1, 1, 1] : tensor<1x8x56x32xf16> into tensor<1x56x56x256xf16>
      %inserted_slice_11 = tensor.insert_slice %30 into %arg11[0, %arg1, 0, 0] [1, 8, 56, 64] [1, 1, 1, 1] : tensor<1x8x56x64xf16> into tensor<1x56x56x64xf16>
      %inserted_slice_12 = tensor.insert_slice %29 into %arg13[0, %13, 0, 0] [1, %15, 56, 64] [1, 1, 1, 1] : tensor<1x?x56x64xf16> into tensor<1x56x56x64xf16>
      %inserted_slice_13 = tensor.insert_slice %26 into %arg14[0, %21, 0, %6] [1, %24, 56, %8] [1, 1, 1, 1] : tensor<1x?x56x?xf16> into tensor<1x56x56x256xf16>
      %inserted_slice_14 = tensor.insert_slice %25 into %arg16[0, %21, 0, %6] [1, %24, 56, %8] [1, 1, 1, 1] : tensor<1x?x56x?xf16> into tensor<1x56x56x256xf16>
      scf.yield %inserted_slice_10, %inserted_slice_11, %inserted_slice, %inserted_slice_12, %inserted_slice_13, %inserted_slice_5, %inserted_slice_14 : tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>
    }
    scf.yield %3#0, %3#1, %3#2, %3#3, %3#4, %3#5, %3#6 : tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>, tensor<1x56x56x64xf16>, tensor<1x56x56x256xf16>
  }
  return %2#0 : tensor<1x56x56x256xf16>
}

```

请参考 [注意力的例子](../attention/) 来理解 FuseExt 算子分块和融合的能力。

增强了 **_逐元素算子融合变换_**

- 以支持融合内中间结果作为输出，
- 以支持生产者消费者式的融合以及输入共享的融合，
- 以支持中间张量的维数简化，
- 通过自动将映射算子转为通用算子以支持映射融合，
- 以支持具有常量结果的索引映射。

这里展示了具有输入共享的融合时的不同。

```
// input.mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.addf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ops, unchanged
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.addf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ext="shared-input"
func.func @input_sharing(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.mulf %in, %in_2 : f32
    linalg.yield %2, %3 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
```

这里展示了支持中间张量维数简化时的不同。

```
// input.mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @may_more_break_outs_dependency(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.mulf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ops, no fusion
func.func @may_more_break_outs_dependency(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %in : f32
    linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  %dim_1 = tensor.dim %1, %c0 : tensor<?x?xf32>
  %dim_2 = tensor.dim %1, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%dim_1, %dim_2) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.mulf %in, %in : f32
    linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// result after linalg-fuse-elementwise-ext, perfect fusion
func.func @may_more_break_outs_dependency(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    %3 = arith.mulf %2, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

```

这里展示了支持具有常量结果的索引映射时的不同。

```
// input.mlir
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @constant_in_affine_map_with_collapse_shape(%arg0: tensor<1x256x1024xf32>, %arg1: tensor<256x1024xf16>, %arg2: tensor<256x1xf32>, %arg3: tensor<256x1xf32>) -> tensor<256x1024xf32> {
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x1024xf16> into tensor<1x256x1024xf16>
  %0 = tensor.empty() : tensor<1x256x1024xf32>
  %1 = tensor.empty() : tensor<256x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0 : tensor<1x256x1024xf16>, tensor<1x256x1024xf32>) outs(%0 : tensor<1x256x1024xf32>) {
  ^bb0(%in: f16, %in_0: f32, %out: f32):
    %4 = arith.extf %in : f16 to f32
    %5 = arith.addf %in_0, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x256x1024xf32>
  %collapsed = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<1x256x1024xf32> into tensor<256x1024xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2, %collapsed : tensor<256x1xf32>, tensor<256x1xf32>, tensor<256x1024xf32>) outs(%1 : tensor<256x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %4 = arith.subf %in_1, %in_0 : f32
    %5 = arith.mulf %4, %in : f32
    linalg.yield %5 : f32
  } -> tensor<256x1024xf32>
  return %3 : tensor<256x1024xf32>
}

// result after linalg-fuse-elementwise-ops, no fusion
func.func @constant_in_affine_map_with_collapse_shape(%arg0: tensor<1x256x1024xf32>, %arg1: tensor<256x1024xf16>, %arg2: tensor<256x1xf32>, %arg3: tensor<256x1xf32>) -> tensor<256x1024xf32> {
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x1024xf16> into tensor<1x256x1024xf16>
  %0 = tensor.empty() : tensor<1x256x1024xf32>
  %1 = tensor.empty() : tensor<256x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0 : tensor<1x256x1024xf16>, tensor<1x256x1024xf32>) outs(%0 : tensor<1x256x1024xf32>) {
  ^bb0(%in: f16, %in_0: f32, %out: f32):
    %4 = arith.extf %in : f16 to f32
    %5 = arith.addf %in_0, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x256x1024xf32>
  %collapsed = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<1x256x1024xf32> into tensor<256x1024xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2, %collapsed : tensor<256x1xf32>, tensor<256x1xf32>, tensor<256x1024xf32>) outs(%1 : tensor<256x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %4 = arith.subf %in_1, %in_0 : f32
    %5 = arith.mulf %4, %in : f32
    linalg.yield %5 : f32
  } -> tensor<256x1024xf32>
  return %3 : tensor<256x1024xf32>
}

// result after linalg-fuse-elementwise-ext, perfect fusion
func.func @constant_in_affine_map_with_collapse_shape(%arg0: tensor<1x256x1024xf32>, %arg1: tensor<256x1024xf16>, %arg2: tensor<256x1xf32>, %arg3: tensor<256x1xf32>) -> tensor<256x1024xf32> {
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x1024xf16> into tensor<1x256x1024xf16>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1], [2]] : tensor<256x1xf32> into tensor<1x256x1xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1], [2]] : tensor<256x1xf32> into tensor<1x256x1xf32>
  %0 = tensor.empty() : tensor<1x256x1024xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0, %expanded_0, %expanded_1 : tensor<1x256x1024xf16>, tensor<1x256x1024xf32>, tensor<1x256x1xf32>, tensor<1x256x1xf32>) outs(%0 : tensor<1x256x1024xf32>) {
  ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
    %2 = arith.extf %in : f16 to f32
    %3 = arith.addf %in_2, %2 : f32
    %4 = arith.subf %3, %in_4 : f32
    %5 = arith.mulf %4, %in_3 : f32
    linalg.yield %5 : f32
  } -> tensor<1x256x1024xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<1x256x1024xf32> into tensor<256x1024xf32>
  return %collapsed : tensor<256x1024xf32>
}
```

## Linalg-ext 算子拓展

### 别名算子 (Alias Op)

Linalg-ext 别名算子是作为辅助算子，帮助输入共享融合而引入的。
它是在Pass内部产生的，并且通常是在通用算子内。
它并不会因为标准化被消去，并且仅仅会因为调用 `populateRemoveLinalgExtAliasPattern` 被移除。
注意：别名算子并不是一个结构化的算子, 并且没有像 `LoopIteratorType` 这样的接口。

### 对角算子 (Diag Op)

Linalg-ext 对角算子是为了表示对角矩阵而引入的。
它是一个结构化的算子，但是目前仅仅是用作输出的中间表示，通常与矩阵相乘算子一起使用。
基于后端, 一个带有对角算子的矩阵相乘通常可以被重写成：

1. 与一个减少了负载的矩阵的矩阵乘
2. 一个稀疏的矩阵乘
3. 一个带有广播的逐元素乘

定义:

- 操作数:
  - input （输入）: 形状为 N 的张量
- 初始值/结果:
  - output （输出）: 形状为 N x N 的张量

### 遍历算子 (Scan Op)

Linalg-ext 遍历算子是为了表示遍历、前缀和、或者 `cumsum` 这样的模式，
它是一个结构化的算子。

定义:

- 操作数:
  - input （输入）: 维数为 N 的张量
- 属性:
  - dimension （维数）: I64ArrayAttr
  - inclusivie （是否包括当前值）: BoolAttr
- 初始值/结果:
  - output （输出）: 维数为 N 的张量
  - accumulator （累积器）: 维数为 N - 1 的张量

### 索引更新算子 （Scatter Op）

Linalg-ext 索引更新算子是为了表示按照索引更新的模式，
它是一个结构化的算子。

定义:

- 操作数:
  - indices （索引）: 张量
  - updates （更新值）: 张量
- 初始值/结果:
  - src （源值）: 张量

这里, `indices` 和 `update` 的前 `rank(indices) - 1` 个维度是匹配的。
`update` 和 `src` 的后 `rank(update) - rank(indices) + 1` 个维度是匹配的。
`indices` 的最后一维表示为 `dim(indices, rank(indices) - 1)`, 并且它应当是静态的。`src` 的秩等于 `dim(indices, rank(indices) - 1) + rank(update) - rank(indices) + 1`。

### 归一化指数函数算子（Softmax Op）

Linalg-ext softmax 算子是为了表示 softmax 模式，
它是一个结构化的算子。

定义:

- 操作数:
  - input （输入）: 维数为 N 的张量
- 属性
  - dimension （维数）: I64ArrayAttr
- 初始值/结果:
  - output （输出）: 维数为 N 的张量, `output_result = exp(input - max_result) / accumulator_result`
  - max （最大值）: 维数为 N - 1 的张量, `max_result = max(max(input, dimension), max_init)`
  - accumulator （累积器）: 维数为 N - 1 的张量, `accumulator_result = accumulator_init * exp(max_init - max_result) + sum(exp(input - max_result), dimension)`
  - scale （标度）: 维数为 N - 1 的张量, `scale_result = accumulator_init * exp(max_init - max_result) / accumulator_result`

在这里, 操作数 `1`, 最大值定义为 `max_result = max(max(input, dimension), max_init)`。
基本上说, 它是对于 `input` 沿着维度 `dimension` 作初始值为 `max_init` 的 `reduce_max` 的结果。
操作数 `2`, 累积器定义为 `accumulator_result = accumulator_init * exp(max_init - max_result) + sum(exp(input - max_result), dimension)`
基本上说, 它是对于 `exp(input - max_result)` 沿着维度 `dimension` 作初始值为 `accumulator_init * exp(max_init - max_result)` 的 `reduce_sum` 的结果。
操作数 `3`, 标度定义为 `scale_result = accumulator_init * exp(max_init - max_result) / accumulator_result`。
最后, 操作数 `0`, 输出定义为 `output_result = exp(input - max_result) / accumulator_result`。

### Topk 算子 （Topk Op）

Linalg-ext topk 算子是用来表示 topk 模式，
它是一个结构化的算子。

定义:

- 操作数:
  - 输入值（input_values）: 维数为 N 的张量
  - 输入索引（input_indices）: 可选的维数为 N 的张量
- 属性：
  - 维度（dimension）: I64ArrayAttr
- 初始值/结果:
  - 输出值（output_values）: 维数为 N 的张量
  - 输出索引（output_indices）: 维数为 N 的张量

