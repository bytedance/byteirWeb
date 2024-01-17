---
title: "ByteIR 附加算子"
date: 2023-09-04
weight: 6
keywords: ["自定义算子"]
description:
---

ByteIR 编译器引入了几个粗粒度的算子来提升编译过程中的模式匹配重写。
ByteIR 通过复用mhlo自定义算子定义的方式实现，它们在call_target_name中带有ByteIR前缀。而不是定义另一种新方言。
ByteIR 在前端实现了这种转换，而不是将其放在 ByteIR 编译器中。

## 理由
### 对于粗粒度算子的需要

引入粗粒度的算子可以提供如下几个好处，
* 无论优化或递降，它都简化了重写期间的模式匹配过程。
* 它允许使用粗粒度的操作对上层信息进行编码，有助于优化；
* 它提供了从前端到中间表示的直观映射，有助于可调试性；
* 它提供了灵活的控制，因为粗粒度的操作可以很容易地分解为细粒度的操作，反之则困难得多。

### 复用mhlo自定义调用的实现

复用在call_target_name中带有ByteIR前缀的mhlo自定义算子可以提供以下几个好处：
* 原始的中间表示仍然合法且定义明确，无需引入额外的新方言或在tablegen中定义新算子；
* 它为所有现有的Pass或模式匹配提供向后支持，不会造成任何破坏；
* 通过合适的定义，无法识别的粗粒度算子可以轻松映射到自定义库或分解为细粒度操作。

### 前端中实现粗粒度的算子转换

在前端实现粗粒度的算子转换可以提供以下几个好处，
* 它避免了在 ByteIR 编译器中发生 N 对 1 的重写，将相应的重写放在每个自己的前端提供了更简洁的实现；
* 不同的前端可能已经定义了自己的方言，提供粗粒度的操作，使得这种转换变得简单而直观；
* 它隔离了现有前端图优化引起的影响，这些影响可能会在每个前端的不同版本之间发生变化。


## 附加算子定义

粗粒度的算子类型是通过前缀定义的。
```call_target_name = "byteir.softmax" or "tf.DynamicPartition"```
如果一个算子是跨前端通用的（这种情况大多发生），它将使用 byteir 前缀。
如果一个算子是特定于某个前端的，它会使用特定前端的前缀，例如 tf 或 pytorch。

对于给定粗粒度算子所需的进一步的信息被编码在一个名为 `byteir_attrs` 的字典属性中，该属性包括所有命名属性。

```Op Attribute: byteir_attrs = {approximate = "none"} or byteir_attrs = {} of if none```

### byteir.layer_norm
- 操作数:
  - input: Tensor
  - weight: Tensor
  - bias: Tensor
- 属性
  - epsilon: F64Attr
  - axis: I64ArrayAttr
  - eps_outside_sqrt: Optional\<BoolAttr>
- 结果(1 或 3):
  - output: Tensor
  - mean: Optional\<Tensor>
  - inv_std_dev: Optional\<Tensor>

### byteir.l2_norm
- 操作数:
  - input: Tensor
- 属性
  - epsilon: F64Attr
  - axis: I64ArrayAttr
- 结果:
  - output: Tensor

### byteir.softmax
- 操作数:
  - input: Tensor
- 属性
  - axis: I64Attr
- 结果:
  - output: Tensor
- 例子:
```
%0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 1 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<4x64xf32>) -> tensor<4x64xf32>
```

### byteir.log_softmax
- 操作数:
  - input: Tensor
- 属性
  - axis: I64Attr
- Result:
  - output: Tensor
- 例子:
```
%0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 1 : i64}, call_target_name = "byteir.log_softmax", called_computations = [], has_side_effect = false} : (tensor<4x64xf32>) -> tensor<4x64xf32>
```

### byteir.gelu
- 操作数:
  - input: Tensor
- 属性:
  - approximate: str
    - none / erf
    - tanh
- 结果:
  - output: Tensor
- 例子:
```
%0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {approximate = "none"}, call_target_name = "byteir.gelu", called_computations = [], has_side_effect = false} : (tensor<4x64xf32>) -> tensor<4x64xf32>
```

### byteir.arg_max/byteir.arg_min
- 操作数:
  - input: Tensor
- 属性
  - axis: I64Attr
  - keep_dims: BoolAttr
  - select_last_index: BoolAttr
- 结果:
  - output: Optional\<Tensor>
  - indices: IntTensor


### byteir.top_k
- 操作数:
  - input: Tensor
- 属性
  - k: I64Attr
  - axis: I64ArrayAttr
  - sorted: BoolAttr
- 结果:
  - output: Tensor
  - indices: IntTensor

### byteir.erf
- 操作数:
  - input: Tensor
- 结果:
  - output: Tensor
- 例子:
```
%0 = "mhlo.custom_call"(%arg0) {call_target_name = "byteir.erf", has_side_effect = false} : (tensor<?x64xf32>) -> tensor<?x64xf32>
```

### byteir.one_hot
- 操作数:
  - indices: IntTensor
- 属性:
  - depth: I64Attr
  - axis: I64Attr
  - on_value: AnyAttr
  - off_value: AnyAttr
- 结果:
  - output: Tensor (on_value 和 off_value 元素类型相同)

### byteir.quantize
- 操作数:
  - input: FloatTensor
  - scale: FloatTensor (对于 per-tensor 量化秩为0, 对于 per-channel 量化秩为1)
  - zero_point: Int8Tensor (形状与 scale 相同)
- 属性
  - axis: I64Attr (可选, 只在per-channel 量化时必须)
- 结果:
  - output: Int8Tensor

### byteir.dequantize
- 操作数:
  - input: Int8Tensor
  - scale: FloatTensor (对于 per-tensor 量化秩为0, 对于 per-channel 量化秩为1)
  - zero_point: Int8Tensor (形状与 scale 相同)
- 属性
  - axis: I64Attr (可选, 只在per-channel 量化时必须, channel 维的索引)
- 结果:
  - output: FloatTensor

### byteir.resize
- 操作数:
  - input: Tensor
  - target (scale/size): FloatTensor/IntTensor (对应)
- 属性:
  - target_mode: StringAttr
    - `scale`
    - `size`
  - mode: StringAttr
    - `nearest`
    - `linear`
  - coordinate_transformation_mode: StringAttr
    - 将 scale 表示为 length_resized / length_original, 变换可以如下描述：

| coordinate_transformation_mode | x_original = |
| ------------------------------ | :----------  |
| `asymmetric` | x_resized / scale |
| `pytorch_half_pixel`| length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0 |
| `half_pixel` | (x_resized + 0.5) / scale - 0.5 |
| `align_corners`| x_resized * (length_original - 1) / (length_resized - 1) |

- 结果:
  - output: Tensor

### byteir.rng_uniform
- 操作数:
  - low: 0dTensor
  - high: 0dTensor
  - seed: 0dTensor
  - offset: 0dTensor
  - shape: Optional<1dTensor>
- 结果:
  - out: Tensor
- 例子:
```
// Static Shape Case: out tensor must have static shape
%high = mhlo.constant dense<1.000000e+00> : tensor<f32>
%low = mhlo.constant dense<0.000000e+00> : tensor<f32>
%seed = byre.compute @GetSeed() : tensor<i64>
%offset = byre.compute @NextOffset() : tensor<i64>
%0 = "mhlo.custom_call"(%low, %high, %seed, %offset) {call_target_name = "byteir.rng_uniform", has_side_effect = false} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>) -> tensor<8x1024x768xf32>
```
```
// Dynamic Shape Case
%high = mhlo.constant dense<1.000000e+00> : tensor<f32>
%low = mhlo.constant dense<0.000000e+00> : tensor<f32>
%seed = byre.compute @GetSeed() : tensor<i64>
%offset = byre.compute @NextOffset() : tensor<i64>
%shape = shape.shape_of %arg0 : tensor<3xindex>
%0 = "mhlo.custom_call"(%low, %high, %seed, %offset, %shape) {call_target_name = "byteir.rng_uniform", has_side_effect = false} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>, tensor<3xindex>) -> tensor<?x?x?xf32>
```
