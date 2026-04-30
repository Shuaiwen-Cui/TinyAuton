# 说明

!!! note "说明"
    INT 量化是 `tiny_ai` 部署阶段的主力路径：通过对称的 min-max 校准，把 float32 权重 / 激活映射到 INT8 或 INT16 整数空间，再使用 INT8 dense kernel 进行纯整数推理。该模块同时提供 C 与 C++ 两套接口。

## 校准（min-max，对称）

```c
tiny_error_t tiny_quant_calibrate_minmax(const float *data, int n,
                                          tiny_dtype_t dtype,
                                          tiny_quant_params_t *params);
```

实现：

\[
\text{abs\_max} = \max_i |x_i|,\quad
\text{scale} = \frac{\text{abs\_max}}{Q_\text{max}},\quad
\text{zero\_point} = 0
\]

`Q_max = 127 (INT8) / 32767 (INT16)`。当 `abs_max < TINY_MATH_MIN_DENOMINATOR` 时回退到 `1.0f`。

## INT8 量化 / 反量化

```c
tiny_error_t tiny_quant_f32_to_int8(const float *src, int8_t *dst, int n,
                                     const tiny_quant_params_t *params);

tiny_error_t tiny_quant_int8_to_f32(const int8_t *src, float *dst, int n,
                                     const tiny_quant_params_t *params);
```

量化：`q = clamp(round(x / scale) + zp, -128, 127)`，反量化：`x = (q - zp) * scale`。

## INT16 量化 / 反量化

API 形式相同，仅类型替换为 `int16_t`，范围 `[-32768, 32767]`。当对精度敏感（如 IMU 中间统计量）时用 INT16。

## INT8 dense forward kernel

```c
tiny_error_t tiny_quant_dense_forward_int8(
    const int8_t  *input,
    const int8_t  *weight,
    const int32_t *bias,
    int8_t        *output,
    int batch, int in_feat, int out_feat,
    float input_scale, float weight_scale, float output_scale);
```

实现要点：

1. **INT32 累加**：`acc = bias[o] + Σ_i input[b][i] * weight[o][i]`。
2. **重量化**：`q_out = round(acc * (input_scale * weight_scale / output_scale))`。
3. **截断**：`output[b][o] = clamp(q_out, -128, 127)`。

输入 / 输出激活共享同一 `output_scale`（隐藏在重量化里），因此可以与下游 INT8 层直接级联，无需中间反量化。

`bias` 是预先量化好的 INT32（典型做法是把浮点 bias 除以 `input_scale * weight_scale` 后取整）。

## C++ 包装：QuantParams + Tensor 接口

```cpp
struct QuantParams
{
    tiny_dtype_t dtype      = TINY_DTYPE_INT8;
    float        scale      = 1.0f;
    int          zero_point = 0;

    QuantParams() = default;
    QuantParams(tiny_dtype_t d, float s, int zp = 0);

    tiny_quant_params_t to_c() const;
};

QuantParams calibrate(const Tensor &t, tiny_dtype_t dtype = TINY_DTYPE_INT8);
tiny_error_t quantize  (const Tensor &src, uint8_t *dst, const QuantParams &params);
tiny_error_t dequantize(const uint8_t *src, Tensor &dst, const QuantParams &params);

int8_t *quantize_weights(const Tensor &t, QuantParams &params);

tiny_error_t requantize_int8(const int8_t *src, int8_t *dst, int n,
                              float src_scale, float dst_scale);
```

`quantize` / `dequantize` 内部根据 `params.dtype` 派发到 INT8 / INT16 / FP8 路径，因此 C++ 端的代码可以保持一致。

`quantize_weights(t, params)`：

1. `params = calibrate(t, TINY_DTYPE_INT8)`。
2. `TINY_AI_MALLOC(t.size * sizeof(int8_t))`。
3. `tiny_quant_f32_to_int8(t.data, buf, t.size, params.to_c())`。
4. 调用方负责 `TINY_AI_FREE(buf)`。

## PTQ 流程示例

```cpp
using namespace tiny;

// 1) 训练后得到 fp32 模型
MLP fp_model({4, 16, 8, 3}, ActType::RELU, true);
trainer.fit(...);

// 2) 取出第一层 Dense 的权重
Dense *fc = (Dense *)fp_model[0];

// 3) 校准 + 量化
QuantParams qp;
int8_t *w_int8 = quantize_weights(fc->weight, qp);

// 4) 量化输入激活
QuantParams in_qp = calibrate(input_batch, TINY_DTYPE_INT8);
int8_t *x_int8 = (int8_t *)TINY_AI_MALLOC(input_batch.size);
quantize(input_batch, (uint8_t *)x_int8, in_qp);

// 5) INT8 dense forward
QuantParams out_qp(TINY_DTYPE_INT8, 0.05f);  // 输出 scale 也需要校准 / 估计
int8_t *y_int8 = (int8_t *)TINY_AI_MALLOC(B * out_feat);
tiny_quant_dense_forward_int8(
    x_int8, w_int8, /*bias=*/nullptr, y_int8,
    B, in_feat, out_feat,
    in_qp.scale, qp.scale, out_qp.scale);
```

## 重量化（INT8 → INT8）

```c
tiny_error_t requantize_int8(const int8_t *src, int8_t *dst, int n,
                              float src_scale, float dst_scale);
```

当两个量化层级联但 `scale` 不同（例如 act → linear → linear），先用此函数把上一层的 INT8 输出重量化到下一层期望的 scale 域。

\[
q' = \mathrm{clamp}\!\big(\mathrm{round}(q \cdot s_\text{src} / s_\text{dst}),\,-128,\,127\big)
\]

## 精度损失

- **INT8**：在 `[-1, 1]` 经过对称量化后误差大致为 `scale ≈ 1/127 ≈ 0.0079`。配合 ReLU / sigmoid 通常 1~3% 精度损失可接受。
- **INT16**：误差降至 `scale ≈ 1/32767`，几乎无损。
- **建议**：先用 INT8 跑一遍 PTQ，若分类下降明显（>3%），改用 QAT 或回退到 INT16。
