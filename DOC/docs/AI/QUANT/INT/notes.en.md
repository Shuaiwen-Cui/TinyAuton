# TinyAI · Quant · INT — Design notes {#notes}

!!! info "Implementation and records"
    APIs in this section are based on `CODE/AIoTNode-TinyAuton-AI/middleware/`. Source excerpts and serial output include historical records; check the project entry and enabled selectors before reproducing a test.

!!! note "Notes"
    INT quantisation is the deployment workhorse of `tiny_ai`: a symmetric min-max calibration maps float32 weights / activations into INT8 or INT16 integer space, after which the INT8 dense kernel runs fully-integer inference. Both C and C++ APIs are provided.

!!! abstract "INT8 — Quantization: Compress FP32 4×, Almost No Loss"
    FP32 → INT8 compresses each 32-bit float into 8 bits. 4× memory savings.

## Intuition {#intuition}

### Quantization = Scale Mapping {#quantization-scale-mapping}

`FP32 [min,max] → INT8 [-128,127]`

\[x_{int8} = \text{round}(x_{fp32} / scale), \quad scale = \max(|min|, |max|) / 127\]

### PTQ Flow {#ptq-flow}

1. **Calibrate**: compute scale from weight min/max
2. **Quantize**: FP32 → INT8 (4× savings)
3. **Inference**: INT8 × INT8 with INT32 accumulation

### Notes {#notes_1}

- Symmetric quantization assumes roughly symmetric data
- INT32 accumulation prevents overflow; result scaled back to FP32

---

## CALIBRATION (min-max, symmetric) {#calibration-min-max-symmetric}

```c
tiny_error_t tiny_quant_calibrate_minmax(const float *data, int n,
                                          tiny_dtype_t dtype,
                                          tiny_quant_params_t *params);
```

Implementation:

\[
\text{abs\_max} = \max_i |x_i|,\quad
\text{scale} = \frac{\text{abs\_max}}{Q_\text{max}},\quad
\text{zero\_point} = 0
\]

`Q_max = 127 (INT8) / 32767 (INT16)`. If `abs_max < TINY_MATH_MIN_DENOMINATOR`, the implementation falls back to `1.0f`.

## INT8 QUANT / DEQUANT {#int8-quant-dequant}

```c
tiny_error_t tiny_quant_f32_to_int8(const float *src, int8_t *dst, int n,
                                     const tiny_quant_params_t *params);

tiny_error_t tiny_quant_int8_to_f32(const int8_t *src, float *dst, int n,
                                     const tiny_quant_params_t *params);
```

Quant: `q = clamp(round(x / scale) + zp, -128, 127)`. Dequant: `x = (q - zp) * scale`.

## INT16 QUANT / DEQUANT {#int16-quant-dequant}

Same shape with `int16_t`, range `[-32768, 32767]`. Use INT16 when precision matters (intermediate IMU stats, etc.).

## INT8 DENSE FORWARD KERNEL {#int8-dense-forward-kernel}

```c
tiny_error_t tiny_quant_dense_forward_int8(
    const int8_t  *input,
    const int8_t  *weight,
    const int32_t *bias,
    int8_t        *output,
    int batch, int in_feat, int out_feat,
    float input_scale, float weight_scale, float output_scale);
```

Steps:

1. **INT32 accumulator**: `acc = bias[o] + Σ_i input[b][i] * weight[o][i]`.
2. **Requantise**: `q_out = round(acc * (input_scale * weight_scale / output_scale))`.
3. **Saturate**: `output[b][o] = clamp(q_out, -128, 127)`.

Input and output activations share the same `output_scale` (folded into the requantise step), so consecutive INT8 layers can be cascaded with no float intermediates.

`bias` is pre-quantised INT32 (typical practice: divide the float bias by `input_scale * weight_scale` and round).

## C++ WRAPPER: QuantParams + Tensor APIs {#c-wrapper-quantparams-tensor-apis}

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

`quantize` / `dequantize` dispatch on `params.dtype` to reach the INT8 / INT16 / FP8 paths, so C++ code stays uniform.

`quantize_weights(t, params)`:

1. `params = calibrate(t, TINY_DTYPE_INT8)`.
2. `TINY_AI_MALLOC(t.size * sizeof(int8_t))`.
3. `tiny_quant_f32_to_int8(t.data, buf, t.size, params.to_c())`.
4. The caller is responsible for `TINY_AI_FREE(buf)`.

## PTQ EXAMPLE {#ptq-example}

```cpp
using namespace tiny;

// 1) Train fp32 model
MLP fp_model({4, 16, 8, 3}, ActType::RELU, true);
trainer.fit(...);

// 2) Pull the first Dense's weight
Dense *fc = (Dense *)fp_model[0];

// 3) Calibrate + quantise weights
QuantParams qp;
int8_t *w_int8 = quantize_weights(fc->weight, qp);

// 4) Quantise input activations
QuantParams in_qp = calibrate(input_batch, TINY_DTYPE_INT8);
int8_t *x_int8 = (int8_t *)TINY_AI_MALLOC(input_batch.size);
quantize(input_batch, (uint8_t *)x_int8, in_qp);

// 5) INT8 dense forward
QuantParams out_qp(TINY_DTYPE_INT8, 0.05f);  // output scale needs calibration / estimation
int8_t *y_int8 = (int8_t *)TINY_AI_MALLOC(B * out_feat);
tiny_quant_dense_forward_int8(
    x_int8, w_int8, /*bias=*/nullptr, y_int8,
    B, in_feat, out_feat,
    in_qp.scale, qp.scale, out_qp.scale);
```

## REQUANTISE (INT8 → INT8) {#requantise-int8-int8}

```c
tiny_error_t requantize_int8(const int8_t *src, int8_t *dst, int n,
                              float src_scale, float dst_scale);
```

When two quantised layers are cascaded with different `scale`s (e.g. act → linear → linear), call this to bring the previous layer's INT8 output into the next layer's scale domain.

\[
q' = \mathrm{clamp}\!\big(\mathrm{round}(q \cdot s_\text{src} / s_\text{dst}),\,-128,\,127\big)
\]

## ACCURACY IMPACT {#accuracy-impact}

- **INT8**: with values in `[-1, 1]` the symmetric error is roughly `scale ≈ 1/127 ≈ 0.0079`. Combined with ReLU / sigmoid this typically costs 1~3% accuracy.
- **INT16**: error drops to `scale ≈ 1/32767`, virtually lossless.
- **Recommendation**: try INT8 PTQ first; if accuracy drops noticeably (>3%), switch to QAT or fall back to INT16.
