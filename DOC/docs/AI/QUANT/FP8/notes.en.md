# Notes

!!! note "Notes"
    `tiny_fp8` provides a pure-software implementation of OCP-spec 8-bit floating-point formats: E4M3FN for weights / activations and E5M2 for gradients. ESP32-S3 has no FP8 ALU, so all values are stored as `uint8_t` and upcast to `float32` for arithmetic.

## FORMAT OVERVIEW

| Format | Bit layout | Bias | Max value | Min normal | Special encodings |
| --- | --- | --- | --- | --- | --- |
| E4M3FN | `S EEEE MMM` | 7 | ±448.0 | 2⁻⁶ | NaN = 0x7F / 0xFF; **no ±inf** |
| E5M2 | `S EEEEE MM` | 15 | ±57344.0 | 2⁻¹⁴ | ±Inf = 0x7C / 0xFC; NaN = 0x7D-0x7F / 0xFD-0xFF |

E4M3FN trades ±inf for four extra normal values — OCP's recommended "fitting normal" format for weights / activations. E5M2 mirrors a subset of IEEE 754 (keeps ±inf / NaN) and is recommended for gradients.

## E4M3FN

```cpp
uint8_t fp32_to_fp8_e4m3 (float val);
float   fp8_e4m3_to_fp32 (uint8_t fp8);
void    fp32_to_fp8_e4m3_batch(const float *src, uint8_t *dst, int n);
void    fp8_e4m3_to_fp32_batch(const uint8_t *src, float *dst, int n);
```

Encode flow:

1. Extract sign / exponent / mantissa.
2. If `exp > 8` → clamp to ±448 (encoding `0x7E`); if `exp < -9` → flush to ±0.
3. Rebias `new_exp = exp + 7`.
4. If `new_exp <= 0`: subnormal, right-shift mantissa by `(1 - new_exp)` extra bits; otherwise round-to-nearest-even to a 3-bit mantissa.
5. If rounding overflows the mantissa, bump exponent and re-check overflow.
6. Pack `S EEEE MMM`.

Decode flow:

- `0x7F / 0xFF` → NaN.
- `exp == 0` → subnormal: `val = (-1)^S · 2⁻⁶ · (mant / 8)`.
- Otherwise → normal: `val = (-1)^S · 2^(exp - 7) · (1 + mant / 8)`.

## E5M2

```cpp
uint8_t fp32_to_fp8_e5m2 (float val);
float   fp8_e5m2_to_fp32 (uint8_t fp8);
void    fp32_to_fp8_e5m2_batch(const float *src, uint8_t *dst, int n);
void    fp8_e5m2_to_fp32_batch(const uint8_t *src, float *dst, int n);
```

Differences vs E4M3:

- Bias = 15, `new_exp = exp + 15`.
- Round-to-nearest-even to a 2-bit mantissa.
- Values beyond ±57344 → encode ±Inf.
- IEEE-style ±Inf / NaN are preserved.

## FORMAT DISPATCH

```cpp
uint8_t fp32_to_fp8(float val, tiny_dtype_t dtype);   // pick E4M3 / E5M2
float   fp8_to_fp32(uint8_t fp8, tiny_dtype_t dtype);
void    fp32_to_fp8_batch(const float *src, uint8_t *dst, int n, tiny_dtype_t dtype);
void    fp8_to_fp32_batch(const uint8_t *src, float *dst, int n, tiny_dtype_t dtype);
```

`tiny::quantize / dequantize` (in `tiny_quant.hpp`) auto-dispatch to these helpers based on `params.dtype`, so application code rarely needs to call the batch functions directly.

## USAGE PATTERNS

### Weight compression

```cpp
QuantParams qp = calibrate(weight, TINY_DTYPE_FP8_E4M3);

uint8_t *buf = (uint8_t *)TINY_AI_MALLOC(weight.size);
quantize(weight, buf, qp);

// ... persist to SPIFFS / NVS / deployment blob ...

// Reload + decompress
Tensor restored = Tensor::zeros_like(weight);
dequantize(buf, restored, qp);
```

`example_cnn.cpp` demonstrates the end-to-end 4× compression flow with error stats — see [EXAMPLES/CNN](../../EXAMPLES/CNN/notes.md).

### Gradient communication / checkpointing

Compress gradients to E5M2 before stashing them in PSRAM:

```cpp
QuantParams qp_g = calibrate(grad, TINY_DTYPE_FP8_E5M2);
uint8_t *gb = (uint8_t *)TINY_AI_MALLOC_PSRAM(grad.size);
quantize(grad, gb, qp_g);
```

Decompress back to fp32 when needed.

## ACCURACY & TRADE-OFFS

- **E4M3FN**: relative error ~1/8 = 12.5%; works well for sparsified weights / activations after ReLU.
- **E5M2**: relative error ~1/4 = 25% but range is 128× larger, suitable for the long-tailed distributions of gradients.
- **Recommendation**: pair with INT8 — when INT8 loses precision on layers with extreme dynamic range (typical for attention weights), switch to E4M3 instead.

## SOFTWARE IMPLEMENTATION COST

ESP32-S3 has no FP8 ALU, so every quant / dequant call goes through pure C++ bit packing (with `expf / powf`). Recommendations:

- Treat FP8 as **storage**, not **compute**: decompress once, do math in fp32.
- Quantise / dequantise off the hot path (one-shot at weight load time).
- Batch functions are inlined per element — feel free to add coarser parallelism on top (FreeRTOS tasks, cache-friendly unrolling).
