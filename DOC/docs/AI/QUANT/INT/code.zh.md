# 代码

## tiny_quant.h（C 接口）

```c
/**
 * @file tiny_quant.h
 * @brief C-compatible interface for tiny_ai quantisation utilities.
 *        Provides min-max calibration, INT8/INT16 quantise/dequantise,
 *        and a simple INT8 dense-layer forward pass.
 */

#pragma once

#include "tiny_quant_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

tiny_error_t tiny_quant_calibrate_minmax(const float *data, int n,
                                          tiny_dtype_t dtype,
                                          tiny_quant_params_t *params);

tiny_error_t tiny_quant_f32_to_int8(const float *src, int8_t *dst, int n,
                                     const tiny_quant_params_t *params);

tiny_error_t tiny_quant_int8_to_f32(const int8_t *src, float *dst, int n,
                                     const tiny_quant_params_t *params);

tiny_error_t tiny_quant_f32_to_int16(const float *src, int16_t *dst, int n,
                                      const tiny_quant_params_t *params);

tiny_error_t tiny_quant_int16_to_f32(const int16_t *src, float *dst, int n,
                                      const tiny_quant_params_t *params);

tiny_error_t tiny_quant_dense_forward_int8(
    const int8_t  *input,
    const int8_t  *weight,
    const int32_t *bias,
    int8_t        *output,
    int batch, int in_feat, int out_feat,
    float input_scale, float weight_scale, float output_scale);

#ifdef __cplusplus
}
#endif
```

## tiny_quant.c（C 实现）

```c
/**
 * @file tiny_quant.c
 * @brief C implementation of INT8/INT16 quantisation utilities.
 */

#include "tiny_quant.h"
#include <math.h>
#include <stdint.h>

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

tiny_error_t tiny_quant_calibrate_minmax(const float *data, int n,
                                          tiny_dtype_t dtype,
                                          tiny_quant_params_t *params)
{
    if (!data || !params || n <= 0) return TINY_ERR_INVALID_ARG;

    float val_max = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float a = fabsf(data[i]);
        if (a > val_max) val_max = a;
    }
    if (val_max < TINY_MATH_MIN_DENOMINATOR) val_max = 1.0f;

    params->dtype      = dtype;
    params->zero_point = 0;

    if (dtype == TINY_DTYPE_INT8)
        params->scale = val_max / (float)TINY_INT8_MAX;
    else if (dtype == TINY_DTYPE_INT16)
        params->scale = val_max / (float)TINY_INT16_MAX;
    else
        return TINY_ERR_AI_INVALID_DTYPE;

    return TINY_OK;
}

tiny_error_t tiny_quant_f32_to_int8(const float *src, int8_t *dst, int n,
                                     const tiny_quant_params_t *params)
{
    if (!src || !dst || !params || n <= 0) return TINY_ERR_INVALID_ARG;
    if (params->scale < TINY_MATH_MIN_DENOMINATOR) return TINY_ERR_AI_QUANT_FAILED;

    float inv_scale = 1.0f / params->scale;
    for (int i = 0; i < n; i++)
    {
        int32_t q = (int32_t)roundf(src[i] * inv_scale) + params->zero_point;
        dst[i] = (int8_t)clamp_i32(q, TINY_INT8_MIN, TINY_INT8_MAX);
    }
    return TINY_OK;
}

tiny_error_t tiny_quant_int8_to_f32(const int8_t *src, float *dst, int n,
                                     const tiny_quant_params_t *params)
{
    if (!src || !dst || !params || n <= 0) return TINY_ERR_INVALID_ARG;
    for (int i = 0; i < n; i++)
        dst[i] = ((float)src[i] - (float)params->zero_point) * params->scale;
    return TINY_OK;
}

tiny_error_t tiny_quant_f32_to_int16(const float *src, int16_t *dst, int n,
                                      const tiny_quant_params_t *params)
{
    if (!src || !dst || !params || n <= 0) return TINY_ERR_INVALID_ARG;
    if (params->scale < TINY_MATH_MIN_DENOMINATOR) return TINY_ERR_AI_QUANT_FAILED;

    float inv_scale = 1.0f / params->scale;
    for (int i = 0; i < n; i++)
    {
        int32_t q = (int32_t)roundf(src[i] * inv_scale) + params->zero_point;
        dst[i] = (int16_t)clamp_i32(q, TINY_INT16_MIN, TINY_INT16_MAX);
    }
    return TINY_OK;
}

tiny_error_t tiny_quant_int16_to_f32(const int16_t *src, float *dst, int n,
                                      const tiny_quant_params_t *params)
{
    if (!src || !dst || !params || n <= 0) return TINY_ERR_INVALID_ARG;
    for (int i = 0; i < n; i++)
        dst[i] = ((float)src[i] - (float)params->zero_point) * params->scale;
    return TINY_OK;
}

tiny_error_t tiny_quant_dense_forward_int8(
    const int8_t  *input,
    const int8_t  *weight,
    const int32_t *bias,
    int8_t        *output,
    int batch, int in_feat, int out_feat,
    float input_scale, float weight_scale, float output_scale)
{
    if (!input || !weight || !output) return TINY_ERR_INVALID_ARG;
    if (output_scale < TINY_MATH_MIN_DENOMINATOR) return TINY_ERR_AI_QUANT_FAILED;

    float combined_scale = (input_scale * weight_scale) / output_scale;

    for (int b = 0; b < batch; b++)
    {
        for (int o = 0; o < out_feat; o++)
        {
            int32_t acc = bias ? bias[o] : 0;
            const int8_t *w_row = weight + o * in_feat;
            const int8_t *x_row = input  + b * in_feat;
            for (int i = 0; i < in_feat; i++)
                acc += (int32_t)x_row[i] * (int32_t)w_row[i];

            int32_t q = (int32_t)roundf((float)acc * combined_scale);
            output[b * out_feat + o] = (int8_t)clamp_i32(q, TINY_INT8_MIN, TINY_INT8_MAX);
        }
    }
    return TINY_OK;
}
```

## tiny_quant.hpp（C++ 接口）

```cpp
/**
 * @file tiny_quant.hpp
 * @brief C++ quantisation utilities for tiny_ai.
 */

#pragma once

#include "tiny_quant_config.h"
#include "tiny_quant.h"
#include "tiny_fp8.hpp"

#ifdef __cplusplus

#include <stdint.h>

namespace tiny { class Tensor; }

namespace tiny
{

struct QuantParams
{
    tiny_dtype_t dtype      = TINY_DTYPE_INT8;
    float        scale      = 1.0f;
    int          zero_point = 0;

    QuantParams() = default;
    QuantParams(tiny_dtype_t d, float s, int zp = 0) : dtype(d), scale(s), zero_point(zp) {}

    tiny_quant_params_t to_c() const
    {
        tiny_quant_params_t p;
        p.dtype      = dtype;
        p.scale      = scale;
        p.zero_point = zero_point;
        return p;
    }
};

QuantParams calibrate(const Tensor &t, tiny_dtype_t dtype = TINY_DTYPE_INT8);

tiny_error_t quantize  (const Tensor &src, uint8_t *dst, const QuantParams &params);
tiny_error_t dequantize(const uint8_t *src, Tensor &dst, const QuantParams &params);

int8_t *quantize_weights(const Tensor &t, QuantParams &params);

tiny_error_t requantize_int8(const int8_t *src, int8_t *dst, int n,
                              float src_scale, float dst_scale);

} // namespace tiny

#endif // __cplusplus
```

## tiny_quant.cpp（C++ 实现）

```cpp
/**
 * @file tiny_quant.cpp
 * @brief C++ quantisation utilities — Tensor-level calibration and PTQ helpers.
 */

#include "tiny_quant.hpp"
#include "tiny_tensor.hpp"

#ifdef __cplusplus

#include <cmath>
#include <cstring>

namespace tiny
{

QuantParams calibrate(const Tensor &t, tiny_dtype_t dtype)
{
    QuantParams qp;
    qp.dtype = dtype;
    qp.zero_point = 0;

    if (dtype == TINY_DTYPE_FP8_E4M3 || dtype == TINY_DTYPE_FP8_E5M2)
    {
        float max_val = (dtype == TINY_DTYPE_FP8_E4M3) ? TINY_FP8_E4M3_MAX : TINY_FP8_E5M2_MAX;
        float abs_max = 0.0f;
        for (int i = 0; i < t.size; i++)
        {
            float a = fabsf(t.data[i]);
            if (a > abs_max) abs_max = a;
        }
        if (abs_max < TINY_MATH_MIN_DENOMINATOR) abs_max = 1.0f;
        qp.scale = abs_max / max_val;
        return qp;
    }

    tiny_quant_params_t cp;
    tiny_quant_calibrate_minmax(t.data, t.size, dtype, &cp);
    qp.scale      = cp.scale;
    qp.zero_point = cp.zero_point;
    return qp;
}

tiny_error_t quantize(const Tensor &src, uint8_t *dst, const QuantParams &params)
{
    if (!dst) return TINY_ERR_AI_INVALID_SHAPE;

    switch (params.dtype)
    {
        case TINY_DTYPE_INT8:
        {
            tiny_quant_params_t cp = params.to_c();
            return tiny_quant_f32_to_int8(src.data, (int8_t *)dst, src.size, &cp);
        }
        case TINY_DTYPE_INT16:
        {
            tiny_quant_params_t cp = params.to_c();
            return tiny_quant_f32_to_int16(src.data, (int16_t *)dst, src.size, &cp);
        }
        case TINY_DTYPE_FP8_E4M3:
            fp32_to_fp8_e4m3_batch(src.data, dst, src.size); return TINY_OK;
        case TINY_DTYPE_FP8_E5M2:
            fp32_to_fp8_e5m2_batch(src.data, dst, src.size); return TINY_OK;
        default:
            return TINY_ERR_AI_INVALID_DTYPE;
    }
}

tiny_error_t dequantize(const uint8_t *src, Tensor &dst, const QuantParams &params)
{
    if (!src) return TINY_ERR_AI_INVALID_SHAPE;

    switch (params.dtype)
    {
        case TINY_DTYPE_INT8:
        {
            tiny_quant_params_t cp = params.to_c();
            return tiny_quant_int8_to_f32((const int8_t *)src, dst.data, dst.size, &cp);
        }
        case TINY_DTYPE_INT16:
        {
            tiny_quant_params_t cp = params.to_c();
            return tiny_quant_int16_to_f32((const int16_t *)src, dst.data, dst.size, &cp);
        }
        case TINY_DTYPE_FP8_E4M3:
            fp8_e4m3_to_fp32_batch(src, dst.data, dst.size); return TINY_OK;
        case TINY_DTYPE_FP8_E5M2:
            fp8_e5m2_to_fp32_batch(src, dst.data, dst.size); return TINY_OK;
        default:
            return TINY_ERR_AI_INVALID_DTYPE;
    }
}

int8_t *quantize_weights(const Tensor &t, QuantParams &params)
{
    params = calibrate(t, TINY_DTYPE_INT8);
    int8_t *buf = (int8_t *)TINY_AI_MALLOC((size_t)t.size * sizeof(int8_t));
    if (!buf) return nullptr;
    tiny_quant_params_t cp = params.to_c();
    tiny_quant_f32_to_int8(t.data, buf, t.size, &cp);
    return buf;
}

tiny_error_t requantize_int8(const int8_t *src, int8_t *dst, int n,
                              float src_scale, float dst_scale)
{
    if (!src || !dst || n <= 0) return TINY_ERR_INVALID_ARG;
    if (dst_scale < TINY_MATH_MIN_DENOMINATOR) return TINY_ERR_AI_QUANT_FAILED;

    float ratio = src_scale / dst_scale;
    for (int i = 0; i < n; i++)
    {
        int32_t v = (int32_t)roundf((float)src[i] * ratio);
        dst[i] = (int8_t)(v < TINY_INT8_MIN ? TINY_INT8_MIN : (v > TINY_INT8_MAX ? TINY_INT8_MAX : v));
    }
    return TINY_OK;
}

} // namespace tiny

#endif // __cplusplus
```
