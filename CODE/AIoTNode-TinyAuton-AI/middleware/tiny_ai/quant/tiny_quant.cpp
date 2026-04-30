/**
 * @file tiny_quant.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief C++ quantisation utilities — Tensor-level calibration and PTQ helpers.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#include "tiny_quant.hpp"
#include "tiny_tensor.hpp"

#ifdef __cplusplus

#include <cmath>
#include <cstring>

namespace tiny
{

// ============================================================================
// Calibration
// ============================================================================

QuantParams calibrate(const Tensor &t, tiny_dtype_t dtype)
{
    QuantParams qp;
    qp.dtype = dtype;
    qp.zero_point = 0;

    if (dtype == TINY_DTYPE_FP8_E4M3 || dtype == TINY_DTYPE_FP8_E5M2)
    {
        // FP8: symmetric, scale = max_abs / fp8_max
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

    // INT8 / INT16 — delegate to C calibration
    tiny_quant_params_t cp;
    tiny_quant_calibrate_minmax(t.data, t.size, dtype, &cp);
    qp.scale      = cp.scale;
    qp.zero_point = cp.zero_point;
    return qp;
}

// ============================================================================
// quantize / dequantize
// ============================================================================

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
            fp32_to_fp8_e4m3_batch(src.data, dst, src.size);
            return TINY_OK;
        case TINY_DTYPE_FP8_E5M2:
            fp32_to_fp8_e5m2_batch(src.data, dst, src.size);
            return TINY_OK;
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
            fp8_e4m3_to_fp32_batch(src, dst.data, dst.size);
            return TINY_OK;
        case TINY_DTYPE_FP8_E5M2:
            fp8_e5m2_to_fp32_batch(src, dst.data, dst.size);
            return TINY_OK;
        default:
            return TINY_ERR_AI_INVALID_DTYPE;
    }
}

// ============================================================================
// quantize_weights — allocate + fill int8 buffer
// ============================================================================

int8_t *quantize_weights(const Tensor &t, QuantParams &params)
{
    params = calibrate(t, TINY_DTYPE_INT8);
    int8_t *buf = (int8_t *)TINY_AI_MALLOC((size_t)t.size * sizeof(int8_t));
    if (!buf) return nullptr;
    tiny_quant_params_t cp = params.to_c();
    tiny_quant_f32_to_int8(t.data, buf, t.size, &cp);
    return buf;
}

// ============================================================================
// requantize_int8
// ============================================================================

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
