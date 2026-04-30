/**
 * @file tiny_quant.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief C implementation of INT8/INT16 quantisation utilities.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#include "tiny_quant.h"
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * Helper: clamp integer to int8 / int16 range
 * ============================================================================ */
static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ============================================================================
 * Calibration — min-max, symmetric (zero_point = 0)
 * ============================================================================ */
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

/* ============================================================================
 * INT8 quantise / dequantise
 * ============================================================================ */
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

/* ============================================================================
 * INT16 quantise / dequantise
 * ============================================================================ */
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

/* ============================================================================
 * Quantised Dense forward (INT8 × INT8 → INT32 accumulate → INT8)
 * ============================================================================ */
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

    // Combined scale: converts INT32 accumulator → output INT8 space
    float combined_scale = (input_scale * weight_scale) / output_scale;
    float inv_out_scale  = 1.0f / output_scale;

    for (int b = 0; b < batch; b++)
    {
        for (int o = 0; o < out_feat; o++)
        {
            int32_t acc = bias ? bias[o] : 0;
            const int8_t *w_row = weight + o * in_feat;
            const int8_t *x_row = input  + b * in_feat;
            for (int i = 0; i < in_feat; i++)
                acc += (int32_t)x_row[i] * (int32_t)w_row[i];

            // Requantise to output scale
            int32_t q = (int32_t)roundf((float)acc * combined_scale);
            output[b * out_feat + o] = (int8_t)clamp_i32(q, TINY_INT8_MIN, TINY_INT8_MAX);
        }
    }
    (void)inv_out_scale;
    return TINY_OK;
}
