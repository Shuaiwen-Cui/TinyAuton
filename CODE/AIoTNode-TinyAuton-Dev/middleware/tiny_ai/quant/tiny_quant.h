/**
 * @file tiny_quant.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief C-compatible interface for tiny_ai quantisation utilities.
 *        Provides min-max calibration, INT8/INT16 quantise/dequantise,
 *        and a simple INT8 dense-layer forward pass.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_quant_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * CALIBRATION
 * ============================================================================ */

/**
 * @brief Compute min-max quantisation parameters from a float array.
 *        Produces symmetric INT8 parameters (zero_point = 0).
 *
 * @param data   Input float array
 * @param n      Number of elements
 * @param dtype  Target quantised dtype (INT8 or INT16)
 * @param params Output quantisation parameters
 * @return TINY_OK on success
 */
tiny_error_t tiny_quant_calibrate_minmax(const float *data, int n,
                                          tiny_dtype_t dtype,
                                          tiny_quant_params_t *params);

/* ============================================================================
 * INT8 QUANTISE / DEQUANTISE
 * ============================================================================ */

/**
 * @brief Quantise a float array to INT8 using the provided parameters.
 *
 * @param src    Input float array  (length n)
 * @param dst    Output int8 array  (length n)
 * @param n      Number of elements
 * @param params Quantisation parameters (scale, zero_point)
 * @return TINY_OK on success
 */
tiny_error_t tiny_quant_f32_to_int8(const float *src, int8_t *dst, int n,
                                     const tiny_quant_params_t *params);

/**
 * @brief Dequantise an INT8 array back to float32.
 *
 * @param src    Input int8 array   (length n)
 * @param dst    Output float array (length n)
 * @param n      Number of elements
 * @param params Quantisation parameters (scale, zero_point)
 * @return TINY_OK on success
 */
tiny_error_t tiny_quant_int8_to_f32(const int8_t *src, float *dst, int n,
                                     const tiny_quant_params_t *params);

/* ============================================================================
 * INT16 QUANTISE / DEQUANTISE
 * ============================================================================ */

tiny_error_t tiny_quant_f32_to_int16(const float *src, int16_t *dst, int n,
                                      const tiny_quant_params_t *params);

tiny_error_t tiny_quant_int16_to_f32(const int16_t *src, float *dst, int n,
                                      const tiny_quant_params_t *params);

/* ============================================================================
 * QUANTISED DENSE LAYER FORWARD (INT8 × INT8 → INT32 accumulate)
 * ============================================================================ */

/**
 * @brief Quantised dense (fully connected) forward pass.
 *        Computes output = input × weight^T + bias in INT8 arithmetic,
 *        then requantises to INT8 output.
 *
 * @param input        [batch × in_feat]  int8 input activations
 * @param weight       [out_feat × in_feat] int8 weights
 * @param bias         [out_feat]          int32 bias (pre-scaled)
 * @param output       [batch × out_feat]  int8 output activations
 * @param batch        Batch size
 * @param in_feat      Input feature dimension
 * @param out_feat     Output feature dimension
 * @param input_scale  Scale factor of input activations
 * @param weight_scale Scale factor of weights
 * @param output_scale Scale factor of output activations
 * @return TINY_OK on success
 */
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
