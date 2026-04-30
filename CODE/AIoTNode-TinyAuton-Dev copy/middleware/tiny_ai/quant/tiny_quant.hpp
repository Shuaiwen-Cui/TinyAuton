/**
 * @file tiny_quant.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief C++ quantisation utilities for tiny_ai — calibration, PTQ helpers,
 *        and quantised tensor operations built on top of the C interface.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_quant_config.h"
#include "tiny_quant.h"
#include "tiny_fp8.hpp"

#ifdef __cplusplus

#include <stdint.h>

// Forward declaration — defined in core/tiny_tensor.hpp
namespace tiny { class Tensor; }

namespace tiny
{

/* ============================================================================
 * QuantParams — C++ wrapper around tiny_quant_params_t
 * ============================================================================ */
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

/* ============================================================================
 * Calibration
 * ============================================================================ */

/**
 * @brief Calibrate a QuantParams from a Tensor using min-max strategy.
 *        Only INT8, INT16, FP8 E4M3 and FP8 E5M2 dtypes are supported.
 */
QuantParams calibrate(const Tensor &t, tiny_dtype_t dtype = TINY_DTYPE_INT8);

/* ============================================================================
 * Float32 ↔ quantised Tensor conversions
 * ============================================================================ */

/**
 * @brief Quantise a float32 Tensor to INT8, storing results in a uint8 buffer.
 *
 * @param src    Source float Tensor
 * @param dst    Pre-allocated uint8_t buffer (size >= src.size)
 * @param params Quantisation parameters
 */
tiny_error_t quantize(const Tensor &src, uint8_t *dst, const QuantParams &params);

/**
 * @brief Dequantise a uint8 buffer back to a float32 Tensor.
 *
 * @param src    Source quantised buffer
 * @param dst    Destination float Tensor (must already be allocated)
 * @param params Quantisation parameters used during quantization
 */
tiny_error_t dequantize(const uint8_t *src, Tensor &dst, const QuantParams &params);

/* ============================================================================
 * Convenience: quantise entire weight Tensor and return INT8 buffer
 * ============================================================================ */

/**
 * @brief Allocate an int8_t buffer and quantise the float Tensor into it.
 *        Caller is responsible for freeing the returned pointer with TINY_AI_FREE.
 *
 * @param t      Source float Tensor
 * @param params Output quantisation params (filled in by this function)
 * @return Pointer to int8_t buffer, or nullptr on failure
 */
int8_t *quantize_weights(const Tensor &t, QuantParams &params);

/* ============================================================================
 * In-place requantise between two scales (INT8 → INT8)
 * ============================================================================ */
tiny_error_t requantize_int8(const int8_t *src, int8_t *dst, int n,
                              float src_scale, float dst_scale);

} // namespace tiny

#endif // __cplusplus
