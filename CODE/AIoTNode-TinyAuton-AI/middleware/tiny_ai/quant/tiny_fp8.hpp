/**
 * @file tiny_fp8.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Software FP8 implementation for tiny_ai.
 *        Supports OCP FP8 E4M3FN (weights/activations) and E5M2 (gradients).
 *        ESP32-S3 has no FP8 hardware; values are stored as uint8_t and
 *        upcasted to float32 for all arithmetic.
 * @version 1.0
 * @date 2026-04-30
 * @copyright Copyright (c) 2026
 */

#pragma once

#include "tiny_quant_config.h"
#include <stdint.h>

#ifdef __cplusplus

#include <cstring>
#include <cmath>

namespace tiny
{

/* ============================================================================
 * FP8 E4M3FN  (1 sign + 4 exponent + 3 mantissa, bias = 7)
 * Exponent all-1 + mantissa all-1 is NaN (0x7F / 0xFF); no infinities.
 * Normal range: ±448.0    Subnormal min: 2^-9
 * ============================================================================ */

/// Encode a float32 value to FP8 E4M3FN (nearest, clamp on overflow)
uint8_t fp32_to_fp8_e4m3(float val);

/// Decode an FP8 E4M3FN byte back to float32
float fp8_e4m3_to_fp32(uint8_t fp8);

/// Batch encode: n float32 values → n FP8 E4M3FN bytes
void fp32_to_fp8_e4m3_batch(const float *src, uint8_t *dst, int n);

/// Batch decode: n FP8 E4M3FN bytes → n float32 values
void fp8_e4m3_to_fp32_batch(const uint8_t *src, float *dst, int n);

/* ============================================================================
 * FP8 E5M2  (1 sign + 5 exponent + 2 mantissa, bias = 15)
 * Supports ±inf (0x7C / 0xFC) and NaN (0x7D-0x7F / 0xFD-0xFF).
 * Normal range: ±57344.0   Subnormal min: 2^-16
 * ============================================================================ */

/// Encode a float32 value to FP8 E5M2 (nearest, clamp on overflow)
uint8_t fp32_to_fp8_e5m2(float val);

/// Decode an FP8 E5M2 byte back to float32
float fp8_e5m2_to_fp32(uint8_t fp8);

/// Batch encode: n float32 values → n FP8 E5M2 bytes
void fp32_to_fp8_e5m2_batch(const float *src, uint8_t *dst, int n);

/// Batch decode: n FP8 E5M2 bytes → n float32 values
void fp8_e5m2_to_fp32_batch(const uint8_t *src, float *dst, int n);

/* ============================================================================
 * Dispatch helpers (select format via tiny_dtype_t)
 * ============================================================================ */

/// Encode one float32 → FP8 (E4M3 or E5M2 based on dtype)
uint8_t fp32_to_fp8(float val, tiny_dtype_t dtype);

/// Decode one FP8 byte → float32 (E4M3 or E5M2 based on dtype)
float fp8_to_fp32(uint8_t fp8, tiny_dtype_t dtype);

/// Batch encode with dtype dispatch
void fp32_to_fp8_batch(const float *src, uint8_t *dst, int n, tiny_dtype_t dtype);

/// Batch decode with dtype dispatch
void fp8_to_fp32_batch(const uint8_t *src, float *dst, int n, tiny_dtype_t dtype);

} // namespace tiny

#endif // __cplusplus
