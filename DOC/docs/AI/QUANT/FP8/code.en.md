# Code

## tiny_fp8.hpp

```cpp
/**
 * @file tiny_fp8.hpp
 * @brief Software FP8 implementation for tiny_ai.
 *        Supports OCP FP8 E4M3FN (weights/activations) and E5M2 (gradients).
 *        ESP32-S3 has no FP8 hardware; values are stored as uint8_t and
 *        upcasted to float32 for all arithmetic.
 */

#pragma once

#include "tiny_quant_config.h"
#include <stdint.h>

#ifdef __cplusplus

#include <cstring>
#include <cmath>

namespace tiny
{

uint8_t fp32_to_fp8_e4m3 (float val);
float   fp8_e4m3_to_fp32 (uint8_t fp8);
void    fp32_to_fp8_e4m3_batch(const float *src, uint8_t *dst, int n);
void    fp8_e4m3_to_fp32_batch(const uint8_t *src, float *dst, int n);

uint8_t fp32_to_fp8_e5m2 (float val);
float   fp8_e5m2_to_fp32 (uint8_t fp8);
void    fp32_to_fp8_e5m2_batch(const float *src, uint8_t *dst, int n);
void    fp8_e5m2_to_fp32_batch(const uint8_t *src, float *dst, int n);

uint8_t fp32_to_fp8(float val, tiny_dtype_t dtype);
float   fp8_to_fp32(uint8_t fp8, tiny_dtype_t dtype);
void    fp32_to_fp8_batch(const float *src, uint8_t *dst, int n, tiny_dtype_t dtype);
void    fp8_to_fp32_batch(const uint8_t *src, float *dst, int n, tiny_dtype_t dtype);

} // namespace tiny

#endif // __cplusplus
```

## tiny_fp8.cpp

```cpp
/**
 * @file tiny_fp8.cpp
 * @brief Software FP8 implementation — E4M3FN and E5M2 formats.
 */

#include "tiny_fp8.hpp"

#ifdef __cplusplus

#include <cstring>
#include <cmath>
#include <stdint.h>

namespace tiny
{

static inline uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, sizeof(u)); return u; }
static inline float    bits_f32(uint32_t u) { float f; memcpy(&f, &u, sizeof(f)); return f; }

static inline float f32_nan()  { return bits_f32(0x7FC00000u); }
static inline float f32_inf()  { return bits_f32(0x7F800000u); }
static inline float f32_ninf() { return bits_f32(0xFF800000u); }

// ============================================================================
// FP8 E4M3FN
// ============================================================================
uint8_t fp32_to_fp8_e4m3(float val)
{
    if (val != val) return TINY_FP8_E4M3_NAN;

    uint32_t bits = f32_bits(val);
    uint8_t  sign = (uint8_t)((bits >> 31) & 1u);
    int      exp  = (int)((bits >> 23) & 0xFFu) - 127;
    uint32_t mant = bits & 0x7FFFFFu;

    if (exp > 8)  return (sign << 7u) | 0x7Eu;
    if (exp < -9) return (sign << 7u);

    int new_exp = exp + 7;

    if (new_exp <= 0)
    {
        uint32_t full_mant = (mant | 0x800000u);
        int shift = 21 + (1 - new_exp);
        if (shift >= 24) return (sign << 7u);
        uint8_t m3 = (uint8_t)((full_mant + (1u << (shift - 1))) >> shift);
        if (m3 > 7u) m3 = 7u;
        return (uint8_t)((sign << 7u) | m3);
    }

    uint32_t round_bit = 1u << 20;
    uint32_t sticky    = mant & (round_bit - 1u);
    uint8_t  m3        = (uint8_t)((mant + round_bit) >> 21);

    if (m3 > 7u) { m3 = 0u; new_exp++; }
    if (new_exp > 15 || (new_exp == 15 && m3 == 7u))
        return (uint8_t)((sign << 7u) | 0x7Eu);

    (void)sticky;
    return (uint8_t)((sign << 7u) | ((uint8_t)(new_exp & 0xFu) << 3u) | (m3 & 0x7u));
}

float fp8_e4m3_to_fp32(uint8_t fp8)
{
    if ((fp8 & 0x7Fu) == 0x7Fu) return f32_nan();

    uint8_t sign = (fp8 >> 7u) & 1u;
    uint8_t exp4 = (fp8 >> 3u) & 0xFu;
    uint8_t mant = fp8 & 0x7u;

    float val;
    if (exp4 == 0u)
        val = (float)(sign ? -1.0f : 1.0f) * powf(2.0f, -6.0f) * ((float)mant / 8.0f);
    else
        val = (float)(sign ? -1.0f : 1.0f) * powf(2.0f, (float)((int)exp4 - 7)) * (1.0f + (float)mant / 8.0f);
    return val;
}

void fp32_to_fp8_e4m3_batch(const float *src, uint8_t *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = fp32_to_fp8_e4m3(src[i]);
}

void fp8_e4m3_to_fp32_batch(const uint8_t *src, float *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = fp8_e4m3_to_fp32(src[i]);
}

// ============================================================================
// FP8 E5M2
// ============================================================================
uint8_t fp32_to_fp8_e5m2(float val)
{
    uint32_t bits = f32_bits(val);
    uint8_t  sign = (uint8_t)((bits >> 31u) & 1u);
    int      exp  = (int)((bits >> 23u) & 0xFFu) - 127;
    uint32_t mant = bits & 0x7FFFFFu;

    if (val != val) return (uint8_t)((sign << 7u) | TINY_FP8_E5M2_NAN);
    if ((bits & 0x7FFFFFFFu) == 0x7F800000u)
        return (uint8_t)((sign << 7u) | TINY_FP8_E5M2_INF);

    if (exp > 15)  return (uint8_t)((sign << 7u) | TINY_FP8_E5M2_INF);
    if (exp < -16) return (sign << 7u);

    int new_exp = exp + 15;

    if (new_exp <= 0)
    {
        uint32_t full_mant = (mant | 0x800000u);
        int shift = 22 + (1 - new_exp);
        if (shift >= 24) return (sign << 7u);
        uint8_t m2 = (uint8_t)((full_mant + (1u << (shift - 1))) >> shift);
        if (m2 > 3u) m2 = 3u;
        return (uint8_t)((sign << 7u) | m2);
    }

    uint32_t round_bit = 1u << 21;
    uint8_t  m2        = (uint8_t)((mant + round_bit) >> 22);

    if (m2 > 3u) { m2 = 0u; new_exp++; }
    if (new_exp > 30 || (new_exp == 30 && m2 == 3u))
        return (uint8_t)((sign << 7u) | TINY_FP8_E5M2_INF);

    return (uint8_t)((sign << 7u) | ((uint8_t)(new_exp & 0x1Fu) << 2u) | (m2 & 0x3u));
}

float fp8_e5m2_to_fp32(uint8_t fp8)
{
    uint8_t sign = (fp8 >> 7u) & 1u;
    uint8_t exp5 = (fp8 >> 2u) & 0x1Fu;
    uint8_t mant = fp8 & 0x3u;

    if (exp5 == 0x1Fu)
    {
        if (mant == 0u) return sign ? f32_ninf() : f32_inf();
        return f32_nan();
    }

    float val;
    if (exp5 == 0u)
        val = (sign ? -1.0f : 1.0f) * powf(2.0f, -14.0f) * ((float)mant / 4.0f);
    else
        val = (sign ? -1.0f : 1.0f) * powf(2.0f, (float)((int)exp5 - 15)) * (1.0f + (float)mant / 4.0f);
    return val;
}

void fp32_to_fp8_e5m2_batch(const float *src, uint8_t *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = fp32_to_fp8_e5m2(src[i]);
}

void fp8_e5m2_to_fp32_batch(const uint8_t *src, float *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = fp8_e5m2_to_fp32(src[i]);
}

// ============================================================================
// Dispatch
// ============================================================================
uint8_t fp32_to_fp8(float val, tiny_dtype_t dtype)
{
    if (dtype == TINY_DTYPE_FP8_E5M2) return fp32_to_fp8_e5m2(val);
    return fp32_to_fp8_e4m3(val);
}

float fp8_to_fp32(uint8_t fp8, tiny_dtype_t dtype)
{
    if (dtype == TINY_DTYPE_FP8_E5M2) return fp8_e5m2_to_fp32(fp8);
    return fp8_e4m3_to_fp32(fp8);
}

void fp32_to_fp8_batch(const float *src, uint8_t *dst, int n, tiny_dtype_t dtype)
{
    if (dtype == TINY_DTYPE_FP8_E5M2) { fp32_to_fp8_e5m2_batch(src, dst, n); return; }
    fp32_to_fp8_e4m3_batch(src, dst, n);
}

void fp8_to_fp32_batch(const uint8_t *src, float *dst, int n, tiny_dtype_t dtype)
{
    if (dtype == TINY_DTYPE_FP8_E5M2) { fp8_e5m2_to_fp32_batch(src, dst, n); return; }
    fp8_e4m3_to_fp32_batch(src, dst, n);
}

} // namespace tiny

#endif // __cplusplus
```
