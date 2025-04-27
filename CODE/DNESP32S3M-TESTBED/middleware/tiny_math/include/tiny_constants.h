/**
 * @file tiny_constants.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the constants used in the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-15
 * @copyright Copyright (c) 2025
 */

#ifndef __TINY_CONSTANTS_H__
#define __TINY_CONSTANTS_H__

#ifdef __cplusplus
extern "C"
{
#endif

// =======================================
//  Logical Constants
// =======================================
#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef NULL
#define NULL ((void *)0)
#endif

// =======================================
//  Math Constants (float/double safe)
// =======================================
#define TINY_PI 3.14159265358979323846f
#define TINY_TWO_PI 6.28318530717958647692f
#define TINY_HALF_PI 1.57079632679489661923f
#define TINY_E 2.71828182845904523536f
#define TINY_SQRT2 1.41421356237309504880f
#define TINY_INV_SQRT2 0.70710678118654752440f

#define TINY_DEG2RAD(x) ((x) * TINY_PI / 180.0f)
#define TINY_RAD2DEG(x) ((x) * 180.0f / TINY_PI)

// =======================================
//  Bitmask & Bit Manipulation
// =======================================

// Bitwise operations
#define TINY_BIT(n) (1U << (n)) // e.g. TINY_BIT(3) = 0b00001000
#define TINY_BIT_SET(x, n) ((x) |= TINY_BIT(n))
#define TINY_BIT_CLEAR(x, n) ((x) &= ~TINY_BIT(n))
#define TINY_BIT_TOGGLE(x, n) ((x) ^= TINY_BIT(n))
#define TINY_BIT_CHECK(x, n) (((x) >> (n)) & 0x1U)

// Common bit masks
#define TINY_MASK_4BIT 0x0FU
#define TINY_MASK_8BIT 0xFFU
#define TINY_MASK_16BIT 0xFFFFU
#define TINY_MASK_32BIT 0xFFFFFFFFU

// =======================================
//  Fixed-Point Scaling Factors
// =======================================
#define TINY_Q7_SCALE 128          // 2^7
#define TINY_Q15_SCALE 32768       // 2^15
#define TINY_Q31_SCALE 2147483648U // 2^31

// =======================================
//  User-Defined Constants (Optional)
// =======================================
#define TINY_MATH_MIN_DENOMINATOR 1e-6f // Minimum denominator for safe division
#define TINY_MATH_MIN_POSITIVE_INPUT_F32 1e-12f // Minimum positive input for float operations



#ifdef __cplusplus
}
#endif

#endif /* __TINY_CONSTANTS_H__ */
