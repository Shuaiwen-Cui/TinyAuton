/**
 * @file tiny_quant_config.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Shared type definitions and constants for the tiny_ai quantization
 *        subsystem (INT8, INT16, FP8 E4M3FN, FP8 E5M2).
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "tiny_ai_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ============================================================================
 * DATA TYPE ENUM
 * ============================================================================ */
typedef enum
{
    TINY_DTYPE_FLOAT32  = 0,  ///< 32-bit IEEE 754 float  (native compute type)
    TINY_DTYPE_INT16    = 1,  ///< Signed 16-bit integer  (fixed-point)
    TINY_DTYPE_INT8     = 2,  ///< Signed 8-bit integer   (most HW-friendly on ESP32-S3)
    TINY_DTYPE_FP8_E4M3 = 3,  ///< 8-bit float E4M3FN: range ±448, good for weights/activations
    TINY_DTYPE_FP8_E5M2 = 4,  ///< 8-bit float E5M2:   range ±57344, good for gradients
} tiny_dtype_t;

/* ============================================================================
 * QUANTISATION PARAMETERS
 * ============================================================================ */
typedef struct
{
    tiny_dtype_t dtype;       ///< Target quantised dtype
    float        scale;       ///< Scale factor:  float_val = scale * (quant_val - zero_point)
    int          zero_point;  ///< Zero point offset (0 for symmetric / FP8)
} tiny_quant_params_t;

/* ============================================================================
 * FORMAT LIMITS
 * ============================================================================ */

// FP8 E4M3FN (OCP spec): no infinities, NaN = 0x7F / 0xFF
#define TINY_FP8_E4M3_MAX   448.0f
#define TINY_FP8_E4M3_MIN  (-448.0f)
#define TINY_FP8_E4M3_NAN   0x7Fu

// FP8 E5M2 (OCP spec): supports ±inf and NaN
#define TINY_FP8_E5M2_MAX  57344.0f
#define TINY_FP8_E5M2_MIN  (-57344.0f)
#define TINY_FP8_E5M2_INF  0x7Cu  ///< +infinity
#define TINY_FP8_E5M2_NAN  0x7Fu

// INT8 symmetric range used for weights
#define TINY_INT8_MAX    127
#define TINY_INT8_MIN   (-128)

// INT16 symmetric range
#define TINY_INT16_MAX   32767
#define TINY_INT16_MIN  (-32768)

#ifdef __cplusplus
}
#endif
