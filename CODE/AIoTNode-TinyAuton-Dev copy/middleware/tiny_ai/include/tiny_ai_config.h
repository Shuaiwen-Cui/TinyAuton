/**
 * @file tiny_ai_config.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief Configuration file for the tiny_ai middleware — platform macros,
 *        feature flags, memory strategy, and AI-specific error codes.
 * @version 1.0
 * @date 2025-05-01
 * @copyright Copyright (c) 2025
 */

#pragma once

/* ============================================================================
 * DEPENDENCIES
 * ============================================================================ */
#include "tiny_math.h"
#include "tiny_dsp.h"

/* ============================================================================
 * FEATURE FLAGS
 * ============================================================================ */

// Set to 0 for inference-only builds: all backward() paths and gradient
// buffers are compiled out, saving significant RAM and code space.
#ifndef TINY_AI_TRAINING_ENABLED
#define TINY_AI_TRAINING_ENABLED   1
#endif

// Quantization format support (each can be disabled independently)
#ifndef TINY_AI_QUANT_INT8
#define TINY_AI_QUANT_INT8         1
#endif

#ifndef TINY_AI_QUANT_INT16
#define TINY_AI_QUANT_INT16        1
#endif

// FP8 is implemented in software (no hardware FP8 on ESP32-S3).
// Both E4M3FN (better for weights/activations) and E5M2 (better for
// gradients) formats are supported.
#ifndef TINY_AI_QUANT_FP8
#define TINY_AI_QUANT_FP8          1
#endif

/* ============================================================================
 * MEMORY STRATEGY
 * ============================================================================ */
// ESP32-S3 WROOM-1U: ~390 KB internal SRAM + 8 MB PSRAM (octal, 80 MHz).
// Large tensors (weights, activation maps) should live in PSRAM; hot
// inference paths and small buffers should stay in internal RAM / IRAM.

#ifndef TINY_AI_USE_PSRAM
#define TINY_AI_USE_PSRAM          1
#endif

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 && TINY_AI_USE_PSRAM
    #include "esp_heap_caps.h"
    // Default heap allocation (prefers internal RAM for speed)
    #define TINY_AI_MALLOC(sz)        heap_caps_malloc((sz), MALLOC_CAP_DEFAULT)
    // PSRAM allocation (for large weight/activation tensors)
    #define TINY_AI_MALLOC_PSRAM(sz)  heap_caps_malloc((sz), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    #define TINY_AI_FREE(ptr)         free(ptr)
    // Section placement attributes
    #define TINY_AI_ATTR_IRAM         IRAM_ATTR
    #define TINY_AI_ATTR_PSRAM        EXT_RAM_ATTR
    #define TINY_AI_ATTR_RODATA       RODATA_ATTR
#else
    #define TINY_AI_MALLOC(sz)        malloc(sz)
    #define TINY_AI_MALLOC_PSRAM(sz)  malloc(sz)
    #define TINY_AI_FREE(ptr)         free(ptr)
    #define TINY_AI_ATTR_IRAM
    #define TINY_AI_ATTR_PSRAM
    #define TINY_AI_ATTR_RODATA
#endif

/* ============================================================================
 * AI ERROR CODES
 * ============================================================================ */
#ifdef __cplusplus
extern "C"
{
#endif

#define TINY_ERR_AI_BASE                0x90000
#define TINY_ERR_AI_INVALID_SHAPE      (TINY_ERR_AI_BASE + 1)  /*!< Shape mismatch or illegal dimensions */
#define TINY_ERR_AI_INVALID_DTYPE      (TINY_ERR_AI_BASE + 2)  /*!< Unsupported data type */
#define TINY_ERR_AI_ALLOC_FAILED       (TINY_ERR_AI_BASE + 3)  /*!< Memory allocation failure */
#define TINY_ERR_AI_FORWARD_FAILED     (TINY_ERR_AI_BASE + 4)  /*!< Forward pass error */
#define TINY_ERR_AI_BACKWARD_FAILED    (TINY_ERR_AI_BASE + 5)  /*!< Backward pass error */
#define TINY_ERR_AI_NOT_COMPILED       (TINY_ERR_AI_BASE + 6)  /*!< Feature disabled at compile time */
#define TINY_ERR_AI_QUANT_FAILED       (TINY_ERR_AI_BASE + 7)  /*!< Quantization / dequantization error */
#define TINY_ERR_AI_INCOMPATIBLE_SHAPE (TINY_ERR_AI_BASE + 8)  /*!< Layers have incompatible shapes */
#define TINY_ERR_AI_OPTIMIZER_UNINIT   (TINY_ERR_AI_BASE + 9)  /*!< Optimizer not initialised */
#define TINY_ERR_AI_NO_CACHE           (TINY_ERR_AI_BASE + 10) /*!< Backward called before forward */

#ifdef __cplusplus
}
#endif
