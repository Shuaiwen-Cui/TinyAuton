/**
 * @file tiny_ai_config.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief The configuration file for the tiny_ai middleware.
 * @version 1.0
 * @date 2025-04-27
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */

// ANSI C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

// Lower level dependencies (include before extern "C" to avoid C++ template issues)
#include "tiny_math.h"
#include "tiny_dsp.h"

// Error types
#include "tiny_error_type.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* CONFIGURATION MACROS */

// Maximum number of dimensions supported (can be extended)
#define TINY_AI_TENSOR_MAX_DIMS 8

// Default alignment for memory allocation (for SIMD optimization)
#define TINY_AI_TENSOR_DEFAULT_ALIGN 16

// Enable gradient storage (for training)
#define TINY_AI_ENABLE_GRADIENTS 1

// Enable tensor views (memory sharing without copying)
#define TINY_AI_ENABLE_VIEWS 0  // Disabled for MVP, enable later

/* ERROR CODES */

#define TINY_ERR_AI_BASE 0x90000
#define TINY_ERR_AI_INVALID_ARG (TINY_ERR_AI_BASE + 1)
#define TINY_ERR_AI_INVALID_SHAPE (TINY_ERR_AI_BASE + 2)
#define TINY_ERR_AI_INVALID_DIM (TINY_ERR_AI_BASE + 3)
#define TINY_ERR_AI_NO_MEM (TINY_ERR_AI_BASE + 4)
#define TINY_ERR_AI_NULL_POINTER (TINY_ERR_AI_BASE + 5)
#define TINY_ERR_AI_UNINITIALIZED (TINY_ERR_AI_BASE + 6)
#define TINY_ERR_AI_SHAPE_MISMATCH (TINY_ERR_AI_BASE + 7)
#define TINY_ERR_AI_NOT_SUPPORTED (TINY_ERR_AI_BASE + 8)
#define TINY_ERR_AI_INVALID_STATE (TINY_ERR_AI_BASE + 9)  // Invalid graph state (e.g., cycle detected, order not built)

#ifdef __cplusplus
}
#endif