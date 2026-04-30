# Code

## tiny_quant_config.h

```c
/**
 * @file tiny_quant_config.h
 * @brief Shared type definitions and constants for the tiny_ai quantization
 *        subsystem (INT8, INT16, FP8 E4M3FN, FP8 E5M2).
 */

#pragma once

#include "tiny_ai_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    TINY_DTYPE_FLOAT32  = 0,
    TINY_DTYPE_INT16    = 1,
    TINY_DTYPE_INT8     = 2,
    TINY_DTYPE_FP8_E4M3 = 3,
    TINY_DTYPE_FP8_E5M2 = 4,
} tiny_dtype_t;

typedef struct
{
    tiny_dtype_t dtype;
    float        scale;
    int          zero_point;
} tiny_quant_params_t;

#define TINY_FP8_E4M3_MAX   448.0f
#define TINY_FP8_E4M3_MIN  (-448.0f)
#define TINY_FP8_E4M3_NAN   0x7Fu

#define TINY_FP8_E5M2_MAX  57344.0f
#define TINY_FP8_E5M2_MIN  (-57344.0f)
#define TINY_FP8_E5M2_INF  0x7Cu
#define TINY_FP8_E5M2_NAN  0x7Fu

#define TINY_INT8_MAX   127
#define TINY_INT8_MIN  (-128)

#define TINY_INT16_MAX  32767
#define TINY_INT16_MIN (-32768)

#ifdef __cplusplus
}
#endif
```
