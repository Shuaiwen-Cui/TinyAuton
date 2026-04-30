# TinyAI Configuration

!!! info
    `tiny_ai_config.h` is the global configuration file for the `tiny_ai` middleware. It centralises platform macros, feature flags, the memory strategy, and AI-specific error codes. Documentation may lag the implementation; treat the source as the source of truth.

## DEPENDENCIES

`tiny_ai_config.h` includes `tiny_math.h` and `tiny_dsp.h`, so it inherits all platform macros from `tiny_math` (`MCU_PLATFORM_SELECTED`, `MCU_PLATFORM_ESP32`, `TINY_PI`, `TINY_MATH_MIN_DENOMINATOR`, …).

```c
#include "tiny_math.h"
#include "tiny_dsp.h"
```

## FEATURE FLAGS

| Macro | Default | Meaning |
| --- | --- | --- |
| `TINY_AI_TRAINING_ENABLED` | `1` | Set to `0` to compile inference-only builds: every `backward()` path and gradient buffer is removed by the preprocessor, saving significant RAM and code space |
| `TINY_AI_QUANT_INT8` | `1` | Enable INT8 quantisation path |
| `TINY_AI_QUANT_INT16` | `1` | Enable INT16 quantisation path |
| `TINY_AI_QUANT_FP8` | `1` | Enable software FP8 (E4M3FN / E5M2) path |

!!! note
    ESP32-S3 has no FP8 hardware; FP8 is fully software-emulated. E4M3FN is preferred for weights/activations, while E5M2 is preferred for gradients.

## MEMORY STRATEGY

ESP32-S3 WROOM-1U ships with ~390 KB of internal SRAM and 8 MB of octal PSRAM (80 MHz). `tiny_ai` exposes two allocation macros to separate the hot path from bulk data:

```c
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 && TINY_AI_USE_PSRAM
    #include "esp_heap_caps.h"
    #define TINY_AI_MALLOC(sz)        heap_caps_malloc((sz), MALLOC_CAP_DEFAULT)
    #define TINY_AI_MALLOC_PSRAM(sz)  heap_caps_malloc((sz), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    #define TINY_AI_FREE(ptr)         free(ptr)
    #define TINY_AI_ATTR_IRAM         IRAM_ATTR
    #define TINY_AI_ATTR_PSRAM        EXT_RAM_ATTR
    #define TINY_AI_ATTR_RODATA       RODATA_ATTR
#else
    #define TINY_AI_MALLOC(sz)        malloc(sz)
    #define TINY_AI_MALLOC_PSRAM(sz)  malloc(sz)
    #define TINY_AI_FREE(ptr)         free(ptr)
    ...
#endif
```

**Guidelines:**

- **Small tensors / hot inference path** → `TINY_AI_MALLOC()`, prefers internal SRAM.
- **Weight / activation tensors** → `TINY_AI_MALLOC_PSRAM()`, lives in PSRAM.
- **Constants / functions** → place explicitly with `TINY_AI_ATTR_IRAM`, `TINY_AI_ATTR_PSRAM`, `TINY_AI_ATTR_RODATA`.

`Tensor` defaults to `TINY_AI_MALLOC`, so medium-size MLPs / CNNs entirely live in SRAM. When the model exceeds the SRAM budget, allocate a PSRAM buffer manually and wrap it via `Tensor::from_data()` (a non-owning view).

## ERROR CODES

Error codes share the global `tiny_error_t` enum used by `tiny_math` / `tiny_dsp`. The AI-specific block starts at `0x90000`:

```c
#define TINY_ERR_AI_BASE                0x90000
#define TINY_ERR_AI_INVALID_SHAPE      (TINY_ERR_AI_BASE + 1)  // shape mismatch / illegal dims
#define TINY_ERR_AI_INVALID_DTYPE      (TINY_ERR_AI_BASE + 2)  // unsupported dtype
#define TINY_ERR_AI_ALLOC_FAILED       (TINY_ERR_AI_BASE + 3)  // memory allocation failure
#define TINY_ERR_AI_FORWARD_FAILED     (TINY_ERR_AI_BASE + 4)  // forward-pass error
#define TINY_ERR_AI_BACKWARD_FAILED    (TINY_ERR_AI_BASE + 5)  // backward-pass error
#define TINY_ERR_AI_NOT_COMPILED       (TINY_ERR_AI_BASE + 6)  // feature disabled at compile time
#define TINY_ERR_AI_QUANT_FAILED       (TINY_ERR_AI_BASE + 7)  // quant / dequant error
#define TINY_ERR_AI_INCOMPATIBLE_SHAPE (TINY_ERR_AI_BASE + 8)  // adjacent layers incompatible
#define TINY_ERR_AI_OPTIMIZER_UNINIT   (TINY_ERR_AI_BASE + 9)  // optimizer not initialised
#define TINY_ERR_AI_NO_CACHE           (TINY_ERR_AI_BASE + 10) // backward called before forward
```

## FULL SOURCE

```c
/**
 * @file tiny_ai_config.h
 * @brief Configuration file for the tiny_ai middleware — platform macros,
 *        feature flags, memory strategy, and AI-specific error codes.
 */

#pragma once

#include "tiny_math.h"
#include "tiny_dsp.h"

#ifndef TINY_AI_TRAINING_ENABLED
#define TINY_AI_TRAINING_ENABLED   1
#endif

#ifndef TINY_AI_QUANT_INT8
#define TINY_AI_QUANT_INT8         1
#endif

#ifndef TINY_AI_QUANT_INT16
#define TINY_AI_QUANT_INT16        1
#endif

#ifndef TINY_AI_QUANT_FP8
#define TINY_AI_QUANT_FP8          1
#endif

#ifndef TINY_AI_USE_PSRAM
#define TINY_AI_USE_PSRAM          1
#endif

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 && TINY_AI_USE_PSRAM
    #include "esp_heap_caps.h"
    #define TINY_AI_MALLOC(sz)        heap_caps_malloc((sz), MALLOC_CAP_DEFAULT)
    #define TINY_AI_MALLOC_PSRAM(sz)  heap_caps_malloc((sz), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    #define TINY_AI_FREE(ptr)         free(ptr)
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

#define TINY_ERR_AI_BASE                0x90000
#define TINY_ERR_AI_INVALID_SHAPE      (TINY_ERR_AI_BASE + 1)
#define TINY_ERR_AI_INVALID_DTYPE      (TINY_ERR_AI_BASE + 2)
#define TINY_ERR_AI_ALLOC_FAILED       (TINY_ERR_AI_BASE + 3)
#define TINY_ERR_AI_FORWARD_FAILED     (TINY_ERR_AI_BASE + 4)
#define TINY_ERR_AI_BACKWARD_FAILED    (TINY_ERR_AI_BASE + 5)
#define TINY_ERR_AI_NOT_COMPILED       (TINY_ERR_AI_BASE + 6)
#define TINY_ERR_AI_QUANT_FAILED       (TINY_ERR_AI_BASE + 7)
#define TINY_ERR_AI_INCOMPATIBLE_SHAPE (TINY_ERR_AI_BASE + 8)
#define TINY_ERR_AI_OPTIMIZER_UNINIT   (TINY_ERR_AI_BASE + 9)
#define TINY_ERR_AI_NO_CACHE           (TINY_ERR_AI_BASE + 10)
```
