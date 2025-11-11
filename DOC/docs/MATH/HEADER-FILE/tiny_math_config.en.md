# TinyMath CONFIGURATION

!!! INFO
    This header file serves to configure the entire TinyMath module, and each submodule includes this header file. It defines the configuration options and macros for TinyMath, allowing users to customize settings as needed. By modifying the configuration options in this header file, users can easily adjust the behavior and functionality of TinyMath to meet specific requirements. The documentation may be updated slowly and may not match the actual code, so please refer to the code for accuracy.

!!! tip
    This component includes macro definitions for selecting platforms, allowing users to choose different platforms for compilation as needed. By switching to the corresponding platform macro, users can leverage platform acceleration features to enhance performance. For example, for the ESP32 platform, TinyMath will automatically select the ESP32 DSP library for compilation, achieving more efficient mathematical operations.

```c
/**
 * @file tiny_math_config.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief The configuration file for the tiny_math middleware.
 * @version 1.0
 * @date 2025-04-14
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

/* DEPENDENCIES */

// ANSI C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

// lower level
#include "tiny_toolbox.h"

// this level
#include "tiny_error_type.h"
#include "tiny_constants.h"

/* PLATFORM SELECTION */

// available platforms
#define MCU_PLATFORM_GENERIC 0
#define MCU_PLATFORM_ESP32 1 // here, we utilize the ESP built-in DSP library, it will automatically select the optimized version
#define MCU_PLATFORM_STM32 2
#define MCU_PLATFORM_RISCV 3

// choose one platform
#define MCU_PLATFORM_SELECTED MCU_PLATFORM_ESP32

#ifdef __cplusplus
}
#endif
```