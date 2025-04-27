# TinyMath 配置

!!! INFO
    这个头文件起到配置整个TinyMath模块的作用，每个子模块都包含了此头文件。它定义了TinyMath的配置选项和宏，允许用户根据需要进行自定义设置。通过修改这个头文件中的配置选项，用户可以轻松地调整TinyMath的行为和功能，以满足特定的需求。文档更新速度较慢，可能会与实际代码不一致，请以代码为准。

!!! tip
    该组件内包括选择平台的宏定义，用户可以根据需要选择不同的平台进行编译。切换到对应平台的宏后，可以利用平台加速的特性来提升性能。例如，对于ESP32平台，TinyMath会自动选择ESP32的DSP库进行编译，从而实现更高效的数学运算。

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

#ifndef __TINY_MATH_CONFIG__
#define __TINY_MATH_CONFIG__

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
#define MCU_PLATFORM_GENERIC     0
#define MCU_PLATFORM_ESP32       1 // here, we utilize the ESP built-in DSP library, it will automatically select the optimized version
#define MCU_PLATFORM_STM32       2
#define MCU_PLATFORM_RISCV       3

// choose one platform
#define MCU_PLATFORM_SELECTED    MCU_PLATFORM_ESP32


#ifdef __cplusplus
}
#endif

#endif /* __TINY_MATH_CONFIG__ */
```