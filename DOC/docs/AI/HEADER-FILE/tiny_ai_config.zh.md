# TinyAI 配置

!!! info
    `tiny_ai_config.h` 是 `tiny_ai` 中间件的全局配置文件，集中定义了平台宏、特性开关、内存策略以及 AI 专用的错误码。文档更新速度较慢，可能与实际代码不一致，请以实际代码为准。

## 依赖

`tiny_ai_config.h` 依赖 `tiny_math.h` 和 `tiny_dsp.h`，因此自动获得来自 `tiny_math` 的平台宏（`MCU_PLATFORM_SELECTED`、`MCU_PLATFORM_ESP32`、`TINY_PI`、`TINY_MATH_MIN_DENOMINATOR` 等）。

```c
#include "tiny_math.h"
#include "tiny_dsp.h"
```

## 特性开关

| 宏 | 默认值 | 含义 |
| --- | --- | --- |
| `TINY_AI_TRAINING_ENABLED` | `1` | 设为 `0` 编译纯推理版本：所有 `backward()` 路径与梯度缓冲被预处理移除，显著节省 RAM 与代码空间 |
| `TINY_AI_QUANT_INT8` | `1` | 启用 INT8 量化路径 |
| `TINY_AI_QUANT_INT16` | `1` | 启用 INT16 量化路径 |
| `TINY_AI_QUANT_FP8` | `1` | 启用软件 FP8（E4M3FN / E5M2）路径 |

!!! note
    ESP32-S3 没有 FP8 硬件，FP8 全部以软件方式实现。E4M3FN 适合权重 / 激活，E5M2 适合梯度。

## 内存策略

ESP32-S3 WROOM-1U 大约具有 390 KB 内部 SRAM 与 8 MB Octal PSRAM (80 MHz)。`tiny_ai` 通过两组分配宏区分热路径与大数据：

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

**使用建议**：

- **小张量 / 推理热路径**：使用 `TINY_AI_MALLOC()`，优先落在内部 SRAM。
- **权重 / 激活图等大张量**：使用 `TINY_AI_MALLOC_PSRAM()`，落在 PSRAM。
- **常数 / 函数段**：通过 `TINY_AI_ATTR_IRAM`、`TINY_AI_ATTR_PSRAM`、`TINY_AI_ATTR_RODATA` 显式放置。

`Tensor` 类内部默认使用 `TINY_AI_MALLOC`，因此中等规模的 MLP / CNN 默认全部驻留 SRAM。当模型超过 SRAM 容量时，可以将权重张量替换为 `TINY_AI_MALLOC_PSRAM` 分配的缓冲区，再通过 `Tensor::from_data()` 包装为非拥有视图。

## 错误码

错误码与 `tiny_math` / `tiny_dsp` 同样基于 `tiny_error_t` 枚举，AI 专属错误码段从 `0x90000` 开始：

```c
#define TINY_ERR_AI_BASE                0x90000
#define TINY_ERR_AI_INVALID_SHAPE      (TINY_ERR_AI_BASE + 1)  // 形状不匹配 / 非法维度
#define TINY_ERR_AI_INVALID_DTYPE      (TINY_ERR_AI_BASE + 2)  // 不支持的数据类型
#define TINY_ERR_AI_ALLOC_FAILED       (TINY_ERR_AI_BASE + 3)  // 内存分配失败
#define TINY_ERR_AI_FORWARD_FAILED     (TINY_ERR_AI_BASE + 4)  // 前向传播错误
#define TINY_ERR_AI_BACKWARD_FAILED    (TINY_ERR_AI_BASE + 5)  // 反向传播错误
#define TINY_ERR_AI_NOT_COMPILED       (TINY_ERR_AI_BASE + 6)  // 该特性在编译期被关闭
#define TINY_ERR_AI_QUANT_FAILED       (TINY_ERR_AI_BASE + 7)  // 量化 / 反量化失败
#define TINY_ERR_AI_INCOMPATIBLE_SHAPE (TINY_ERR_AI_BASE + 8)  // 层间形状不兼容
#define TINY_ERR_AI_OPTIMIZER_UNINIT   (TINY_ERR_AI_BASE + 9)  // 优化器未初始化
#define TINY_ERR_AI_NO_CACHE           (TINY_ERR_AI_BASE + 10) // 在 forward 之前调用了 backward
```

## 完整源码

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
