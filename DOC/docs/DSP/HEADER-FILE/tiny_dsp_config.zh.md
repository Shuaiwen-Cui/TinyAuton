# TinyDSP 配置

!!! INFO
    这个头文件起到配置整个TinyDSP模块的作用，每个子模块都包含了此头文件。它定义了TinyDSP的配置选项和宏，允许用户根据需要进行自定义设置。通过修改这个头文件中的配置选项，用户可以轻松地调整TinyDSP的行为和功能，以满足特定的需求。文档更新速度较慢，可能会与实际代码不一致，请以代码为准。

!!! tip
    平台加速选项请到TinyMath配置文件中进行设置。

```c
/**
 * @file tiny_dsp_config.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief The configuration file for the tiny_dsp middleware.
 * @version 1.0
 * @date 2025-04-27
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

/* DEPENDENCIES */
#include "tiny_math.h"


#ifdef __cplusplus
}
#endif
```