# TinyDSP 头文件

!!! INFO
    这是TinyDSP库的主头文件。它包含所有必要的头文件，并提供了一个统一的接口来使用库的功能。在项目中完成该库的移植后，在需要使用相关函数的地方插入该头文件即可使用库内的所有函数。文档更新速度较慢，可能与实际代码不一致，请以实际代码为准。

```c
/**
 * @file tiny_dsp.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_dsp | entrance file
 * @version 1.0
 * @date 2025-04-28
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */

// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// [signal]

// signal - convolution
#include "tiny_conv.h"
#include "tiny_conv_test.h"

// signal - correlation & crosss correlation
#include "tiny_corr.h"
#include "tiny_corr_test.h"

// signal - resample & decimate
#include "tiny_resample.h"
#include "tiny_resample_test.h"

// [filter]

// [transform]

// transform - discrete wavelet transform
#include "tiny_dwt.h"
#include "tiny_dwt_test.h"

// [support]




#ifdef __cplusplus
extern "C"
{
#endif

    

#ifdef __cplusplus
}
#endif

```
