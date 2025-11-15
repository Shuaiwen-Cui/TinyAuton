# TinyDSP HEADER FILE

!!! INFO
    This is the main header file of the TinyDSP library. It includes all necessary header files and provides a unified interface to use the functions of the library. After completing the porting of this library in the project, you can insert this header file where you want to use the relevant functions to use all functions in the library. The documentation update speed is slow and may not be consistent with the actual code, please refer to the actual code.

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