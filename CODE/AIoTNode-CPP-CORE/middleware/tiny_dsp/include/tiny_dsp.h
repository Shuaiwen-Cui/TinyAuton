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
