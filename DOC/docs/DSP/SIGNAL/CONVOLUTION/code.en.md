# CODE

## tiny_conv.h

```c
/**
 * @file tiny_conv.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_conv | code | header 
 * @version 1.0
 * @date 2025-04-27
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// ESP32 DSP Library for Acceleration
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 // ESP32 DSP library

#include "dsps_conv.h"
#include "dspi_conv.h"

#endif

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @name: tiny_conv_f32
 * @brief Convolution function
 * 
 * @param Signal The input signal array
 * @param siglen The length of the input signal array
 * @param Kernel The input kernel array
 * @param kernlen The length of the input kernel array
 * @param convout The output array for the convolution result
 * 
 * @return tiny_error_t 
 */
tiny_error_t tiny_conv_f32(const float *Signal, const int siglen, const float *Kernel, const int kernlen, float *convout);

typedef enum
{
    TINY_PADDING_ZERO = 0,      // Zero padding
    TINY_PADDING_SYMMETRIC = 1, // Symmetric reflection
    TINY_PADDING_PERIODIC = 2   // Periodic extension
} tiny_padding_mode_t;

typedef enum
{
    TINY_CONV_FULL = 0,   // Full convolution (len = siglen + kernlen - 1)
    TINY_CONV_HEAD = 1,   // Head mode (first lkern points)
    TINY_CONV_CENTER = 2, // Centered mode (output siglen points)
    TINY_CONV_TAIL = 3    // Tail mode (last lkern points)
} tiny_conv_mode_t;

/**
 * @name: tiny_conv_ex_f32
 * @brief Extended convolution function with padding and mode options
 *
 * @param Signal The input signal array
 * @param siglen The length of the input signal array
 * @param Kernel The input kernel array
 * @param kernlen The length of the input kernel array
 * @param convout The output array for the convolution result
 * @param padding_mode Padding mode (zero, symmetric, periodic)
 * @param conv_mode Convolution mode (full, head, center, tail)
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_conv_ex_f32(const float *Signal, const int siglen, const float *Kernel, const int kernlen, float *convout, tiny_padding_mode_t padding_mode, tiny_conv_mode_t conv_mode);

#ifdef __cplusplus
}
#endif

```

## tiny_conv.c

```c
/**
 * @file tiny_conv.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_conv | code | source
 * @version 1.0
 * @date 2025-04-27
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_conv.h"

/**
 * @name: tiny_conv_f32
 * @brief Convolution function
 *
 * @param Signal The input signal array
 * @param siglen The length of the input signal array
 * @param Kernel The input kernel array
 * @param kernlen The length of the input kernel array
 * @param convout The output array for the convolution result
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_conv_f32(const float *Signal, const int siglen, const float *Kernel, const int kernlen, float *convout)
{
    if (NULL == Signal || NULL == Kernel || NULL == convout)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }
    if (siglen <= 0 || kernlen <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }
    if (siglen < kernlen)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // ESP32 DSP library
    dsps_conv_f32(Signal, siglen, Kernel, kernlen, convout);
#else
    float *sig = (float *)Signal;
    float *kern = (float *)Kernel;
    int lsig = siglen;
    int lkern = kernlen;

    // stage I
    for (int n = 0; n < lkern; n++)
    {
        size_t k;

        convout[n] = 0;

        for (k = 0; k <= n; k++)
        {
            convout[n] += sig[k] * kern[n - k];
        }
    }

    // stage II
    for (int n = lkern; n < lsig; n++)
    {
        size_t kmin, kmax, k;

        convout[n] = 0;

        kmin = n - lkern + 1;
        kmax = n;
        for (k = kmin; k <= kmax; k++)
        {
            convout[n] += sig[k] * kern[n - k];
        }
    }

    // stage III
    for (int n = lsig; n < lsig + lkern - 1; n++)
    {
        size_t kmin, kmax, k;

        convout[n] = 0;

        kmin = n - lkern + 1;
        kmax = lsig - 1;
        for (k = kmin; k <= kmax; k++)
        {
            convout[n] += sig[k] * kern[n - k];
        }
    }
#endif

    return TINY_OK;
}

/**
 * @name: tiny_conv_ex_f32
 * @brief Extended convolution function with padding and mode options
 *
 * @param Signal The input signal array
 * @param siglen The length of the input signal array
 * @param Kernel The input kernel array
 * @param kernlen The length of the input kernel array
 * @param convout The output array for the convolution result
 * @param padding_mode Padding mode (zero, symmetric, periodic)
 * @param conv_mode Convolution mode (full, head, center, tail)
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_conv_ex_f32(const float *Signal, const int siglen,
                              const float *Kernel, const int kernlen,
                              float *convout,
                              tiny_padding_mode_t padding_mode,
                              tiny_conv_mode_t conv_mode)
{
    if (NULL == Signal || NULL == Kernel || NULL == convout)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }
    if (siglen <= 0 || kernlen <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }
    if (siglen < kernlen)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    if (padding_mode == TINY_PADDING_ZERO && conv_mode == TINY_CONV_FULL)
    {
        dsps_conv_f32(Signal, siglen, Kernel, kernlen, convout);
        return TINY_OK;
    }
#endif

    int pad_len = kernlen - 1;
    int padded_len = siglen + 2 * pad_len;
    float *padded_signal = (float *)calloc(padded_len, sizeof(float));
    if (padded_signal == NULL)
    {
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    // Fill padded signal
    switch (padding_mode)
    {
    case TINY_PADDING_ZERO:
        // Middle copy only, left and right are zeros (calloc already zeroed)
        memcpy(padded_signal + pad_len, Signal, sizeof(float) * siglen);
        break;

    case TINY_PADDING_SYMMETRIC:
        for (int i = 0; i < pad_len; i++)
        {
            padded_signal[pad_len - 1 - i] = Signal[i];                   // Mirror left
            padded_signal[pad_len + siglen + i] = Signal[siglen - 1 - i]; // Mirror right
        }
        memcpy(padded_signal + pad_len, Signal, sizeof(float) * siglen); // Copy center
        break;

    case TINY_PADDING_PERIODIC:
        for (int i = 0; i < pad_len; i++)
        {
            padded_signal[pad_len - 1 - i] = Signal[(siglen - pad_len + i) % siglen]; // Wrap left
            padded_signal[pad_len + siglen + i] = Signal[i % siglen];                 // Wrap right
        }
        memcpy(padded_signal + pad_len, Signal, sizeof(float) * siglen); // Copy center
        break;

    default:
        free(padded_signal);
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Full convolution
    int convlen_full = siglen + kernlen - 1;
    for (int n = 0; n < convlen_full; n++)
    {
        float sum = 0.0f;
        for (int k = 0; k < kernlen; k++)
        {
            sum += padded_signal[n + k] * Kernel[kernlen - 1 - k]; // Convolution is flip+slide
        }
        convout[n] = sum;
    }

    free(padded_signal);

    // Handle output mode
    if (conv_mode == TINY_CONV_FULL)
    {
        return TINY_OK;
    }
    else
    {
        int start_idx = 0;
        int out_len = 0;

        switch (conv_mode)
        {
        case TINY_CONV_HEAD:
            start_idx = 0;
            out_len = kernlen;
            break;
        case TINY_CONV_CENTER:
            start_idx = (kernlen - 1) / 2;
            out_len = siglen;
            break;
        case TINY_CONV_TAIL:
            start_idx = siglen - 1;
            out_len = kernlen;
            break;
        default:
            return TINY_ERR_DSP_INVALID_MODE;
        }

        // Copy the selected part to the beginning
        for (int i = 0; i < out_len; i++)
        {
            convout[i] = convout[start_idx + i];
        }
    }

    return TINY_OK;
}
```