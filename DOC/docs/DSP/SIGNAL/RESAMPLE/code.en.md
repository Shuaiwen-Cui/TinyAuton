# CODE

## tiny_resample.h

```c
/**
 * @file tiny_resample.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_resample | code | header
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// ESP32 DSP Library for Acceleration
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 // ESP32 DSP library

#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @name tiny_downsample_skip_f32
     * @brief Downsample a signal by a given factor using skipping
     *
     * @param input pointer to the input signal array
     * @param input_len length of the input signal array
     * @param output pointer to the output signal array
     * @param output_len pointer to the length of the output signal array
     * @param keep number of samples to keep
     * @param skip number of samples to skip
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_downsample_skip_f32(const float *input, int input_len, float *output, int *output_len, int keep, int skip);

    /**
     * @name tiny_upsample_zero_f32
     * @brief Upsample a signal using zero-insertion between samples
     *
     * @param input pointer to the input signal array
     * @param input_len length of the input signal array
     * @param output pointer to the output signal array
     * @param target_len target length for the output signal array
     * @return tiny_error_t
     */
    tiny_error_t tiny_upsample_zero_f32(const float *input, int input_len, float *output, int target_len);

    /**
     * @name: tiny_resample_f32
     * @brief Resample a signal to a target length
     *
     * @param input pointer to the input signal array
     * @param input_len length of the input signal array
     * @param output pointer to the output signal array
     * @param target_len target length for the output signal array
     * @return tiny_error_t
     */
    tiny_error_t tiny_resample_f32(const float *input,
                                   int input_len,
                                   float *output,
                                   int target_len);

#ifdef __cplusplus
}
#endif


```

## tiny_resample.c

```c
/**
 * @file tiny_resample.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_resample | code | source
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_resample.h" // tiny_resample header

/**
 * @name tiny_downsample_skip_f32
 * @brief Downsample a signal by a given factor using skipping
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param output_len pointer to the length of the output signal array
 * @param keep number of samples to keep
 * @param skip number of samples to skip
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_downsample_skip_f32(const float *input, int input_len, float *output, int *output_len, int keep, int skip)
{
    if (!input || !output || !output_len)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || keep <= 0 || skip <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    int out_len = input_len / skip;
    *output_len = out_len;

    for (int i = 0; i < out_len; i++)
    {
        output[i] = input[i * skip];
    }

    return TINY_OK;
}

/**
 * @name tiny_upsample_zero_f32
 * @brief Upsample a signal using zero-insertion between samples
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param target_len target length for the output signal array
 * @return tiny_error_t
 */
tiny_error_t tiny_upsample_zero_f32(const float *input, int input_len, float *output, int target_len)
{
    if (!input || !output)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || target_len <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    int factor = target_len / input_len;
    if (factor <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    for (int i = 0; i < target_len; i++)
    {
        output[i] = (i % factor == 0) ? input[i / factor] : 0.0f;
    }

    return TINY_OK;
}


/**
 * @name: tiny_resample_f32
 * @brief Resample a signal to a target length
 *
 * @param input pointer to the input signal array
 * @param input_len length of the input signal array
 * @param output pointer to the output signal array
 * @param target_len target length for the output signal array
 * @return tiny_error_t
 */
tiny_error_t tiny_resample_f32(const float *input,
                               int input_len,
                               float *output,
                               int target_len)
{
    if (!input || !output)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || target_len <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    float ratio = (float)(target_len) / (float)(input_len);

    for (int i = 0; i < target_len; i++)
    {
        float pos = i / ratio;
        int index = (int)floorf(pos);
        float frac = pos - index;

        if (index >= input_len - 1)
            output[i] = input[input_len - 1]; // Clamp at end
        else
            output[i] = input[index] * (1.0f - frac) + input[index + 1] * frac;
    }

    return TINY_OK;
}


```