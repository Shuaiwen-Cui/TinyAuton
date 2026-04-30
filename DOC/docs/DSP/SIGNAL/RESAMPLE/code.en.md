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
 * @brief Downsample by alternately keeping and skipping samples.
 *
 * The pattern is: copy `keep` consecutive samples, advance past the
 * next `skip` samples, repeat until the input is exhausted. The total
 * number of samples written is reported back through @p output_len.
 *
 * Examples (input = [0..9], length 10):
 *   keep=1, skip=1 -> stride 2  -> output = [0, 2, 4, 6, 8]      (5)
 *   keep=1, skip=2 -> stride 3  -> output = [0, 3, 6, 9]         (4)
 *   keep=2, skip=1 -> period 3  -> output = [0, 1, 3, 4, 6, 7, 9](7)
 *   keep=3, skip=2 -> period 5  -> output = [0, 1, 2, 5, 6, 7]   (6)
 *
 * @param input      Input signal
 * @param input_len  Length of the input signal (> 0)
 * @param output     Output buffer; caller must size it to at least
 *                   ceil(input_len * keep / (keep + skip)) elements.
 * @param output_len [out] Number of samples actually written
 * @param keep       Samples to copy in each cycle (>= 1)
 * @param skip       Samples to drop in each cycle (>= 1)
 *
 * @return tiny_error_t
 */
tiny_error_t tiny_downsample_skip_f32(const float *input, int input_len, float *output, int *output_len, int keep, int skip)
{
    if (!input || !output || !output_len)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || keep <= 0 || skip <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    int in_idx  = 0;
    int out_idx = 0;
    while (in_idx < input_len)
    {
        /* Copy up to `keep` consecutive samples. */
        int copy_n = keep;
        if (copy_n > input_len - in_idx) copy_n = input_len - in_idx;
        for (int k = 0; k < copy_n; ++k)
        {
            output[out_idx++] = input[in_idx + k];
        }
        in_idx += copy_n;

        /* Skip the next `skip` samples (no bound check needed; in_idx is
         * clamped by the outer while). */
        in_idx += skip;
    }

    *output_len = out_idx;
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

    /* When target_len is not an exact multiple of input_len, the trailing
     * positions whose source index would land past the end of `input` must
     * be zero filled rather than indexed out of bounds. */
    for (int i = 0; i < target_len; ++i)
    {
        int src = i / factor;
        output[i] = (i % factor == 0 && src < input_len) ? input[src] : 0.0f;
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