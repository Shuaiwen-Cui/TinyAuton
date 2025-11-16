# 代码

## tiny_fir.h

```c
/**
 * @file tiny_fir.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fir | FIR (Finite Impulse Response) Filter | header
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 * @details
 * FIR Filter Implementation
 * - Always stable (no poles, only zeros)
 * - Linear phase response possible
 * - Implemented via convolution
 * - Support for low-pass, high-pass, band-pass, band-stop filters
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// tiny_dsp submodules
#include "tiny_conv.h" // FIR filtering uses convolution

// ESP32 DSP Library for Acceleration
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 // ESP32 DSP library
#include "dsps_fir.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief FIR filter types
     */
    typedef enum
    {
        TINY_FIR_LOWPASS = 0,   // Low-pass filter
        TINY_FIR_HIGHPASS,      // High-pass filter
        TINY_FIR_BANDPASS,      // Band-pass filter
        TINY_FIR_BANDSTOP,      // Band-stop (notch) filter
        TINY_FIR_COUNT
    } tiny_fir_type_t;

    /**
     * @brief FIR filter design methods
     */
    typedef enum
    {
        TINY_FIR_DESIGN_WINDOW = 0,        // Window method (Hamming, Hanning, etc.)
        TINY_FIR_DESIGN_EQUIRIPPLE,        // Equiripple (Parks-McClellan) - future
        TINY_FIR_DESIGN_FREQ_SAMPLING,     // Frequency sampling - future
        TINY_FIR_DESIGN_COUNT
    } tiny_fir_design_method_t;

    /**
     * @brief Window functions for FIR design
     */
    typedef enum
    {
        TINY_FIR_WINDOW_RECTANGULAR = 0, // Rectangular (no window)
        TINY_FIR_WINDOW_HAMMING,         // Hamming window
        TINY_FIR_WINDOW_HANNING,         // Hanning window
        TINY_FIR_WINDOW_BLACKMAN,        // Blackman window
        TINY_FIR_WINDOW_KAISER,          // Kaiser window - future
        TINY_FIR_WINDOW_COUNT
    } tiny_fir_window_t;

    /**
     * @brief FIR filter structure
     * @note For real-time filtering, use this structure to maintain state
     */
    typedef struct
    {
        float *coefficients;  // Filter coefficients (taps)
        int num_taps;         // Number of filter taps (coefficients)
        float *delay_line;    // Delay line for real-time filtering
        int delay_index;      // Current position in delay line
        int initialized;      // Initialization flag
    } tiny_fir_filter_t;

    /* ============================================================================
     * FIR FILTER DESIGN FUNCTIONS
     * ============================================================================ */

    /**
     * @name tiny_fir_design_lowpass
     * @brief Design a low-pass FIR filter using window method
     *
     * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5, where 0.5 = Nyquist)
     * @param num_taps Number of filter taps (should be odd for linear phase)
     * @param window Window function to use
     * @param coefficients Output array for filter coefficients (size: num_taps)
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_fir_design_lowpass(float cutoff_freq, int num_taps,
                                         tiny_fir_window_t window,
                                         float *coefficients);

    /**
     * @name tiny_fir_design_highpass
     * @brief Design a high-pass FIR filter using window method
     *
     * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
     * @param num_taps Number of filter taps (should be odd)
     * @param window Window function to use
     * @param coefficients Output array for filter coefficients (size: num_taps)
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_fir_design_highpass(float cutoff_freq, int num_taps,
                                           tiny_fir_window_t window,
                                           float *coefficients);

    /**
     * @name tiny_fir_design_bandpass
     * @brief Design a band-pass FIR filter using window method
     *
     * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
     * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
     * @param num_taps Number of filter taps (should be odd)
     * @param window Window function to use
     * @param coefficients Output array for filter coefficients (size: num_taps)
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_fir_design_bandpass(float low_freq, float high_freq,
                                           int num_taps,
                                           tiny_fir_window_t window,
                                           float *coefficients);

    /**
     * @name tiny_fir_design_bandstop
     * @brief Design a band-stop (notch) FIR filter using window method
     *
     * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
     * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
     * @param num_taps Number of filter taps (should be odd)
     * @param window Window function to use
     * @param coefficients Output array for filter coefficients (size: num_taps)
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_fir_design_bandstop(float low_freq, float high_freq,
                                           int num_taps,
                                           tiny_fir_window_t window,
                                           float *coefficients);

    /* ============================================================================
     * FIR FILTER APPLICATION FUNCTIONS
     * ============================================================================ */

    /**
     * @name tiny_fir_filter_f32
     * @brief Apply FIR filter to a signal (batch processing)
     * @note This function uses convolution internally
     *
     * @param input Input signal array
     * @param input_len Length of input signal
     * @param coefficients FIR filter coefficients (taps)
     * @param num_taps Number of filter taps
     * @param output Output filtered signal array (size: input_len)
     * @param padding_mode Padding mode for boundary handling
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_fir_filter_f32(const float *input, int input_len,
                                      const float *coefficients, int num_taps,
                                      float *output,
                                      tiny_padding_mode_t padding_mode);

    /**
     * @name tiny_fir_init
     * @brief Initialize FIR filter structure for real-time filtering
     *
     * @param filter Pointer to FIR filter structure
     * @param coefficients Filter coefficients (will be copied internally)
     * @param num_taps Number of filter taps
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_fir_init(tiny_fir_filter_t *filter,
                                const float *coefficients, int num_taps);

    /**
     * @name tiny_fir_deinit
     * @brief Deinitialize FIR filter and free allocated memory
     *
     * @param filter Pointer to FIR filter structure
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_fir_deinit(tiny_fir_filter_t *filter);

    /**
     * @name tiny_fir_process_sample
     * @brief Process a single sample through FIR filter (real-time)
     *
     * @param filter Pointer to initialized FIR filter structure
     * @param input Input sample value
     *
     * @return Filtered output sample
     */
    float tiny_fir_process_sample(tiny_fir_filter_t *filter, float input);

    /**
     * @name tiny_fir_reset
     * @brief Reset FIR filter state (clear delay line)
     *
     * @param filter Pointer to FIR filter structure
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_fir_reset(tiny_fir_filter_t *filter);

#ifdef __cplusplus
}
#endif

```

## tiny_fir.c

```c
/**
 * @file tiny_fir.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fir | FIR (Finite Impulse Response) Filter | source
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_fir.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Apply window function to filter coefficients
 */
static void apply_window(float *coeffs, int num_taps, tiny_fir_window_t window)
{
    if (window == TINY_FIR_WINDOW_RECTANGULAR)
    {
        // No window applied (already rectangular)
        return;
    }

    for (int i = 0; i < num_taps; i++)
    {
        float w = 1.0f;
        float n = (float)i;
        float N = (float)(num_taps - 1);

        switch (window)
        {
        case TINY_FIR_WINDOW_HAMMING:
            w = 0.54f - 0.46f * cosf(2.0f * M_PI * n / N);
            break;
        case TINY_FIR_WINDOW_HANNING:
            w = 0.5f * (1.0f - cosf(2.0f * M_PI * n / N));
            break;
        case TINY_FIR_WINDOW_BLACKMAN:
            w = 0.42f - 0.5f * cosf(2.0f * M_PI * n / N) + 0.08f * cosf(4.0f * M_PI * n / N);
            break;
        default:
            w = 1.0f;
            break;
        }

        coeffs[i] *= w;
    }
}

/**
 * @brief Generate ideal low-pass filter impulse response
 */
static void generate_ideal_lowpass(float *coeffs, int num_taps, float cutoff_freq)
{
    int center = (num_taps - 1) / 2;

    for (int i = 0; i < num_taps; i++)
    {
        int n = i - center;
        if (n == 0)
        {
            coeffs[i] = 2.0f * cutoff_freq;
        }
        else
        {
            coeffs[i] = sinf(2.0f * M_PI * cutoff_freq * n) / (M_PI * n);
        }
    }
}

/**
 * @brief Generate ideal high-pass filter impulse response
 */
static void generate_ideal_highpass(float *coeffs, int num_taps, float cutoff_freq)
{
    int center = (num_taps - 1) / 2;

    for (int i = 0; i < num_taps; i++)
    {
        int n = i - center;
        if (n == 0)
        {
            coeffs[i] = 1.0f - 2.0f * cutoff_freq;
        }
        else
        {
            coeffs[i] = -sinf(2.0f * M_PI * cutoff_freq * n) / (M_PI * n);
        }
    }
}

/* ============================================================================
 * FIR FILTER DESIGN FUNCTIONS
 * ============================================================================ */

tiny_error_t tiny_fir_design_lowpass(float cutoff_freq, int num_taps,
                                      tiny_fir_window_t window,
                                      float *coefficients)
{
    if (coefficients == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (num_taps <= 0 || cutoff_freq <= 0.0f || cutoff_freq >= 0.5f)
        return TINY_ERR_DSP_INVALID_PARAM;

    if (num_taps % 2 == 0)
        return TINY_ERR_DSP_INVALID_PARAM; // Should be odd for linear phase

    // Generate ideal low-pass filter
    generate_ideal_lowpass(coefficients, num_taps, cutoff_freq);

    // Apply window
    apply_window(coefficients, num_taps, window);

    return TINY_OK;
}

tiny_error_t tiny_fir_design_highpass(float cutoff_freq, int num_taps,
                                       tiny_fir_window_t window,
                                       float *coefficients)
{
    if (coefficients == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (num_taps <= 0 || cutoff_freq <= 0.0f || cutoff_freq >= 0.5f)
        return TINY_ERR_DSP_INVALID_PARAM;

    if (num_taps % 2 == 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // Generate ideal high-pass filter
    generate_ideal_highpass(coefficients, num_taps, cutoff_freq);

    // Apply window
    apply_window(coefficients, num_taps, window);

    return TINY_OK;
}

tiny_error_t tiny_fir_design_bandpass(float low_freq, float high_freq,
                                       int num_taps,
                                       tiny_fir_window_t window,
                                       float *coefficients)
{
    if (coefficients == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (num_taps <= 0 || low_freq <= 0.0f || high_freq <= 0.0f ||
        low_freq >= 0.5f || high_freq >= 0.5f || low_freq >= high_freq)
        return TINY_ERR_DSP_INVALID_PARAM;

    if (num_taps % 2 == 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    int center = (num_taps - 1) / 2;
    float center_freq = (low_freq + high_freq) / 2.0f;
    float bandwidth = high_freq - low_freq;

    // Generate band-pass filter (low-pass shifted to center frequency)
    for (int i = 0; i < num_taps; i++)
    {
        int n = i - center;
        if (n == 0)
        {
            coefficients[i] = 2.0f * bandwidth;
        }
        else
        {
            coefficients[i] = 2.0f * bandwidth * cosf(2.0f * M_PI * center_freq * n) *
                              sinf(M_PI * bandwidth * n) / (M_PI * n);
        }
    }

    // Apply window
    apply_window(coefficients, num_taps, window);

    return TINY_OK;
}

tiny_error_t tiny_fir_design_bandstop(float low_freq, float high_freq,
                                       int num_taps,
                                       tiny_fir_window_t window,
                                       float *coefficients)
{
    if (coefficients == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (num_taps <= 0 || low_freq <= 0.0f || high_freq <= 0.0f ||
        low_freq >= 0.5f || high_freq >= 0.5f || low_freq >= high_freq)
        return TINY_ERR_DSP_INVALID_PARAM;

    if (num_taps % 2 == 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // Band-stop = All-pass - Band-pass
    // First generate low-pass and high-pass, then combine
    float *lp_coeffs = (float *)malloc(num_taps * sizeof(float));
    float *hp_coeffs = (float *)malloc(num_taps * sizeof(float));

    if (!lp_coeffs || !hp_coeffs)
    {
        free(lp_coeffs);
        free(hp_coeffs);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    // Generate low-pass at low_freq
    generate_ideal_lowpass(lp_coeffs, num_taps, low_freq);
    apply_window(lp_coeffs, num_taps, window);

    // Generate high-pass at high_freq
    generate_ideal_highpass(hp_coeffs, num_taps, high_freq);
    apply_window(hp_coeffs, num_taps, window);

    // Combine: band-stop = low-pass + high-pass
    for (int i = 0; i < num_taps; i++)
    {
        coefficients[i] = lp_coeffs[i] + hp_coeffs[i];
    }

    free(lp_coeffs);
    free(hp_coeffs);

    return TINY_OK;
}

/* ============================================================================
 * FIR FILTER APPLICATION FUNCTIONS
 * ============================================================================ */

tiny_error_t tiny_fir_filter_f32(const float *input, int input_len,
                                   const float *coefficients, int num_taps,
                                   float *output,
                                   tiny_padding_mode_t padding_mode)
{
    if (input == NULL || coefficients == NULL || output == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || num_taps <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // tiny_conv_ex_f32 with TINY_CONV_CENTER mode internally performs full convolution
    // and writes to output buffer, which may cause buffer overflow if output buffer
    // is smaller than input_len + num_taps - 1. To avoid this, we use TINY_CONV_FULL
    // mode with a temporary buffer, then extract the center portion.
    int conv_full_len = input_len + num_taps - 1;
    float *temp_buffer = (float *)malloc(conv_full_len * sizeof(float));
    if (temp_buffer == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;

    // Perform full convolution
    tiny_error_t err = tiny_conv_ex_f32(input, input_len, coefficients, num_taps,
                                         temp_buffer, padding_mode, TINY_CONV_FULL);
    if (err != TINY_OK)
    {
        free(temp_buffer);
        return err;
    }

    // Extract center portion (equivalent to TINY_CONV_CENTER mode)
    int center_start = (num_taps - 1) / 2;
    for (int i = 0; i < input_len; i++)
    {
        output[i] = temp_buffer[center_start + i];
    }

    free(temp_buffer);
    return TINY_OK;
}

tiny_error_t tiny_fir_init(tiny_fir_filter_t *filter,
                             const float *coefficients, int num_taps)
{
    if (filter == NULL || coefficients == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (num_taps <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // Allocate memory for coefficients
    filter->coefficients = (float *)malloc(num_taps * sizeof(float));
    if (filter->coefficients == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;

    // Allocate memory for delay line
    filter->delay_line = (float *)calloc(num_taps, sizeof(float));
    if (filter->delay_line == NULL)
    {
        free(filter->coefficients);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    // Copy coefficients
    memcpy(filter->coefficients, coefficients, num_taps * sizeof(float));

    filter->num_taps = num_taps;
    filter->delay_index = 0;
    filter->initialized = 1;

    return TINY_OK;
}

tiny_error_t tiny_fir_deinit(tiny_fir_filter_t *filter)
{
    if (filter == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (filter->initialized)
    {
        free(filter->coefficients);
        free(filter->delay_line);
        filter->coefficients = NULL;
        filter->delay_line = NULL;
        filter->num_taps = 0;
        filter->delay_index = 0;
        filter->initialized = 0;
    }

    return TINY_OK;
}

float tiny_fir_process_sample(tiny_fir_filter_t *filter, float input)
{
    if (filter == NULL || !filter->initialized)
        return 0.0f;

    // Add input to delay line
    filter->delay_line[filter->delay_index] = input;

    // Compute output (convolution)
    float output = 0.0f;
    int idx = filter->delay_index;
    for (int i = 0; i < filter->num_taps; i++)
    {
        output += filter->coefficients[i] * filter->delay_line[idx];
        idx = (idx + 1) % filter->num_taps; // Circular buffer
    }

    // Update delay index
    filter->delay_index = (filter->delay_index + filter->num_taps - 1) % filter->num_taps;

    return output;
}

tiny_error_t tiny_fir_reset(tiny_fir_filter_t *filter)
{
    if (filter == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (!filter->initialized)
        return TINY_ERR_DSP_UNINITIALIZED;

    // Clear delay line
    memset(filter->delay_line, 0, filter->num_taps * sizeof(float));
    filter->delay_index = 0;

    return TINY_OK;
}


```

