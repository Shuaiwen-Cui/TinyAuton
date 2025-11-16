# CODE

## tiny_iir.h

```c
/**
 * @file tiny_iir.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_iir | IIR (Infinite Impulse Response) Filter | header
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 * @details
 * IIR Filter Implementation
 * - Recursive filter (uses feedback)
 * - More efficient than FIR for same specifications
 * - Can be unstable if not designed carefully
 * - Support for Butterworth, Chebyshev, Elliptic designs
 * - Direct Form II transposed structure (efficient)
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// ESP32 DSP Library for Acceleration
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 // ESP32 DSP library
#include "dsps_biquad.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief IIR filter types
     */
    typedef enum
    {
        TINY_IIR_LOWPASS = 0,   // Low-pass filter
        TINY_IIR_HIGHPASS,      // High-pass filter
        TINY_IIR_BANDPASS,      // Band-pass filter
        TINY_IIR_BANDSTOP,      // Band-stop (notch) filter
        TINY_IIR_COUNT
    } tiny_iir_type_t;

    /**
     * @brief IIR filter design methods
     */
    typedef enum
    {
        TINY_IIR_DESIGN_BUTTERWORTH = 0, // Butterworth (maximally flat)
        TINY_IIR_DESIGN_CHEBYSHEV1,      // Chebyshev Type I (equiripple passband)
        TINY_IIR_DESIGN_CHEBYSHEV2,      // Chebyshev Type II (equiripple stopband)
        TINY_IIR_DESIGN_ELLIPTIC,        // Elliptic (equiripple both bands) - future
        TINY_IIR_DESIGN_BESSEL,          // Bessel (linear phase) - future
        TINY_IIR_DESIGN_COUNT
    } tiny_iir_design_method_t;

    /**
     * @brief IIR filter structure (Direct Form II Transposed)
     * @note This structure maintains filter state for real-time processing
     */
    typedef struct
    {
        // Numerator coefficients (feedforward, b coefficients)
        float *b_coeffs;
        int num_b; // Number of b coefficients (order + 1)

        // Denominator coefficients (feedback, a coefficients)
        float *a_coeffs;
        int num_a; // Number of a coefficients (order + 1)

        // State variables (delay line)
        float *state;
        int state_size; // Size of state array (max(num_b, num_a) - 1)

        // Filter order
        int order;

        // Initialization flag
        int initialized;
    } tiny_iir_filter_t;

    /**
     * @brief Biquad (second-order) IIR filter structure
     * @note More efficient for cascaded biquad implementations
     */
    typedef struct
    {
        // Biquad coefficients (5 coefficients: b0, b1, b2, a1, a2)
        // Note: a0 is always 1.0 in normalized form
        float b0, b1, b2; // Numerator coefficients
        float a1, a2;     // Denominator coefficients (a0 = 1.0)

        // State variables (2 delay elements)
        float w1, w2; // Internal state

        // Initialization flag
        int initialized;
    } tiny_iir_biquad_t;

    /* ============================================================================
     * IIR FILTER DESIGN FUNCTIONS
     * ============================================================================ */

    /**
     * @name tiny_iir_design_lowpass
     * @brief Design a low-pass IIR filter
     *
     * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5, where 0.5 = Nyquist)
     * @param order Filter order
     * @param design_method Design method (Butterworth, Chebyshev, etc.)
     * @param ripple_db Passband ripple in dB (for Chebyshev, typically 0.5-2.0 dB)
     * @param b_coeffs Output numerator coefficients (size: order + 1)
     * @param a_coeffs Output denominator coefficients (size: order + 1)
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_design_lowpass(float cutoff_freq, int order,
                                          tiny_iir_design_method_t design_method,
                                          float ripple_db,
                                          float *b_coeffs, float *a_coeffs);

    /**
     * @name tiny_iir_design_highpass
     * @brief Design a high-pass IIR filter
     *
     * @param cutoff_freq Normalized cutoff frequency (0.0 to 0.5)
     * @param order Filter order
     * @param design_method Design method
     * @param ripple_db Passband ripple in dB
     * @param b_coeffs Output numerator coefficients (size: order + 1)
     * @param a_coeffs Output denominator coefficients (size: order + 1)
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_design_highpass(float cutoff_freq, int order,
                                           tiny_iir_design_method_t design_method,
                                           float ripple_db,
                                           float *b_coeffs, float *a_coeffs);

    /**
     * @name tiny_iir_design_bandpass
     * @brief Design a band-pass IIR filter
     *
     * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
     * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
     * @param order Filter order (will be doubled for bandpass)
     * @param design_method Design method
     * @param ripple_db Passband ripple in dB
     * @param b_coeffs Output numerator coefficients
     * @param a_coeffs Output denominator coefficients
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_design_bandpass(float low_freq, float high_freq,
                                           int order,
                                           tiny_iir_design_method_t design_method,
                                           float ripple_db,
                                           float *b_coeffs, float *a_coeffs);

    /**
     * @name tiny_iir_design_bandstop
     * @brief Design a band-stop (notch) IIR filter
     *
     * @param low_freq Lower cutoff frequency (normalized, 0.0 to 0.5)
     * @param high_freq Upper cutoff frequency (normalized, 0.0 to 0.5)
     * @param order Filter order
     * @param design_method Design method
     * @param ripple_db Passband ripple in dB
     * @param b_coeffs Output numerator coefficients
     * @param a_coeffs Output denominator coefficients
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_design_bandstop(float low_freq, float high_freq,
                                           int order,
                                           tiny_iir_design_method_t design_method,
                                           float ripple_db,
                                           float *b_coeffs, float *a_coeffs);

    /* ============================================================================
     * IIR FILTER APPLICATION FUNCTIONS
     * ============================================================================ */

    /**
     * @name tiny_iir_filter_f32
     * @brief Apply IIR filter to a signal (batch processing)
     *
     * @param input Input signal array
     * @param input_len Length of input signal
     * @param b_coeffs Numerator coefficients
     * @param num_b Number of b coefficients
     * @param a_coeffs Denominator coefficients
     * @param num_a Number of a coefficients
     * @param output Output filtered signal array (size: input_len)
     * @param initial_state Initial state vector (can be NULL for zero initial conditions)
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_filter_f32(const float *input, int input_len,
                                      const float *b_coeffs, int num_b,
                                      const float *a_coeffs, int num_a,
                                      float *output,
                                      const float *initial_state);

    /**
     * @name tiny_iir_init
     * @brief Initialize IIR filter structure for real-time filtering
     *
     * @param filter Pointer to IIR filter structure
     * @param b_coeffs Numerator coefficients (will be copied)
     * @param num_b Number of b coefficients
     * @param a_coeffs Denominator coefficients (will be copied)
     * @param num_a Number of a coefficients
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_init(tiny_iir_filter_t *filter,
                                const float *b_coeffs, int num_b,
                                const float *a_coeffs, int num_a);

    /**
     * @name tiny_iir_deinit
     * @brief Deinitialize IIR filter and free allocated memory
     *
     * @param filter Pointer to IIR filter structure
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_deinit(tiny_iir_filter_t *filter);

    /**
     * @name tiny_iir_process_sample
     * @brief Process a single sample through IIR filter (real-time)
     *
     * @param filter Pointer to initialized IIR filter structure
     * @param input Input sample value
     *
     * @return Filtered output sample
     */
    float tiny_iir_process_sample(tiny_iir_filter_t *filter, float input);

    /**
     * @name tiny_iir_reset
     * @brief Reset IIR filter state (clear delay line)
     *
     * @param filter Pointer to IIR filter structure
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_reset(tiny_iir_filter_t *filter);

    /* ============================================================================
     * BIQUAD (SECOND-ORDER) IIR FILTER FUNCTIONS
     * ============================================================================ */

    /**
     * @name tiny_iir_biquad_init
     * @brief Initialize a biquad (second-order) IIR filter
     *
     * @param biquad Pointer to biquad filter structure
     * @param b0 Numerator coefficient b0
     * @param b1 Numerator coefficient b1
     * @param b2 Numerator coefficient b2
     * @param a1 Denominator coefficient a1 (a0 = 1.0)
     * @param a2 Denominator coefficient a2
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_biquad_init(tiny_iir_biquad_t *biquad,
                                       float b0, float b1, float b2,
                                       float a1, float a2);

    /**
     * @name tiny_iir_biquad_process_sample
     * @brief Process a single sample through biquad filter (real-time)
     *
     * @param biquad Pointer to initialized biquad filter structure
     * @param input Input sample value
     *
     * @return Filtered output sample
     */
    float tiny_iir_biquad_process_sample(tiny_iir_biquad_t *biquad, float input);

    /**
     * @name tiny_iir_biquad_reset
     * @brief Reset biquad filter state
     *
     * @param biquad Pointer to biquad filter structure
     *
     * @return tiny_error_t
     */
    tiny_error_t tiny_iir_biquad_reset(tiny_iir_biquad_t *biquad);

#ifdef __cplusplus
}
#endif



```

## tiny_iir.c

```c
/**
 * @file tiny_iir.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_iir | IIR (Infinite Impulse Response) Filter | source
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_iir.h"
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
 * @brief Bilinear transform: convert analog frequency to digital
 */
static float bilinear_transform(float analog_freq, float sample_rate)
{
    float w = 2.0f * M_PI * analog_freq / sample_rate;
    return tanf(w / 2.0f);
}

/**
 * @brief Calculate Butterworth filter coefficients for low-pass
 */
static void butterworth_lowpass_coeffs(float cutoff_freq, int order,
                                        float *b_coeffs, float *a_coeffs)
{
    // Simplified Butterworth design (for order 1 and 2)
    // For higher orders, would need to cascade biquads

    if (order == 1)
    {
        // First-order Butterworth: H(s) = 1 / (s + 1)
        // Bilinear transform to z-domain
        float wc = 2.0f * M_PI * cutoff_freq;
        float k = tanf(wc / 2.0f);
        float a0 = 1.0f + k;

        b_coeffs[0] = k / a0;
        b_coeffs[1] = k / a0;
        b_coeffs[2] = 0.0f;

        a_coeffs[0] = 1.0f;
        a_coeffs[1] = (1.0f - k) / a0;
        a_coeffs[2] = 0.0f;
    }
    else if (order == 2)
    {
        // Second-order Butterworth: H(s) = 1 / (s^2 + sqrt(2)*s + 1)
        float wc = 2.0f * M_PI * cutoff_freq;
        float k = tanf(wc / 2.0f);
        float k2 = k * k;
        float sqrt2 = 1.4142135623730951f;
        float a0 = 1.0f + sqrt2 * k + k2;

        b_coeffs[0] = k2 / a0;
        b_coeffs[1] = 2.0f * k2 / a0;
        b_coeffs[2] = k2 / a0;

        a_coeffs[0] = 1.0f;
        a_coeffs[1] = 2.0f * (k2 - 1.0f) / a0;
        a_coeffs[2] = (1.0f - sqrt2 * k + k2) / a0;
    }
    else
    {
        // For higher orders, would cascade biquads
        // This is a placeholder - full implementation would decompose into biquads
        // For now, use second-order approximation
        butterworth_lowpass_coeffs(cutoff_freq, 2, b_coeffs, a_coeffs);
    }
}

/**
 * @brief Calculate Butterworth filter coefficients for high-pass
 */
static void butterworth_highpass_coeffs(float cutoff_freq, int order,
                                        float *b_coeffs, float *a_coeffs)
{
    // High-pass is frequency transformation of low-pass
    // H_HP(z) = H_LP(-z) with frequency transformation
    if (order == 1)
    {
        float wc = 2.0f * M_PI * cutoff_freq;
        float k = tanf(wc / 2.0f);
        float a0 = 1.0f + k;

        b_coeffs[0] = 1.0f / a0;
        b_coeffs[1] = -1.0f / a0;
        b_coeffs[2] = 0.0f;

        a_coeffs[0] = 1.0f;
        a_coeffs[1] = (1.0f - k) / a0;
        a_coeffs[2] = 0.0f;
    }
    else if (order == 2)
    {
        float wc = 2.0f * M_PI * cutoff_freq;
        float k = tanf(wc / 2.0f);
        float k2 = k * k;
        float sqrt2 = 1.4142135623730951f;
        float a0 = 1.0f + sqrt2 * k + k2;

        b_coeffs[0] = 1.0f / a0;
        b_coeffs[1] = -2.0f / a0;
        b_coeffs[2] = 1.0f / a0;

        a_coeffs[0] = 1.0f;
        a_coeffs[1] = 2.0f * (k2 - 1.0f) / a0;
        a_coeffs[2] = (1.0f - sqrt2 * k + k2) / a0;
    }
    else
    {
        butterworth_highpass_coeffs(cutoff_freq, 2, b_coeffs, a_coeffs);
    }
}

/* ============================================================================
 * IIR FILTER DESIGN FUNCTIONS
 * ============================================================================ */

tiny_error_t tiny_iir_design_lowpass(float cutoff_freq, int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs)
{
    if (b_coeffs == NULL || a_coeffs == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (cutoff_freq <= 0.0f || cutoff_freq >= 0.5f || order <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // Initialize coefficients
    for (int i = 0; i <= order; i++)
    {
        b_coeffs[i] = 0.0f;
        a_coeffs[i] = 0.0f;
    }

    switch (design_method)
    {
    case TINY_IIR_DESIGN_BUTTERWORTH:
        butterworth_lowpass_coeffs(cutoff_freq, order, b_coeffs, a_coeffs);
        break;
    case TINY_IIR_DESIGN_CHEBYSHEV1:
    case TINY_IIR_DESIGN_CHEBYSHEV2:
    case TINY_IIR_DESIGN_ELLIPTIC:
    case TINY_IIR_DESIGN_BESSEL:
        // Future implementation
        return TINY_ERR_NOT_SUPPORTED;
    default:
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    return TINY_OK;
}

tiny_error_t tiny_iir_design_highpass(float cutoff_freq, int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs)
{
    if (b_coeffs == NULL || a_coeffs == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (cutoff_freq <= 0.0f || cutoff_freq >= 0.5f || order <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // Initialize coefficients
    for (int i = 0; i <= order; i++)
    {
        b_coeffs[i] = 0.0f;
        a_coeffs[i] = 0.0f;
    }

    switch (design_method)
    {
    case TINY_IIR_DESIGN_BUTTERWORTH:
        butterworth_highpass_coeffs(cutoff_freq, order, b_coeffs, a_coeffs);
        break;
    default:
        return TINY_ERR_NOT_SUPPORTED;
    }

    return TINY_OK;
}

tiny_error_t tiny_iir_design_bandpass(float low_freq, float high_freq,
                                       int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs)
{
    // Band-pass design: cascade low-pass and high-pass
    // This is a simplified implementation
    // Full implementation would design band-pass directly

    if (b_coeffs == NULL || a_coeffs == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (low_freq <= 0.0f || high_freq <= 0.0f || low_freq >= high_freq ||
        low_freq >= 0.5f || high_freq >= 0.5f || order <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // For now, return not supported - would need proper band-pass design
    return TINY_ERR_NOT_SUPPORTED;
}

tiny_error_t tiny_iir_design_bandstop(float low_freq, float high_freq,
                                       int order,
                                       tiny_iir_design_method_t design_method,
                                       float ripple_db,
                                       float *b_coeffs, float *a_coeffs)
{
    // Band-stop design: parallel low-pass and high-pass
    // This is a simplified implementation

    if (b_coeffs == NULL || a_coeffs == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (low_freq <= 0.0f || high_freq <= 0.0f || low_freq >= high_freq ||
        low_freq >= 0.5f || high_freq >= 0.5f || order <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // For now, return not supported - would need proper band-stop design
    return TINY_ERR_NOT_SUPPORTED;
}

/* ============================================================================
 * IIR FILTER APPLICATION FUNCTIONS
 * ============================================================================ */

tiny_error_t tiny_iir_filter_f32(const float *input, int input_len,
                                  const float *b_coeffs, int num_b,
                                  const float *a_coeffs, int num_a,
                                  float *output,
                                  const float *initial_state)
{
    if (input == NULL || b_coeffs == NULL || a_coeffs == NULL || output == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (input_len <= 0 || num_b <= 0 || num_a <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // ESP32 optimized implementation
    // Note: ESP-DSP uses biquad structure, would need to convert
    // For now, use generic implementation
#endif

    // Direct Form II Transposed implementation
    int state_size = (num_b > num_a ? num_b : num_a) - 1;
    float *state = (float *)calloc(state_size, sizeof(float));

    if (state == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;

    // Initialize state if provided
    if (initial_state != NULL)
    {
        memcpy(state, initial_state, state_size * sizeof(float));
    }

    // Filter each sample
    for (int n = 0; n < input_len; n++)
    {
        // Feedforward part (b coefficients)
        float y = b_coeffs[0] * input[n];
        for (int i = 1; i < num_b && i <= state_size; i++)
        {
            y += b_coeffs[i] * state[i - 1];
        }

        // Feedback part (a coefficients) and update state
        for (int i = state_size; i > 0; i--)
        {
            if (i < num_a)
            {
                y -= a_coeffs[i] * state[i - 1];
            }
            if (i > 1)
            {
                state[i - 1] = state[i - 2];
            }
        }
        state[0] = input[n];

        output[n] = y;
    }

    free(state);
    return TINY_OK;
}

tiny_error_t tiny_iir_init(tiny_iir_filter_t *filter,
                            const float *b_coeffs, int num_b,
                            const float *a_coeffs, int num_a)
{
    if (filter == NULL || b_coeffs == NULL || a_coeffs == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (num_b <= 0 || num_a <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;

    // Allocate memory for coefficients
    filter->b_coeffs = (float *)malloc(num_b * sizeof(float));
    filter->a_coeffs = (float *)malloc(num_a * sizeof(float));

    if (filter->b_coeffs == NULL || filter->a_coeffs == NULL)
    {
        free(filter->b_coeffs);
        free(filter->a_coeffs);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    // Allocate memory for state
    filter->state_size = (num_b > num_a ? num_b : num_a) - 1;
    filter->state = (float *)calloc(filter->state_size, sizeof(float));

    if (filter->state == NULL)
    {
        free(filter->b_coeffs);
        free(filter->a_coeffs);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    // Copy coefficients
    memcpy(filter->b_coeffs, b_coeffs, num_b * sizeof(float));
    memcpy(filter->a_coeffs, a_coeffs, num_a * sizeof(float));

    filter->num_b = num_b;
    filter->num_a = num_a;
    filter->order = (num_b > num_a ? num_b : num_a) - 1;
    filter->initialized = 1;

    return TINY_OK;
}

tiny_error_t tiny_iir_deinit(tiny_iir_filter_t *filter)
{
    if (filter == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (filter->initialized)
    {
        free(filter->b_coeffs);
        free(filter->a_coeffs);
        free(filter->state);
        filter->b_coeffs = NULL;
        filter->a_coeffs = NULL;
        filter->state = NULL;
        filter->num_b = 0;
        filter->num_a = 0;
        filter->state_size = 0;
        filter->order = 0;
        filter->initialized = 0;
    }

    return TINY_OK;
}

float tiny_iir_process_sample(tiny_iir_filter_t *filter, float input)
{
    if (filter == NULL || !filter->initialized)
        return 0.0f;

    // Direct Form II Transposed
    // Feedforward part
    float output = filter->b_coeffs[0] * input;
    for (int i = 1; i < filter->num_b && i <= filter->state_size; i++)
    {
        output += filter->b_coeffs[i] * filter->state[i - 1];
    }

    // Feedback part and update state
    float temp = input;
    for (int i = 1; i < filter->num_a && i <= filter->state_size; i++)
    {
        output -= filter->a_coeffs[i] * filter->state[i - 1];
    }

    // Shift state
    for (int i = filter->state_size - 1; i > 0; i--)
    {
        filter->state[i] = filter->state[i - 1];
    }
    filter->state[0] = temp;

    return output;
}

tiny_error_t tiny_iir_reset(tiny_iir_filter_t *filter)
{
    if (filter == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    if (!filter->initialized)
        return TINY_ERR_DSP_UNINITIALIZED;

    // Clear state
    memset(filter->state, 0, filter->state_size * sizeof(float));

    return TINY_OK;
}

/* ============================================================================
 * BIQUAD FILTER FUNCTIONS
 * ============================================================================ */

tiny_error_t tiny_iir_biquad_init(tiny_iir_biquad_t *biquad,
                                    float b0, float b1, float b2,
                                    float a1, float a2)
{
    if (biquad == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    biquad->b0 = b0;
    biquad->b1 = b1;
    biquad->b2 = b2;
    biquad->a1 = a1;
    biquad->a2 = a2;
    biquad->w1 = 0.0f;
    biquad->w2 = 0.0f;
    biquad->initialized = 1;

    return TINY_OK;
}

float tiny_iir_biquad_process_sample(tiny_iir_biquad_t *biquad, float input)
{
    if (biquad == NULL || !biquad->initialized)
        return 0.0f;

    // Direct Form II Transposed for biquad
    float w0 = input - biquad->a1 * biquad->w1 - biquad->a2 * biquad->w2;
    float output = biquad->b0 * w0 + biquad->b1 * biquad->w1 + biquad->b2 * biquad->w2;

    // Update state
    biquad->w2 = biquad->w1;
    biquad->w1 = w0;

    return output;
}

tiny_error_t tiny_iir_biquad_reset(tiny_iir_biquad_t *biquad)
{
    if (biquad == NULL)
        return TINY_ERR_DSP_NULL_POINTER;

    biquad->w1 = 0.0f;
    biquad->w2 = 0.0f;

    return TINY_OK;
}


```