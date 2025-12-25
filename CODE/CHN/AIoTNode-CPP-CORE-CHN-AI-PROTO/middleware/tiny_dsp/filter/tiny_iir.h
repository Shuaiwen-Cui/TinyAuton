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

