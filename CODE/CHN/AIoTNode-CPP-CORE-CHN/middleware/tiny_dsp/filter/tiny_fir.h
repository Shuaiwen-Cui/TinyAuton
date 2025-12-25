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

