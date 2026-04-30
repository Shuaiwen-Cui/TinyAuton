/**
 * @file tiny_fft.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fft | code | header
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// ESP32 DSP Library for Acceleration
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32 // ESP32 DSP library
#include "dsps_fft2r.h"
#include "dsps_wind_hann.h"
#include "dsps_wind_blackman.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Window function types for FFT preprocessing
     */
    typedef enum
    {
        TINY_FFT_WINDOW_NONE = 0,    // No window (rectangular)
        TINY_FFT_WINDOW_HANNING,     // Hanning window
        TINY_FFT_WINDOW_HAMMING,     // Hamming window
        TINY_FFT_WINDOW_BLACKMAN,    // Blackman window
        TINY_FFT_WINDOW_COUNT
    } tiny_fft_window_t;

    /**
     * @name: tiny_fft_init
     * @brief Initialize FFT tables (required before using FFT functions)
     * @note This function should be called once at startup
     * @param fft_size Maximum FFT size to support (must be power of 2)
     * @return tiny_error_t
     */
    tiny_error_t tiny_fft_init(int fft_size);

    /**
     * @name: tiny_fft_deinit
     * @brief Deinitialize FFT tables and free resources
     * @return tiny_error_t
     */
    tiny_error_t tiny_fft_deinit(void);

    /**
     * @name: tiny_fft_f32
     * @brief Perform FFT on real-valued input signal
     * @param input Input signal array (real values)
     * @param input_len Length of input signal (must be power of 2)
     * @param output_fft Output FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
     *                   Size must be at least input_len * 2
     * @param window Window function to apply before FFT (optional)
     * @return tiny_error_t
     */
    tiny_error_t tiny_fft_f32(const float *input, int input_len, float *output_fft, tiny_fft_window_t window);

    /**
     * @name: tiny_fft_ifft_f32
     * @brief Perform inverse FFT to reconstruct time-domain signal
     * @param input_fft Input FFT array (complex: [Re0, Im0, Re1, Im1, ...])
     * @param fft_len Length of FFT (number of complex points)
     * @param output Output reconstructed signal (real values)
     *               Size must be at least fft_len
     * @return tiny_error_t
     */
    tiny_error_t tiny_fft_ifft_f32(const float *input_fft, int fft_len, float *output);

    /**
     * @name: tiny_fft_magnitude_f32
     * @brief Calculate magnitude spectrum from FFT result
     * @param fft_result FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
     * @param fft_len Length of FFT (number of complex points)
     * @param magnitude Output magnitude spectrum (real values)
     *                  Size must be at least fft_len
     * @return tiny_error_t
     */
    tiny_error_t tiny_fft_magnitude_f32(const float *fft_result, int fft_len, float *magnitude);

    /**
     * @name: tiny_fft_power_spectrum_f32
     * @brief Calculate power spectrum density (PSD) from FFT result
     * @param fft_result FFT result (complex array: [Re0, Im0, Re1, Im1, ...])
     * @param fft_len Length of FFT (number of complex points)
     * @param power Output power spectrum (real values)
     *              Size must be at least fft_len
     * @return tiny_error_t
     */
    tiny_error_t tiny_fft_power_spectrum_f32(const float *fft_result, int fft_len, float *power);

    /**
     * @name: tiny_fft_find_peak_frequency
     * @brief Find the frequency with maximum power (useful for structural health monitoring)
     * @param power_spectrum Power spectrum array
     * @param fft_len Length of power spectrum
     * @param sample_rate Sampling rate of the original signal (Hz)
     * @param peak_freq Output peak frequency (Hz)
     * @param peak_power Output peak power value
     * @return tiny_error_t
     */
    tiny_error_t tiny_fft_find_peak_frequency(const float *power_spectrum, int fft_len, float sample_rate, float *peak_freq, float *peak_power);

    /**
     * @name: tiny_fft_find_top_frequencies
     * @brief Find top N frequencies with highest power
     * @param power_spectrum Power spectrum array
     * @param fft_len Length of power spectrum
     * @param sample_rate Sampling rate of the original signal (Hz)
     * @param top_n Number of top frequencies to find
     * @param frequencies Output array for frequencies (Hz), size must be at least top_n
     * @param powers Output array for power values, size must be at least top_n
     * @return tiny_error_t
     */
    tiny_error_t tiny_fft_find_top_frequencies(const float *power_spectrum, int fft_len, float sample_rate, int top_n, float *frequencies, float *powers);

#ifdef __cplusplus
}
#endif

