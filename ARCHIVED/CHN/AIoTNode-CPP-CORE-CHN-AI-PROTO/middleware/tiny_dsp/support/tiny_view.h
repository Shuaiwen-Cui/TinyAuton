/**
 * @file tiny_view.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_view | code | header
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @name: tiny_view_signal_f32
     * @brief Visualize a signal in ASCII format (like oscilloscope)
     * @param data Input signal array
     * @param len Length of the signal
     * @param width Width of the plot in characters (default: 64)
     * @param height Height of the plot in lines (default: 16)
     * @param min Minimum Y-axis value (auto-detect if min == max)
     * @param max Maximum Y-axis value (auto-detect if min == max)
     * @param title Optional title for the plot (NULL for no title)
     * @return tiny_error_t
     */
    tiny_error_t tiny_view_signal_f32(const float *data, int len, int width, int height, float min, float max, const char *title);

    /**
     * @name: tiny_view_spectrum_f32
     * @brief Visualize power spectrum in ASCII format (optimized for frequency domain)
     * @param power_spectrum Power spectrum array
     * @param len Length of the spectrum
     * @param sample_rate Sampling rate (Hz) for frequency axis labels
     * @param title Optional title for the plot (NULL for no title)
     * @return tiny_error_t
     */
    tiny_error_t tiny_view_spectrum_f32(const float *power_spectrum, int len, float sample_rate, const char *title);

    /**
     * @name: tiny_view_array_f32
     * @brief Print array values in a formatted table
     * @param data Input array
     * @param len Length of the array
     * @param name Name/label for the array
     * @param precision Number of decimal places (default: 3)
     * @param items_per_line Number of items per line (default: 8)
     * @return tiny_error_t
     */
    tiny_error_t tiny_view_array_f32(const float *data, int len, const char *name, int precision, int items_per_line);

    /**
     * @name: tiny_view_statistics_f32
     * @brief Print statistical information about a signal
     * @param data Input signal array
     * @param len Length of the signal
     * @param name Name/label for the signal
     * @return tiny_error_t
     */
    tiny_error_t tiny_view_statistics_f32(const float *data, int len, const char *name);

#ifdef __cplusplus
}
#endif

