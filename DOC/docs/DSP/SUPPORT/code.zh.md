# 代码

## tiny_view.h

```c
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


```

## tiny_view.c

```c
/**
 * @file tiny_view.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_view | code | source
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_view.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/**
 * @brief Find min and max values in array
 */
static void find_min_max(const float *data, int len, float *min_val, float *max_val)
{
    *min_val = data[0];
    *max_val = data[0];
    for (int i = 1; i < len; i++)
    {
        if (data[i] < *min_val)
            *min_val = data[i];
        if (data[i] > *max_val)
            *max_val = data[i];
    }
}

/**
 * @name: tiny_view_signal_f32
 * @brief Visualize a signal in ASCII format
 */
tiny_error_t tiny_view_signal_f32(const float *data, int len, int width, int height, float min, float max, const char *title)
{
    if (NULL == data || len <= 0 || width <= 0 || height <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Auto-detect min/max if they are equal
    float min_val = min;
    float max_val = max;
    if (min == max)
    {
        find_min_max(data, len, &min_val, &max_val);
        // Add small margin
        float margin = (max_val - min_val) * 0.1f;
        if (margin == 0.0f)
            margin = 0.1f;
        min_val -= margin;
        max_val += margin;
    }

    // Allocate buffer for plot
    char *plot = (char *)malloc(width * height * sizeof(char));
    if (plot == NULL)
    {
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }

    // Initialize plot with spaces
    memset(plot, ' ', width * height);

    // Calculate scaling factors
    float x_scale = (len > 1) ? (float)(len - 1) / (width - 1) : 0.0f;  // Inverse: data index per pixel
    float y_scale = (max_val > min_val) ? (float)(height - 1) / (max_val - min_val) : 1.0f;

    // High-resolution drawing: for each pixel column, interpolate the signal value
    for (int x = 0; x < width; x++)
    {
        // Calculate corresponding data index (with interpolation)
        float data_idx = x * x_scale;
        int idx0 = (int)data_idx;
        int idx1 = idx0 + 1;
        float frac = data_idx - idx0;
        
        // Clamp indices
        if (idx0 < 0) idx0 = 0;
        if (idx0 >= len) idx0 = len - 1;
        if (idx1 < 0) idx1 = 0;
        if (idx1 >= len) idx1 = len - 1;
        
        // Linear interpolation between data points
        float val;
        if (len == 1)
        {
            val = data[0];
        }
        else if (idx0 == len - 1)
        {
            val = data[idx0];
        }
        else
        {
            val = data[idx0] * (1.0f - frac) + data[idx1] * frac;
        }
        
        // Clamp value to range
        if (val < min_val)
            val = min_val;
        if (val > max_val)
            val = max_val;
        
        // Calculate Y position
        int y = (int)((max_val - val) * y_scale + 0.5f);
        if (y < 0)
            y = 0;
        if (y >= height)
            y = height - 1;
        
        // Draw point
        plot[y * width + x] = '*';
        
        // For better visualization, also draw adjacent points if there's a significant change
        // This helps show the signal shape more clearly
        if (x > 0)
        {
            float prev_data_idx = (x - 1) * x_scale;
            int prev_idx0 = (int)prev_data_idx;
            int prev_idx1 = prev_idx0 + 1;
            float prev_frac = prev_data_idx - prev_idx0;
            
            if (prev_idx0 < 0) prev_idx0 = 0;
            if (prev_idx0 >= len) prev_idx0 = len - 1;
            if (prev_idx1 < 0) prev_idx1 = 0;
            if (prev_idx1 >= len) prev_idx1 = len - 1;
            
            float prev_val;
            if (len == 1)
            {
                prev_val = data[0];
            }
            else if (prev_idx0 == len - 1)
            {
                prev_val = data[prev_idx0];
            }
            else
            {
                prev_val = data[prev_idx0] * (1.0f - prev_frac) + data[prev_idx1] * prev_frac;
            }
            
            if (prev_val < min_val) prev_val = min_val;
            if (prev_val > max_val) prev_val = max_val;
            
            int prev_y = (int)((max_val - prev_val) * y_scale + 0.5f);
            if (prev_y < 0) prev_y = 0;
            if (prev_y >= height) prev_y = height - 1;
            
            // Draw line between previous and current point
            int dy = y - prev_y;
            int steps = abs(dy);
            if (steps > 1)
            {
                for (int s = 1; s < steps; s++)
                {
                    int py = prev_y + (dy * s) / steps;
                    if (py >= 0 && py < height)
                    {
                        plot[py * width + (x - 1)] = '*';
                    }
                }
            }
        }
    }

    // Print title
    if (title != NULL)
    {
        printf("\n%s\n", title);
    }

    // Print Y-axis labels and plot
    printf("Value\n");
    for (int y = 0; y < height; y++)
    {
        // Calculate Y value correctly: from max_val at top (y=0) to min_val at bottom (y=height-1)
        float y_val = (height > 1) ? (max_val - (float)y * (max_val - min_val) / (height - 1)) : max_val;
        printf("%6.2f |", y_val);
        for (int x = 0; x < width; x++)
        {
            printf("%c", plot[y * width + x]);
        }
        printf("\n");
    }

    // Print X-axis
    printf("       ");
    for (int x = 0; x < width; x++)
    {
        printf("-");
    }
    printf("\n");
    printf("       ");
    for (int x = 0; x < width; x += width / 8)
    {
        printf("%-*d", width / 8, x);
    }
    printf(" (Sample Index)\n");

    printf("Range: [%.3f, %.3f], Length: %d\n\n", min_val, max_val, len);

    free(plot);
    return TINY_OK;
}

/**
 * @name: tiny_view_spectrum_f32
 * @brief Visualize power spectrum
 */
tiny_error_t tiny_view_spectrum_f32(const float *power_spectrum, int len, float sample_rate, const char *title)
{
    if (NULL == power_spectrum || len <= 0 || sample_rate <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    const int width = 64;
    const int height = 16;

    // Find min/max
    float min_val, max_val;
    find_min_max(power_spectrum, len, &min_val, &max_val);
    if (max_val == min_val)
    {
        max_val = min_val + 1.0f;
    }

    // Allocate buffer
    char *plot = (char *)malloc(width * height * sizeof(char));
    if (plot == NULL)
    {
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    memset(plot, ' ', width * height);

    // Calculate scaling (only use first half for real signals)
    // Power spectrum length is typically half of FFT length
    // So FFT length = 2 * len, frequency resolution = sample_rate / (2 * len)
    int plot_len = len / 2;
    int fft_len = 2 * len;  // FFT length is typically 2x power spectrum length
    float x_scale = (plot_len > 1) ? (float)(plot_len - 1) / (width - 1) : 0.0f;  // Inverse: data index per pixel
    float y_scale = (max_val > min_val) ? (float)(height - 1) / (max_val - min_val) : 1.0f;

    // High-resolution drawing: for each pixel column, interpolate the spectrum value
    for (int x = 0; x < width; x++)
    {
        // Calculate corresponding spectrum index (with interpolation)
        float spec_idx = x * x_scale;
        int idx0 = (int)spec_idx;
        int idx1 = idx0 + 1;
        float frac = spec_idx - idx0;
        
        // Clamp indices
        if (idx0 < 0) idx0 = 0;
        if (idx0 >= plot_len) idx0 = plot_len - 1;
        if (idx1 < 0) idx1 = 0;
        if (idx1 >= plot_len) idx1 = plot_len - 1;
        
        // Linear interpolation between spectrum points
        float val;
        if (plot_len == 1)
        {
            val = power_spectrum[0];
        }
        else if (idx0 == plot_len - 1)
        {
            val = power_spectrum[idx0];
        }
        else
        {
            val = power_spectrum[idx0] * (1.0f - frac) + power_spectrum[idx1] * frac;
        }
        
        // Clamp value to range
        if (val < min_val)
            val = min_val;
        if (val > max_val)
            val = max_val;
        
        // Calculate Y position
        int y = (int)((max_val - val) * y_scale + 0.5f);
        if (y < 0)
            y = 0;
        if (y >= height)
            y = height - 1;
        
        // Draw bar from bottom to value
        for (int bar_y = height - 1; bar_y >= y; bar_y--)
        {
            plot[bar_y * width + x] = '|';
        }
    }

    // Print title
    if (title != NULL)
    {
        printf("\n%s\n", title);
    }

    // Print plot
    printf("Power\n");
    for (int y = 0; y < height; y++)
    {
        // Calculate Y value correctly: from max_val at top (y=0) to min_val at bottom (y=height-1)
        float y_val = (height > 1) ? (max_val - (float)y * (max_val - min_val) / (height - 1)) : max_val;
        printf("%6.2f |", y_val);
        for (int x = 0; x < width; x++)
        {
            printf("%c", plot[y * width + x]);
        }
        printf("\n");
    }

    // Print X-axis with frequency labels
    printf("       ");
    for (int x = 0; x < width; x++)
    {
        printf("-");
    }
    printf("\n");
    
    // Print frequency labels aligned with X-axis positions
    if (x_scale > 0.0f && plot_len > 0)
    {
        // Create a buffer for frequency labels
        char *freq_line = (char *)calloc(width + 20, sizeof(char));
        if (freq_line != NULL)
        {
            memset(freq_line, ' ', width + 20);
            freq_line[width + 19] = '\0';
            
            // Place frequency labels at 8 evenly spaced positions
            int fft_len = 2 * len;  // FFT length is 2x power spectrum length
            for (int label_idx = 0; label_idx < 8; label_idx++)
            {
                int x_pos = (label_idx * width) / 8;
                // Convert x position back to array index (x_scale is now inverse: data_idx per pixel)
                float array_idx = (float)x_pos * x_scale;
                int idx = (int)(array_idx + 0.5f);
                if (idx >= plot_len)
                    idx = plot_len - 1;
                if (idx < 0)
                    idx = 0;
                // Frequency = idx * sample_rate / fft_len
                float freq = (float)idx * sample_rate / fft_len;
                
                // Format frequency string with appropriate precision
                char freq_str[12];
                if (freq < 10.0f)
                {
                    snprintf(freq_str, sizeof(freq_str), "%.1f", freq);
                }
                else if (freq < 100.0f)
                {
                    snprintf(freq_str, sizeof(freq_str), "%.0f", freq);
                }
                else
                {
                    snprintf(freq_str, sizeof(freq_str), "%.0f", freq);
                }
                
                // Place string at x_pos (centered if possible)
                int str_len = strlen(freq_str);
                int start_pos = x_pos - str_len / 2;
                if (start_pos < 0)
                    start_pos = 0;
                if (start_pos + str_len > width)
                    start_pos = width - str_len;
                
                memcpy(freq_line + start_pos, freq_str, str_len);
            }
            
            printf("       %s (Hz)\n", freq_line);
            free(freq_line);
        }
        else
        {
            // Fallback: simple printing
            int fft_len = 2 * len;  // FFT length is 2x power spectrum length
            printf("       ");
            for (int label_idx = 0; label_idx < 8; label_idx++)
            {
                int x_pos = (label_idx * width) / 8;
                // Convert x position back to array index (x_scale is now inverse: data_idx per pixel)
                float array_idx = (float)x_pos * x_scale;
                int idx = (int)(array_idx + 0.5f);
                if (idx >= plot_len)
                    idx = plot_len - 1;
                if (idx < 0)
                    idx = 0;
                // Frequency = idx * sample_rate / fft_len
                float freq = (float)idx * sample_rate / fft_len;
                if (freq < 10.0f)
                {
                    printf("%6.1f", freq);
                }
                else
                {
                    printf("%7.0f", freq);
                }
            }
            printf(" (Hz)\n");
        }
    }
    else
    {
        // Fallback: just print frequency range
        printf("       0    %5.0f (Hz)\n", sample_rate / 2.0f);
    }

    printf("Range: [%.3f, %.3f], Nyquist: %.1f Hz\n\n", min_val, max_val, sample_rate / 2.0f);

    free(plot);
    return TINY_OK;
}

/**
 * @name: tiny_view_array_f32
 * @brief Print array in formatted table
 */
tiny_error_t tiny_view_array_f32(const float *data, int len, const char *name, int precision, int items_per_line)
{
    if (NULL == data || len <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    if (precision < 0)
        precision = 3;
    if (items_per_line <= 0)
        items_per_line = 8;

    if (name != NULL)
    {
        printf("\n%s [%d elements]:\n", name, len);
    }
    else
    {
        printf("\nArray [%d elements]:\n", len);
    }

    for (int i = 0; i < len; i++)
    {
        if (i % items_per_line == 0)
        {
            printf("  [%3d] ", i);
        }
        printf("%*.*f ", precision + 4, precision, data[i]);
        if ((i + 1) % items_per_line == 0 || i == len - 1)
        {
            printf("\n");
        }
    }
    printf("\n");

    return TINY_OK;
}

/**
 * @name: tiny_view_statistics_f32
 * @brief Print statistical information
 */
tiny_error_t tiny_view_statistics_f32(const float *data, int len, const char *name)
{
    if (NULL == data || len <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Calculate statistics and find positions in one pass
    float min_val = data[0];
    float max_val = data[0];
    int min_idx = 0;
    int max_idx = 0;
    int peak_idx = 0;
    float peak_val = fabsf(data[0]);
    
    float sum = data[0];
    float sum_sq = data[0] * data[0];
    
    for (int i = 1; i < len; i++)
    {
        // Update min/max and their positions
        if (data[i] < min_val)
        {
            min_val = data[i];
            min_idx = i;
        }
        if (data[i] > max_val)
        {
            max_val = data[i];
            max_idx = i;
        }
        
        // Update peak (absolute value)
        float abs_val = fabsf(data[i]);
        if (abs_val > peak_val)
        {
            peak_val = abs_val;
            peak_idx = i;
        }
        
        // Accumulate for mean/variance
        sum += data[i];
        sum_sq += data[i] * data[i];
    }

    float mean = sum / len;
    float variance = (sum_sq / len) - (mean * mean);
    float std_dev = sqrtf(variance > 0 ? variance : 0);

    // Print statistics
    if (name != NULL)
    {
        printf("\n=== Statistics: %s ===\n", name);
    }
    else
    {
        printf("\n=== Statistics ===\n");
    }
    printf("  Length:     %d samples\n", len);
    printf("  Min:        %.6f (at index %d)\n", min_val, min_idx);
    printf("  Max:        %.6f (at index %d)\n", max_val, max_idx);
    printf("  Peak:       %.6f (at index %d)\n", data[peak_idx], peak_idx);
    printf("  Mean:       %.6f\n", mean);
    printf("  Std Dev:    %.6f\n", std_dev);
    printf("  Variance:   %.6f\n", variance);
    printf("  Range:      %.6f\n", max_val - min_val);
    printf("========================\n\n");

    return TINY_OK;
}

```
