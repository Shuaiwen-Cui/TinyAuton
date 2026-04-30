# CODE

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
 * @version 1.1
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_view.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================
 * Internal helpers
 * ============================================================ */

/**
 * @brief Find min and max values in array (single pass).
 */
static void find_min_max(const float *data, int len, float *min_val, float *max_val)
{
    float lo = data[0];
    float hi = data[0];
    for (int i = 1; i < len; ++i)
    {
        if (data[i] < lo) lo = data[i];
        if (data[i] > hi) hi = data[i];
    }
    *min_val = lo;
    *max_val = hi;
}

/**
 * @brief Linearly interpolate sample at fractional position pos in [0, len-1].
 */
static inline float interp_linear(const float *data, int len, float pos)
{
    if (len <= 1) return data[0];
    int i0 = (int)pos;
    if (i0 < 0) return data[0];
    if (i0 >= len - 1) return data[len - 1];
    float frac = pos - (float)i0;
    return data[i0] * (1.0f - frac) + data[i0 + 1] * frac;
}

/**
 * @brief Map a value in [min_val, max_val] to a row index.
 *        Row 0 is at the top (max_val), height-1 at the bottom (min_val).
 */
static inline int value_to_row(float val, float min_val, float max_val, int height)
{
    if (val < min_val) val = min_val;
    if (val > max_val) val = max_val;
    float scale = (max_val > min_val) ? (float)(height - 1) / (max_val - min_val) : 0.0f;
    int y = (int)((max_val - val) * scale + 0.5f);
    if (y < 0) y = 0;
    if (y >= height) y = height - 1;
    return y;
}

/**
 * @brief Fill plot column x from row y0 to row y1 (inclusive) with character ch.
 */
static inline void plot_fill_col(char *plot, int width, int x, int y0, int y1, char ch)
{
    if (y0 > y1) { int t = y0; y0 = y1; y1 = t; }
    for (int y = y0; y <= y1; ++y)
        plot[y * width + x] = ch;
}

/**
 * @brief Print a horizontal axis line of given width.
 */
static inline void print_axis_line(int width, const char *prefix)
{
    fputs(prefix, stdout);
    for (int x = 0; x < width; ++x) putchar('-');
    putchar('\n');
}

/* ============================================================
 * tiny_view_signal_f32
 * ============================================================ */

/**
 * @name: tiny_view_signal_f32
 * @brief Visualize a signal in ASCII format
 */
tiny_error_t tiny_view_signal_f32(const float *data, int len, int width, int height,
                                  float min, float max, const char *title)
{
    if (NULL == data || len <= 0 || width <= 0 || height <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    /* Resolve Y range (auto when min == max). */
    float min_val = min;
    float max_val = max;
    if (min_val == max_val)
    {
        find_min_max(data, len, &min_val, &max_val);
        float margin = (max_val - min_val) * 0.1f;
        if (margin == 0.0f) margin = 0.1f;
        min_val -= margin;
        max_val += margin;
    }
    /* Defensive: caller could pass an inverted or degenerate range. */
    if (max_val <= min_val) max_val = min_val + 1.0f;

    char *plot = (char *)malloc((size_t)width * (size_t)height);
    if (plot == NULL)
    {
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    memset(plot, ' ', (size_t)width * (size_t)height);

    /* x_scale: data index per pixel column. */
    const float x_scale = (width > 1) ? (float)(len - 1) / (float)(width - 1) : 0.0f;

    int prev_y = -1;
    for (int x = 0; x < width; ++x)
    {
        float pos = (float)x * x_scale;
        float val = interp_linear(data, len, pos);
        int   y   = value_to_row(val, min_val, max_val, height);

        /* Connect previous sample to current one with a vertical run on
         * the previous column so the trace stays visually continuous. */
        if (prev_y >= 0 && prev_y != y)
        {
            plot_fill_col(plot, width, x - 1, prev_y, y, '*');
        }
        plot[y * width + x] = '*';
        prev_y = y;
    }

    if (title != NULL) printf("\n%s\n", title);

    printf("Value\n");
    for (int y = 0; y < height; ++y)
    {
        float y_val = (height > 1)
                    ? (max_val - (float)y * (max_val - min_val) / (float)(height - 1))
                    : max_val;
        printf("%6.2f |", y_val);
        fwrite(plot + (size_t)y * (size_t)width, 1, (size_t)width, stdout);
        putchar('\n');
    }

    print_axis_line(width, "       ");

    /* Sample-index labels: guard against width < 8 (would divide by zero). */
    {
        int step = (width >= 8) ? width / 8 : (width > 0 ? width : 1);
        printf("       ");
        for (int x = 0; x < width; x += step)
        {
            printf("%-*d", step, x);
        }
        printf(" (Sample Index)\n");
    }

    printf("Range: [%.3f, %.3f], Length: %d\n\n", min_val, max_val, len);

    free(plot);
    return TINY_OK;
}

/* ============================================================
 * tiny_view_spectrum_f32
 * ============================================================ */

/**
 * @name: tiny_view_spectrum_f32
 * @brief Visualize power spectrum
 *
 * The caller is expected to pass the single-sided spectrum (length = N/2,
 * covering DC up to ~Nyquist). The original FFT length is therefore
 * approximated as fft_len = 2 * len and the frequency of bin k is
 * f_k = k * sample_rate / fft_len.
 */
tiny_error_t tiny_view_spectrum_f32(const float *power_spectrum, int len,
                                    float sample_rate, const char *title)
{
    if (NULL == power_spectrum || len <= 0 || sample_rate <= 0.0f)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    const int width  = 64;
    const int height = 16;

    float min_val, max_val;
    find_min_max(power_spectrum, len, &min_val, &max_val);
    if (max_val <= min_val) max_val = min_val + 1.0f;

    char *plot = (char *)malloc((size_t)width * (size_t)height);
    if (plot == NULL)
    {
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    memset(plot, ' ', (size_t)width * (size_t)height);

    const int   fft_len = 2 * len;
    const float x_scale = (width > 1) ? (float)(len - 1) / (float)(width - 1) : 0.0f;

    /* Render bars: each pixel column maps to an interpolated spectrum bin. */
    for (int x = 0; x < width; ++x)
    {
        float pos = (float)x * x_scale;
        float val = interp_linear(power_spectrum, len, pos);
        int   y   = value_to_row(val, min_val, max_val, height);
        plot_fill_col(plot, width, x, y, height - 1, '|');
    }

    if (title != NULL) printf("\n%s\n", title);

    printf("Power\n");
    for (int y = 0; y < height; ++y)
    {
        float y_val = (height > 1)
                    ? (max_val - (float)y * (max_val - min_val) / (float)(height - 1))
                    : max_val;
        printf("%6.2f |", y_val);
        fwrite(plot + (size_t)y * (size_t)width, 1, (size_t)width, stdout);
        putchar('\n');
    }

    print_axis_line(width, "       ");

    /* Frequency labels: 8 evenly spaced tick positions. */
    {
        char freq_line[256];
        memset(freq_line, ' ', sizeof(freq_line));
        freq_line[width] = '\0';

        for (int label_idx = 0; label_idx < 8; ++label_idx)
        {
            int   x_pos = (label_idx * width) / 8;
            int   idx   = (int)((float)x_pos * x_scale + 0.5f);
            if (idx < 0)        idx = 0;
            if (idx > len - 1)  idx = len - 1;

            float freq = (float)idx * sample_rate / (float)fft_len;

            char  freq_str[16];
            int   n = (freq < 10.0f)
                    ? snprintf(freq_str, sizeof(freq_str), "%.1f", freq)
                    : snprintf(freq_str, sizeof(freq_str), "%.0f", freq);
            if (n < 0) n = 0;
            if (n > (int)sizeof(freq_str) - 1) n = (int)sizeof(freq_str) - 1;

            int start_pos = x_pos - n / 2;
            if (start_pos < 0)             start_pos = 0;
            if (start_pos + n > width)     start_pos = width - n;

            memcpy(freq_line + start_pos, freq_str, (size_t)n);
        }
        printf("       %s (Hz)\n", freq_line);
    }

    printf("Range: [%.3f, %.3f], Nyquist: %.1f Hz\n\n",
           min_val, max_val, sample_rate / 2.0f);

    free(plot);
    return TINY_OK;
}

/* ============================================================
 * tiny_view_array_f32
 * ============================================================ */

/**
 * @name: tiny_view_array_f32
 * @brief Print array in formatted table
 */
tiny_error_t tiny_view_array_f32(const float *data, int len, const char *name,
                                 int precision, int items_per_line)
{
    if (NULL == data || len <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    if (precision < 0)        precision      = 3;
    if (items_per_line <= 0)  items_per_line = 8;

    /* Reserve room for: optional sign + a few integer digits + '.' + decimals + spacing. */
    const int field_w = precision + 8;

    printf("\n%s [%d elements]:\n",
           (name != NULL) ? name : "Array", len);

    for (int i = 0; i < len; ++i)
    {
        if (i % items_per_line == 0)
        {
            printf("  [%4d] ", i);
        }
        printf("%*.*f ", field_w, precision, data[i]);
        if ((i + 1) % items_per_line == 0 || i == len - 1)
        {
            putchar('\n');
        }
    }
    putchar('\n');

    return TINY_OK;
}

/* ============================================================
 * tiny_view_statistics_f32
 * ============================================================ */

/**
 * @name: tiny_view_statistics_f32
 * @brief Print statistical information
 *
 * Two-pass implementation: the first pass collects min/max/peak and a
 * running sum (in double) for the mean; the second pass accumulates
 * squared deviations to keep variance numerically stable on signals
 * with large DC offsets.
 */
tiny_error_t tiny_view_statistics_f32(const float *data, int len, const char *name)
{
    if (NULL == data || len <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    float  min_val  = data[0];
    float  max_val  = data[0];
    int    min_idx  = 0;
    int    max_idx  = 0;
    int    peak_idx = 0;
    float  peak_abs = fabsf(data[0]);
    double sum      = 0.0;

    for (int i = 0; i < len; ++i)
    {
        float v = data[i];
        if (v < min_val) { min_val = v; min_idx = i; }
        if (v > max_val) { max_val = v; max_idx = i; }
        float av = fabsf(v);
        if (av > peak_abs) { peak_abs = av; peak_idx = i; }
        sum += (double)v;
    }
    float mean = (float)(sum / (double)len);

    double sse = 0.0;
    for (int i = 0; i < len; ++i)
    {
        double d = (double)data[i] - (double)mean;
        sse += d * d;
    }
    float variance = (float)(sse / (double)len);
    if (variance < 0.0f) variance = 0.0f;
    float std_dev = sqrtf(variance);

    if (name != NULL) printf("\n=== Statistics: %s ===\n", name);
    else              printf("\n=== Statistics ===\n");

    printf("  Length:     %d samples\n", len);
    printf("  Min:        %.6f (at index %d)\n", min_val, min_idx);
    printf("  Max:        %.6f (at index %d)\n", max_val, max_idx);
    printf("  Peak |x|:   %.6f (value %.6f at index %d)\n",
           peak_abs, data[peak_idx], peak_idx);
    printf("  Mean:       %.6f\n", mean);
    printf("  Std Dev:    %.6f\n", std_dev);
    printf("  Variance:   %.6f\n", variance);
    printf("  Range:      %.6f\n", max_val - min_val);
    printf("========================\n\n");

    return TINY_OK;
}


```
