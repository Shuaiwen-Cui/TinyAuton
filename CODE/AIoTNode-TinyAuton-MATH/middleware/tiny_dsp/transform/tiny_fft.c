/**
 * @file tiny_fft.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fft | code | source
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_fft.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
#include "esp_heap_caps.h"
#endif

/* STATIC VARIABLES */
static int g_fft_initialized = 0;
static int g_fft_size = 0;

/* STATIC FUNCTIONS FOR NON-ESP32 PLATFORM */
#if MCU_PLATFORM_SELECTED != MCU_PLATFORM_ESP32
/**
 * @brief Bit-reverse an integer (for FFT)
 */
static unsigned int bit_reverse(unsigned int x, int log2n)
{
    unsigned int n = 0;
    for (int i = 0; i < log2n; i++)
    {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

/**
 * @brief Calculate log2 of a power-of-2 number
 */
static int log2_power2(int n)
{
    int log2n = 0;
    while (n > 1)
    {
        n >>= 1;
        log2n++;
    }
    return log2n;
}

/**
 * @brief Perform Radix-2 FFT on complex data
 * @param data Complex array [Re0, Im0, Re1, Im1, ...]
 * @param n Number of complex points (must be power of 2)
 * @param inverse If 1, perform IFFT; if 0, perform FFT
 */
static void fft_radix2_f32(float *data, int n, int inverse)
{
    int log2n = log2_power2(n);
    
    // Bit-reverse the input
    for (int i = 0; i < n; i++)
    {
        unsigned int j = bit_reverse(i, log2n);
        if (j > i)
        {
            // Swap real parts
            float temp = data[i * 2];
            data[i * 2] = data[j * 2];
            data[j * 2] = temp;
            // Swap imaginary parts
            temp = data[i * 2 + 1];
            data[i * 2 + 1] = data[j * 2 + 1];
            data[j * 2 + 1] = temp;
        }
    }
    
    // FFT butterfly operations
    float sign = inverse ? -1.0f : 1.0f;
    for (int stage = 1; stage <= log2n; stage++)
    {
        int m = 1 << stage;  // 2^stage
        int m2 = m >> 1;     // m/2
        
        float wm_real = cosf(sign * 2.0f * M_PI / m);
        float wm_imag = sinf(sign * 2.0f * M_PI / m);
        
        for (int k = 0; k < n; k += m)
        {
            float w_real = 1.0f;
            float w_imag = 0.0f;
            
            for (int j = 0; j < m2; j++)
            {
                int t = k + j;
                int u = t + m2;
                
                float t_real = data[t * 2];
                float t_imag = data[t * 2 + 1];
                float u_real = data[u * 2];
                float u_imag = data[u * 2 + 1];
                
                // Multiply u by twiddle factor
                float u_real_new = u_real * w_real - u_imag * w_imag;
                float u_imag_new = u_real * w_imag + u_imag * w_real;
                
                // Butterfly operation
                data[t * 2] = t_real + u_real_new;
                data[t * 2 + 1] = t_imag + u_imag_new;
                data[u * 2] = t_real - u_real_new;
                data[u * 2 + 1] = t_imag - u_imag_new;
                
                // Update twiddle factor
                float w_real_new = w_real * wm_real - w_imag * wm_imag;
                float w_imag_new = w_real * wm_imag + w_imag * wm_real;
                w_real = w_real_new;
                w_imag = w_imag_new;
            }
        }
    }
    
    // Scale for IFFT
    if (inverse)
    {
        float scale = 1.0f / n;
        for (int i = 0; i < n; i++)
        {
            data[i * 2] *= scale;
            data[i * 2 + 1] *= scale;
        }
    }
}
#endif

/**
 * @brief Check if a number is power of 2
 */
static int is_power_of_2(int n)
{
    return (n > 0) && ((n & (n - 1)) == 0);
}

/**
 * @name: tiny_fft_init
 * @brief Initialize FFT tables
 */
tiny_error_t tiny_fft_init(int fft_size)
{
    if (!is_power_of_2(fft_size))
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    if (g_fft_initialized)
    {
        return TINY_ERR_DSP_REINITIALIZED;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, fft_size);
    if (ret != ESP_OK)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }
    g_fft_size = fft_size;
    g_fft_initialized = 1;
    return TINY_OK;
#else
    // For non-ESP32 platforms, FFT initialization is not required
    // but we mark it as initialized for compatibility
    g_fft_size = fft_size;
    g_fft_initialized = 1;
    return TINY_OK;
#endif
}

/**
 * @name: tiny_fft_deinit
 * @brief Deinitialize FFT tables
 */
tiny_error_t tiny_fft_deinit(void)
{
    if (!g_fft_initialized)
    {
        return TINY_ERR_DSP_UNINITIALIZED;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    dsps_fft2r_deinit_fc32();
#endif

    g_fft_initialized = 0;
    g_fft_size = 0;
    return TINY_OK;
}

/**
 * @brief Apply window function to input signal
 */
static void apply_window(const float *input, int len, float *output, tiny_fft_window_t window)
{
    if (window == TINY_FFT_WINDOW_NONE)
    {
        memcpy(output, input, len * sizeof(float));
        return;
    }

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Use ESP32 DSP window functions
    float *window_coeffs = (float *)malloc(len * sizeof(float));
    if (window_coeffs == NULL)
    {
        memcpy(output, input, len * sizeof(float));
        return;
    }

    switch (window)
    {
    case TINY_FFT_WINDOW_HANNING:
        dsps_wind_hann_f32(window_coeffs, len);
        break;
    case TINY_FFT_WINDOW_HAMMING:
        // ESP-DSP doesn't have Hamming, use Hann as approximation
        dsps_wind_hann_f32(window_coeffs, len);
        break;
    case TINY_FFT_WINDOW_BLACKMAN:
        dsps_wind_blackman_f32(window_coeffs, len);
        break;
    default:
        free(window_coeffs);
        memcpy(output, input, len * sizeof(float));
        return;
    }
    // Multiply input by window
    for (int i = 0; i < len; i++)
    {
        output[i] = input[i] * window_coeffs[i];
    }
    free(window_coeffs);
#else
    // Simple window implementation for non-ESP32 platforms
    for (int i = 0; i < len; i++)
    {
        float w = 1.0f;
        switch (window)
        {
        case TINY_FFT_WINDOW_HANNING:
            w = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (len - 1)));
            break;
        case TINY_FFT_WINDOW_HAMMING:
            w = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (len - 1));
            break;
        case TINY_FFT_WINDOW_BLACKMAN:
            w = 0.42f - 0.5f * cosf(2.0f * M_PI * i / (len - 1)) + 0.08f * cosf(4.0f * M_PI * i / (len - 1));
            break;
        default:
            w = 1.0f;
            break;
        }
        output[i] = input[i] * w;
    }
#endif
}

/**
 * @name: tiny_fft_f32
 * @brief Perform FFT on real-valued input signal
 */
tiny_error_t tiny_fft_f32(const float *input, int input_len, float *output_fft, tiny_fft_window_t window)
{
    if (!g_fft_initialized)
    {
        return TINY_ERR_DSP_UNINITIALIZED;
    }

    if (NULL == input || NULL == output_fft)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    if (!is_power_of_2(input_len) || input_len > g_fft_size)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Apply window function
    float *windowed_input = (float *)malloc(input_len * sizeof(float));
    if (windowed_input == NULL)
    {
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    apply_window(input, input_len, windowed_input, window);

    // Convert real input to complex format [Re0, Im0, Re1, Im1, ...]
    for (int i = 0; i < input_len; i++)
    {
        output_fft[i * 2] = windowed_input[i];     // Real part
        output_fft[i * 2 + 1] = 0.0f;               // Imaginary part
    }
    free(windowed_input);

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Perform FFT using ESP32 optimized library
    // ESP32 FFT requires: FFT -> bit reverse
    // Note: dsps_cplx2reC_fc32 is for two real signals, not needed for single real signal
    esp_err_t ret = dsps_fft2r_fc32(output_fft, input_len);
    if (ret != ESP_OK)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }
    
    // Bit reverse
    ret = dsps_bit_rev_fc32(output_fft, input_len);
    if (ret != ESP_OK)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }
    
    return TINY_OK;
#else
    // Perform FFT using Radix-2 algorithm (non-ESP32 platforms)
    fft_radix2_f32(output_fft, input_len, 0);  // 0 = forward FFT
    return TINY_OK;
#endif
}

/**
 * @name: tiny_fft_ifft_f32
 * @brief Perform inverse FFT
 */
tiny_error_t tiny_fft_ifft_f32(const float *input_fft, int fft_len, float *output)
{
    if (!g_fft_initialized)
    {
        return TINY_ERR_DSP_UNINITIALIZED;
    }

    if (NULL == input_fft || NULL == output)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    if (!is_power_of_2(fft_len) || fft_len > g_fft_size)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Copy input to temporary buffer
    // ESP32 DSP library requires 16-byte aligned memory
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    float *temp_fft = (float *)heap_caps_aligned_alloc(16, fft_len * 2 * sizeof(float), MALLOC_CAP_DEFAULT);
    if (temp_fft == NULL)
    {
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
#else
    float *temp_fft = (float *)malloc(fft_len * 2 * sizeof(float));
    if (temp_fft == NULL)
    {
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
#endif
    memcpy(temp_fft, input_fft, fft_len * 2 * sizeof(float));

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    // Perform IFFT using ESP32 optimized library
    // Note: Input FFT result is already in reC format (from FFT function)
    // For IFFT, we need to reverse the process:
    // 1. The input is already in reC format, so we can work with it directly
    // 2. IFFT = conj(FFT(conj(X))) / N
    
    // First, conjugate the input (since it's already processed by FFT)
    for (int i = 0; i < fft_len; i++)
    {
        temp_fft[i * 2 + 1] = -temp_fft[i * 2 + 1]; // Conjugate
    }

    // Perform FFT (which gives us IFFT after conjugation)
    esp_err_t ret = dsps_fft2r_fc32(temp_fft, fft_len);
    if (ret != ESP_OK)
    {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
        heap_caps_free(temp_fft);
#else
        free(temp_fft);
#endif
        return TINY_ERR_DSP_INVALID_PARAM;
    }
    
    // Bit reverse
    ret = dsps_bit_rev_fc32(temp_fft, fft_len);
    if (ret != ESP_OK)
    {
#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
        heap_caps_free(temp_fft);
#else
        free(temp_fft);
#endif
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Conjugate again and scale
    float scale = 1.0f / fft_len;
    for (int i = 0; i < fft_len; i++)
    {
        output[i] = temp_fft[i * 2] * scale; // Take real part and scale
    }
#else
    // Perform IFFT using Radix-2 algorithm (non-ESP32 platforms)
    fft_radix2_f32(temp_fft, fft_len, 1);  // 1 = inverse FFT
    
    // Extract real part (IFFT of real signal should have zero imaginary part)
    for (int i = 0; i < fft_len; i++)
    {
        output[i] = temp_fft[i * 2];  // Take real part
    }
#endif

#if MCU_PLATFORM_SELECTED == MCU_PLATFORM_ESP32
    heap_caps_free(temp_fft);
#else
    free(temp_fft);
#endif
    return TINY_OK;
}

/**
 * @name: tiny_fft_magnitude_f32
 * @brief Calculate magnitude spectrum
 */
tiny_error_t tiny_fft_magnitude_f32(const float *fft_result, int fft_len, float *magnitude)
{
    if (NULL == fft_result || NULL == magnitude)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    for (int i = 0; i < fft_len; i++)
    {
        float re = fft_result[i * 2];
        float im = fft_result[i * 2 + 1];
        magnitude[i] = sqrtf(re * re + im * im);
    }

    return TINY_OK;
}

/**
 * @name: tiny_fft_power_spectrum_f32
 * @brief Calculate power spectrum density
 */
tiny_error_t tiny_fft_power_spectrum_f32(const float *fft_result, int fft_len, float *power)
{
    if (NULL == fft_result || NULL == power)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    // Calculate power spectrum with normalization
    // Normalize by FFT length to get proper power values
    float norm = 1.0f / (float)fft_len;
    for (int i = 0; i < fft_len; i++)
    {
        float re = fft_result[i * 2];
        float im = fft_result[i * 2 + 1];
        power[i] = (re * re + im * im) * norm;
    }

    return TINY_OK;
}

/**
 * @name: tiny_fft_find_peak_frequency
 * @brief Find peak frequency
 */
tiny_error_t tiny_fft_find_peak_frequency(const float *power_spectrum, int fft_len, float sample_rate, float *peak_freq, float *peak_power)
{
    if (NULL == power_spectrum || NULL == peak_freq || NULL == peak_power)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    if (sample_rate <= 0)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Find maximum power (skip DC component at index 0)
    int max_idx = 1;
    float max_power = power_spectrum[1];
    for (int i = 2; i < fft_len / 2; i++) // Only check first half (Nyquist)
    {
        if (power_spectrum[i] > max_power)
        {
            max_power = power_spectrum[i];
            max_idx = i;
        }
    }

    // Use parabolic interpolation for sub-bin accuracy
    // This improves frequency estimation when peak is between bins
    float refined_idx = (float)max_idx;
    if (max_idx > 0 && max_idx < (fft_len / 2 - 1))
    {
        float y0 = power_spectrum[max_idx - 1];
        float y1 = power_spectrum[max_idx];      // Peak
        float y2 = power_spectrum[max_idx + 1];
        
        // Parabolic interpolation: find peak of parabola through three points
        // Formula: offset = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        float denominator = y0 - 2.0f * y1 + y2;
        if (fabsf(denominator) > 1e-6f)  // Avoid division by zero
        {
            float offset = 0.5f * (y0 - y2) / denominator;
            refined_idx = (float)max_idx + offset;
            
            // Clamp to valid range
            if (refined_idx < 0.0f)
                refined_idx = 0.0f;
            if (refined_idx >= (float)(fft_len / 2))
                refined_idx = (float)(fft_len / 2 - 1);
        }
    }

    // Convert refined index to frequency
    *peak_freq = refined_idx * sample_rate / fft_len;
    *peak_power = max_power;

    return TINY_OK;
}

/**
 * @name: tiny_fft_find_top_frequencies
 * @brief Find top N frequencies
 */
tiny_error_t tiny_fft_find_top_frequencies(const float *power_spectrum, int fft_len, float sample_rate, int top_n, float *frequencies, float *powers)
{
    if (NULL == power_spectrum || NULL == frequencies || NULL == powers)
    {
        return TINY_ERR_DSP_NULL_POINTER;
    }

    if (sample_rate <= 0 || top_n <= 0 || top_n > fft_len / 2)
    {
        return TINY_ERR_DSP_INVALID_PARAM;
    }

    // Find peaks first, then select top N peaks
    // This avoids selecting multiple bins from the same frequency peak
    int max_peaks = fft_len / 4;  // Maximum possible peaks
    int *peak_indices = (int *)malloc(max_peaks * sizeof(int));
    float *peak_powers = (float *)malloc(max_peaks * sizeof(float));
    if (peak_indices == NULL || peak_powers == NULL)
    {
        if (peak_indices) free(peak_indices);
        if (peak_powers) free(peak_powers);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    int num_peaks = 0;
    
    // Find local peaks (points higher than neighbors)
    // Skip DC and first few bins to avoid noise
    for (int i = 2; i < fft_len / 2 - 1; i++)
    {
        // Check if this is a local maximum
        if (power_spectrum[i] > power_spectrum[i - 1] && 
            power_spectrum[i] > power_spectrum[i + 1])
        {
            // Only consider significant peaks (above threshold)
            // Threshold: at least 1% of maximum power
            float max_power = 0.0f;
            for (int j = 1; j < fft_len / 2; j++)
            {
                if (power_spectrum[j] > max_power)
                    max_power = power_spectrum[j];
            }
            
            if (power_spectrum[i] > max_power * 0.01f)  // 1% threshold
            {
                peak_indices[num_peaks] = i;
                peak_powers[num_peaks] = power_spectrum[i];
                num_peaks++;
                
                if (num_peaks >= max_peaks)
                    break;
            }
        }
    }
    
    // Sort peaks by power (descending)
    for (int i = 0; i < num_peaks - 1; i++)
    {
        for (int j = i + 1; j < num_peaks; j++)
        {
            if (peak_powers[i] < peak_powers[j])
            {
                // Swap
                int temp_idx = peak_indices[i];
                float temp_power = peak_powers[i];
                peak_indices[i] = peak_indices[j];
                peak_powers[i] = peak_powers[j];
                peak_indices[j] = temp_idx;
                peak_powers[j] = temp_power;
            }
        }
    }
    
    // Merge nearby peaks (within 2 bins) - keep the stronger one
    int *merged_indices = (int *)malloc(num_peaks * sizeof(int));
    float *merged_powers = (float *)malloc(num_peaks * sizeof(float));
    int num_merged = 0;
    
    if (merged_indices == NULL || merged_powers == NULL)
    {
        free(peak_indices);
        free(peak_powers);
        if (merged_indices) free(merged_indices);
        if (merged_powers) free(merged_powers);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    for (int i = 0; i < num_peaks; i++)
    {
        int idx = peak_indices[i];
        int is_merged = 0;
        
        // Check if this peak is too close to an already merged peak
        for (int j = 0; j < num_merged; j++)
        {
            int freq_diff = abs(idx - merged_indices[j]);
            if (freq_diff <= 2)  // Within 2 bins (~7.8 Hz at 1000 Hz sample rate)
            {
                // Keep the stronger peak
                if (peak_powers[i] > merged_powers[j])
                {
                    merged_indices[j] = idx;
                    merged_powers[j] = peak_powers[i];
                }
                is_merged = 1;
                break;
            }
        }
        
        if (!is_merged)
        {
            merged_indices[num_merged] = idx;
            merged_powers[num_merged] = peak_powers[i];
            num_merged++;
        }
    }
    
    // Re-sort merged peaks by power (descending) since merging may have changed powers
    for (int i = 0; i < num_merged - 1; i++)
    {
        for (int j = i + 1; j < num_merged; j++)
        {
            if (merged_powers[i] < merged_powers[j])
            {
                // Swap
                int temp_idx = merged_indices[i];
                float temp_power = merged_powers[i];
                merged_indices[i] = merged_indices[j];
                merged_powers[i] = merged_powers[j];
                merged_indices[j] = temp_idx;
                merged_powers[j] = temp_power;
            }
        }
    }
    
    // Select top N from merged peaks
    int n_to_return = (top_n < num_merged) ? top_n : num_merged;
    int *indices = (int *)malloc(n_to_return * sizeof(int));
    if (indices == NULL)
    {
        free(peak_indices);
        free(peak_powers);
        free(merged_indices);
        free(merged_powers);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    for (int i = 0; i < n_to_return; i++)
    {
        indices[i] = merged_indices[i];
    }
    
    free(peak_indices);
    free(peak_powers);
    free(merged_indices);
    free(merged_powers);

    // Convert to frequencies and powers (with parabolic interpolation for better accuracy)
    for (int i = 0; i < n_to_return; i++)
    {
        int idx = indices[i];
        float refined_idx = (float)idx;
        
        // Apply parabolic interpolation if possible
        if (idx > 0 && idx < (fft_len / 2 - 1))
        {
            float y0 = power_spectrum[idx - 1];
            float y1 = power_spectrum[idx];
            float y2 = power_spectrum[idx + 1];
            
            float denominator = y0 - 2.0f * y1 + y2;
            if (fabsf(denominator) > 1e-6f)
            {
                float offset = 0.5f * (y0 - y2) / denominator;
                refined_idx = (float)idx + offset;
                
                // Clamp to valid range
                if (refined_idx < 0.0f)
                    refined_idx = 0.0f;
                if (refined_idx >= (float)(fft_len / 2))
                    refined_idx = (float)(fft_len / 2 - 1);
            }
        }
        
        frequencies[i] = refined_idx * sample_rate / fft_len;
        powers[i] = power_spectrum[idx];
    }
    
    // If we found fewer peaks than requested, set remaining to zero
    for (int i = n_to_return; i < top_n; i++)
    {
        frequencies[i] = 0.0f;
        powers[i] = 0.0f;
    }

    free(indices);
    return TINY_OK;
}

