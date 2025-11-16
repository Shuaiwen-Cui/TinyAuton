# TESTS

## tiny_fft_test.h

```c
/**
 * @file tiny_fft_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fft | test | header
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_fft.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_fft_test(void);

#ifdef __cplusplus
}
#endif

```

## tiny_fft_test.c

```c
/**
 * @file tiny_fft_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fft | test | source
 * @version 1.0
 * @date 2025-04-29
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_fft_test.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Generate a test signal with known frequency components
 */
static void generate_test_signal(float *signal, int len, float sample_rate)
{
    // Generate signal: sin(2*pi*10*t) + 0.5*sin(2*pi*50*t)
    // Frequencies: 10 Hz and 50 Hz
    for (int i = 0; i < len; i++)
    {
        float t = (float)i / sample_rate;
        signal[i] = sinf(2.0f * M_PI * 10.0f * t) + 0.5f * sinf(2.0f * M_PI * 50.0f * t);
    }
}

void tiny_fft_test(void)
{
    printf("========== TinyFFT Test ==========\n\n");

    const int fft_size = 256;
    const float sample_rate = 1000.0f; // 1 kHz sampling rate
    const int signal_len = fft_size;

    // Initialize FFT
    printf("1. FFT Initialization:\n");
    tiny_error_t ret = tiny_fft_init(fft_size);
    if (ret != TINY_OK)
    {
        printf("  ✗ FFT initialization failed: %d\n", ret);
        return;
    }
    printf("  ✓ FFT initialized (max size: %d)\n\n", fft_size);

    // Generate test signal
    float *input_signal = (float *)malloc(signal_len * sizeof(float));
    float *fft_result = (float *)malloc(signal_len * 2 * sizeof(float)); // Complex output
    float *magnitude = (float *)malloc(signal_len * sizeof(float));
    float *power = (float *)malloc(signal_len * sizeof(float));
    float *reconstructed = (float *)malloc(signal_len * sizeof(float));

    if (!input_signal || !fft_result || !magnitude || !power || !reconstructed)
    {
        printf("  ✗ Memory allocation failed\n");
        goto cleanup;
    }

    generate_test_signal(input_signal, signal_len, sample_rate);

    printf("2. Test Signal Generation:\n");
    printf("  Input: Signal with frequencies 10 Hz and 50 Hz\n");
    printf("  Sample rate: %.1f Hz\n", sample_rate);
    printf("  Signal length: %d samples\n", signal_len);
    printf("  First 10 samples: ");
    for (int i = 0; i < 10 && i < signal_len; i++)
    {
        printf("%.3f ", input_signal[i]);
    }
    printf("\n\n");

    // Test FFT without window
    printf("3. FFT (No Window):\n");
    printf("  Input: Test signal (length=%d)\n", signal_len);
    ret = tiny_fft_f32(input_signal, signal_len, fft_result, TINY_FFT_WINDOW_NONE);
    if (ret != TINY_OK)
    {
        printf("  ✗ FFT failed: %d\n", ret);
        goto cleanup;
    }
    printf("  ✓ FFT completed\n");

    // Calculate magnitude
    ret = tiny_fft_magnitude_f32(fft_result, signal_len, magnitude);
    if (ret != TINY_OK)
    {
        printf("  ✗ Magnitude calculation failed\n");
        goto cleanup;
    }

    // Calculate power spectrum
    ret = tiny_fft_power_spectrum_f32(fft_result, signal_len, power);
    if (ret != TINY_OK)
    {
        printf("  ✗ Power spectrum calculation failed\n");
        goto cleanup;
    }

    printf("  Output: FFT result (complex, length=%d)\n", signal_len);
    printf("  Magnitude spectrum: First 10 values: ");
    for (int i = 0; i < 10 && i < signal_len; i++)
    {
        printf("%.3f ", magnitude[i]);
    }
    printf("\n\n");

    // Find peak frequency
    printf("4. Peak Frequency Detection:\n");
    printf("  Input: Power spectrum (length=%d)\n", signal_len);
    float peak_freq, peak_power;
    ret = tiny_fft_find_peak_frequency(power, signal_len, sample_rate, &peak_freq, &peak_power);
    if (ret != TINY_OK)
    {
        printf("  ✗ Peak detection failed\n");
        goto cleanup;
    }
    printf("  Output: Peak frequency = %.2f Hz (power = %.3f)\n", peak_freq, peak_power);
    printf("  Expected: ~10 Hz or ~50 Hz (strongest component)\n\n");

    // Find top frequencies
    printf("5. Top Frequencies Detection:\n");
    printf("  Input: Power spectrum (length=%d)\n", signal_len);
    const int top_n = 3;
    float *top_freqs = (float *)malloc(top_n * sizeof(float));
    float *top_powers = (float *)malloc(top_n * sizeof(float));
    if (!top_freqs || !top_powers)
    {
        printf("  ✗ Memory allocation failed\n");
        goto cleanup;
    }

    ret = tiny_fft_find_top_frequencies(power, signal_len, sample_rate, top_n, top_freqs, top_powers);
    if (ret != TINY_OK)
    {
        printf("  ✗ Top frequencies detection failed\n");
        goto cleanup;
    }
    printf("  Output: Top %d frequencies:\n", top_n);
    for (int i = 0; i < top_n; i++)
    {
        printf("    [%d] %.2f Hz (power = %.3f)\n", i + 1, top_freqs[i], top_powers[i]);
    }
    printf("  Expected: ~10 Hz and ~50 Hz should be in top frequencies\n\n");

    // Test IFFT
    printf("6. IFFT (Signal Reconstruction):\n");
    printf("  Input: FFT result (complex, length=%d)\n", signal_len);
    ret = tiny_fft_ifft_f32(fft_result, signal_len, reconstructed);
    if (ret != TINY_OK)
    {
        printf("  ✗ IFFT failed: %d\n", ret);
        goto cleanup;
    }
    printf("  Output: Reconstructed signal (length=%d)\n", signal_len);
    printf("  First 10 samples: ");
    for (int i = 0; i < 10 && i < signal_len; i++)
    {
        printf("%.3f ", reconstructed[i]);
    }
    printf("\n");

    // Verify reconstruction (should match original, accounting for window effects)
    float max_diff = 0.0f;
    for (int i = 0; i < signal_len; i++)
    {
        float diff = fabsf(reconstructed[i] - input_signal[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    printf("  Max difference from original: %.6f\n", max_diff);
    printf("  ✓ IFFT reconstruction completed\n\n");

    // Test with window
    printf("7. FFT with Hanning Window:\n");
    printf("  Input: Test signal (length=%d) with Hanning window\n", signal_len);
    ret = tiny_fft_f32(input_signal, signal_len, fft_result, TINY_FFT_WINDOW_HANNING);
    if (ret != TINY_OK)
    {
        printf("  ✗ Windowed FFT failed\n");
        goto cleanup;
    }
    ret = tiny_fft_power_spectrum_f32(fft_result, signal_len, power);
    if (ret != TINY_OK)
    {
        printf("  ✗ Power spectrum calculation failed\n");
        goto cleanup;
    }
    ret = tiny_fft_find_peak_frequency(power, signal_len, sample_rate, &peak_freq, &peak_power);
    if (ret != TINY_OK)
    {
        printf("  ✗ Peak detection failed\n");
        goto cleanup;
    }
    printf("  Output: Peak frequency = %.2f Hz (power = %.3f)\n", peak_freq, peak_power);
    printf("  Note: Window reduces spectral leakage, improving frequency resolution\n\n");

    free(top_freqs);
    free(top_powers);

cleanup:
    if (input_signal)
        free(input_signal);
    if (fft_result)
        free(fft_result);
    if (magnitude)
        free(magnitude);
    if (power)
        free(power);
    if (reconstructed)
        free(reconstructed);

    // Deinitialize
    tiny_fft_deinit();
    printf("8. FFT Deinitialization:\n");
    printf("  ✓ FFT deinitialized\n");

    printf("\n========================================\n");
}

```


## TEST RESULTS

```
========== TinyFFT Test ==========

1. FFT Initialization:
  ✓ FFT initialized (max size: 256)

2. Test Signal Generation:
  Input: Signal with frequencies 10 Hz and 50 Hz
  Sample rate: 1000.0 Hz
  Signal length: 256 samples
  First 10 samples: 0.000 0.217 0.419 0.592 0.724 0.809 0.844 0.830 0.776 0.690 

3. FFT (No Window):
  Input: Test signal (length=256)
  ✓ FFT completed
  Output: FFT result (complex, length=256)
  Magnitude spectrum: First 10 values: 32.216 37.858 81.243 82.696 20.529 9.805 5.459 3.052 1.398 0.332 

4. Peak Frequency Detection:
  Input: Power spectrum (length=256)
  Output: Peak frequency = 9.91 Hz (power = 26.714)
  Expected: ~10 Hz or ~50 Hz (strongest component)

5. Top Frequencies Detection:
  Input: Power spectrum (length=256)
  Output: Top 3 frequencies:
    [1] 9.91 Hz (power = 26.714)
    [2] 50.77 Hz (power = 14.756)
    [3] 0.00 Hz (power = 0.000)
  Expected: ~10 Hz and ~50 Hz should be in top frequencies

6. IFFT (Signal Reconstruction):
  Input: FFT result (complex, length=256)
  Output: Reconstructed signal (length=256)
  First 10 samples: 0.000 0.217 0.419 0.592 0.724 0.809 0.844 0.830 0.776 0.690 
  Max difference from original: 0.000003
  ✓ IFFT reconstruction completed

7. FFT with Hanning Window:
  Input: Test signal (length=256) with Hanning window
  Output: Peak frequency = 10.32 Hz (power = 12.407)
  Note: Window reduces spectral leakage, improving frequency resolution

8. FFT Deinitialization:
  ✓ FFT deinitialized

========================================
```
