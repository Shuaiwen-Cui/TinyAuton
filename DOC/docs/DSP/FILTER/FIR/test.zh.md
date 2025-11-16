# 测试

## tiny_fir_test.h

```c
/**
 * @file tiny_fir_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fir | test | header
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_fir.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_fir_test(void);

#ifdef __cplusplus
}
#endif


```

## tiny_fir_test.c

```c
/**
 * @file tiny_fir_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_fir | test | source
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_fir_test.h"
#include "tiny_view.h" // For signal visualization
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define EPSILON 1e-4f // Tolerance for floating-point comparison

/**
 * @brief Generate a test signal with multiple frequency components
 */
static void generate_test_signal(float *signal, int len, float sample_rate)
{
    // Generate signal: DC + 10Hz + 50Hz + 100Hz
    for (int i = 0; i < len; i++)
    {
        float t = (float)i / sample_rate;
        signal[i] = 1.0f +                                    // DC component
                    sinf(2.0f * M_PI * 10.0f * t) +          // 10 Hz
                    0.5f * sinf(2.0f * M_PI * 50.0f * t) +   // 50 Hz
                    0.3f * sinf(2.0f * M_PI * 100.0f * t);   // 100 Hz
    }
}

/**
 * @brief Test FIR filter design
 */
static void test_fir_design(void)
{
    printf("========== FIR Filter Design Test ==========\n\n");

    const int num_taps = 51;
    float *coeffs = (float *)malloc(num_taps * sizeof(float));
    if (!coeffs)
    {
        printf("  ✗ Memory allocation failed\n");
        return;
    }

    // Test 1: Low-pass filter design
    printf("Test 1: Low-Pass Filter Design\n");
    printf("  Parameters: cutoff=0.1 (normalized), taps=%d, window=Hamming\n", num_taps);
    tiny_error_t err = tiny_fir_design_lowpass(0.1f, num_taps, TINY_FIR_WINDOW_HAMMING, coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Low-pass design failed: %d\n", err);
        free(coeffs);
        return;
    }
    printf("  ✓ Low-pass filter designed successfully\n");
    printf("  Coefficient range: [%.6f, %.6f]\n",
           coeffs[0], coeffs[num_taps / 2]);
    printf("\n");

    // Test 2: High-pass filter design
    printf("Test 2: High-Pass Filter Design\n");
    printf("  Parameters: cutoff=0.2 (normalized), taps=%d, window=Hanning\n", num_taps);
    err = tiny_fir_design_highpass(0.2f, num_taps, TINY_FIR_WINDOW_HANNING, coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ High-pass design failed: %d\n", err);
        free(coeffs);
        return;
    }
    printf("  ✓ High-pass filter designed successfully\n");
    printf("\n");

    // Test 3: Band-pass filter design
    printf("Test 3: Band-Pass Filter Design\n");
    printf("  Parameters: low=0.1, high=0.3 (normalized), taps=%d, window=Blackman\n", num_taps);
    err = tiny_fir_design_bandpass(0.1f, 0.3f, num_taps, TINY_FIR_WINDOW_BLACKMAN, coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Band-pass design failed: %d\n", err);
        free(coeffs);
        return;
    }
    printf("  ✓ Band-pass filter designed successfully\n");
    printf("\n");

    // Test 4: Band-stop filter design
    printf("Test 4: Band-Stop Filter Design\n");
    printf("  Parameters: low=0.1, high=0.3 (normalized), taps=%d, window=Hamming\n", num_taps);
    err = tiny_fir_design_bandstop(0.1f, 0.3f, num_taps, TINY_FIR_WINDOW_HAMMING, coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Band-stop design failed: %d\n", err);
        free(coeffs);
        return;
    }
    printf("  ✓ Band-stop filter designed successfully\n");
    printf("\n");

    free(coeffs);
    printf("========================================\n\n");
}

/**
 * @brief Test FIR filter application (batch processing)
 */
static void test_fir_batch_filtering(void)
{
    printf("========== FIR Batch Filtering Test ==========\n\n");

    const int signal_len = 256;
    const float sample_rate = 1000.0f; // 1 kHz
    const int num_taps = 51;
    const float cutoff_freq = 0.1f; // Normalized (100 Hz at 1 kHz sample rate)

    // Allocate memory
    float *input = (float *)malloc(signal_len * sizeof(float));
    float *output = (float *)malloc(signal_len * sizeof(float));
    float *coeffs = (float *)malloc(num_taps * sizeof(float));

    if (!input || !output || !coeffs)
    {
        printf("  ✗ Memory allocation failed\n");
        free(input);
        free(output);
        free(coeffs);
        return;
    }

    // Generate test signal
    generate_test_signal(input, signal_len, sample_rate);

    // Design low-pass filter
    printf("Test: Low-Pass FIR Filtering\n");
    printf("  Input signal: DC + 10Hz + 50Hz + 100Hz components\n");
    printf("  Filter: Low-pass, cutoff=%.1f Hz (normalized=%.3f)\n",
           cutoff_freq * sample_rate, cutoff_freq);
    printf("  Taps: %d, Window: Hamming\n\n", num_taps);

    tiny_error_t err = tiny_fir_design_lowpass(cutoff_freq, num_taps,
                                                TINY_FIR_WINDOW_HAMMING, coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter design failed: %d\n", err);
        goto cleanup;
    }

    // Apply filter
    err = tiny_fir_filter_f32(input, signal_len, coeffs, num_taps,
                               output, TINY_PADDING_SYMMETRIC);
    if (err != TINY_OK)
    {
        printf("  ✗ Filtering failed: %d\n", err);
        goto cleanup;
    }

    printf("  ✓ Filtering completed successfully\n\n");

    // Visualize signals
    printf("Signal Visualization:\n");
    tiny_view_signal_f32(input, signal_len, 64, 12, 0, 0, "Original Signal");
    tiny_view_signal_f32(output, signal_len, 64, 12, 0, 0, "Filtered Signal (Low-Pass)");

    // Calculate statistics
    float input_mean = 0.0f, output_mean = 0.0f;
    float input_power = 0.0f, output_power = 0.0f;
    for (int i = 0; i < signal_len; i++)
    {
        input_mean += input[i];
        output_mean += output[i];
        input_power += input[i] * input[i];
        output_power += output[i] * output[i];
    }
    input_mean /= signal_len;
    output_mean /= signal_len;
    input_power /= signal_len;
    output_power /= signal_len;

    printf("\nStatistics:\n");
    printf("  Input:  mean=%.4f, power=%.4f\n", input_mean, input_power);
    printf("  Output: mean=%.4f, power=%.4f\n", output_mean, output_power);
    printf("  Power reduction: %.2f%%\n", (1.0f - output_power / input_power) * 100.0f);

cleanup:
    free(input);
    free(output);
    free(coeffs);
    printf("\n========================================\n\n");
}

/**
 * @brief Test FIR real-time filtering
 */
static void test_fir_realtime_filtering(void)
{
    printf("========== FIR Real-Time Filtering Test ==========\n\n");

    const int num_taps = 21;
    const float cutoff_freq = 0.1f;

    // Design filter
    float *coeffs = (float *)malloc(num_taps * sizeof(float));
    if (!coeffs)
    {
        printf("  ✗ Memory allocation failed\n");
        return;
    }

    tiny_error_t err = tiny_fir_design_lowpass(cutoff_freq, num_taps,
                                                TINY_FIR_WINDOW_HAMMING, coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter design failed: %d\n", err);
        free(coeffs);
        return;
    }

    // Initialize filter
    tiny_fir_filter_t filter;
    err = tiny_fir_init(&filter, coeffs, num_taps);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter initialization failed: %d\n", err);
        free(coeffs);
        return;
    }

    printf("Test: Real-Time FIR Filtering\n");
    printf("  Filter: Low-pass, taps=%d\n", num_taps);
    printf("  Processing samples one by one...\n\n");

    // Process test samples
    const int num_samples = 20;
    float test_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,
                          0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, 0.0f};
    float *output = (float *)malloc(num_samples * sizeof(float));

    if (!output)
    {
        printf("  ✗ Memory allocation failed\n");
        tiny_fir_deinit(&filter);
        free(coeffs);
        return;
    }

    printf("  Input samples: ");
    for (int i = 0; i < num_samples; i++)
    {
        printf("%.1f ", test_input[i]);
    }
    printf("\n");

    printf("  Output samples: ");
    for (int i = 0; i < num_samples; i++)
    {
        output[i] = tiny_fir_process_sample(&filter, test_input[i]);
        printf("%.3f ", output[i]);
    }
    printf("\n\n");

    // Test reset
    printf("  Testing filter reset...\n");
    err = tiny_fir_reset(&filter);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter reset failed: %d\n", err);
    }
    else
    {
        printf("  ✓ Filter reset successful\n");
    }

    // Cleanup
    tiny_fir_deinit(&filter);
    free(coeffs);
    free(output);

    printf("\n========================================\n\n");
}

/**
 * @brief Main FIR test function
 */
void tiny_fir_test(void)
{
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║          TinyFIR Filter Test Suite                      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Run all tests
    test_fir_design();
    test_fir_batch_filtering();
    test_fir_realtime_filtering();

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║          All FIR Tests Completed                          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
}


```

## 输出结果

```c
╔══════════════════════════════════════════════════════════╗
║          TinyFIR Filter Test Suite                      ║
╚══════════════════════════════════════════════════════════╝

========== FIR Filter Design Test ==========

Test 1: Low-Pass Filter Design
  Parameters: cutoff=0.1 (normalized), taps=51, window=Hamming
  ✓ Low-pass filter designed successfully
  Coefficient range: [-0.000000, 0.200000]

Test 2: High-Pass Filter Design
  Parameters: cutoff=0.2 (normalized), taps=51, window=Hanning
  ✓ High-pass filter designed successfully

Test 3: Band-Pass Filter Design
  Parameters: low=0.1, high=0.3 (normalized), taps=51, window=Blackman
  ✓ Band-pass filter designed successfully

Test 4: Band-Stop Filter Design
  Parameters: low=0.1, high=0.3 (normalized), taps=51, window=Hamming
  ✓ Band-stop filter designed successfully

========================================

========== FIR Batch Filtering Test ==========

Test: Low-Pass FIR Filtering
  Input signal: DC + 10Hz + 50Hz + 100Hz components
  Filter: Low-pass, cutoff=100.0 Hz (normalized=0.100)
  Taps: 51, Window: Hamming

  ✓ Filtering completed successfully

Signal Visualization:

Original Signal
Value
  3.02 |                                                                
  2.65 |      *                                                *        
  2.28 |     **                       **                      * *   *   
  1.92 | *   * **  *              *  *  *  *              *   *  * **   
  1.55 |* ***   * **             * ***   ** *            * ****  * **   
  1.18 |*   *    *  *           *    *    *  *           *        ** *  
  0.82 |*            *  *    *  *            *  *    *   *            * 
  0.45 |             * **   * ***             ** *  * ****            * 
  0.08 |              *  ***    *              *  * *                  *
 -0.28 |                  **                       **                   
 -0.65 |                   *                        *                   
 -1.02 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.018, 3.018], Length: 256


Filtered Signal (Low-Pass)
Value
  2.88 |                                                                
  2.54 |      *                        *                       *        
  2.20 |     * *                      **                      * *       
  1.85 |  *  *  *  *              *  *  *  **              *  *  *  *   
  1.51 |** **   * **             * ***   ** *             * ***   ** *  
  1.17 |    *    *  *           *         *  *           *           *  
  0.83 |            *   *    *  *            *  *        *            * 
  0.49 |             * **   * * *             ** *   *** *             *
  0.15 |              *  *  *  **              *  * *   *               
 -0.19 |                  **                      * *                   
 -0.53 |                   *                       **                   
 -0.87 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-0.872, 2.877], Length: 256


Statistics:
  Input:  mean=1.1294, power=1.9174
  Output: mean=1.1321, power=1.8899
  Power reduction: 1.43%

========================================

========== FIR Real-Time Filtering Test ==========

Test: Real-Time FIR Filtering
  Filter: Low-pass, taps=21
  Processing samples one by one...

  Input samples: 1.0 2.0 3.0 4.0 5.0 4.0 3.0 2.0 1.0 0.0 0.0 1.0 2.0 3.0 4.0 3.0 2.0 1.0 0.0 0.0 
  Output samples: 0.000 -0.002 -0.011 -0.031 -0.063 -0.096 -0.092 0.006 0.265 0.733 1.400 2.184 2.934 3.472 3.652 3.419 2.845 2.109 1.446 1.063 

  Testing filter reset...
  ✓ Filter reset successful

========================================

╔══════════════════════════════════════════════════════════╗
║          All FIR Tests Completed                          ║
╚══════════════════════════════════════════════════════════╝
```