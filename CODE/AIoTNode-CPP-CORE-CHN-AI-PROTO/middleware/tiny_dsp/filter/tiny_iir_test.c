/**
 * @file tiny_iir_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_iir | test | source
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_iir_test.h"
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
 * @brief Test IIR filter design
 */
static void test_iir_design(void)
{
    printf("========== IIR Filter Design Test ==========\n\n");

    const int order = 2;
    const int num_coeffs = order + 1;
    float *b_coeffs = (float *)malloc(num_coeffs * sizeof(float));
    float *a_coeffs = (float *)malloc(num_coeffs * sizeof(float));

    if (!b_coeffs || !a_coeffs)
    {
        printf("  ✗ Memory allocation failed\n");
        free(b_coeffs);
        free(a_coeffs);
        return;
    }

    // Test 1: Low-pass Butterworth filter design
    printf("Test 1: Low-Pass Butterworth Filter Design\n");
    printf("  Parameters: cutoff=0.1 (normalized), order=%d\n", order);
    tiny_error_t err = tiny_iir_design_lowpass(0.1f, order,
                                                TINY_IIR_DESIGN_BUTTERWORTH, 0.0f,
                                                b_coeffs, a_coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Low-pass design failed: %d\n", err);
        goto cleanup;
    }
    printf("  ✓ Low-pass filter designed successfully\n");
    printf("  B coefficients: ");
    for (int i = 0; i < num_coeffs; i++)
    {
        printf("%.6f ", b_coeffs[i]);
    }
    printf("\n  A coefficients: ");
    for (int i = 0; i < num_coeffs; i++)
    {
        printf("%.6f ", a_coeffs[i]);
    }
    printf("\n");
    printf("  Note: a[0] should be 1.0 (normalized)\n");
    printf("\n");

    // Test 2: High-pass Butterworth filter design
    printf("Test 2: High-Pass Butterworth Filter Design\n");
    printf("  Parameters: cutoff=0.2 (normalized), order=%d\n", order);
    err = tiny_iir_design_highpass(0.2f, order,
                                    TINY_IIR_DESIGN_BUTTERWORTH, 0.0f,
                                    b_coeffs, a_coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ High-pass design failed: %d\n", err);
        goto cleanup;
    }
    printf("  ✓ High-pass filter designed successfully\n");
    printf("  B coefficients: ");
    for (int i = 0; i < num_coeffs; i++)
    {
        printf("%.6f ", b_coeffs[i]);
    }
    printf("\n  A coefficients: ");
    for (int i = 0; i < num_coeffs; i++)
    {
        printf("%.6f ", a_coeffs[i]);
    }
    printf("\n\n");

    // Test 3: Band-pass (should return not supported for now)
    printf("Test 3: Band-Pass Filter Design\n");
    printf("  Parameters: low=0.1, high=0.3 (normalized), order=%d\n", order);
    err = tiny_iir_design_bandpass(0.1f, 0.3f, order,
                                    TINY_IIR_DESIGN_BUTTERWORTH, 0.0f,
                                    b_coeffs, a_coeffs);
    if (err == TINY_ERR_NOT_SUPPORTED)
    {
        printf("  ⚠ Band-pass design not yet implemented (expected)\n");
    }
    else if (err == TINY_OK)
    {
        printf("  ✓ Band-pass filter designed successfully\n");
    }
    else
    {
        printf("  ✗ Band-pass design failed: %d\n", err);
    }
    printf("\n");

cleanup:
    free(b_coeffs);
    free(a_coeffs);
    printf("========================================\n\n");
}

/**
 * @brief Test IIR filter application (batch processing)
 */
static void test_iir_batch_filtering(void)
{
    printf("========== IIR Batch Filtering Test ==========\n\n");

    const int signal_len = 256;
    const float sample_rate = 1000.0f; // 1 kHz
    const int order = 2;
    const int num_coeffs = order + 1;
    const float cutoff_freq = 0.1f; // Normalized (100 Hz at 1 kHz sample rate)

    // Allocate memory
    float *input = (float *)malloc(signal_len * sizeof(float));
    float *output = (float *)malloc(signal_len * sizeof(float));
    float *b_coeffs = (float *)malloc(num_coeffs * sizeof(float));
    float *a_coeffs = (float *)malloc(num_coeffs * sizeof(float));

    if (!input || !output || !b_coeffs || !a_coeffs)
    {
        printf("  ✗ Memory allocation failed\n");
        free(input);
        free(output);
        free(b_coeffs);
        free(a_coeffs);
        return;
    }

    // Generate test signal
    generate_test_signal(input, signal_len, sample_rate);

    // Design low-pass filter
    printf("Test: Low-Pass IIR Filtering (Butterworth)\n");
    printf("  Input signal: DC + 10Hz + 50Hz + 100Hz components\n");
    printf("  Filter: Low-pass Butterworth, cutoff=%.1f Hz (normalized=%.3f)\n",
           cutoff_freq * sample_rate, cutoff_freq);
    printf("  Order: %d\n\n", order);

    tiny_error_t err = tiny_iir_design_lowpass(cutoff_freq, order,
                                                TINY_IIR_DESIGN_BUTTERWORTH, 0.0f,
                                                b_coeffs, a_coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter design failed: %d\n", err);
        goto cleanup;
    }

    // Apply filter
    err = tiny_iir_filter_f32(input, signal_len, b_coeffs, num_coeffs,
                              a_coeffs, num_coeffs, output, NULL);
    if (err != TINY_OK)
    {
        printf("  ✗ Filtering failed: %d\n", err);
        goto cleanup;
    }

    printf("  ✓ Filtering completed successfully\n\n");

    // Visualize signals
    printf("Signal Visualization:\n");
    tiny_view_signal_f32(input, signal_len, 64, 12, 0, 0, "Original Signal");
    tiny_view_signal_f32(output, signal_len, 64, 12, 0, 0, "Filtered Signal (Low-Pass IIR)");

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
    free(b_coeffs);
    free(a_coeffs);
    printf("\n========================================\n\n");
}

/**
 * @brief Test IIR real-time filtering
 */
static void test_iir_realtime_filtering(void)
{
    printf("========== IIR Real-Time Filtering Test ==========\n\n");

    const int order = 2;
    const int num_coeffs = order + 1;
    const float cutoff_freq = 0.1f;

    // Design filter
    float *b_coeffs = (float *)malloc(num_coeffs * sizeof(float));
    float *a_coeffs = (float *)malloc(num_coeffs * sizeof(float));

    if (!b_coeffs || !a_coeffs)
    {
        printf("  ✗ Memory allocation failed\n");
        return;
    }

    tiny_error_t err = tiny_iir_design_lowpass(cutoff_freq, order,
                                                TINY_IIR_DESIGN_BUTTERWORTH, 0.0f,
                                                b_coeffs, a_coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter design failed: %d\n", err);
        free(b_coeffs);
        free(a_coeffs);
        return;
    }

    // Initialize filter
    tiny_iir_filter_t filter;
    err = tiny_iir_init(&filter, b_coeffs, num_coeffs, a_coeffs, num_coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter initialization failed: %d\n", err);
        free(b_coeffs);
        free(a_coeffs);
        return;
    }

    printf("Test: Real-Time IIR Filtering\n");
    printf("  Filter: Low-pass Butterworth, order=%d\n", order);
    printf("  Processing samples one by one...\n\n");

    // Process test samples
    const int num_samples = 20;
    float test_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,
                          0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, 0.0f};
    float *output = (float *)malloc(num_samples * sizeof(float));

    if (!output)
    {
        printf("  ✗ Memory allocation failed\n");
        tiny_iir_deinit(&filter);
        free(b_coeffs);
        free(a_coeffs);
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
        output[i] = tiny_iir_process_sample(&filter, test_input[i]);
        printf("%.3f ", output[i]);
    }
    printf("\n\n");

    // Test reset
    printf("  Testing filter reset...\n");
    err = tiny_iir_reset(&filter);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter reset failed: %d\n", err);
    }
    else
    {
        printf("  ✓ Filter reset successful\n");
    }

    // Cleanup
    tiny_iir_deinit(&filter);
    free(b_coeffs);
    free(a_coeffs);
    free(output);

    printf("\n========================================\n\n");
}

/**
 * @brief Test IIR biquad filter
 */
static void test_iir_biquad(void)
{
    printf("========== IIR Biquad Filter Test ==========\n\n");

    // Design a simple low-pass biquad (second-order Butterworth)
    const float cutoff_freq = 0.1f;
    float b_coeffs[3], a_coeffs[3];

    tiny_error_t err = tiny_iir_design_lowpass(cutoff_freq, 2,
                                                TINY_IIR_DESIGN_BUTTERWORTH, 0.0f,
                                                b_coeffs, a_coeffs);
    if (err != TINY_OK)
    {
        printf("  ✗ Filter design failed: %d\n", err);
        return;
    }

    // Initialize biquad
    tiny_iir_biquad_t biquad;
    err = tiny_iir_biquad_init(&biquad, b_coeffs[0], b_coeffs[1], b_coeffs[2],
                                a_coeffs[1], a_coeffs[2]);
    if (err != TINY_OK)
    {
        printf("  ✗ Biquad initialization failed: %d\n", err);
        return;
    }

    printf("Test: Biquad (Second-Order) IIR Filter\n");
    printf("  Coefficients: b0=%.6f, b1=%.6f, b2=%.6f, a1=%.6f, a2=%.6f\n",
           biquad.b0, biquad.b1, biquad.b2, biquad.a1, biquad.a2);
    printf("\n");

    // Process test samples
    const int num_samples = 10;
    float test_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f};

    printf("  Input samples: ");
    for (int i = 0; i < num_samples; i++)
    {
        printf("%.1f ", test_input[i]);
    }
    printf("\n");

    printf("  Output samples: ");
    for (int i = 0; i < num_samples; i++)
    {
        float output = tiny_iir_biquad_process_sample(&biquad, test_input[i]);
        printf("%.3f ", output);
    }
    printf("\n\n");

    // Test reset
    err = tiny_iir_biquad_reset(&biquad);
    if (err != TINY_OK)
    {
        printf("  ✗ Biquad reset failed: %d\n", err);
    }
    else
    {
        printf("  ✓ Biquad reset successful\n");
    }

    printf("\n========================================\n\n");
}

/**
 * @brief Main IIR test function
 */
void tiny_iir_test(void)
{
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║          TinyIIR Filter Test Suite                       ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Run all tests
    test_iir_design();
    test_iir_batch_filtering();
    test_iir_realtime_filtering();
    test_iir_biquad();

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║          All IIR Tests Completed                          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
}

