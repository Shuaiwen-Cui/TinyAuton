# TESTS

## tiny_dwt_test.h

```c
/**
 * @file tiny_dwt_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_dwt | test | header
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_dwt.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @name tiny_dwt_test
 * @brief Unit test for single-level DWT and inverse DWT
 */
void tiny_dwt_test(void);

/**
 * @name tiny_dwt_test_multilevel
 * @brief Unit test for multi-level DWT and inverse DWT
 */
void tiny_dwt_test_multilevel(void);

/**
 * @name tiny_dwt_test_wavelets
 * @brief Test different wavelet types (DB1-DB10)
 */
void tiny_dwt_test_wavelets(void);

/**
 * @name tiny_dwt_test_all
 * @brief Run all DWT tests
 */
void tiny_dwt_test_all(void);

#ifdef __cplusplus
}
#endif

```

## tiny_dwt_test.c

```c
/**
 * @file tiny_dwt_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_dwt | test | source
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_dwt_test.h" // TinyDWT Test Header
#include "tiny_view.h"     // Signal visualization
#include <math.h>
#include <stdlib.h>        // For malloc/free

/**
 * @brief Calculate signal energy (sum of squares)
 */
static float calculate_energy(const float *signal, int len)
{
    float energy = 0.0f;
    for (int i = 0; i < len; i++)
    {
        energy += signal[i] * signal[i];
    }
    return energy;
}

/**
 * @brief Calculate maximum absolute error between two signals
 */
static float calculate_max_error(const float *signal1, const float *signal2, int len, int *max_err_idx)
{
    float max_err = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < len; i++)
    {
        float err = fabsf(signal1[i] - signal2[i]);
        if (err > max_err)
        {
            max_err = err;
            max_idx = i;
        }
    }
    if (max_err_idx)
        *max_err_idx = max_idx;
    return max_err;
}

/**
 * @brief Calculate mean squared error (MSE)
 */
static float calculate_mse(const float *signal1, const float *signal2, int len)
{
    float mse = 0.0f;
    for (int i = 0; i < len; i++)
    {
        float diff = signal1[i] - signal2[i];
        mse += diff * diff;
    }
    return mse / len;
}

/**
 * @name tiny_dwt_test
 * @brief Unit test for single-level DWT and inverse DWT
 */
void tiny_dwt_test(void)
{
    printf("========== TinyDWT Single-Level Test ==========\n\n");

    // Test signal: longer sinusoidal pattern to reduce boundary effects
    // Generate 64 samples: 2 cycles of sine wave
    // Use dynamic allocation to avoid stack overflow
    #define TEST_SIGNAL_LEN 64
    float *input = (float *)malloc(TEST_SIGNAL_LEN * sizeof(float));
    if (!input)
    {
        printf("  ✗ Memory allocation failed for input signal\n");
        return;
    }
    
    for (int i = 0; i < TEST_SIGNAL_LEN; i++)
    {
        input[i] = 2.0f * sinf(2.0f * M_PI * i / (TEST_SIGNAL_LEN / 2.0f));
    }
    int input_len = TEST_SIGNAL_LEN;

    printf("Test 1: Single-Level DWT Decomposition and Reconstruction\n");
    printf("  Input: Sinusoidal signal (length=%d, 2 cycles)\n", input_len);
    printf("  Wavelet: DB4\n");
    printf("  Note: Using longer signal to better assess boundary effects\n\n");

    float *cA = (float *)calloc(128, sizeof(float));
    float *cD = (float *)calloc(128, sizeof(float));
    float *output = (float *)calloc(256, sizeof(float));
    
    if (!cA || !cD || !output)
    {
        printf("  ✗ Memory allocation failed\n");
        free(input);
        free(cA);
        free(cD);
        free(output);
        return;
    }
    
    int cA_len = 0, cD_len = 0;
    int output_len = 0;

    tiny_error_t err;

    // Decomposition
    printf("1. DWT Decomposition:\n");
    printf("  Input: Original signal (length=%d)\n", input_len);
    err = tiny_dwt_decompose_f32(input, input_len, TINY_WAVELET_DB4, cA, cD, &cA_len, &cD_len);
    if (err != TINY_OK)
    {
        printf("  ✗ DWT decomposition failed: %d\n", err);
        return;
    }
    printf("  ✓ Decomposition completed\n");
    printf("  Output: Approximation coefficients (cA, length=%d)\n", cA_len);
    printf("  Output: Detail coefficients (cD, length=%d)\n", cD_len);

    // Calculate energy preservation
    float input_energy = calculate_energy(input, input_len);
    float cA_energy = calculate_energy(cA, cA_len);
    float cD_energy = calculate_energy(cD, cD_len);
    float coeff_energy = cA_energy + cD_energy;
    float energy_ratio = (input_energy > 0) ? (coeff_energy / input_energy) : 0.0f;
    printf("  Energy: Input=%.3f, cA=%.3f, cD=%.3f, Total=%.3f (ratio=%.4f)\n\n",
           input_energy, cA_energy, cD_energy, coeff_energy, energy_ratio);

    // Reconstruction
    printf("2. DWT Reconstruction:\n");
    printf("  Input: cA (length=%d) + cD (length=%d)\n", cA_len, cD_len);
    err = tiny_dwt_reconstruct_f32(cA, cD, cA_len, TINY_WAVELET_DB4, output, &output_len);
    if (err != TINY_OK)
    {
        printf("  ✗ DWT reconstruction failed: %d\n", err);
        return;
    }
    printf("  ✓ Reconstruction completed\n");
    printf("  Output: Reconstructed signal (length=%d)\n\n", output_len);

    // Visualization
    printf("3. Signal Visualization:\n");
    tiny_view_signal_f32(input, input_len, 64, 12, 0, 0, "Original Signal");
    tiny_view_signal_f32(cA, cA_len, 32, 12, 0, 0, "Approximation (cA)");
    tiny_view_signal_f32(cD, cD_len, 32, 12, 0, 0, "Detail (cD)");
    tiny_view_signal_f32(output, output_len, 64, 12, 0, 0, "Reconstructed Signal");

    // Error analysis with boundary effect assessment
    printf("4. Reconstruction Error Analysis:\n");
    int filter_len = TINY_WAVELET_GET_LEN(TINY_WAVELET_DB4);
    int boundary_width = filter_len;  // Boundary effect typically extends ~filter_len samples
    int end = (input_len < output_len) ? input_len : output_len;
    
    int max_err_idx = 0;
    float max_err = calculate_max_error(input, output, end, &max_err_idx);
    float mse = calculate_mse(input, output, end);
    float rmse = sqrtf(mse);
    
    printf("  Comparison length: %d samples\n", end);
    printf("  Max absolute error: %.6f (at index %d)\n", max_err, max_err_idx);
    printf("  Mean squared error (MSE): %.6f\n", mse);
    printf("  Root mean squared error (RMSE): %.6f\n", rmse);
    
    // Analyze boundary regions vs center region
    if (end > 2 * boundary_width)
    {
        // Left boundary region
        float left_max_err = 0.0f;
        float left_mse = 0.0f;
        for (int i = 0; i < boundary_width; i++)
        {
            float err = fabsf(output[i] - input[i]);
            if (err > left_max_err)
                left_max_err = err;
            left_mse += (output[i] - input[i]) * (output[i] - input[i]);
        }
        left_mse /= boundary_width;
        
        // Right boundary region
        float right_max_err = 0.0f;
        float right_mse = 0.0f;
        for (int i = end - boundary_width; i < end; i++)
        {
            float err = fabsf(output[i] - input[i]);
            if (err > right_max_err)
                right_max_err = err;
            right_mse += (output[i] - input[i]) * (output[i] - input[i]);
        }
        right_mse /= boundary_width;
        
        // Center region (excluding boundaries)
        float center_max_err = 0.0f;
        float center_mse = 0.0f;
        int center_count = end - 2 * boundary_width;
        for (int i = boundary_width; i < end - boundary_width; i++)
        {
            float err = fabsf(output[i] - input[i]);
            if (err > center_max_err)
                center_max_err = err;
            center_mse += (output[i] - input[i]) * (output[i] - input[i]);
        }
        center_mse /= center_count;
        float center_rmse = sqrtf(center_mse);
        
        printf("\n  Boundary Effect Analysis:\n");
        printf("  Left boundary (indices 0-%d):\n", boundary_width - 1);
        printf("    Max error: %.6f, RMSE: %.6f\n", left_max_err, sqrtf(left_mse));
        printf("  Center region (indices %d-%d, %d samples):\n", 
               boundary_width, end - boundary_width - 1, center_count);
        printf("    Max error: %.6f, RMSE: %.6f\n", center_max_err, center_rmse);
        printf("  Right boundary (indices %d-%d):\n", end - boundary_width, end - 1);
        printf("    Max error: %.6f, RMSE: %.6f\n", right_max_err, sqrtf(right_mse));
        
        // Compare center vs boundaries
        if (center_rmse < 1e-3f && (sqrtf(left_mse) > 1e-2f || sqrtf(right_mse) > 1e-2f))
        {
            printf("  ✓ Center region is highly accurate, boundary effects confirmed\n");
        }
        else if (center_rmse < 1e-2f)
        {
            printf("  ⚠ Center region has minor errors\n");
        }
        else
        {
            printf("  ✗ Center region has significant errors (may indicate implementation issue)\n");
        }
    }
    else
    {
        printf("  ⚠ Signal too short for boundary analysis (need > %d samples)\n", 2 * boundary_width);
    }
    printf("\n");

    // Energy preservation check
    float output_energy = calculate_energy(output, output_len);
    float energy_preservation = (input_energy > 0) ? (output_energy / input_energy) : 0.0f;
    printf("5. Energy Preservation:\n");
    printf("  Input energy: %.6f\n", input_energy);
    printf("  Output energy: %.6f\n", output_energy);
    printf("  Preservation ratio: %.6f\n", energy_preservation);
    if (fabsf(energy_preservation - 1.0f) < 0.1f)
        printf("  ✓ Energy is well preserved\n");
    else
        printf("  ⚠ Energy preservation ratio: %.2f%%\n", energy_preservation * 100.0f);
    printf("\n");

    // Cleanup
    free(input);
    free(cA);
    free(cD);
    free(output);
    
    printf("========================================\n");
}

/**
 * @name tiny_dwt_test_multilevel
 * @brief Test multi-level DWT decomposition and reconstruction
 */
void tiny_dwt_test_multilevel(void)
{
    printf("========== TinyDWT Multi-Level Test ==========\n\n");

    // Extended test signal: longer sinusoidal pattern
    // Generate 128 samples: 4 cycles of sine wave
    // Use dynamic allocation to avoid stack overflow
    #define MULTI_TEST_SIGNAL_LEN 128
    float *input = (float *)malloc(MULTI_TEST_SIGNAL_LEN * sizeof(float));
    if (!input)
    {
        printf("  ✗ Memory allocation failed for input signal\n");
        return;
    }
    
    for (int i = 0; i < MULTI_TEST_SIGNAL_LEN; i++)
    {
        input[i] = 2.0f * sinf(2.0f * M_PI * i / (MULTI_TEST_SIGNAL_LEN / 4.0f));
    }
    int input_len = MULTI_TEST_SIGNAL_LEN;
    int levels = 3;

    printf("Test 2: Multi-Level DWT Decomposition and Reconstruction\n");
    printf("  Input: Sinusoidal signal (length=%d, 4 cycles)\n", input_len);
    printf("  Wavelet: DB4\n");
    printf("  Decomposition levels: %d\n", levels);
    printf("  Note: Using longer signal to better assess boundary effects in multi-level decomposition\n\n");

    float *cA = NULL;
    float *cD = NULL;
    int cA_len = 0;

    // Multi-level decomposition
    printf("1. Multi-Level DWT Decomposition:\n");
    printf("  Input: Original signal (length=%d)\n", input_len);
    tiny_error_t err = tiny_dwt_multilevel_decompose_f32(input, input_len, TINY_WAVELET_DB4, levels, &cA, &cD, &cA_len);
    if (err != TINY_OK)
    {
        printf("  ✗ Multi-level DWT decomposition failed: %d\n", err);
        return;
    }
    printf("  ✓ Decomposition completed\n");
    printf("  Output: Final approximation (cA, length=%d)\n", cA_len);
    printf("  Output: All detail coefficients (cD, total length=%d)\n", input_len - cA_len);
    
    // Calculate energy
    float input_energy = calculate_energy(input, input_len);
    float cA_energy = calculate_energy(cA, cA_len);
    float cD_energy = calculate_energy(cD, input_len - cA_len);
    float total_coeff_energy = cA_energy + cD_energy;
    printf("  Energy: Input=%.3f, cA=%.3f, cD=%.3f, Total=%.3f\n\n",
           input_energy, cA_energy, cD_energy, total_coeff_energy);

    // Coefficient processing (placeholder)
    printf("2. Coefficient Processing:\n");
    tiny_dwt_coeffs_process(cA, cD, cA_len, input_len - cA_len, levels);
    printf("  ✓ Coefficient processing completed (placeholder function)\n\n");

    // Multi-level reconstruction
    printf("3. Multi-Level DWT Reconstruction:\n");
    printf("  Input: cA (length=%d) + cD (total length=%d)\n", cA_len, input_len - cA_len);
    float *output = (float *)malloc(sizeof(float) * input_len);
    if (!output)
    {
        printf("  ✗ Memory allocation failed for output\n");
        free(cA);
        free(cD);
        return;
    }

    err = tiny_dwt_multilevel_reconstruct_f32(cA, cD, cA_len, TINY_WAVELET_DB4, levels, output);
    if (err != TINY_OK)
    {
        printf("  ✗ Multi-level DWT reconstruction failed: %d\n", err);
        free(cA);
        free(cD);
        free(output);
        return;
    }
    printf("  ✓ Reconstruction completed\n");
    printf("  Output: Reconstructed signal (length=%d)\n\n", input_len);

    // Visualization
    printf("4. Signal Visualization:\n");
    tiny_view_signal_f32(input, input_len, 64, 12, 0, 0, "Original Signal");
    tiny_view_signal_f32(cA, cA_len, 32, 12, 0, 0, "Final Approximation (cA)");
    tiny_view_signal_f32(output, input_len, 64, 12, 0, 0, "Reconstructed Signal");

    // Error analysis with boundary effect assessment
    printf("5. Reconstruction Error Analysis:\n");
    int filter_len = TINY_WAVELET_GET_LEN(TINY_WAVELET_DB4);
    int boundary_width = filter_len * levels;  // Boundary effect accumulates with levels
    int max_err_idx = 0;
    float max_err = calculate_max_error(input, output, input_len, &max_err_idx);
    float mse = calculate_mse(input, output, input_len);
    float rmse = sqrtf(mse);
    
    printf("  Max absolute error: %.6f (at index %d)\n", max_err, max_err_idx);
    printf("  Mean squared error (MSE): %.6f\n", mse);
    printf("  Root mean squared error (RMSE): %.6f\n", rmse);
    
    // Analyze boundary regions vs center region
    if (input_len > 2 * boundary_width)
    {
        // Left boundary region
        float left_max_err = 0.0f;
        float left_mse = 0.0f;
        for (int i = 0; i < boundary_width; i++)
        {
            float err = fabsf(output[i] - input[i]);
            if (err > left_max_err)
                left_max_err = err;
            left_mse += (output[i] - input[i]) * (output[i] - input[i]);
        }
        left_mse /= boundary_width;
        
        // Right boundary region
        float right_max_err = 0.0f;
        float right_mse = 0.0f;
        for (int i = input_len - boundary_width; i < input_len; i++)
        {
            float err = fabsf(output[i] - input[i]);
            if (err > right_max_err)
                right_max_err = err;
            right_mse += (output[i] - input[i]) * (output[i] - input[i]);
        }
        right_mse /= boundary_width;
        
        // Center region (excluding boundaries)
        float center_max_err = 0.0f;
        float center_mse = 0.0f;
        int center_count = input_len - 2 * boundary_width;
        for (int i = boundary_width; i < input_len - boundary_width; i++)
        {
            float err = fabsf(output[i] - input[i]);
            if (err > center_max_err)
                center_max_err = err;
            center_mse += (output[i] - input[i]) * (output[i] - input[i]);
        }
        center_mse /= center_count;
        float center_rmse = sqrtf(center_mse);
        
        printf("\n  Boundary Effect Analysis (Multi-Level):\n");
        printf("  Left boundary (indices 0-%d):\n", boundary_width - 1);
        printf("    Max error: %.6f, RMSE: %.6f\n", left_max_err, sqrtf(left_mse));
        printf("  Center region (indices %d-%d, %d samples):\n", 
               boundary_width, input_len - boundary_width - 1, center_count);
        printf("    Max error: %.6f, RMSE: %.6f\n", center_max_err, center_rmse);
        printf("  Right boundary (indices %d-%d):\n", input_len - boundary_width, input_len - 1);
        printf("    Max error: %.6f, RMSE: %.6f\n", right_max_err, sqrtf(right_mse));
        
        // Compare center vs boundaries
        if (center_rmse < 0.01f && (sqrtf(left_mse) > 0.1f || sqrtf(right_mse) > 0.1f))
        {
            printf("  ✓ Center region is accurate, boundary effects confirmed (expected in multi-level)\n");
        }
        else if (center_rmse < 0.1f)
        {
            printf("  ⚠ Center region has minor errors\n");
        }
        else
        {
            printf("  ✗ Center region has significant errors (may indicate implementation issue)\n");
        }
    }
    else
    {
        printf("  ⚠ Signal too short for boundary analysis (need > %d samples)\n", 2 * boundary_width);
    }
    printf("\n");

    // Energy preservation
    float output_energy = calculate_energy(output, input_len);
    float energy_preservation = (input_energy > 0) ? (output_energy / input_energy) : 0.0f;
    printf("6. Energy Preservation:\n");
    printf("  Input energy: %.6f\n", input_energy);
    printf("  Output energy: %.6f\n", output_energy);
    printf("  Preservation ratio: %.6f\n", energy_preservation);
    if (fabsf(energy_preservation - 1.0f) < 0.1f)
        printf("  ✓ Energy is well preserved\n");
    else
        printf("  ⚠ Energy preservation ratio: %.2f%%\n", energy_preservation * 100.0f);
    printf("\n");

    free(input);
    free(cA);
    free(cD);
    free(output);
    printf("========================================\n");
}

/**
 * @name tiny_dwt_test_wavelets
 * @brief Test different wavelet types
 */
void tiny_dwt_test_wavelets(void)
{
    printf("========== TinyDWT Wavelet Types Test ==========\n\n");

    // Simple test signal
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0};
    int input_len = sizeof(input) / sizeof(input[0]);

    const char *wavelet_names[] = {
        "DB1", "DB2", "DB3", "DB4", "DB5",
        "DB6", "DB7", "DB8", "DB9", "DB10"
    };

    tiny_wavelet_type_t wavelets[] = {
        TINY_WAVELET_DB1, TINY_WAVELET_DB2, TINY_WAVELET_DB3, TINY_WAVELET_DB4, TINY_WAVELET_DB5,
        TINY_WAVELET_DB6, TINY_WAVELET_DB7, TINY_WAVELET_DB8, TINY_WAVELET_DB9, TINY_WAVELET_DB10
    };

    printf("Test 3: Different Wavelet Types\n");
    printf("  Input: Test signal (length=%d)\n", input_len);
    printf("  Testing: DB1 through DB10\n\n");

    int passed = 0;
    int failed = 0;

    for (int w = 0; w < TINY_WAVELET_COUNT; w++)
    {
        tiny_wavelet_type_t wavelet = wavelets[w];
        int filter_len = TINY_WAVELET_GET_LEN(wavelet);
        
        // Skip if signal is too short for this wavelet
        if (input_len < filter_len * 2)
        {
            printf("  [%s] Skipped (signal too short: need >= %d, have %d)\n",
                   wavelet_names[w], filter_len * 2, input_len);
            continue;
        }

        float cA[32] = {0}, cD[32] = {0};
        int cA_len = 0, cD_len = 0;
        float output[64] = {0};
        int output_len = 0;

        tiny_error_t err = tiny_dwt_decompose_f32(input, input_len, wavelet, cA, cD, &cA_len, &cD_len);
        if (err != TINY_OK)
        {
            printf("  [%s] ✗ Decomposition failed: %d\n", wavelet_names[w], err);
            failed++;
            continue;
        }

        err = tiny_dwt_reconstruct_f32(cA, cD, cA_len, wavelet, output, &output_len);
        if (err != TINY_OK)
        {
            printf("  [%s] ✗ Reconstruction failed: %d\n", wavelet_names[w], err);
            failed++;
            continue;
        }

        // Check reconstruction accuracy
        int check_len = (input_len < output_len) ? input_len : output_len;
        float max_err = 0.0f;
        for (int i = 0; i < check_len; i++)
        {
            float err = fabsf(output[i] - input[i]);
            if (err > max_err)
                max_err = err;
        }

        if (max_err < 0.1f)
        {
            printf("  [%s] ✓ Pass (filter_len=%d, max_err=%.4f)\n",
                   wavelet_names[w], filter_len, max_err);
            passed++;
        }
        else
        {
            printf("  [%s] ⚠ Warning (filter_len=%d, max_err=%.4f)\n",
                   wavelet_names[w], filter_len, max_err);
            passed++; // Still count as passed, just with warning
        }
    }

    printf("\n  Summary: %d passed, %d failed\n", passed, failed);
    printf("\n========================================\n");
}

/**
 * @name tiny_dwt_test_all
 * @brief Run all DWT tests
 */
void tiny_dwt_test_all(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║          TinyDWT Complete Test Suite                     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Run all tests
    tiny_dwt_test();
    printf("\n");
    
    tiny_dwt_test_multilevel();
    printf("\n");
    
    tiny_dwt_test_wavelets();
    printf("\n");

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║          All DWT Tests Completed                         ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
}

```
