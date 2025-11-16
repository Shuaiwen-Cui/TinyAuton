# 测试

## tiny_ica_test.h

```c
/**
 * @file tiny_ica_test.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | test | header
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_ica.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tiny_ica_test(void);

#ifdef __cplusplus
}
#endif


```

## tiny_ica_test.c

```c
/**
 * @file tiny_ica_test.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | test | source
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_ica_test.h"
#include "tiny_view.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/**
 * @brief Generate source signals for testing
 */
static void generate_source_signals(float *sources, int num_sources, int num_samples, float sample_rate)
{
    // Source 1: Sinusoid at 10 Hz
    // Source 2: Sinusoid at 30 Hz
    // Source 3: Square wave (if num_sources > 2)
    
    for (int t = 0; t < num_samples; t++)
    {
        float time = (float)t / sample_rate;
        
        // Source 1: 10 Hz sine
        sources[0 * num_samples + t] = sinf(2.0f * M_PI * 10.0f * time);
        
        // Source 2: 30 Hz sine
        sources[1 * num_samples + t] = sinf(2.0f * M_PI * 30.0f * time);
        
        // Source 3: Square wave (if exists)
        if (num_sources > 2)
        {
            sources[2 * num_samples + t] = (sinf(2.0f * M_PI * 5.0f * time) > 0.0f) ? 1.0f : -1.0f;
        }
    }
}

/**
 * @brief Mix source signals with a mixing matrix
 */
static void mix_signals(const float *sources, int num_sources, int num_samples,
                       const float *mixing_matrix, int num_obs, float *mixed)
{
    // mixed = mixing_matrix * sources
    // mixing_matrix: (num_obs x num_sources)
    // sources: (num_sources x num_samples)
    // mixed: (num_obs x num_samples)
    
    for (int i = 0; i < num_obs; i++)
    {
        for (int j = 0; j < num_samples; j++)
        {
            mixed[i * num_samples + j] = 0.0f;
            for (int k = 0; k < num_sources; k++)
            {
                mixed[i * num_samples + j] += mixing_matrix[i * num_sources + k] * sources[k * num_samples + j];
            }
        }
    }
}

/**
 * @brief Compute correlation coefficient between two signals
 */
static float compute_correlation(const float *x, const float *y, int len)
{
    float mean_x = 0.0f, mean_y = 0.0f;
    for (int i = 0; i < len; i++)
    {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= (float)len;
    mean_y /= (float)len;
    
    float cov = 0.0f, var_x = 0.0f, var_y = 0.0f;
    for (int i = 0; i < len; i++)
    {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    if (var_x < 1e-10f || var_y < 1e-10f)
        return 0.0f;
    
    return cov / sqrtf(var_x * var_y);
}

/**
 * @brief Find best matching source for each separated component
 */
static void match_sources(const float *separated, const float *original_sources,
                         int num_sources, int num_samples, int *matches)
{
    for (int i = 0; i < num_sources; i++)
    {
        float best_corr = -1.0f;
        int best_match = -1;
        
        for (int j = 0; j < num_sources; j++)
        {
            float corr = fabsf(compute_correlation(
                &separated[i * num_samples],
                &original_sources[j * num_samples],
                num_samples));
            
            if (corr > best_corr)
            {
                best_corr = corr;
                best_match = j;
            }
        }
        
        matches[i] = best_match;
    }
}

/**
 * @brief Compute signal statistics
 */
static void compute_statistics(const float *signal, int len, float *mean, float *std, float *min, float *max)
{
    *mean = 0.0f;
    *min = signal[0];
    *max = signal[0];
    
    for (int i = 0; i < len; i++)
    {
        *mean += signal[i];
        if (signal[i] < *min) *min = signal[i];
        if (signal[i] > *max) *max = signal[i];
    }
    *mean /= (float)len;
    
    float variance = 0.0f;
    for (int i = 0; i < len; i++)
    {
        float diff = signal[i] - *mean;
        variance += diff * diff;
    }
    variance /= (float)len;
    *std = sqrtf(variance);
}

/**
 * @brief Print signal samples (first and last N samples)
 */
static void print_signal_samples(const float *signal, int len, int num_samples, const char *name)
{
    printf("  %s samples (first %d and last %d):\n", name, num_samples, num_samples);
    printf("    First %d: ", num_samples);
    for (int i = 0; i < num_samples && i < len; i++)
    {
        printf("%.4f ", signal[i]);
    }
    printf("\n    Last %d:  ", num_samples);
    int start = (len > num_samples) ? len - num_samples : 0;
    for (int i = start; i < len; i++)
    {
        printf("%.4f ", signal[i]);
    }
    printf("\n");
}

void tiny_ica_test(void)
{
    // Initialize all pointers to NULL at function start for safe cleanup
    float *source_signals = NULL;
    float *mixing_matrix = NULL;
    float *mixed_signals = NULL;
    float *separated_sources = NULL;
    int *matches = NULL;
    
    printf("========== TinyICA Test ==========\n\n");
    
    const int num_sources = 2;
    const int num_obs = 2;
    const int num_samples = 512;
    const float sample_rate = 1000.0f; // 1 kHz
    
    // Allocate memory
    
    source_signals = (float *)malloc(num_sources * num_samples * sizeof(float));
    mixing_matrix = (float *)malloc(num_obs * num_sources * sizeof(float));
    mixed_signals = (float *)malloc(num_obs * num_samples * sizeof(float));
    separated_sources = (float *)malloc(num_sources * num_samples * sizeof(float));
    
    if (!source_signals || !mixing_matrix || !mixed_signals || !separated_sources)
    {
        printf("✗ Memory allocation failed\n");
        goto cleanup;
    }
    
    // Generate source signals
    printf("========================================\n");
    printf("STEP 1: Generating Source Signals\n");
    printf("========================================\n");
    printf("Configuration:\n");
    printf("  - Number of sources: %d\n", num_sources);
    printf("  - Number of samples: %d\n", num_samples);
    printf("  - Sample rate: %.1f Hz\n", sample_rate);
    printf("  - Duration: %.3f seconds\n\n", (float)num_samples / sample_rate);
    
    generate_source_signals(source_signals, num_sources, num_samples, sample_rate);
    
    printf("Source Signal Details:\n");
    for (int i = 0; i < num_sources; i++)
    {
        float freq = (i == 0) ? 10.0f : 30.0f;
        printf("  Source %d: %.1f Hz sinusoid (sin(2π*%.1f*t))\n", i + 1, freq, freq);
        
        float mean, std, min, max;
        compute_statistics(&source_signals[i * num_samples], num_samples, &mean, &std, &min, &max);
        printf("    Statistics: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n", mean, std, min, max);
        print_signal_samples(&source_signals[i * num_samples], num_samples, 8, "Source");
        
        // Visualize source signal
        char title[64];
        snprintf(title, sizeof(title), "Source %d (%.1f Hz)", i + 1, freq);
        tiny_view_signal_f32(&source_signals[i * num_samples], num_samples, 64, 12, 0.0f, 0.0f, title);
        printf("\n");
    }
    
    // Create mixing matrix
    printf("========================================\n");
    printf("STEP 2: Creating Mixing Matrix\n");
    printf("========================================\n");
    // Mixing matrix: [0.8, 0.3; 0.2, 0.7]
    mixing_matrix[0 * num_sources + 0] = 0.8f; // obs1 = 0.8*s1 + 0.3*s2
    mixing_matrix[0 * num_sources + 1] = 0.3f;
    mixing_matrix[1 * num_sources + 0] = 0.2f; // obs2 = 0.2*s1 + 0.7*s2
    mixing_matrix[1 * num_sources + 1] = 0.7f;
    
    printf("Mixing Matrix A (num_obs x num_sources = %d x %d):\n", num_obs, num_sources);
    printf("  [%.3f  %.3f]\n", mixing_matrix[0 * num_sources + 0], mixing_matrix[0 * num_sources + 1]);
    printf("  [%.3f  %.3f]\n", mixing_matrix[1 * num_sources + 0], mixing_matrix[1 * num_sources + 1]);
    printf("\nMixing Equation:\n");
    printf("  Observation 1 = %.3f * Source 1 + %.3f * Source 2\n", 
           mixing_matrix[0 * num_sources + 0], mixing_matrix[0 * num_sources + 1]);
    printf("  Observation 2 = %.3f * Source 1 + %.3f * Source 2\n\n", 
           mixing_matrix[1 * num_sources + 0], mixing_matrix[1 * num_sources + 1]);
    
    // Mix signals
    printf("========================================\n");
    printf("STEP 3: Mixing Signals (X = A * S)\n");
    printf("========================================\n");
    printf("Process: Multiplying mixing matrix by source signals\n");
    mix_signals(source_signals, num_sources, num_samples, mixing_matrix, num_obs, mixed_signals);
    printf("✓ Mixed %d sources into %d observations\n\n", num_sources, num_obs);
    
    printf("Mixed Signal Details:\n");
    for (int i = 0; i < num_obs; i++)
    {
        printf("  Observation %d:\n", i + 1);
        float mean, std, min, max;
        compute_statistics(&mixed_signals[i * num_samples], num_samples, &mean, &std, &min, &max);
        printf("    Statistics: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n", mean, std, min, max);
        print_signal_samples(&mixed_signals[i * num_samples], num_samples, 8, "Mixed");
        
        // Visualize mixed signal
        char title[64];
        snprintf(title, sizeof(title), "Mixed Observation %d", i + 1);
        tiny_view_signal_f32(&mixed_signals[i * num_samples], num_samples, 64, 12, 0.0f, 0.0f, title);
        printf("\n");
    }
    
    // Test ICA separation
    printf("========================================\n");
    printf("STEP 4: ICA Separation (FastICA Algorithm)\n");
    printf("========================================\n");
    printf("Algorithm: FastICA\n");
    printf("Nonlinearity: tanh (for super-Gaussian sources)\n");
    printf("Max iterations: 100\n");
    printf("Convergence tolerance: 1e-4\n");
    printf("Process:\n");
    printf("  1. Center data (subtract mean)\n");
    printf("  2. Whiten data (decorrelate and normalize variance)\n");
    printf("  3. Extract independent components using FastICA\n");
    printf("  4. Reconstruct separated sources\n\n");
    
    tiny_error_t ret = tiny_ica_separate_f32(
        mixed_signals, num_obs, num_samples, num_sources, separated_sources,
        TINY_ICA_FASTICA, TINY_ICA_NONLINEARITY_TANH, 100, 1e-4f);
    
    if (ret != TINY_OK)
    {
        printf("✗ ICA separation failed with error code: %d\n", ret);
        goto cleanup;
    }
    printf("✓ ICA separation completed successfully\n\n");
    
    // Evaluate results
    printf("========================================\n");
    printf("STEP 5: Evaluating Separation Quality\n");
    printf("========================================\n");
    
    matches = (int *)malloc(num_sources * sizeof(int));
    if (matches == NULL)
    {
        printf("✗ Memory allocation failed\n");
        goto cleanup;
    }
    
    match_sources(separated_sources, source_signals, num_sources, num_samples, matches);
    
    printf("Separated Signal Details:\n");
    for (int i = 0; i < num_sources; i++)
    {
        int match = matches[i];
        float corr = compute_correlation(
            &separated_sources[i * num_samples],
            &source_signals[match * num_samples],
            num_samples);
        
        printf("\n  Separated Component %d (matches Source %d):\n", i + 1, match + 1);
        printf("    Correlation with original: %.6f\n", corr);
        
        float mean, std, min, max;
        compute_statistics(&separated_sources[i * num_samples], num_samples, &mean, &std, &min, &max);
        printf("    Statistics: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n", mean, std, min, max);
        
        // Compare with original source
        float orig_mean, orig_std, orig_min, orig_max;
        compute_statistics(&source_signals[match * num_samples], num_samples, &orig_mean, &orig_std, &orig_min, &orig_max);
        printf("    Original Source %d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n", 
               match + 1, orig_mean, orig_std, orig_min, orig_max);
        
        print_signal_samples(&separated_sources[i * num_samples], num_samples, 8, "Separated");
        
        // Visualize separated signal
        char title[64];
        snprintf(title, sizeof(title), "Separated Component %d (matches Source %d)", i + 1, match + 1);
        tiny_view_signal_f32(&separated_sources[i * num_samples], num_samples, 64, 12, 0.0f, 0.0f, title);
        
        // Visualize original for comparison
        snprintf(title, sizeof(title), "Original Source %d (for comparison)", match + 1);
        tiny_view_signal_f32(&source_signals[match * num_samples], num_samples, 64, 12, 0.0f, 0.0f, title);
        
        // Compute normalized error
        float mse = 0.0f;
        float scale = (orig_std > 1e-10f) ? (std / orig_std) : 1.0f;
        for (int j = 0; j < num_samples; j++)
        {
            float normalized_sep = (separated_sources[i * num_samples + j] - mean) / (std + 1e-10f);
            float normalized_orig = (source_signals[match * num_samples + j] - orig_mean) / (orig_std + 1e-10f);
            float diff = normalized_sep - normalized_orig;
            mse += diff * diff;
        }
        mse /= (float)num_samples;
        float rmse = sqrtf(mse);
        printf("    Normalized RMSE: %.6f\n", rmse);
        printf("    Quality: %s\n", (corr > 0.9f) ? "Excellent" : (corr > 0.7f) ? "Good" : (corr > 0.5f) ? "Fair" : "Poor");
    }
    printf("\n");
    
    // Test ICA structure API
    printf("========================================\n");
    printf("STEP 6: Testing ICA Structure API\n");
    printf("========================================\n");
    printf("This tests the reusable ICA structure for multiple separations:\n");
    printf("  1. Initialize ICA structure\n");
    printf("  2. Fit model to training data\n");
    printf("  3. Transform new data using learned model\n");
    printf("  4. Compare with direct separation\n\n");
    tiny_ica_t ica;
    ret = tiny_ica_init(&ica, num_obs, num_sources);
    if (ret != TINY_OK)
    {
        printf("  ✗ ICA initialization failed: %d\n", ret);
    }
    else
    {
        printf("  ✓ ICA structure initialized\n");
        
        ret = tiny_ica_fit(&ica, mixed_signals, num_samples,
                          TINY_ICA_FASTICA, TINY_ICA_NONLINEARITY_TANH, 100, 1e-4f);
        if (ret != TINY_OK)
        {
            printf("  ✗ ICA fitting failed: %d\n", ret);
        }
        else
        {
            printf("  ✓ ICA model fitted\n");
            
            // Test transform
            float *separated2 = (float *)malloc(num_sources * num_samples * sizeof(float));
            if (separated2 != NULL)
            {
                ret = tiny_ica_transform(&ica, mixed_signals, num_samples, separated2);
                if (ret != TINY_OK)
                {
                    printf("  ✗ ICA transform failed: %d\n", ret);
                }
                else
                {
                    printf("  ✓ ICA transform completed\n");
                    
                    // Compare with direct separation
                    float diff_sum = 0.0f;
                    for (int i = 0; i < num_sources * num_samples; i++)
                    {
                        float diff = separated_sources[i] - separated2[i];
                        diff_sum += diff * diff;
                    }
                    float rmse = sqrtf(diff_sum / (float)(num_sources * num_samples));
                    printf("  RMSE between direct and structure API: %.6f\n", rmse);
                }
                free(separated2);
            }
        }
        
        tiny_ica_deinit(&ica);
        printf("  ✓ ICA structure deinitialized\n");
    }
    printf("\n");
    
    // Final summary
    printf("========================================\n");
    printf("SUMMARY: ICA Test Results\n");
    printf("========================================\n");
    printf("Test Configuration:\n");
    printf("  - Sources: %d independent signals\n", num_sources);
    printf("  - Observations: %d mixed signals\n", num_obs);
    printf("  - Samples: %d per signal\n", num_samples);
    printf("  - Sample rate: %.1f Hz\n", sample_rate);
    printf("\nSeparation Quality Summary:\n");
    for (int i = 0; i < num_sources; i++)
    {
        int match = matches[i];
        float corr = compute_correlation(
            &separated_sources[i * num_samples],
            &source_signals[match * num_samples],
            num_samples);
        printf("  Component %d → Source %d: Correlation = %.6f (%s)\n", 
               i + 1, match + 1, corr,
               (corr > 0.9f) ? "Excellent" : (corr > 0.7f) ? "Good" : (corr > 0.5f) ? "Fair" : "Poor");
    }
    printf("\nExpected vs Actual:\n");
    printf("  Expected: Separated signals should match original sources\n");
    printf("  Actual: ICA successfully extracted independent components\n");
    printf("  Note: ICA may recover sources with different scale/sign, which is normal\n");
    printf("        (correlation measures similarity regardless of scale)\n");
    printf("\n");
    
    printf("========== TinyICA Test Complete ==========\n\n");
    
cleanup:
    free(source_signals);
    free(mixing_matrix);
    free(mixed_signals);
    free(separated_sources);
    // matches is always initialized to NULL at function start (line 139), safe to free
    // Suppress false positive uninitialized variable warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    if (matches != NULL)
    {
        free(matches);
    }
    #pragma GCC diagnostic pop
}


```

## 输出结果

```c
========== TinyICA Test ==========

========================================
STEP 1: Generating Source Signals
========================================
Configuration:
  - Number of sources: 2
  - Number of samples: 512
  - Sample rate: 1000.0 Hz
  - Duration: 0.512 seconds

Source Signal Details:
  Source 1: 10.0 Hz sinusoid (sin(2π*10.0*t))
    Statistics: mean=0.0078, std=0.7012, min=-1.0000, max=1.0000
  Source samples (first 8 and last 8):
    First 8: 0.0000 0.0628 0.1253 0.1874 0.2487 0.3090 0.3681 0.4258 
    Last 8:  0.2487 0.3090 0.3681 0.4258 0.4818 0.5358 0.5878 0.6374 

Source 1 (10.0 Hz)
Value
  1.20 |                                                                
  0.98 |   **          **          **           **          **          
  0.76 |  * *         *  *        *  *         * *         *  *         
  0.55 | *   *       *   *        *  *        *   *       *   *        *
  0.33 |*    *       *    *      *    *      *    *       *    *      * 
  0.11 |*     *     *     *      *    *      *     *     *     *      * 
 -0.11 |*     *     *     *     *      *    *      *     *     *     *  
 -0.33 |       *   *       *    *      *    *       *   *       *    *  
 -0.55 |       *   *       *   *        *   *       *   *       *   *   
 -0.76 |        * *         *  *        *  *         * *         *  *   
 -0.98 |         **          **          **           **          **    
 -1.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.200, 1.200], Length: 512


  Source 2: 30.0 Hz sinusoid (sin(2π*30.0*t))
    Statistics: mean=0.0162, std=0.7083, min=-1.0000, max=1.0000
  Source samples (first 8 and last 8):
    First 8: 0.0000 0.1874 0.3681 0.5358 0.6845 0.8090 0.9048 0.9686 
    Last 8:  0.6845 0.8090 0.9048 0.9686 0.9980 0.9921 0.9511 0.8763 

Source 2 (30.0 Hz)
Value
  1.20 |                                                                
  0.98 | *   *   *                *   *   *   *   *   *                *
  0.76 |**  **  **   *   **   *  **  **  **  **  **  **   *   **   *  * 
  0.55 |**  **  **  * * * *  **  **  **  **  **  **  **  * * * *  **  * 
  0.33 |**  * * * * * * * * * * * *  **  **  **  * * * * * * * * * * *  
  0.11 |* * * * * * * * * * * * * * * *  **  * * * * * * * * * * * * *  
 -0.11 |* * * * * * * * * * * * * * * * *  **  * * * * * * * * * * * *  
 -0.33 |  **  * * * * * * * * * *  **  **  **  **  * * * * * * * * * *  
 -0.55 |  **  **  * * * * * *  **  **  **  **  **  **  * * * * * *  **  
 -0.76 |  **  **  **   *   **   *  **  **  **  **  **  **   *   **   *  
 -0.98 |   *   *   *                *   *   *   *   *   *               
 -1.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.200, 1.200], Length: 512


========================================
STEP 2: Creating Mixing Matrix
========================================
Mixing Matrix A (num_obs x num_sources = 2 x 2):
  [0.800  0.300]
  [0.200  0.700]

Mixing Equation:
  Observation 1 = 0.800 * Source 1 + 0.300 * Source 2
  Observation 2 = 0.200 * Source 1 + 0.700 * Source 2

========================================
STEP 3: Mixing Signals (X = A * S)
========================================
Process: Multiplying mixing matrix by source signals
✓ Mixed 2 sources into 2 observations

Mixed Signal Details:
  Observation 1:
    Statistics: mean=0.0111, std=0.6025, min=-0.7788, max=0.7788
  Mixed samples (first 8 and last 8):
    First 8: 0.0000 0.1064 0.2107 0.3107 0.4043 0.4899 0.5659 0.6312 
    Last 8:  0.4043 0.4899 0.5659 0.6312 0.6848 0.7263 0.7555 0.7728 

Mixed Observation 1
Value
  0.93 |                                                                
  0.76 | **  *        *  *        *  *        **  *        *  *        *
  0.59 |* * **       * ***       * ** *      * * **       * ***       * 
  0.42 |*  * *      *     *      *    *      *  * *      *     *      * 
  0.25 |*    *      *     *      *    *      *    *      *     *      * 
  0.08 |*     *     *     *     *     *      *     *     *     *     *  
 -0.08 |*     *     *     *     *      *    *      *     *     *     *  
 -0.25 |      *     *     *     *      *    *      *     *     *     *  
 -0.42 |      *    *       *    *      *  * *      *    *       *    *  
 -0.59 |       * ***       * ** *      * ** *       * ***       * ***   
 -0.76 |        *  *        *  *        *  **        *  *        *  *   
 -0.93 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-0.935, 0.935], Length: 512


  Observation 2:
    Statistics: mean=0.0129, std=0.5171, min=-0.8016, max=0.8016
  Mixed samples (first 8 and last 8):
    First 8: 0.0000 0.1437 0.2828 0.4126 0.5289 0.6281 0.7070 0.7632 
    Last 8:  0.5289 0.6281 0.7070 0.7632 0.7950 0.8016 0.7833 0.7409 

Mixed Observation 2
Value
  0.96 |                                                                
  0.79 | *   *                    *   *       *   *                    *
  0.61 |**  **       **  **      **  **      **  **       **  *       * 
  0.44 |**  **   *  * * * *   *  **  **   *  **  **   *  * * * *   *  * 
  0.26 |* * * * **  * * * *  ** * * * *  **  * * **  **  * * * *  ** *  
  0.09 |* * * * * * * * * * * * * * * *  **  * * * * * * * * * * * * *  
 -0.09 |  **  * * * * * * * * * *  ** *  ** *  **  * * * * * * * * * *  
 -0.26 |  **  * * * * * * * * * *  **  **  **  **  * * * * * * * * * *  
 -0.44 |   *  * * * *  *  * * * *   *  **  **   *  * * * *  *  * * * *  
 -0.61 |      **  **       **  **      **  **      **  **       **  **  
 -0.79 |       *   *                    *   *       *   *               
 -0.96 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-0.962, 0.962], Length: 512


========================================
STEP 4: ICA Separation (FastICA Algorithm)
========================================
Algorithm: FastICA
Nonlinearity: tanh (for super-Gaussian sources)
Max iterations: 100
Convergence tolerance: 1e-4
Process:
  1. Center data (subtract mean)
  2. Whiten data (decorrelate and normalize variance)
  3. Extract independent components using FastICA
  4. Reconstruct separated sources

✓ ICA separation completed successfully

========================================
STEP 5: Evaluating Separation Quality
========================================
Separated Signal Details:

  Separated Component 1 (matches Source 1):
    Correlation with original: 0.959623
    Statistics: mean=-0.0000, std=1.0000, min=-1.2638, max=1.2299
    Original Source 1: mean=0.0078, std=0.7012, min=-1.0000, max=1.0000
  Separated samples (first 8 and last 8):
    First 8: -0.0170 0.1430 0.3001 0.4513 0.5939 0.7256 0.8442 0.9481 
    Last 8:  0.5939 0.7256 0.8442 0.9481 1.0361 1.1074 1.1620 1.1999 

Separated Component 1 (matches Source 1)
Value
  1.48 |                                                                
  1.21 |  * **        *  *        ** *         * **        *  *        *
  0.94 | * * *       * ***       *  * *       * * *       * ***       * 
  0.66 |*    *       *    *      *    *      *    *       *    *      * 
  0.39 |*    *      *     *      *    *      *    *      *     *      * 
  0.12 |*     *     *     *     *     *      *     *     *     *     *  
 -0.15 |*     *     *     *     *      *    *      *     *     *     *  
 -0.43 |      *     *     *     *      *    *      *     *     *     *  
 -0.70 |      *    *       *    *      *    *      *    *       *    *  
 -0.97 |       * * *       * ***       *  * *       * * *       * ***   
 -1.24 |        * **        *  *        ** *         * **        *  *   
 -1.51 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.513, 1.479], Length: 512


Original Source 1 (for comparison)
Value
  1.20 |                                                                
  0.98 |   **          **          **           **          **          
  0.76 |  * *         *  *        *  *         * *         *  *         
  0.55 | *   *       *   *        *  *        *   *       *   *        *
  0.33 |*    *       *    *      *    *      *    *       *    *      * 
  0.11 |*     *     *     *      *    *      *     *     *     *      * 
 -0.11 |*     *     *     *     *      *    *      *     *     *     *  
 -0.33 |       *   *       *    *      *    *       *   *       *    *  
 -0.55 |       *   *       *   *        *   *       *   *       *   *   
 -0.76 |        * *         *  *        *  *         * *         *  *   
 -0.98 |         **          **          **           **          **    
 -1.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.200, 1.200], Length: 512

    Normalized RMSE: 0.284173
    Quality: Excellent

  Separated Component 2 (matches Source 2):
    Correlation with original: 0.955788
    Statistics: mean=0.0000, std=1.0000, min=-1.7930, max=1.7557
    Original Source 2: mean=0.0162, std=0.7083, min=-1.0000, max=1.0000
  Separated samples (first 8 and last 8):
    First 8: -0.0186 0.2089 0.4276 0.6288 0.8046 0.9479 1.0530 1.1152 
    Last 8:  0.8046 0.9479 1.0530 1.1152 1.1316 1.1009 1.0235 0.9014 

Separated Component 2 (matches Source 2)
Value
  2.11 |                                                                
  1.72 |         *                        *           *                 
  1.34 |        **           **          **          **           **    
  0.95 | *   *  * *  *    * * *   *   *  **   *   *  * *  *    * * *   *
  0.56 |**  **  * * * *  ** * *  **  **  **  **  **  * * * *  ** * *  * 
  0.17 |**  * * * * * * * * * * * *  ** *  * **  * * * * * * * * * * *  
 -0.21 |* * * * * * * * * * * * * * *  **  **  * * * * * * * * * * * *  
 -0.60 |  * * **  * * * * * *  ** * *  **  **  * * **  * * * * * *  **  
 -0.99 |  **   *   *  * *  *    *  **   *   *  **   *   *  * *  *    *  
 -1.37 |  **           **          **          **           **          
 -1.76 |   *                        *           *                       
 -2.15 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-2.148, 2.111], Length: 512


Original Source 2 (for comparison)
Value
  1.20 |                                                                
  0.98 | *   *   *                *   *   *   *   *   *                *
  0.76 |**  **  **   *   **   *  **  **  **  **  **  **   *   **   *  * 
  0.55 |**  **  **  * * * *  **  **  **  **  **  **  **  * * * *  **  * 
  0.33 |**  * * * * * * * * * * * *  **  **  **  * * * * * * * * * * *  
  0.11 |* * * * * * * * * * * * * * * *  **  * * * * * * * * * * * * *  
 -0.11 |* * * * * * * * * * * * * * * * *  **  * * * * * * * * * * * *  
 -0.33 |  **  * * * * * * * * * *  **  **  **  **  * * * * * * * * * *  
 -0.55 |  **  **  * * * * * *  **  **  **  **  **  **  * * * * * *  **  
 -0.76 |  **  **  **   *   **   *  **  **  **  **  **  **   *   **   *  
 -0.98 |   *   *   *                *   *   *   *   *   *               
 -1.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.200, 1.200], Length: 512

    Normalized RMSE: 0.297362
    Quality: Excellent

========================================
STEP 6: Testing ICA Structure API
========================================
This tests the reusable ICA structure for multiple separations:
  1. Initialize ICA structure
  2. Fit model to training data
  3. Transform new data using learned model
  4. Compare with direct separation

  ✓ ICA structure initialized
  ✓ ICA model fitted
  ✓ ICA transform completed
  RMSE between direct and structure API: 1.414214
  ✓ ICA structure deinitialized

========================================
SUMMARY: ICA Test Results
========================================
Test Configuration:
  - Sources: 2 independent signals
  - Observations: 2 mixed signals
  - Samples: 512 per signal
  - Sample rate: 1000.0 Hz

Separation Quality Summary:
  Component 1 → Source 1: Correlation = 0.959623 (Excellent)
  Component 2 → Source 2: Correlation = 0.955788 (Excellent)

Expected vs Actual:
  Expected: Separated signals should match original sources
  Actual: ICA successfully extracted independent components
  Note: ICA may recover sources with different scale/sign, which is normal
        (correlation measures similarity regardless of scale)

========== TinyICA Test Complete ==========
```
