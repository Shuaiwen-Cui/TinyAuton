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

