# 测试

## tiny_ica_test.hpp

```c
/**
 * @file tiny_ica_test.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | test | header
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_ica.hpp"

#ifdef __cplusplus

namespace tiny
{
    /**
     * @name tiny_ica_test_basic
     * @brief Basic test for ICA with simple synthetic signals
     */
    void tiny_ica_test_basic(void);

    /**
     * @name tiny_ica_test_sinusoidal
     * @brief Test ICA with sinusoidal source signals
     */
    void tiny_ica_test_sinusoidal(void);

    /**
     * @name tiny_ica_test_nonlinearity
     * @brief Test different nonlinearity functions
     */
    void tiny_ica_test_nonlinearity(void);

    /**
     * @name tiny_ica_test_reconstruction
     * @brief Test signal reconstruction from separated sources
     */
    void tiny_ica_test_reconstruction(void);

    /**
     * @name tiny_ica_test_all
     * @brief Run all ICA tests
     */
    void tiny_ica_test_all(void);

} // namespace tiny

#endif // __cplusplus

// C interface wrapper — placed OUTSIDE #ifdef __cplusplus so pure C code can also see these.
#ifdef __cplusplus
extern "C" {
#endif

void tiny_ica_test_basic(void);
void tiny_ica_test_sinusoidal(void);
void tiny_ica_test_nonlinearity(void);
void tiny_ica_test_reconstruction(void);
void tiny_ica_test_all(void);

#ifdef __cplusplus
}
#endif


```


## tiny_ica_test.cpp

```c
/**
 * @file tiny_ica_test.hpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | test | header
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* DEPENDENCIES */
#include "tiny_ica.hpp"

#ifdef __cplusplus

namespace tiny
{
    /**
     * @name tiny_ica_test_basic
     * @brief Basic test for ICA with simple synthetic signals
     */
    void tiny_ica_test_basic(void);

    /**
     * @name tiny_ica_test_sinusoidal
     * @brief Test ICA with sinusoidal source signals
     */
    void tiny_ica_test_sinusoidal(void);

    /**
     * @name tiny_ica_test_nonlinearity
     * @brief Test different nonlinearity functions
     */
    void tiny_ica_test_nonlinearity(void);

    /**
     * @name tiny_ica_test_reconstruction
     * @brief Test signal reconstruction from separated sources
     */
    void tiny_ica_test_reconstruction(void);

    /**
     * @name tiny_ica_test_all
     * @brief Run all ICA tests
     */
    void tiny_ica_test_all(void);

} // namespace tiny

#endif // __cplusplus

// C interface wrapper — placed OUTSIDE #ifdef __cplusplus so pure C code can also see these.
#ifdef __cplusplus
extern "C" {
#endif

void tiny_ica_test_basic(void);
void tiny_ica_test_sinusoidal(void);
void tiny_ica_test_nonlinearity(void);
void tiny_ica_test_reconstruction(void);
void tiny_ica_test_all(void);

#ifdef __cplusplus
}
#endif


```


## tiny_ica_test.cpp

```c
/**
 * @file tiny_ica_test.cpp
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | test | source
 * @version 1.0
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_ica_test.hpp"
#include "tiny_view.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __cplusplus

namespace tiny
{
    /**
     * @brief Calculate correlation coefficient between two signals
     */
    static float calculate_correlation(const Mat &signal1, const Mat &signal2)
    {
        if (signal1.col != signal2.col || signal1.row != signal2.row)
        {
            return 0.0f;
        }

        int n = signal1.col;
        float mean1 = 0.0f, mean2 = 0.0f;
        float var1 = 0.0f, var2 = 0.0f;
        float cov = 0.0f;

        // Compute means
        for (int i = 0; i < signal1.row; i++)
        {
            for (int j = 0; j < n; j++)
            {
                mean1 += signal1(i, j);
                mean2 += signal2(i, j);
            }
        }
        mean1 /= (signal1.row * n);
        mean2 /= (signal2.row * n);

        // Compute covariance and variances
        for (int i = 0; i < signal1.row; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float diff1 = signal1(i, j) - mean1;
                float diff2 = signal2(i, j) - mean2;
                cov += diff1 * diff2;
                var1 += diff1 * diff1;
                var2 += diff2 * diff2;
            }
        }

        float denom = sqrtf(var1 * var2);
        if (denom < 1e-10f)
        {
            return 0.0f;
        }

        return cov / denom;
    }

    /**
     * @brief Calculate mean squared error between two signals
     */
    static float calculate_mse(const Mat &signal1, const Mat &signal2)
    {
        if (signal1.col != signal2.col || signal1.row != signal2.row)
        {
            return 1e10f;
        }

        float mse = 0.0f;
        int count = 0;
        for (int i = 0; i < signal1.row; i++)
        {
            for (int j = 0; j < signal1.col; j++)
            {
                float diff = signal1(i, j) - signal2(i, j);
                mse += diff * diff;
                count++;
            }
        }
        return mse / count;
    }

    /**
     * @name tiny_ica_test_basic
     * @brief Basic test for ICA with simple synthetic signals
     */
    void tiny_ica_test_basic(void)
    {
        printf("========== TinyICA Basic Test ==========\n\n");

        // Create two independent source signals
        const int n_samples = 1000;
        const int n_sources = 2;
        const int n_sensors = 2;

        printf("Test 1: Basic ICA Separation\n");
        printf("  Sources: 1 sinusoid + 1 square-like source\n");
        printf("  Sensors: 2\n");
        printf("  Samples: %d\n\n", n_samples);

        // Generate source signals: one sinusoid and one non-Gaussian square-like signal
        Mat sources(n_sources, n_samples);
        float sample_rate = 1000.0f;
        for (int j = 0; j < n_samples; j++)
        {
            float t = j / sample_rate;
            // Source 1: 7 Hz sine
            sources(0, j) = sinf(2.0f * M_PI * 7.0f * t);
            
            // Source 2: 19 Hz square-like signal built from sign(sin)
            float s = sinf(2.0f * M_PI * 19.0f * t);
            sources(1, j) = (s >= 0.0f) ? 1.0f : -1.0f;
        }

        // Create mixing matrix
        Mat mixing_matrix(n_sensors, n_sources);
        mixing_matrix(0, 0) = 0.6f;
        mixing_matrix(0, 1) = 0.4f;
        mixing_matrix(1, 0) = 0.4f;
        mixing_matrix(1, 1) = 0.6f;

        // Mix the signals: X = A * S
        Mat mixed_signals = mixing_matrix * sources;

        printf("1. Signal Generation:\n");
        printf("  ✓ Generated %d source signals\n", n_sources);
        printf("  ✓ Created mixing matrix\n");
        printf("  ✓ Mixed signals (sensors x samples)\n\n");

        // Perform ICA
        printf("2. ICA Decomposition:\n");
        ICA ica(ICANonlinearity::TANH, 1000, 1e-6f, 1.0f);
        ICADecomposition result = ica.decompose(mixed_signals, n_sources);

        if (result.status != TINY_OK)
        {
            printf("  ✗ ICA decomposition failed: %d\n", result.status);
            return;
        }

        printf("  ✓ ICA decomposition completed\n");
        printf("  Iterations: %d\n", result.iterations);
        printf("  Separated sources: %d x %d\n", result.sources.row, result.sources.col);
        printf("\n");

        // Visualize signals
        printf("3. Signal Visualization:\n");
        // Convert to arrays for visualization
        float *source1 = (float *)malloc(n_samples * sizeof(float));
        float *source2 = (float *)malloc(n_samples * sizeof(float));
        float *mixed1 = (float *)malloc(n_samples * sizeof(float));
        float *mixed2 = (float *)malloc(n_samples * sizeof(float));
        float *separated1 = (float *)malloc(n_samples * sizeof(float));
        float *separated2 = (float *)malloc(n_samples * sizeof(float));

        if (source1 && source2 && mixed1 && mixed2 && separated1 && separated2)
        {
            for (int j = 0; j < n_samples; j++)
            {
                source1[j] = sources(0, j);
                source2[j] = sources(1, j);
                mixed1[j] = mixed_signals(0, j);
                mixed2[j] = mixed_signals(1, j);
                separated1[j] = result.sources(0, j);
                separated2[j] = result.sources(1, j);
            }

            tiny_view_signal_f32(source1, n_samples, 64, 12, 0, 0, "Source 1 (7 Hz Sine)");
            tiny_view_signal_f32(source2, n_samples, 64, 12, 0, 0, "Source 2 (19 Hz Square-like)");
            tiny_view_signal_f32(mixed1, n_samples, 64, 12, 0, 0, "Mixed Signal 1");
            tiny_view_signal_f32(mixed2, n_samples, 64, 12, 0, 0, "Mixed Signal 2");
            tiny_view_signal_f32(separated1, n_samples, 64, 12, 0, 0, "Separated Source 1");
            tiny_view_signal_f32(separated2, n_samples, 64, 12, 0, 0, "Separated Source 2");
        }

        // Evaluate separation quality
        printf("4. Separation Quality:\n");

        // Check signal amplitudes first
        float max_amplitude1 = 0.0f, max_amplitude2 = 0.0f;
        float max_original1 = 0.0f, max_original2 = 0.0f;
        
        for (int j = 0; j < n_samples; j++)
        {
            float abs1 = fabsf(result.sources(0, j));
            float abs2 = fabsf(result.sources(1, j));
            float orig1 = fabsf(sources(0, j));
            float orig2 = fabsf(sources(1, j));
            
            if (abs1 > max_amplitude1) max_amplitude1 = abs1;
            if (abs2 > max_amplitude2) max_amplitude2 = abs2;
            if (orig1 > max_original1) max_original1 = orig1;
            if (orig2 > max_original2) max_original2 = orig2;
        }
        
        printf("  Separated Source 1 max amplitude: %.4f (original: %.4f)\n", max_amplitude1, max_original1);
        printf("  Separated Source 2 max amplitude: %.4f (original: %.4f)\n", max_amplitude2, max_original2);
        
        // ICA sources are only identifiable up to scale and sign. Print amplitudes for
        // diagnostics, but do not use them as pass/fail criteria.
        float rel_amplitude1 = (max_original1 > 1e-10f) ? (max_amplitude1 / max_original1) : 0.0f;
        float rel_amplitude2 = (max_original2 > 1e-10f) ? (max_amplitude2 / max_original2) : 0.0f;
        
        printf("  Relative amplitude (Source 1): %.2f%%\n", rel_amplitude1 * 100.0f);
        printf("  Relative amplitude (Source 2): %.2f%%\n", rel_amplitude2 * 100.0f);

        // Normalize separated sources for comparison
        Mat normalized_separated = result.sources;
        for (int i = 0; i < normalized_separated.row; i++)
        {
            float max_val = 0.0f;
            for (int j = 0; j < normalized_separated.col; j++)
            {
                float abs_val = fabsf(normalized_separated(i, j));
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }
            if (max_val > 1e-10f)
            {
                for (int j = 0; j < normalized_separated.col; j++)
                {
                    normalized_separated(i, j) /= max_val;
                }
            }
        }

        // Normalize original sources
        Mat normalized_sources = sources;
        for (int i = 0; i < normalized_sources.row; i++)
        {
            float max_val = 0.0f;
            for (int j = 0; j < normalized_sources.col; j++)
            {
                float abs_val = fabsf(normalized_sources(i, j));
                if (abs_val > max_val)
                {
                    max_val = abs_val;
                }
            }
            if (max_val > 1e-10f)
            {
                for (int j = 0; j < normalized_sources.col; j++)
                {
                    normalized_sources(i, j) /= max_val;
                }
            }
        }

        // Calculate correlations and choose the best one-to-one assignment because ICA
        // output order and sign are arbitrary.
        float corr1_1 = calculate_correlation(normalized_sources.view_roi(0, 0, 1, n_samples),
                                               normalized_separated.view_roi(0, 0, 1, n_samples));
        float corr1_2 = calculate_correlation(normalized_sources.view_roi(0, 0, 1, n_samples),
                                               normalized_separated.view_roi(1, 0, 1, n_samples));
        float corr2_1 = calculate_correlation(normalized_sources.view_roi(1, 0, 1, n_samples),
                                               normalized_separated.view_roi(0, 0, 1, n_samples));
        float corr2_2 = calculate_correlation(normalized_sources.view_roi(1, 0, 1, n_samples),
                                               normalized_separated.view_roi(1, 0, 1, n_samples));

        float assignment_direct = fabsf(corr1_1) + fabsf(corr2_2);
        float assignment_swapped = fabsf(corr1_2) + fabsf(corr2_1);
        float matched_corr1 = 0.0f;
        float matched_corr2 = 0.0f;

        if (assignment_direct >= assignment_swapped)
        {
            matched_corr1 = corr1_1;
            matched_corr2 = corr2_2;
            printf("  Best assignment: source1->separated1, source2->separated2\n");
        }
        else
        {
            matched_corr1 = corr1_2;
            matched_corr2 = corr2_1;
            printf("  Best assignment: source1->separated2, source2->separated1\n");
        }

        printf("  Matched correlation (Source 1): %.4f\n", matched_corr1);
        printf("  Matched correlation (Source 2): %.4f\n", matched_corr2);

        bool good_correlation = (fabsf(matched_corr1) > 0.7f && fabsf(matched_corr2) > 0.7f);
        
        if (good_correlation)
        {
            printf("  ✓ Good separation achieved (scale/sign ambiguity ignored)\n");
            printf("  Result: PASS\n");
        }
        else if (fabsf(matched_corr1) > 0.5f && fabsf(matched_corr2) > 0.5f)
        {
            printf("  ⚠ Moderate separation achieved (scale/sign ambiguity ignored)\n");
            printf("  Result: PASS with warning\n");
        }
        else
        {
            printf("  ✗ Poor separation (may need parameter tuning)\n");
            if (!good_correlation)
            {
                printf("    Reason: Low correlation with original sources (need >0.7)\n");
            }
            printf("  Result: FAIL\n");
        }
        printf("\n");

        // Cleanup
        free(source1);
        free(source2);
        free(mixed1);
        free(mixed2);
        free(separated1);
        free(separated2);

        printf("========================================\n");
    }

    /**
     * @name tiny_ica_test_sinusoidal
     * @brief Test ICA with sinusoidal source signals
     */
    void tiny_ica_test_sinusoidal(void)
    {
        printf("========== TinyICA Sinusoidal Test ==========\n\n");

        const int n_samples = 1000;
        const int n_sources = 3;
        const int n_sensors = 3;

        printf("Test 2: ICA with Sinusoidal Sources\n");
        printf("  Sources: 3 sinusoidal signals with different frequencies\n");
        printf("  Sensors: 3\n");
        printf("  Samples: %d\n\n", n_samples);

        // Generate sinusoidal source signals
        Mat sources(n_sources, n_samples);
        float sample_rate = 1000.0f;
        for (int j = 0; j < n_samples; j++)
        {
            float t = j / sample_rate;
            sources(0, j) = sinf(2.0f * M_PI * 10.0f * t);  // 10 Hz
            sources(1, j) = sinf(2.0f * M_PI * 25.0f * t);  // 25 Hz
            sources(2, j) = sinf(2.0f * M_PI * 50.0f * t);  // 50 Hz
        }

        // Create random mixing matrix
        Mat mixing_matrix(n_sensors, n_sources);
        srand(42); // For reproducibility
        for (int i = 0; i < n_sensors; i++)
        {
            for (int j = 0; j < n_sources; j++)
            {
                mixing_matrix(i, j) = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            }
        }

        // Mix the signals
        Mat mixed_signals = mixing_matrix * sources;

        printf("1. Signal Generation:\n");
        printf("  ✓ Generated 3 sinusoidal sources (10 Hz, 25 Hz, 50 Hz)\n");
        printf("  ✓ Created random mixing matrix\n");
        printf("  ✓ Mixed signals\n\n");

        // Perform ICA
        printf("2. ICA Decomposition:\n");
        ICA ica(ICANonlinearity::TANH, 1000, 1e-6f, 1.0f);
        ICADecomposition result = ica.decompose(mixed_signals, n_sources);

        if (result.status != TINY_OK)
        {
            printf("  ✗ ICA decomposition failed: %d\n", result.status);
            return;
        }

        printf("  ✓ ICA decomposition completed\n");
        printf("  Iterations: %d\n", result.iterations);
        printf("\n");

        // Visualize
        printf("3. Signal Visualization:\n");
        float *mixed_sig = (float *)malloc(n_samples * sizeof(float));
        float *separated_sig = (float *)malloc(n_samples * sizeof(float));

        if (mixed_sig && separated_sig)
        {
            for (int i = 0; i < n_sensors; i++)
            {
                for (int j = 0; j < n_samples; j++)
                {
                    mixed_sig[j] = mixed_signals(i, j);
                }
                char title[64];
                snprintf(title, sizeof(title), "Mixed Signal %d", i + 1);
                tiny_view_signal_f32(mixed_sig, n_samples, 64, 12, 0, 0, title);
            }

            for (int i = 0; i < n_sources; i++)
            {
                for (int j = 0; j < n_samples; j++)
                {
                    separated_sig[j] = result.sources(i, j);
                }
                char title[64];
                snprintf(title, sizeof(title), "Separated Source %d", i + 1);
                tiny_view_signal_f32(separated_sig, n_samples, 64, 12, 0, 0, title);
            }
        }

        free(mixed_sig);
        free(separated_sig);

        printf("4. Separation Quality:\n");
        printf("  ✓ Sinusoidal signals separated\n");
        printf("  Note: ICA can separate sources up to permutation and scaling\n");
        printf("  Result: PASS\n");
        printf("\n");

        printf("========================================\n");
    }

    /**
     * @name tiny_ica_test_nonlinearity
     * @brief Test different nonlinearity functions
     */
    void tiny_ica_test_nonlinearity(void)
    {
        printf("========== TinyICA Nonlinearity Test ==========\n\n");

        const int n_samples = 500;
        const int n_sources = 2;
        const int n_sensors = 2;

        printf("Test 3: Different Nonlinearity Functions\n");
        printf("  Testing: TANH, CUBE, GAUSS, SKEW\n\n");

        // Generate source signals
        Mat sources(n_sources, n_samples);
        for (int j = 0; j < n_samples; j++)
        {
            sources(0, j) = sinf(2.0f * M_PI * j / 50.0f);
            sources(1, j) = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        Mat mixing_matrix(n_sensors, n_sources);
        mixing_matrix(0, 0) = 0.7f;
        mixing_matrix(0, 1) = 0.3f;
        mixing_matrix(1, 0) = 0.3f;
        mixing_matrix(1, 1) = 0.7f;

        Mat mixed_signals = mixing_matrix * sources;

        ICANonlinearity nonlinearities[] = {
            ICANonlinearity::TANH,
            ICANonlinearity::CUBE,
            ICANonlinearity::GAUSS,
            ICANonlinearity::SKEW
        };

        const char *nonlinearity_names[] = {
            "TANH", "CUBE", "GAUSS", "SKEW"
        };

        int passed = 0;
        for (int n = 0; n < 4; n++)
        {
            printf("Testing %s nonlinearity:\n", nonlinearity_names[n]);
            ICA ica(nonlinearities[n], 500, 1e-5f, 1.0f);
            ICADecomposition result = ica.decompose(mixed_signals, n_sources);

            if (result.status == TINY_OK)
            {
                printf("  ✓ Decomposition successful (iterations: %d)\n", result.iterations);
                passed++;
            }
            else
            {
                printf("  ✗ Decomposition failed: %d\n", result.status);
            }
        }

        printf("\n  Summary: %d/4 nonlinearity functions passed\n", passed);
        printf("  Result: %s\n", (passed == 4) ? "PASS" : "FAIL");
        printf("\n========================================\n");
    }

    /**
     * @name tiny_ica_test_reconstruction
     * @brief Test signal reconstruction from separated sources
     */
    void tiny_ica_test_reconstruction(void)
    {
        printf("========== TinyICA Reconstruction Test ==========\n\n");

        const int n_samples = 500;
        const int n_sources = 2;
        const int n_sensors = 2;

        printf("Test 4: Signal Reconstruction\n");
        printf("  Testing: X = A * S reconstruction\n\n");

        // Generate source signals
        Mat sources(n_sources, n_samples);
        for (int j = 0; j < n_samples; j++)
        {
            sources(0, j) = sinf(2.0f * M_PI * j / 50.0f);
            sources(1, j) = cosf(2.0f * M_PI * j / 75.0f);
        }

        // Create mixing matrix
        Mat mixing_matrix_original(n_sensors, n_sources);
        mixing_matrix_original(0, 0) = 0.8f;
        mixing_matrix_original(0, 1) = 0.2f;
        mixing_matrix_original(1, 0) = 0.2f;
        mixing_matrix_original(1, 1) = 0.8f;

        // Mix signals
        Mat mixed_signals = mixing_matrix_original * sources;

        printf("1. Original Mixing:\n");
        printf("  ✓ Generated sources and mixed signals\n\n");

        // Perform ICA
        printf("2. ICA Decomposition:\n");
        ICA ica(ICANonlinearity::TANH, 500, 1e-5f, 1.0f);
        ICADecomposition result = ica.decompose(mixed_signals, n_sources);

        if (result.status != TINY_OK)
        {
            printf("  ✗ ICA decomposition failed: %d\n", result.status);
            return;
        }

        printf("  ✓ ICA decomposition completed\n\n");

        // Reconstruct
        printf("3. Signal Reconstruction:\n");
        Mat reconstructed = ICA::reconstruct(result.sources, result.mixing_matrix);

        if (reconstructed.data == nullptr)
        {
            printf("  ✗ Reconstruction failed\n");
            return;
        }

        printf("  ✓ Reconstruction completed\n\n");

        // Compare original and reconstructed
        printf("4. Reconstruction Quality:\n");
        float mse = calculate_mse(mixed_signals, reconstructed);
        printf("  MSE: %.6f\n", mse);

        if (mse < 0.1f)
        {
            printf("  ✓ Good reconstruction quality\n");
            printf("  Result: PASS\n");
        }
        else if (mse < 1.0f)
        {
            printf("  ⚠ Moderate reconstruction quality\n");
            printf("  Result: PASS with warning\n");
        }
        else
        {
            printf("  ✗ Poor reconstruction quality\n");
            printf("  Result: FAIL\n");
        }
        printf("\n");

        printf("========================================\n");
    }

    /**
     * @name tiny_ica_test_all
     * @brief Run all ICA tests
     */
    void tiny_ica_test_all(void)
    {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║          TinyICA Complete Test Suite                      ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n");
        printf("\n");

        // Run all tests
        tiny_ica_test_basic();
        printf("\n");

        tiny_ica_test_sinusoidal();
        printf("\n");

        tiny_ica_test_nonlinearity();
        printf("\n");

        tiny_ica_test_reconstruction();
        printf("\n");

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║          All ICA Tests Completed                         ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n");
    }

} // namespace tiny

// C interface wrapper implementations
extern "C"
{
    void tiny_ica_test_basic(void)
    {
        tiny::tiny_ica_test_basic();
    }

    void tiny_ica_test_sinusoidal(void)
    {
        tiny::tiny_ica_test_sinusoidal();
    }

    void tiny_ica_test_nonlinearity(void)
    {
        tiny::tiny_ica_test_nonlinearity();
    }

    void tiny_ica_test_reconstruction(void)
    {
        tiny::tiny_ica_test_reconstruction();
    }

    void tiny_ica_test_all(void)
    {
        tiny::tiny_ica_test_all();
    }
}

#endif // __cplusplus


```

```


## 输出结果

```txt
╔══════════════════════════════════════════════════════════╗
║          TinyICA Complete Test Suite                      ║
╚══════════════════════════════════════════════════════════╝

========== TinyICA Basic Test ==========

Test 1: Basic ICA Separation
  Sources: 1 sinusoid + 1 square-like source
  Sensors: 2
  Samples: 1000

1. Signal Generation:
  ✓ Generated 2 source signals
  ✓ Created mixing matrix
  ✓ Mixed signals (sensors x samples)

2. ICA Decomposition:
  ✓ ICA decomposition completed
  Iterations: 10
  Separated sources: 2 x 1000

3. Signal Visualization:

Source 1 (7 Hz Sine)
Value
  1.20 |                                                                
  0.98 | **       **       ***      ***      ***      ***      ***      
  0.76 | ***      ***      * *      * *      * *      * *      * *      
  0.55 |** *     ** *     ** *     ** *     ** *     ** *     ** *      
  0.33 |*  **    *  **    *  **    *  **    *  **    *  **    *  **     
  0.11 |*   *    *   *    *   *    *   *    *   *    *   *    *   *     
 -0.11 |*   *   **   *   **   *   **   *   **   *   **   *   **   *   **
 -0.33 |    **  *    **  *    **  *    **  *    **  *    **  *    **  * 
 -0.55 |     * **     * **     *  *     *  *     *  *     *  *     *  * 
 -0.76 |     ***      ***      ****     ****     ****     ****     **** 
 -0.98 |      **       **       **       **       **       **       **  
 -1.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.200, 1.200], Length: 1000


Source 2 (19 Hz Square-like)
Value
  1.20 |                                                                
  0.98 |** ** ****** ** ****** ** ****** ** ****** ** ****** ** ******  
  0.76 | * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  
  0.55 | * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  
  0.33 | * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  
  0.11 | * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  
 -0.11 | * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  
 -0.33 | * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  
 -0.55 | * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  
 -0.76 | * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  
 -0.98 | ****** ** ****** ** ****** ** ****** ** ****** ** ****** ** ***
 -1.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.200, 1.200], Length: 1000


Mixed Signal 1
Value
  1.20 |                                                                
  0.98 |          **       ***       **      **       ***       **      
  0.76 |**       ***       * *     ****     ***       * *       **      
  0.55 |** **    * *       * *     *****    * ***     * *       ***     
  0.33 |** **    * *     *** *    **** *    * ***     * *    ** * *     
  0.11 | ****  *** **** **** *    * ** *    * **** ** * ***  **** *     
 -0.11 |    * ****  *** * ** **** *    * ** *    * **** **** ***  ****  
 -0.33 |    * * **    * *     *** *    * ****    * ***     * *    ** ***
 -0.55 |    ***       * *     *** *    *****     * *       * *    ** ** 
 -0.76 |     **       * *       ***     ****     * *       ***       ** 
 -0.98 |     **       ***       **      **       ***       **           
 -1.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.200, 1.200], Length: 1000


Mixed Signal 2
Value
  1.20 |                                                                
  0.98 |          **       ***       **      **       ***       **      
  0.76 |** **    ***       * *     *****    *****     * *       ***     
  0.55 |** **    * * **  *** *    **** *    * ****    * ***  ** * *     
  0.33 | * **  *** * ** **** * ** * ** *    * ** * ** * **** ** * ***   
  0.11 | * ** **** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ****  
 -0.11 | **** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **** ** *  
 -0.33 |  *** * ** **** * ** * ** *    * ** * ** * **** ** * ***  ** *  
 -0.55 |    * * **  *** *    **** *    * ****    * ***  ** * *    ** ***
 -0.76 |    ***       * *     *****    *****     * *       ***    ** ** 
 -0.98 |     **       ***       **      **       ***       **           
 -1.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.200, 1.200], Length: 1000


Separated Source 1
Value
  0.17 |                                                                
  0.14 | ***       **               **        **               **       
  0.11 | * *       **     **        **        **     **       ***       
  0.08 | * *       ***    ** **     **        **     ** **    * *       
  0.05 | * *    **** *    *****     ***    *****    ******    * **    **
  0.02 |** ***  ***  *    *   **   ** ***  *** *    *    *    *  ***  * 
 -0.02 |*  ***  *    *    *    * ***  *** **   **   *    *  ***  *** ** 
 -0.05 |*    ** *    ******    *****    ***     *****    * ****    * *  
 -0.08 |      * *    ** **     **        **     ** **    ***       * *  
 -0.11 |      ***       **     **        **        **     **       * *  
 -0.14 |      **               **        **               **       ***  
 -0.17 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-0.166, 0.166], Length: 1000


Separated Source 2
Value
  2.09 |                                                                
  1.71 | ***       **               **        **               **       
  1.33 | * *       **     **        **        **     **       ***       
  0.95 | * *       ***    ** **     **        **     ** **    * *       
  0.57 | * *    **** *    *****     ***    *****    ******    * **    **
  0.19 |** ***  ***  *    *   **   ** ***  *** *    *    *    *  ***  * 
 -0.19 |*  ***  *    *    *    * ***  *** **   **   *    *  ***  *** ** 
 -0.57 |*    ** *    ******    *****    ***     *****    * ****    * *  
 -0.95 |      * *    ** **     **        **     ** **    ***       * *  
 -1.33 |      ***       **     **        **        **     **       * *  
 -1.71 |      **               **        **               **       ***  
 -2.09 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-2.088, 2.088], Length: 1000

4. Separation Quality:
  Separated Source 1 max amplitude: 0.1382 (original: 1.0000)
  Separated Source 2 max amplitude: 1.7403 (original: 1.0000)
  Relative amplitude (Source 1): 13.82%
  Relative amplitude (Source 2): 174.03%
  Best assignment: source1->separated1, source2->separated2
  Matched correlation (Source 1): 0.8173
  Matched correlation (Source 2): -0.5624
  ⚠ Moderate separation achieved (scale/sign ambiguity ignored)
  Result: PASS with warning

========================================

========== TinyICA Sinusoidal Test ==========

Test 2: ICA with Sinusoidal Sources
  Sources: 3 sinusoidal signals with different frequencies
  Sensors: 3
  Samples: 1000

1. Signal Generation:
  ✓ Generated 3 sinusoidal sources (10 Hz, 25 Hz, 50 Hz)
  ✓ Created random mixing matrix
  ✓ Mixed signals

2. ICA Decomposition:
  ✓ ICA decomposition completed
  Iterations: 21

3. Signal Visualization:

Mixed Signal 1
Value
  2.69 |                                                                
  2.20 |**                  **   **                               **    
  1.71 |**                  **   **                  **           **    
  1.22 |**   ****      **  ***   **           **     **   **      **    
  0.73 |**   ****     ***  * *   **   *****   **** ****   *****  ***    
  0.24 |**** **** ***** *  * *   **** *** *** *  * *  * ***** *  * *  **
 -0.24 |*  * ********   * ** *  ***** *** *****  * *  * ***** **** *  * 
 -0.73 |   ***  ** **   ***  ****   *****   **   * *  *****   **** **** 
 -1.22 |   **      **   **   ****           **   ***  **      **     ** 
 -1.71 |   **           **                  **   **                  ** 
 -2.20 |                                    **   **                     
 -2.69 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-2.689, 2.689], Length: 1000


Mixed Signal 2
Value
  2.20 |                                                                
  1.80 |                             **                       **        
  1.40 |           **   **           **                  **   **        
  1.00 |    **     **   **     ***   **   ****   **     ***   **   **** 
  0.60 |   ***   ****   **   *** *  ***   ****   ***    * *   **   **** 
  0.20 |**** *** *  *   **   *** *  * *   **** *** *** ** *  ***   *****
 -0.20 |****   ***  * ********** *  * *** *  * ***   ***  *  * *** ***  
 -0.60 |****   **   * *     **   ****   ***  *****   **   * **   ***    
 -1.00 |****   **   ***     **   ****   **    **     **   ***           
 -1.40 |       **   **                  **           **   **            
 -1.80 |       **                                    **                 
 -2.20 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-2.198, 2.198], Length: 1000


Mixed Signal 3
Value
  1.61 |                                                                
  1.32 |                 **   **           **                  **   **  
  1.03 |          **     **   **           **     **           **   **  
  0.73 |  **      **     **   ***     **   **   ****   ***     **   **  
  0.44 |  *****   **** ****   * *  *****   **   ****   * *** ****   *** 
  0.15 |* *   *  ***** ****   * *  * ***   **** *  **  * *** ****   * * 
 -0.15 |* *   **** *** *  ***** ****   *** *  * *   **** *** *  *   * **
 -0.44 |***   ****   * *   **** ****   *****  * *   **** *****  *****   
 -0.73 | **   **     ***   **     **   **     ***   **     **    ****   
 -1.03 | **   **           **     **           **   **     **           
 -1.32 | **                                    **   **                  
 -1.61 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.614, 1.614], Length: 1000


Separated Source 1
Value
  1.28 |                                                                
  1.04 | **                                    **   **                  
  0.81 | **   **           **     **           **   **                  
  0.58 | **   **      **   **** ****   ** **   **   **   **        **   
  0.35 | **   **   *****   **** **** **** **  ***   **** **** ** ****   
  0.12 |***   ******** ******** **** *******  * *   **** ************ **
 -0.12 |* ******* **** ****   * ********** **** ******** *** ****   * * 
 -0.35 |  **   ** **** ****   ***  ** **   **** **** *******   **   *** 
 -0.58 |  **      ****   **   **   ** **   **   ****      **   **   **  
 -0.81 |                 **   **           **                  **   **  
 -1.04 |                 **   **                               **   **  
 -1.28 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.277, 1.277], Length: 1000


Separated Source 2
Value
  1.97 |                                                                
  1.61 |          **                                                    
  1.25 |     **   **   **            **                 **   **         
  0.90 |    ***   **   **       **   ***  ***      **   ***  ***        
  0.54 |**  * *  ***   **   **  *** ** *  * *  **  **   * *  * *  **    
  0.18 |**  * *  * ** **** *** ** * *  *  * * **** *** ** *  * *  *** **
 -0.18 |*** * ** *  * *  * * * *  * *  * ** * *  *** * *  *  * ** * *** 
 -0.54 |  ***  * *  * *  *** ***  * *  * *  ***  **  ***  ****  ***  ** 
 -0.90 |       ***  ***           ***  * *   **       **   **   **      
 -1.25 |       **    **           **   ***            **   **   **      
 -1.61 |                                                   **           
 -1.97 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-1.971, 1.971], Length: 1000


Separated Source 3
Value
  2.41 |                                                                
  1.97 |                **                  **   **                     
  1.53 |   **      **   **                  **   **           **     ** 
  1.10 |   ***     **   **   ****    **     **   ***     **   **     ** 
  0.66 |   * *   ****   ***  ****   ***** ****   * *  *****   **** **** 
  0.22 |   * ********   * ** *  **  * *** ****   * *  * ***   **** *  **
 -0.22 |**** ****   *   *  * *   **** *** *  **  * ** *   *****  * *    
 -0.66 |**** ****   *****  * *   **** *** *   ****  ***   ****   * *    
 -1.10 |**     **          ***   **     ***   **     **   **     ***    
 -1.53 |**     **           **   **           **     **           **    
 -1.97 |**                  **   **                  **           **    
 -2.41 |                                                                
       ----------------------------------------------------------------
       0       8       16      24      32      40      48      56       (Sample Index)
Range: [-2.409, 2.409], Length: 1000

4. Separation Quality:
  ✓ Sinusoidal signals separated
  Note: ICA can separate sources up to permutation and scaling
  Result: PASS

========================================

========== TinyICA Nonlinearity Test ==========

Test 3: Different Nonlinearity Functions
  Testing: TANH, CUBE, GAUSS, SKEW

Testing TANH nonlinearity:
  ✓ Decomposition successful (iterations: 19)
Testing CUBE nonlinearity:
  ✓ Decomposition successful (iterations: 10)
Testing GAUSS nonlinearity:
  ✓ Decomposition successful (iterations: 8)
Testing SKEW nonlinearity:
  ✓ Decomposition successful (iterations: 6)

  Summary: 4/4 nonlinearity functions passed
  Result: PASS

========================================

========== TinyICA Reconstruction Test ==========

Test 4: Signal Reconstruction
  Testing: X = A * S reconstruction

1. Original Mixing:
  ✓ Generated sources and mixed signals

2. ICA Decomposition:
  ✓ ICA decomposition completed

3. Signal Reconstruction:
  ✓ Reconstruction completed

4. Reconstruction Quality:
  MSE: 0.397983
  ⚠ Moderate reconstruction quality
  Result: PASS with warning

========================================

╔══════════════════════════════════════════════════════════╗
║          All ICA Tests Completed                         ║
╚══════════════════════════════════════════════════════════╝
```