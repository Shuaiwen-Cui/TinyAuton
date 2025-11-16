# 代码

## tiny_ica.h

```c
/**
 * @file tiny_ica.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | Independent Component Analysis (ICA) | header
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 * @details
 * Independent Component Analysis (ICA) Implementation
 * - Blind source separation: X = A * S, where X is mixed signals, A is mixing matrix, S is sources
 * - FastICA algorithm implementation
 * - Support for multiple sources and observations
 * - Whitening and centering preprocessing
 * - Reuses tiny_math matrix operations
 */

#pragma once

/* DEPENDENCIES */
// tiny_dsp configuration file
#include "tiny_dsp_config.h"

// tiny_math for matrix operations
#include "tiny_mat.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief ICA algorithm types
     */
    typedef enum
    {
        TINY_ICA_FASTICA = 0,  // FastICA algorithm (default)
        TINY_ICA_INFOMAX,      // Infomax algorithm (future)
        TINY_ICA_COUNT
    } tiny_ica_algorithm_t;

    /**
     * @brief Nonlinearity function types for FastICA
     */
    typedef enum
    {
        TINY_ICA_NONLINEARITY_TANH = 0,  // tanh (default, good for super-Gaussian)
        TINY_ICA_NONLINEARITY_EXP,        // exp(-u^2/2) (good for sub-Gaussian)
        TINY_ICA_NONLINEARITY_CUBE,       // u^3 (good for super-Gaussian)
        TINY_ICA_NONLINEARITY_COUNT
    } tiny_ica_nonlinearity_t;

    /**
     * @brief ICA structure for maintaining state
     */
    typedef struct
    {
        float *mixing_matrix;      // Estimated mixing matrix (num_obs x num_sources)
        float *unmixing_matrix;     // Estimated unmixing matrix (num_sources x num_obs)
        float *whitening_matrix;    // Whitening matrix (num_sources x num_obs)
        float *mean;                // Mean of input data (num_obs)
        int num_obs;                // Number of observations (mixed signals)
        int num_sources;            // Number of sources to extract
        int initialized;           // Initialization flag
    } tiny_ica_t;

    /**
     * @name: tiny_ica_separate_f32
     * @brief Perform ICA separation on mixed signals
     * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
     *                      Each row is one observation (mixed signal)
     * @param num_obs Number of observations (mixed signals)
     * @param num_samples Number of samples per signal
     * @param num_sources Number of independent sources to extract
     * @param separated_sources Output separated sources (num_sources x num_samples, row-major)
     *                          Each row is one independent source
     * @param algorithm ICA algorithm to use (default: TINY_ICA_FASTICA)
     * @param nonlinearity Nonlinearity function for FastICA (default: TINY_ICA_NONLINEARITY_TANH)
     * @param max_iter Maximum number of iterations (default: 100)
     * @param tolerance Convergence tolerance (default: 1e-4)
     * @return tiny_error_t
     * @note This function performs complete ICA: preprocessing, separation, and source extraction
     */
    tiny_error_t tiny_ica_separate_f32(const float *mixed_signals,
                                       int num_obs,
                                       int num_samples,
                                       int num_sources,
                                       float *separated_sources,
                                       tiny_ica_algorithm_t algorithm,
                                       tiny_ica_nonlinearity_t nonlinearity,
                                       int max_iter,
                                       float tolerance);

    /**
     * @name: tiny_ica_init
     * @brief Initialize ICA structure for repeated use
     * @param ica Pointer to ICA structure
     * @param num_obs Number of observations (mixed signals)
     * @param num_sources Number of sources to extract
     * @return tiny_error_t
     * @note This allocates memory for matrices. Call tiny_ica_deinit to free.
     */
    tiny_error_t tiny_ica_init(tiny_ica_t *ica, int num_obs, int num_sources);

    /**
     * @name: tiny_ica_fit
     * @brief Fit ICA model to mixed signals (learn unmixing matrix)
     * @param ica Pointer to initialized ICA structure
     * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
     * @param num_samples Number of samples per signal
     * @param algorithm ICA algorithm to use
     * @param nonlinearity Nonlinearity function for FastICA
     * @param max_iter Maximum number of iterations
     * @param tolerance Convergence tolerance
     * @return tiny_error_t
     * @note After fitting, use tiny_ica_transform to separate new signals
     */
    tiny_error_t tiny_ica_fit(tiny_ica_t *ica,
                              const float *mixed_signals,
                              int num_samples,
                              tiny_ica_algorithm_t algorithm,
                              tiny_ica_nonlinearity_t nonlinearity,
                              int max_iter,
                              float tolerance);

    /**
     * @name: tiny_ica_transform
     * @brief Apply learned ICA model to separate signals
     * @param ica Pointer to fitted ICA structure
     * @param mixed_signals Input mixed signals (num_obs x num_samples, row-major)
     * @param num_samples Number of samples per signal
     * @param separated_sources Output separated sources (num_sources x num_samples, row-major)
     * @return tiny_error_t
     * @note Requires ica to be fitted first using tiny_ica_fit
     */
    tiny_error_t tiny_ica_transform(const tiny_ica_t *ica,
                                    const float *mixed_signals,
                                    int num_samples,
                                    float *separated_sources);

    /**
     * @name: tiny_ica_deinit
     * @brief Deinitialize ICA structure and free memory
     * @param ica Pointer to ICA structure
     * @return tiny_error_t
     */
    tiny_error_t tiny_ica_deinit(tiny_ica_t *ica);

#ifdef __cplusplus
}
#endif


```

## tiny_ica.c

```c
/**
 * @file tiny_ica.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief tiny_ica | Independent Component Analysis (ICA) | source
 * @version 1.0
 * @date 2025-11-16
 * @copyright Copyright (c) 2025
 *
 */

/* DEPENDENCIES */
#include "tiny_ica.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Compute mean of each row (observation)
 */
static void compute_row_mean(const float *data, int rows, int cols, float *mean)
{
    for (int i = 0; i < rows; i++)
    {
        mean[i] = 0.0f;
        for (int j = 0; j < cols; j++)
        {
            mean[i] += data[i * cols + j];
        }
        mean[i] /= (float)cols;
    }
}

/**
 * @brief Center data by subtracting row means
 */
static void center_data(const float *data, int rows, int cols, const float *mean, float *centered)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            centered[i * cols + j] = data[i * cols + j] - mean[i];
        }
    }
}

/**
 * @brief Compute covariance matrix: C = (1/N) * X * X^T
 */
static tiny_error_t compute_covariance(const float *data, int rows, int cols, float *cov)
{
    // cov = (1/N) * data * data^T
    // data is (rows x cols), data^T is (cols x rows)
    // cov is (rows x rows)
    
    // First compute data * data^T
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < cols; k++)
            {
                sum += data[i * cols + k] * data[j * cols + k];
            }
            cov[i * rows + j] = sum / (float)cols;
        }
    }
    
    return TINY_OK;
}

/**
 * @brief Simple eigenvalue decomposition for symmetric matrix (for whitening)
 * Uses Jacobi method for small matrices
 */
static tiny_error_t eigendecompose_symmetric(const float *A, int n, float *eigenvalues, float *eigenvectors, float tolerance, int max_iter)
{
    // Initialize eigenvectors as identity
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            eigenvectors[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Copy A to working matrix
    float *B = (float *)malloc(n * n * sizeof(float));
    if (B == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;
    
    memcpy(B, A, n * n * sizeof(float));
    
    // Jacobi iteration
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        float max_val = fabsf(B[1]);
        
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                float val = fabsf(B[i * n + j]);
                if (val > max_val)
                {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        
        // Check convergence
        if (max_val < tolerance)
            break;
        
        // Compute rotation angle
        float a_pq = B[p * n + q];
        float a_pp = B[p * n + p];
        float a_qq = B[q * n + q];
        
        float tau = (a_qq - a_pp) / (2.0f * a_pq);
        float t = (tau >= 0.0f) ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
                                 : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
        float c = 1.0f / sqrtf(1.0f + t * t);
        float s = t * c;
        
        // Apply rotation to B
        for (int i = 0; i < n; i++)
        {
            if (i != p && i != q)
            {
                float b_ip = B[i * n + p];
                float b_iq = B[i * n + q];
                B[i * n + p] = c * b_ip - s * b_iq;
                B[i * n + q] = s * b_ip + c * b_iq;
                B[p * n + i] = B[i * n + p];
                B[q * n + i] = B[i * n + q];
            }
        }
        
        float b_pp = B[p * n + p];
        float b_pq = B[p * n + q];
        float b_qq = B[q * n + q];
        
        B[p * n + p] = c * c * b_pp - 2.0f * c * s * b_pq + s * s * b_qq;
        B[q * n + q] = s * s * b_pp + 2.0f * c * s * b_pq + c * c * b_qq;
        B[p * n + q] = 0.0f;
        B[q * n + p] = 0.0f;
        
        // Update eigenvectors
        for (int i = 0; i < n; i++)
        {
            float v_ip = eigenvectors[i * n + p];
            float v_iq = eigenvectors[i * n + q];
            eigenvectors[i * n + p] = c * v_ip - s * v_iq;
            eigenvectors[i * n + q] = s * v_ip + c * v_iq;
        }
    }
    
    // Extract eigenvalues from diagonal
    for (int i = 0; i < n; i++)
    {
        eigenvalues[i] = B[i * n + i];
    }
    
    free(B);
    return TINY_OK;
}

/**
 * @brief Whitening: Z = D^(-1/2) * E^T * X
 * where D is eigenvalues, E is eigenvectors of covariance matrix
 */
static tiny_error_t whiten_data(const float *data, int rows, int cols, int num_sources, float *whitened, float *whitening_matrix)
{
    // Compute covariance matrix
    float *cov = (float *)malloc(rows * rows * sizeof(float));
    if (cov == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;
    
    tiny_error_t err = compute_covariance(data, rows, cols, cov);
    if (err != TINY_OK)
    {
        free(cov);
        return err;
    }
    
    // Eigenvalue decomposition
    float *eigenvalues = (float *)malloc(rows * sizeof(float));
    float *eigenvectors = (float *)malloc(rows * rows * sizeof(float));
    
    if (eigenvalues == NULL || eigenvectors == NULL)
    {
        free(cov);
        free(eigenvalues);
        free(eigenvectors);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    err = eigendecompose_symmetric(cov, rows, eigenvalues, eigenvectors, 1e-6f, 100);
    if (err != TINY_OK)
    {
        free(cov);
        free(eigenvalues);
        free(eigenvectors);
        return err;
    }
    
    // Compute whitening matrix: D^(-1/2) * E^T
    // Use only top num_sources components (largest eigenvalues)
    
    // Sort eigenvalues in descending order (simple bubble sort for small matrices)
    int *indices = (int *)malloc(rows * sizeof(int));
    if (indices == NULL)
    {
        free(cov);
        free(eigenvalues);
        free(eigenvectors);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    for (int i = 0; i < rows; i++)
        indices[i] = i;
    
    // Simple bubble sort
    for (int i = 0; i < rows - 1; i++)
    {
        for (int j = 0; j < rows - 1 - i; j++)
        {
            if (eigenvalues[indices[j]] < eigenvalues[indices[j + 1]])
            {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
    
    // Build whitening matrix: D^(-1/2) * E^T
    // whitening_matrix is (num_sources x rows)
    for (int i = 0; i < num_sources; i++)
    {
        int idx = indices[i];
        float lambda = eigenvalues[idx];
        if (lambda > 1e-10f) // Avoid division by zero
        {
            float scale = 1.0f / sqrtf(lambda);
            for (int j = 0; j < rows; j++)
            {
                whitening_matrix[i * rows + j] = scale * eigenvectors[j * rows + idx];
            }
        }
        else
        {
            // Zero eigenvalue, set row to zero
            for (int j = 0; j < rows; j++)
            {
                whitening_matrix[i * rows + j] = 0.0f;
            }
        }
    }
    
    // Apply whitening: whitened = whitening_matrix * data
    // whitening_matrix is (num_sources x rows), data is (rows x cols)
    // result is (num_sources x cols)
    err = tiny_mat_mult_f32(whitening_matrix, data, whitened, num_sources, rows, cols);
    
    free(cov);
    free(eigenvalues);
    free(eigenvectors);
    free(indices);
    
    return err;
}

/**
 * @brief Apply nonlinearity function
 */
static float apply_nonlinearity(float x, tiny_ica_nonlinearity_t type)
{
    switch (type)
    {
    case TINY_ICA_NONLINEARITY_TANH:
        return tanhf(x);
    case TINY_ICA_NONLINEARITY_EXP:
        return x * expf(-0.5f * x * x);
    case TINY_ICA_NONLINEARITY_CUBE:
        return x * x * x;
    default:
        return tanhf(x);
    }
}

/**
 * @brief Apply nonlinearity derivative
 */
static float apply_nonlinearity_derivative(float x, tiny_ica_nonlinearity_t type)
{
    switch (type)
    {
    case TINY_ICA_NONLINEARITY_TANH:
    {
        float t = tanhf(x);
        return 1.0f - t * t;
    }
    case TINY_ICA_NONLINEARITY_EXP:
        return (1.0f - x * x) * expf(-0.5f * x * x);
    case TINY_ICA_NONLINEARITY_CUBE:
        return 3.0f * x * x;
    default:
    {
        float t = tanhf(x);
        return 1.0f - t * t;
    }
    }
}

/**
 * @brief FastICA algorithm: extract one independent component
 */
static tiny_error_t fastica_extract_one(const float *whitened, int num_sources, int num_samples,
                                        float *w, tiny_ica_nonlinearity_t nonlinearity,
                                        int max_iter, float tolerance)
{
    // Initialize w randomly
    for (int i = 0; i < num_sources; i++)
    {
        w[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Normalize w
    float norm = 0.0f;
    for (int i = 0; i < num_sources; i++)
    {
        norm += w[i] * w[i];
    }
    norm = sqrtf(norm);
    if (norm < 1e-10f)
        return TINY_ERR_DSP_INVALID_PARAM;
    
    for (int i = 0; i < num_sources; i++)
    {
        w[i] /= norm;
    }
    
    // Iterate
    float *w_old = (float *)malloc(num_sources * sizeof(float));
    if (w_old == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;
    
    for (int iter = 0; iter < max_iter; iter++)
    {
        memcpy(w_old, w, num_sources * sizeof(float));
        
        // Compute w_new = E{g(w^T * x) * x} - E{g'(w^T * x)} * w
        float *w_new = (float *)calloc(num_sources, sizeof(float));
        if (w_new == NULL)
        {
            free(w_old);
            return TINY_ERR_DSP_MEMORY_ALLOC;
        }
        
        float mean_g_prime = 0.0f;
        
        for (int t = 0; t < num_samples; t++)
        {
            // Compute w^T * x_t
            float wx = 0.0f;
            for (int i = 0; i < num_sources; i++)
            {
                wx += w[i] * whitened[i * num_samples + t];
            }
            
            float g = apply_nonlinearity(wx, nonlinearity);
            float g_prime = apply_nonlinearity_derivative(wx, nonlinearity);
            
            for (int i = 0; i < num_sources; i++)
            {
                w_new[i] += g * whitened[i * num_samples + t];
            }
            
            mean_g_prime += g_prime;
        }
        
        mean_g_prime /= (float)num_samples;
        for (int i = 0; i < num_sources; i++)
        {
            w_new[i] /= (float)num_samples;
            w_new[i] -= mean_g_prime * w[i];
        }
        
        // Normalize
        norm = 0.0f;
        for (int i = 0; i < num_sources; i++)
        {
            norm += w_new[i] * w_new[i];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f)
        {
            for (int i = 0; i < num_sources; i++)
            {
                w[i] = w_new[i] / norm;
            }
        }
        
        free(w_new);
        
        // Check convergence
        float change = 0.0f;
        for (int i = 0; i < num_sources; i++)
        {
            float diff = w[i] - w_old[i];
            change += diff * diff;
        }
        change = sqrtf(change);
        
        if (change < tolerance)
            break;
    }
    
    free(w_old);
    return TINY_OK;
}

/**
 * @brief Orthogonalize w against previous components
 */
static void orthogonalize(float *w, const float *W, int num_components, int num_sources)
{
    // w = w - W^T * W * w
    for (int i = 0; i < num_components; i++)
    {
        // Compute dot product: W[i]^T * w
        float dot = 0.0f;
        for (int j = 0; j < num_sources; j++)
        {
            dot += W[i * num_sources + j] * w[j];
        }
        
        // Subtract projection: w = w - dot * W[i]
        for (int j = 0; j < num_sources; j++)
        {
            w[j] -= dot * W[i * num_sources + j];
        }
    }
    
    // Normalize
    float norm = 0.0f;
    for (int i = 0; i < num_sources; i++)
    {
        norm += w[i] * w[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-10f)
    {
        for (int i = 0; i < num_sources; i++)
        {
            w[i] /= norm;
        }
    }
}

/* ============================================================================
 * ICA IMPLEMENTATION
 * ============================================================================ */

tiny_error_t tiny_ica_separate_f32(const float *mixed_signals,
                                   int num_obs,
                                   int num_samples,
                                   int num_sources,
                                   float *separated_sources,
                                   tiny_ica_algorithm_t algorithm,
                                   tiny_ica_nonlinearity_t nonlinearity,
                                   int max_iter,
                                   float tolerance)
{
    if (mixed_signals == NULL || separated_sources == NULL)
        return TINY_ERR_DSP_NULL_POINTER;
    
    if (num_obs <= 0 || num_samples <= 0 || num_sources <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;
    
    if (num_sources > num_obs)
        return TINY_ERR_DSP_INVALID_PARAM; // Cannot extract more sources than observations
    
    if (algorithm != TINY_ICA_FASTICA)
        return TINY_ERR_NOT_SUPPORTED; // Only FastICA implemented
    
    // Default parameters
    if (max_iter <= 0)
        max_iter = 100;
    if (tolerance <= 0.0f)
        tolerance = 1e-4f;
    
    // Step 1: Center data
    float *mean = (float *)malloc(num_obs * sizeof(float));
    float *centered = (float *)malloc(num_obs * num_samples * sizeof(float));
    
    if (mean == NULL || centered == NULL)
    {
        free(mean);
        free(centered);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    compute_row_mean(mixed_signals, num_obs, num_samples, mean);
    center_data(mixed_signals, num_obs, num_samples, mean, centered);
    
    // Step 2: Whiten data
    float *whitening_matrix = (float *)malloc(num_sources * num_obs * sizeof(float));
    float *whitened = (float *)malloc(num_sources * num_samples * sizeof(float));
    
    if (whitening_matrix == NULL || whitened == NULL)
    {
        free(mean);
        free(centered);
        free(whitening_matrix);
        free(whitened);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    tiny_error_t err = whiten_data(centered, num_obs, num_samples, num_sources, whitened, whitening_matrix);
    if (err != TINY_OK)
    {
        free(mean);
        free(centered);
        free(whitening_matrix);
        free(whitened);
        return err;
    }
    
    // Step 3: FastICA - extract independent components
    float *W = (float *)malloc(num_sources * num_sources * sizeof(float));
    if (W == NULL)
    {
        free(mean);
        free(centered);
        free(whitening_matrix);
        free(whitened);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    float *w = (float *)malloc(num_sources * sizeof(float));
    if (w == NULL)
    {
        free(mean);
        free(centered);
        free(whitening_matrix);
        free(whitened);
        free(W);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    // Extract each component
    for (int comp = 0; comp < num_sources; comp++)
    {
        err = fastica_extract_one(whitened, num_sources, num_samples, w, nonlinearity, max_iter, tolerance);
        if (err != TINY_OK)
        {
            free(mean);
            free(centered);
            free(whitening_matrix);
            free(whitened);
            free(W);
            free(w);
            return err;
        }
        
        // Orthogonalize against previous components
        if (comp > 0)
        {
            orthogonalize(w, W, comp, num_sources);
        }
        
        // Store in W
        for (int i = 0; i < num_sources; i++)
        {
            W[comp * num_sources + i] = w[i];
        }
    }
    
    // Step 4: Compute separated sources: S = W * Z
    // W is (num_sources x num_sources), whitened is (num_sources x num_samples)
    // separated_sources is (num_sources x num_samples)
    err = tiny_mat_mult_f32(W, whitened, separated_sources, num_sources, num_sources, num_samples);
    
    free(mean);
    free(centered);
    free(whitening_matrix);
    free(whitened);
    free(W);
    free(w);
    
    return err;
}

tiny_error_t tiny_ica_init(tiny_ica_t *ica, int num_obs, int num_sources)
{
    if (ica == NULL)
        return TINY_ERR_DSP_NULL_POINTER;
    
    if (num_obs <= 0 || num_sources <= 0 || num_sources > num_obs)
        return TINY_ERR_DSP_INVALID_PARAM;
    
    ica->num_obs = num_obs;
    ica->num_sources = num_sources;
    
    ica->mixing_matrix = (float *)calloc(num_obs * num_sources, sizeof(float));
    ica->unmixing_matrix = (float *)calloc(num_sources * num_obs, sizeof(float));
    ica->whitening_matrix = (float *)calloc(num_sources * num_obs, sizeof(float));
    ica->mean = (float *)calloc(num_obs, sizeof(float));
    
    if (ica->mixing_matrix == NULL || ica->unmixing_matrix == NULL ||
        ica->whitening_matrix == NULL || ica->mean == NULL)
    {
        tiny_ica_deinit(ica);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    ica->initialized = 1;
    return TINY_OK;
}

tiny_error_t tiny_ica_fit(tiny_ica_t *ica,
                          const float *mixed_signals,
                          int num_samples,
                          tiny_ica_algorithm_t algorithm,
                          tiny_ica_nonlinearity_t nonlinearity,
                          int max_iter,
                          float tolerance)
{
    if (ica == NULL || mixed_signals == NULL)
        return TINY_ERR_DSP_NULL_POINTER;
    
    if (!ica->initialized)
        return TINY_ERR_DSP_UNINITIALIZED;
    
    if (num_samples <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;
    
    // Center data
    compute_row_mean(mixed_signals, ica->num_obs, num_samples, ica->mean);
    
    float *centered = (float *)malloc(ica->num_obs * num_samples * sizeof(float));
    if (centered == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;
    
    center_data(mixed_signals, ica->num_obs, num_samples, ica->mean, centered);
    
    // Whiten data
    float *whitened = (float *)malloc(ica->num_sources * num_samples * sizeof(float));
    if (whitened == NULL)
    {
        free(centered);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    tiny_error_t err = whiten_data(centered, ica->num_obs, num_samples, ica->num_sources, whitened, ica->whitening_matrix);
    if (err != TINY_OK)
    {
        free(centered);
        free(whitened);
        return err;
    }
    
    // FastICA - extract components
    float *W = (float *)malloc(ica->num_sources * ica->num_sources * sizeof(float));
    if (W == NULL)
    {
        free(centered);
        free(whitened);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    float *w = (float *)malloc(ica->num_sources * sizeof(float));
    if (w == NULL)
    {
        free(centered);
        free(whitened);
        free(W);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    for (int comp = 0; comp < ica->num_sources; comp++)
    {
        err = fastica_extract_one(whitened, ica->num_sources, num_samples, w, nonlinearity, max_iter, tolerance);
        if (err != TINY_OK)
        {
            free(centered);
            free(whitened);
            free(W);
            free(w);
            return err;
        }
        
        if (comp > 0)
        {
            orthogonalize(w, W, comp, ica->num_sources);
        }
        
        for (int i = 0; i < ica->num_sources; i++)
        {
            W[comp * ica->num_sources + i] = w[i];
        }
    }
    
    // Store unmixing matrix: W_unmix = W * whitening_matrix
    err = tiny_mat_mult_f32(W, ica->whitening_matrix, ica->unmixing_matrix,
                            ica->num_sources, ica->num_sources, ica->num_obs);
    
    // Compute mixing matrix (pseudo-inverse of unmixing matrix)
    // For now, use simple approach: A = (W_unmix^T * W_unmix)^(-1) * W_unmix^T
    // Simplified: assume square and use transpose
    // TODO: Implement proper pseudo-inverse
    
    free(centered);
    free(whitened);
    free(W);
    free(w);
    
    return err;
}

tiny_error_t tiny_ica_transform(const tiny_ica_t *ica,
                                const float *mixed_signals,
                                int num_samples,
                                float *separated_sources)
{
    if (ica == NULL || mixed_signals == NULL || separated_sources == NULL)
        return TINY_ERR_DSP_NULL_POINTER;
    
    if (!ica->initialized)
        return TINY_ERR_DSP_UNINITIALIZED;
    
    if (num_samples <= 0)
        return TINY_ERR_DSP_INVALID_PARAM;
    
    // Center data
    float *centered = (float *)malloc(ica->num_obs * num_samples * sizeof(float));
    if (centered == NULL)
        return TINY_ERR_DSP_MEMORY_ALLOC;
    
    center_data(mixed_signals, ica->num_obs, num_samples, ica->mean, centered);
    
    // Apply whitening
    float *whitened = (float *)malloc(ica->num_sources * num_samples * sizeof(float));
    if (whitened == NULL)
    {
        free(centered);
        return TINY_ERR_DSP_MEMORY_ALLOC;
    }
    
    tiny_error_t err = tiny_mat_mult_f32(ica->whitening_matrix, centered, whitened,
                                        ica->num_sources, ica->num_obs, num_samples);
    if (err != TINY_OK)
    {
        free(centered);
        free(whitened);
        return err;
    }
    
    // Apply unmixing matrix
    // Extract W from unmixing_matrix (W = unmixing_matrix * whitening_matrix^(-1))
    // For simplicity, use unmixing_matrix directly on whitened data
    // Actually, unmixing_matrix = W * whitening_matrix, so we need to extract W
    // Simplified: use unmixing_matrix on centered data directly
    err = tiny_mat_mult_f32(ica->unmixing_matrix, centered, separated_sources,
                            ica->num_sources, ica->num_obs, num_samples);
    
    free(centered);
    free(whitened);
    
    return err;
}

tiny_error_t tiny_ica_deinit(tiny_ica_t *ica)
{
    if (ica == NULL)
        return TINY_ERR_DSP_NULL_POINTER;
    
    if (ica->initialized)
    {
        free(ica->mixing_matrix);
        free(ica->unmixing_matrix);
        free(ica->whitening_matrix);
        free(ica->mean);
        
        ica->mixing_matrix = NULL;
        ica->unmixing_matrix = NULL;
        ica->whitening_matrix = NULL;
        ica->mean = NULL;
        ica->num_obs = 0;
        ica->num_sources = 0;
        ica->initialized = 0;
    }
    
    return TINY_OK;
}


```
