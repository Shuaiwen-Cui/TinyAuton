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

